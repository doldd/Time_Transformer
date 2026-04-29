#!/usr/bin/env python3
import argparse
import copy
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from qwen_fineTuning_PtoC_fixCps import (
    Config,
    _setup_training_env,
    _prepare_datasets,
    setup_metrics,
)
from src.utils import load_checkpoint_new

# Configure environment
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

LOGITS_DIR = "funct_smooth/logits_backup"
os.makedirs(LOGITS_DIR, exist_ok=True)


def clean_dataset_name(ds_name):
    return "_".join(ds_name.split(".")[0].split("_")[:-1])


def get_config_diff(base, current):
    """Recursively find differences between two nested dictionaries."""
    diff = {}
    for k, v in current.items():
        if k not in base:
            diff[k] = v
        elif isinstance(v, dict) and isinstance(base[k], dict):
            nested_diff = get_config_diff(base[k], v)
            if nested_diff:
                diff[k] = nested_diff
        elif base[k] != v:
            diff[k] = v
    # Also check if keys were removed
    for k in base:
        if k not in current:
            diff[k] = "<REMOVED>"
    return diff


def get_model_name(run):
    mp = run.config.get("model_params", {})
    n_curve_segment = mp.get("num_curve_segment", 0)
    curve_degree = mp.get("SegDeg", 0)
    is_fixed = mp.get("Pretraining", False)
    n = n_curve_segment + 1
    m = curve_degree - 1
    prefix = "ALC" if is_fixed else "FLC"
    return f"{prefix}({n},{m})"


def load_config_and_data(run, local_path="artifacts"):
    run.config["net_kwargs"]["model_path"] = f"{local_path}/qwen2.5_7B_bfloat16:v0"
    run.config["data"]["dataset_path"] = (
        f"{local_path}/{run.config['data']['dataset_path'].split('/')[-1]}"
    )
    rng_key = jax.random.PRNGKey(0)
    data, unique_target_ids, rng_key = _prepare_datasets(
        rng_key,
        dataset_path=run.config["data"]["dataset_path"],
        val_percentage=run.config["data"].get("val_percentage", 0.0),
        batch_size=run.config["train_hyper"]["batch_size"],
        smoke_test=False,
        logger=False,
    )
    config = Config.from_dict(run.config, data=data)
    config.net_kwargs["target_token_ids"] = unique_target_ids
    return config, data, rng_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--num-segments", type=int, default=4)
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=None,
        help="Optional list of specific run IDs to process",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional filter for dataset name (e.g. ARC-Challenge)",
    )
    args = parser.parse_args()

    api = wandb.Api()
    project = "qwen_PtoC_newDS"
    entity = "ddold"

    TARGET_SEED = 1
    TARGET_LORA_RHO = 0.25
    SUB_MODEL = "lora_category"
    TARGET_NUM_SEGMENTS = args.num_segments
    SEG_DEGS = [1, 2, 3]

    print(f"Scanning project {entity}/{project}...")
    all_runs = api.runs(f"{entity}/{project}")

    runs_to_process = []
    base_config = None

    print("First pass: Discovering runs...")
    for run in all_runs:
        cfg = run.config
        if (
            cfg.get("rng_seed") != TARGET_SEED
            or cfg.get("model_params", {}).get("lora_params", {}).get("lora_rho")
            != TARGET_LORA_RHO
            or cfg.get("model_params", {}).get("num_curve_segment")
            != TARGET_NUM_SEGMENTS
            or cfg.get("model_params", {}).get("subspace_model") != SUB_MODEL
            or cfg.get("model_params", {}).get("SegDeg") not in SEG_DEGS
        ):
            continue

        ds_name = cfg.get("data", {}).get("dataset_path", "").split("/")[-1]
        if not ds_name:
            continue

        clean_ds = clean_dataset_name(ds_name)
        if args.dataset and args.dataset not in clean_ds:
            continue

        if args.run_ids and run.id not in args.run_ids:
            continue

        model_name = get_model_name(run)
        filename = f"{clean_ds}_seed_{TARGET_SEED}_{model_name}_{run.id}.npz"
        filepath = os.path.join(LOGITS_DIR, filename)

        if "ALC(" in model_name:
            filename2 = f"{clean_ds}_seed_{TARGET_SEED}_{model_name.replace('ALC', 'DE')}_{run.id}.npz"
            filepath2 = os.path.join(LOGITS_DIR, filename2)
            if os.path.exists(filepath) and os.path.exists(filepath2):
                # print(f"Skipping {filename}, already exists.")
                continue
        else:
            if os.path.exists(filepath):
                # print(f"Skipping {filename}, already exists.")
                continue

        sweep_id = run.sweep.id if run.sweep else "None"
        print(
            f"Found run to process: {run.id} | Sweep: {sweep_id} | Model: {model_name} | Dataset: {clean_ds}"
        )

        # Sanitize config for printing
        clean_cfg = copy.deepcopy(run.config)
        clean_cfg.pop("_wandb", None)
        if "model_params" in clean_cfg:
            m_params = clean_cfg["model_params"]
            if isinstance(m_params, dict):
                m_params.pop("filter_masks", None)
                if "lora_params" in m_params and isinstance(
                    m_params["lora_params"], dict
                ):
                    m_params["lora_params"].pop("filter_masks", None)

        if base_config is None:
            base_config = clean_cfg
            print(
                f"Base Config (first matching run): {json.dumps(clean_cfg, indent=2)}\n"
            )
        else:
            diff = get_config_diff(base_config, clean_cfg)
            if diff:
                print(f"Config Diff vs Base: {json.dumps(diff, indent=2)}\n")
            else:
                print("Config matches Base.\n")

        runs_to_process.append((run, ds_name, clean_ds, filepath))

    print(f"\nTotal runs found to process: {len(runs_to_process)}")

    if args.n_samples == 0:
        print(
            "Dry run complete. No runs will be processed. Use --n-samples > 0 to compute logits."
        )
        return

    print("\nSecond pass: Processing runs...")
    for run, ds_name, clean_ds, filepath in runs_to_process:
        print(f"Processing {run.id} for dataset {ds_name}...")
        try:
            # Download artifacts if needed
            local_path = os.path.abspath("artifacts")
            os.makedirs(local_path, exist_ok=True)
            dataset_artifact = ds_name if ":" in ds_name else f"{ds_name}:v0"

            # Check for existing qwen model and dataset
            qwen_found = any("qwen2.5_7B_bfloat16" in d for d in os.listdir(local_path))
            ds_found = any(clean_ds in d for d in os.listdir(local_path))

            if not qwen_found:
                print(f"Downloading artifact qwen2.5_7B_bfloat16:v0...")
                art = api.artifact(f"ddold/{project}/qwen2.5_7B_bfloat16:v0")
                art.download(local_path)
            else:
                print("Qwen artifact found, skipping download.")

            if not ds_found:
                print(f"Downloading artifact {dataset_artifact}...")
                art = api.artifact(f"ddold/{project}/{dataset_artifact}")
                art.download(local_path)
            else:
                print(f"Dataset artifact for {clean_ds} found, skipping download.")

            config, data, rng_key = load_config_and_data(run, local_path=local_path)
            print("Config and data loaded.")
            
            x_test, attn_test, labels = data.get("test")
            num_test = len(labels)

            # Identify ALC models (Pretraining=True) and compute Deep Ensemble (DE) logits
            is_alc = run.config.get("model_params", {}).get("Pretraining", False)

            env, params, rng_key = _setup_training_env(config, data, rng_key)
            params, _, _, _, pretrained_params, _ = load_checkpoint_new(run, params, env.s_model)
            if is_alc and pretrained_params is not None:
                print("Computing DE logits for ALC model...")
                # Filter out False entries
                valid_pts = [p for p in pretrained_params if p is not False]
                if valid_pts:
                    de_logits_list = []
                    with config.as_pretrain() as pt_config:
                        # Re-initialize environment for pretrain mode
                        env, params, rng_key = _setup_training_env(
                            pt_config, data, rng_key
                        )
                        params = env.s_model.set_lora_rho(
                            rho_s=0.0, rho_w=0.0, params=params
                        )
                        # n_samples=1 for discrete logits
                        # Use a small batch size to avoid OOM
                        de_pred_fn = setup_metrics(
                            env.s_model,
                            4,
                            x_test,
                            attn_test,
                            labels,
                            n_samples=1,
                            use_linspace=True,
                        )

                        for pt_idx, pt_p in enumerate(valid_pts):
                            print(
                                f"Inference for checkpoint {pt_idx + 1}/{len(valid_pts)}..."
                            )
                            # Inject checkpoint into parameters using train_mask
                            params['params'] = jax.tree.map(
                                lambda m, p, pl: pl if m else p,
                                env.s_model.train_mask,
                                params["params"],
                                pt_p,
                            )
                            # Run inference
                            _, pt_logits = de_pred_fn(jax.random.PRNGKey(0), params)
                            # pt_logits shape: (1, n_valid, n_classes)
                            de_logits_list.append(np.array(pt_logits))

                    # Concatenate into (N_checkpoints, n_valid, n_classes)
                    de_logits = np.concatenate(de_logits_list, axis=0)
                    
                    de_filepath = filepath.replace("ALC", "DE")
                    np.savez(
                        de_filepath,
                        logits=de_logits,
                        labels=np.array(labels),
                        n_checkpoints=len(valid_pts),
                        wandb_path=run.path,
                    )
                    print(f"Saved DE logits to {de_filepath}")

            # Revert manual chunking and use setup_metrics directly with a fixed small batch size

            print("Training curve env setup.")
            env, params, rng_key = _setup_training_env(config, data, rng_key)
            params, _, _, _, _, _ = load_checkpoint_new(run, params, env.s_model)
            params = env.s_model.set_lora_rho(rho_s=0.0, rho_w=0.0, params=params)

            print(f"Setting up metrics for {num_test} examples...")
            pred_fn = setup_metrics(
                env.s_model,
                4,  # Fixed small batch size to avoid OOM
                x_test,
                attn_test,
                labels,
                n_samples=args.n_samples,
                use_linspace=True,
            )

            print("Running prediction...")
            # To avoid timeout, we might need to process in smaller chunks or use a more efficient scan.
            # But setup_metrics is already scanning. Let's try to see if it's just the first compilation that is slow.
            _, logits = pred_fn(jax.random.PRNGKey(0), params)

            # Check for NaNs
            if np.any(np.isnan(logits)):
                print("Warning: NaNs detected in logits!")

            print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")

            np.savez(
                filepath,
                logits=np.array(logits),
                labels=np.array(labels),
                n_samples=args.n_samples,
                t_max=float(env.s_model.t_max),
                wandb_path=run.path,
            )
            print(f"Saved {filepath}")

        except Exception as e:
            print(f"Error processing run {run.id}: {e}")


if __name__ == "__main__":
    main()
