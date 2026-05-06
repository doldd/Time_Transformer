#!/usr/bin/env python3
import argparse
import copy
import gc
import json
import os
import yaml

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from transformers import AutoTokenizer
from qwen_fineTuning_PtoC_fixCps import (
    Config,
    DataSplits,
    _setup_training_env,
    _prepare_datasets,
    setup_metrics,
)
from src.utils import load_checkpoint_new

# Configure environment
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

LOGITS_DIR = "dataset/logits_backup"
os.makedirs(LOGITS_DIR, exist_ok=True)

OOD_DATA_PATH = "dataset/my_ood_data.yaml"
MODEL_NAME = "Qwen/Qwen2.5-7B"


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
    run.config["data"][
        "dataset_path"
    ] = f"{local_path}/{run.config['data']['dataset_path'].split('/')[-1]}"
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


def load_ood_data(tokenizer):
    """Load and tokenize OOD prompts/labels from the YAML file."""
    with open(OOD_DATA_PATH, "r") as f:
        data = yaml.safe_load(f)

    prompts = [ex["prompt"] for ex in data["examples"]]
    labels = [ex["label"] for ex in data["examples"]]

    encodings = tokenizer(prompts, padding=True, return_tensors="np")
    x_ood = jnp.array(encodings["input_ids"])
    attn_ood = jnp.array(encodings["attention_mask"])

    return x_ood, attn_ood, labels


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
    entity = "[name]"

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

        filepath_ood = filepath.replace(".npz", "_ood.npz")
        if "ALC(" in model_name:
            de_filepath = os.path.join(
                LOGITS_DIR,
                f"{clean_ds}_seed_{TARGET_SEED}_{model_name.replace('ALC', 'DE')}_{run.id}.npz",
            )
            de_filepath_ood = de_filepath.replace(".npz", "_ood.npz")
            if all(
                os.path.exists(f)
                for f in [filepath, filepath_ood, de_filepath, de_filepath_ood]
            ):
                continue
        else:
            if os.path.exists(filepath) and os.path.exists(filepath_ood):
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

    # Lazy-load OOD data once if any run needs it
    needs_ood_globally = any(
        not os.path.exists(fp.replace(".npz", "_ood.npz"))
        for _, _, _, fp in runs_to_process
    )
    x_ood = attn_ood = labels_ood = dummy_labels_ood = None
    if needs_ood_globally:
        print(f"\nLoading OOD tokenizer ({MODEL_NAME})...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Loading OOD data from {OOD_DATA_PATH}...")
        x_ood, attn_ood, labels_ood = load_ood_data(tokenizer)
        dummy_labels_ood = jnp.array([-1] * len(labels_ood))
        print(f"OOD data loaded. Shape: {x_ood.shape}")

    print("\nSecond pass: Processing runs...")
    for run, ds_name, clean_ds, filepath in runs_to_process:
        print(f"Processing {run.id} for dataset {ds_name}...")
        try:
            # ------------------------------------------------------------------ #
            # Artifact download
            # ------------------------------------------------------------------ #
            local_path = os.path.abspath("artifacts")
            os.makedirs(local_path, exist_ok=True)
            dataset_artifact = ds_name if ":" in ds_name else f"{ds_name}:v0"

            qwen_found = any("qwen2.5_7B_bfloat16" in d for d in os.listdir(local_path))
            ds_found = any(clean_ds in d for d in os.listdir(local_path))

            if not qwen_found:
                print("Downloading artifact qwen2.5_7B_bfloat16:v0...")
                art = api.artifact(f"[name]/{project}/qwen2.5_7B_bfloat16:v0")
                art.download(local_path)
            else:
                print("Qwen artifact found, skipping download.")

            if not ds_found:
                print(f"Downloading artifact {dataset_artifact}...")
                art = api.artifact(f"[name]/{project}/{dataset_artifact}")
                art.download(local_path)
            else:
                print(f"Dataset artifact for {clean_ds} found, skipping download.")

            # ------------------------------------------------------------------ #
            # Load config / ID data (needed for OOD too — provides target_token_ids)
            # ------------------------------------------------------------------ #
            config, data, rng_key = load_config_and_data(run, local_path=local_path)
            print("Config and data loaded.")

            x_test, attn_test, labels = data.get("test")
            is_alc = run.config.get("model_params", {}).get("Pretraining", False)

            # Determine what still needs computing
            filepath_ood = filepath.replace(".npz", "_ood.npz")
            need_id = not os.path.exists(filepath)
            need_ood = not os.path.exists(filepath_ood)
            need_de_id = need_de_ood = False
            de_filepath = de_filepath_ood = None
            if is_alc:
                de_filepath = filepath.replace("ALC", "DE")
                de_filepath_ood = de_filepath.replace(".npz", "_ood.npz")
                need_de_id = not os.path.exists(de_filepath)
                need_de_ood = not os.path.exists(de_filepath_ood)

            # Build OOD DataSplits if needed (reuses pre-loaded OOD arrays)
            data_splits_ood = None
            if need_ood or need_de_ood:
                data_splits_ood = DataSplits(
                    train_input_ids=x_ood,
                    train_attention_mask=attn_ood,
                    train_labels=dummy_labels_ood,
                    test_input_ids=x_ood,
                    test_attention_mask=attn_ood,
                    test_labels=dummy_labels_ood,
                )

            # ------------------------------------------------------------------ #
            # DE block (ALC only) — single checkpoint loop, collect ID + OOD
            # ------------------------------------------------------------------ #
            env = params = None
            if is_alc and (need_de_id or need_de_ood):
                print("Loading checkpoint for DE computation...")
                env, params, rng_key = _setup_training_env(config, data, rng_key)
                params, _, _, _, pretrained_params, _ = load_checkpoint_new(
                    run, params, env.s_model
                )
                valid_pts = (
                    [p for p in pretrained_params if p is not False]
                    if pretrained_params is not None
                    else []
                )

                if valid_pts:
                    de_logits_id_list = [] if need_de_id else None
                    de_logits_ood_list = [] if need_de_ood else None

                    with config.as_pretrain() as pt_config:
                        env_pt, params_pt, rng_key = _setup_training_env(
                            pt_config, data, rng_key
                        )
                        params_pt = env_pt.s_model.set_lora_rho(
                            rho_s=0.0, rho_w=0.0, params=params_pt
                        )

                        if need_de_id:
                            de_pred_fn_id = setup_metrics(
                                env_pt.s_model,
                                4,
                                x_test,
                                attn_test,
                                labels,
                                n_samples=1,
                                use_linspace=True,
                            )
                        if need_de_ood:
                            de_pred_fn_ood = setup_metrics(
                                env_pt.s_model,
                                4,
                                x_ood,
                                attn_ood,
                                dummy_labels_ood,
                                n_samples=1,
                                use_linspace=True,
                            )

                        for pt_idx, pt_p in enumerate(valid_pts):
                            print(
                                f"DE inference for checkpoint"
                                f" {pt_idx + 1}/{len(valid_pts)}..."
                            )
                            params_pt["params"] = jax.tree.map(
                                lambda m, p, pl: pl if m else p,
                                env_pt.s_model.train_mask,
                                params_pt["params"],
                                pt_p,
                            )
                            if need_de_id:
                                _, pt_logits = de_pred_fn_id(
                                    jax.random.PRNGKey(0), params_pt
                                )
                                de_logits_id_list.append(np.array(pt_logits))
                            if need_de_ood:
                                _, pt_logits_ood = de_pred_fn_ood(
                                    jax.random.PRNGKey(0), params_pt
                                )
                                de_logits_ood_list.append(np.array(pt_logits_ood))

                    if need_de_id:
                        de_logits = np.concatenate(de_logits_id_list, axis=0)
                        np.savez(
                            de_filepath,
                            logits=de_logits,
                            labels=np.array(labels),
                            n_checkpoints=len(valid_pts),
                            wandb_path=run.path,
                        )
                        print(f"Saved DE logits to {de_filepath}")

                    if need_de_ood:
                        de_logits_ood = np.concatenate(de_logits_ood_list, axis=0)
                        labels_ood_save = np.empty(len(labels_ood), dtype=object)
                        for i, lbl in enumerate(labels_ood):
                            labels_ood_save[i] = lbl
                        np.savez(
                            de_filepath_ood,
                            logits=de_logits_ood,
                            labels=labels_ood_save,
                            n_checkpoints=len(valid_pts),
                            wandb_path=run.path,
                        )
                        print(f"Saved DE OOD logits to {de_filepath_ood}")

                # Clear DE state before curve passes
                env = params = None
                try:
                    env_pt = params_pt = None
                    de_pred_fn_id = de_pred_fn_ood = None
                except NameError:
                    pass
                gc.collect()

            # ------------------------------------------------------------------ #
            # Curve ID block
            # ------------------------------------------------------------------ #
            if need_id:
                print("Setting up curve ID environment...")
                env, params, rng_key = _setup_training_env(config, data, rng_key)
                params, _, _, _, _, _ = load_checkpoint_new(run, params, env.s_model)
                params = env.s_model.set_lora_rho(rho_s=0.0, rho_w=0.0, params=params)

                print(f"Running curve ID prediction ({len(labels)} examples)...")
                pred_fn = setup_metrics(
                    env.s_model,
                    4,
                    x_test,
                    attn_test,
                    labels,
                    n_samples=args.n_samples,
                    use_linspace=True,
                )
                _, logits = pred_fn(jax.random.PRNGKey(0), params)

                if np.any(np.isnan(logits)):
                    print("Warning: NaNs detected in ID logits!")
                print(f"ID logits shape: {logits.shape}")

                np.savez(
                    filepath,
                    logits=np.array(logits),
                    labels=np.array(labels),
                    n_samples=args.n_samples,
                    t_max=float(env.s_model.t_max),
                    wandb_path=run.path,
                )
                print(f"Saved {filepath}")

                if need_ood:
                    env = params = pred_fn = None
                    gc.collect()

            # ------------------------------------------------------------------ #
            # Curve OOD block
            # ------------------------------------------------------------------ #
            if need_ood:
                print("Setting up curve OOD environment...")
                env, params, rng_key = _setup_training_env(
                    config, data_splits_ood, rng_key
                )
                params, _, _, _, _, _ = load_checkpoint_new(run, params, env.s_model)
                params = env.s_model.set_lora_rho(rho_s=0.0, rho_w=0.0, params=params)

                print(f"Running curve OOD prediction ({len(labels_ood)} examples)...")
                pred_fn = setup_metrics(
                    env.s_model,
                    4,
                    x_ood,
                    attn_ood,
                    dummy_labels_ood,
                    n_samples=args.n_samples,
                    use_linspace=True,
                )
                _, logits_ood = pred_fn(jax.random.PRNGKey(0), params)

                if np.any(np.isnan(logits_ood)):
                    print("Warning: NaNs detected in OOD logits!")
                print(f"OOD logits shape: {logits_ood.shape}")

                labels_ood_save = np.empty(len(labels_ood), dtype=object)
                for i, lbl in enumerate(labels_ood):
                    labels_ood_save[i] = lbl

                np.savez(
                    filepath_ood,
                    logits=np.array(logits_ood),
                    labels=labels_ood_save,
                    n_samples=args.n_samples,
                    t_max=float(env.s_model.t_max),
                    wandb_path=run.path,
                )
                print(f"Saved {filepath_ood}")

                env = params = pred_fn = None
                gc.collect()

        except Exception as e:
            print(f"Error processing run {run.id}: {e}")


if __name__ == "__main__":
    main()
