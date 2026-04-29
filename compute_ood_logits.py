#!/usr/bin/env python3
# DEPRECATED: OOD computation has been merged into compute_curve_logits.py.
# That script now handles both ID and OOD in a single pass and is the
# canonical entry point.  This file is kept for reference only.
import warnings

warnings.warn(
    "compute_ood_logits.py is deprecated. Use compute_curve_logits.py instead.",
    DeprecationWarning,
    stacklevel=1,
)

import argparse
import copy
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
    setup_metrics,
)
from src.utils import load_checkpoint_new

# Configure environment
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

LOGITS_DIR = "funct_smooth/logits_backup"
os.makedirs(LOGITS_DIR, exist_ok=True)
OOD_DATA_PATH = "funct_smooth/my_ood_data.yaml"
MODEL_NAME = "Qwen/Qwen2.5-7B"


def get_model_name(run):
    mp = run.config.get("model_params", {})
    n_curve_segment = mp.get("num_curve_segment", 0)
    curve_degree = mp.get("SegDeg", 0)
    is_fixed = mp.get("Pretraining", False)
    n = n_curve_segment + 1
    m = curve_degree - 1
    prefix = "ALC" if is_fixed else "FLC"
    return f"{prefix}({n},{m})"


def load_ood_data(tokenizer):
    with open(OOD_DATA_PATH, "r") as f:
        data = yaml.safe_load(f)

    prompts = [ex["prompt"] for ex in data["examples"]]
    labels = [ex["label"] for ex in data["examples"]]

    # Tokenize prompts
    encodings = tokenizer(prompts, padding=True, return_tensors="np")
    x_test = jnp.array(encodings["input_ids"])
    attn_test = jnp.array(encodings["attention_mask"])

    return x_test, attn_test, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()

    api = wandb.Api()
    project = "qwen_PtoC_newDS"
    entity = "ddold"

    print(f"Loading tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading OOD data from {OOD_DATA_PATH}...")
    x_ood, attn_ood, labels_ood = load_ood_data(tokenizer)
    print(f"OOD data loaded. Shape: {x_ood.shape}")

    for run_id in args.run_ids:
        print(f"\nProcessing run {run_id}...")
        try:
            # Setup directory for artifacts
            local_path = os.path.abspath("artifacts")
            os.makedirs(local_path, exist_ok=True)

            run_path = f"{entity}/{project}/{run_id}"
            run = api.run(run_path)

            # Use target_token_ids from run config if available, otherwise try to load from dataset artifact
            target_token_ids = run.config.get("net_kwargs", {}).get("target_token_ids")
            if target_token_ids is not None:
                target_token_ids = jnp.array(target_token_ids)
            else:
                print(
                    f"target_token_ids not found in config for {run_id}, attempting to load from dataset artifact..."
                )
                ds_path_ref = run.config.get("data", {}).get("dataset_path")
                if not ds_path_ref:
                    print(f"Error: dataset_path not found in run {run_id} config.")
                    continue

                # We need to resolve the artifact path
                if ":" not in ds_path_ref:
                    ds_path_ref = f"{ds_path_ref.split('/')[-1]}:v0"
                else:
                    ds_path_ref = ds_path_ref.split("/")[-1]

                try:
                    art = api.artifact(f"{entity}/{project}/{ds_path_ref}")
                    ds_dir = art.download(local_path)
                    test_data_path = os.path.join(ds_dir, "test_data.npz")
                    with np.load(test_data_path) as d_np:
                        target_token_ids = jnp.array(d_np["target_ids"])
                    print(
                        f"Successfully loaded target_token_ids from artifact: {target_token_ids}"
                    )
                except Exception as e:
                    print(f"Error loading target_token_ids from artifact: {e}")
                    continue

            model_name = get_model_name(run)
            # Extract dataset name for new naming convention
            ds_name = run.config.get("data", {}).get("dataset_path", "").split("/")[-1]
            clean_ds = clean_dataset_name(ds_name)
            filename = f"{clean_ds}_seed_{run.config.get('rng_seed', 1)}_{model_name}_{run.id}_ood.npz"
            filepath = os.path.join(LOGITS_DIR, filename)

            if os.path.exists(filepath):
                print(f"Skipping {filename}, already exists.")
                continue

            # Ensure base model exists
            cfg_dict = run.config.copy()
            cfg_dict["net_kwargs"]["model_path"] = os.path.join(
                local_path, "qwen2.5_7B_bfloat16:v0"
            )

            # Check if base model artifact exists
            if not any("qwen2.5_7B_bfloat16" in d for d in os.listdir(local_path)):
                print("Downloading base model artifact...")
                art = api.artifact(f"{entity}/{project}/qwen2.5_7B_bfloat16:v0")
                art.download(local_path)

            # Create Config object
            # Note: _setup_training_env expects a DataSplits object.
            # We'll pass dummy data since we are doing manual inference.
            data_splits = DataSplits(
                train_input_ids=x_ood,  # Need something for from_config_dict assertion
                train_attention_mask=attn_ood,
                train_labels=jnp.array(
                    [-1] * len(labels_ood)
                ),  # Dummy labels as jnp array
                test_input_ids=x_ood,
                test_attention_mask=attn_ood,
                test_labels=jnp.array([-1] * len(labels_ood)),
            )
            config = Config.from_dict(cfg_dict, data=data_splits)
            config.net_kwargs["target_token_ids"] = target_token_ids

            # Setup environment
            rng_key = jax.random.PRNGKey(run.config.get("rng_seed", 1))
            env, params, rng_key = _setup_training_env(config, data_splits, rng_key)

            # Load checkpoint
            params, _, _, _, pretrained_params, _ = load_checkpoint_new(
                run, params, env.s_model
            )
            params = env.s_model.set_lora_rho(rho_s=0.0, rho_w=0.0, params=params)

            # Identify ALC models (Pretraining=True) and compute Deep Ensemble (DE) logits
            is_alc = run.config.get("model_params", {}).get("Pretraining", False)
            if is_alc and pretrained_params is not None:
                print("Computing DE logits for ALC model on OOD data...")
                # Filter out False entries
                valid_pts = [p for p in pretrained_params if p is not False]
                if valid_pts:
                    # Save pretrained params and env info before clearing to free memory
                    stored_valid_pts = valid_pts
                    de_logits_list = []
                    with config.as_pretrain() as pt_config:
                        # Re-initialize environment for pretrain mode
                        env_pt, params_pt, rng_key = _setup_training_env(
                            pt_config, data_splits, rng_key
                        )
                        # Clear old params and env to save memory
                        params = None
                        env = None
                        import gc

                        gc.collect()

                        params_pt = env_pt.s_model.set_lora_rho(
                            rho_s=0.0, rho_w=0.0, params=params_pt
                        )
                        # n_samples=1 for discrete logits
                        # Use a small batch size to avoid OOM
                        de_pred_fn = setup_metrics(
                            env_pt.s_model,
                            4,  # Fixed batch size
                            x_ood,
                            attn_ood,
                            jnp.array(
                                [-1] * len(labels_ood)
                            ),  # Dummy labels for metrics
                            n_samples=1,
                            use_linspace=True,
                        )

                        for pt_idx, pt_p in enumerate(stored_valid_pts):
                            print(
                                f"OOD inference for checkpoint {pt_idx + 1}/{len(stored_valid_pts)}..."
                            )
                            # Inject checkpoint into parameters using train_mask
                            new_inner_params = jax.tree.map(
                                lambda m, p, pl: pl if m else p,
                                env_pt.s_model.train_mask,
                                params_pt["params"],
                                pt_p,
                            )
                            pt_params = {"params": new_inner_params}
                            # Run inference
                            _, pt_logits = de_pred_fn(jax.random.PRNGKey(0), pt_params)
                            de_logits_list.append(np.array(pt_logits))

                    # Concatenate into (N_checkpoints, n_valid, n_classes)
                    de_logits = np.concatenate(de_logits_list, axis=0)

                    de_filename = f"{clean_ds}_seed_{run.config.get('rng_seed', 1)}_DE({len(stored_valid_pts)},{config.model_params.SegDeg - 1})_{run.id}_ood.npz"
                    de_filepath = os.path.join(LOGITS_DIR, de_filename)
                    # Convert labels to object array to preserve list structure in npz
                    labels_to_save = np.array(labels_ood, dtype=object)
                    np.savez(
                        de_filepath,
                        logits=de_logits,
                        labels=labels_to_save,
                        n_checkpoints=len(stored_valid_pts),
                        wandb_path=run.path,
                    )
                    print(f"Saved DE OOD logits to {de_filepath}")

                # RE-INITIALIZE main env and params for curve prediction after clearing them
                print("Re-initializing main environment for curve prediction...")
                env, params, rng_key = _setup_training_env(config, data_splits, rng_key)
                params, _, _, _, _, _ = load_checkpoint_new(run, params, env.s_model)
                params = env.s_model.set_lora_rho(rho_s=0.0, rho_w=0.0, params=params)

            print(f"Setting up metrics for {len(labels_ood)} examples...")
            pred_fn = setup_metrics(
                env.s_model,
                4,  # Fixed batch size
                x_ood,
                attn_ood,
                jnp.array([-1] * len(labels_ood)),  # Dummy labels for metrics
                n_samples=args.n_samples,
                use_linspace=True,
            )

            print("Running prediction...")
            _, logits = pred_fn(jax.random.PRNGKey(0), params)

            # Convert labels to object array to preserve list structure in npz
            labels_to_save = np.array(labels_ood, dtype=object)

            np.savez(
                filepath,
                logits=np.array(logits),
                labels=labels_to_save,
                n_samples=args.n_samples,
                t_max=float(env.s_model.t_max),
                wandb_path=run.path,
            )
            print(f"Saved {filepath}")

        except Exception as e:
            print(f"Error processing run {run_id}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
