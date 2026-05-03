#!/usr/bin/env python3
import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
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
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

LOGITS_DIR = "funct_smooth/logits_backup"
os.makedirs(LOGITS_DIR, exist_ok=True)


def clean_dataset_name(ds_name):
    return "_".join(ds_name.split(".")[0].split("_")[:-1])


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
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional filter for dataset name (e.g. ARC-Challenge)",
    )
    args = parser.parse_args()

    api = wandb.Api()
    project = "qwen_PtoC_newDS"
    sweep_ids = ["w0kaq8i0", "we8wq8n7"]

    TARGET_SEED = 1
    TARGET_LORA_RHO = 0.25
    TARGET_NUM_SEGMENTS = 4

    for sweep_id in sweep_ids:
        print(f"Checking sweep {sweep_id}...")
        try:
            sweep = api.sweep(f"ddold/{project}/{sweep_id}")
            # Access runs once to avoid multiple API calls in loop
            run_list = [r for r in sweep.runs]
        except Exception as e:
            print(f"Error accessing sweep {sweep_id}: {e}")
            continue

        for run in run_list:
            cfg = run.config
            # print(f"Checking run {run.id}...") # Too verbose
            if (
                cfg.get("rng_seed") != TARGET_SEED
                or cfg.get("model_params", {}).get("lora_params", {}).get("lora_rho")
                != TARGET_LORA_RHO
                or cfg.get("model_params", {}).get("num_curve_segment")
                != TARGET_NUM_SEGMENTS
            ):
                continue

            ds_name = cfg.get("data", {}).get("dataset_path", "").split("/")[-1]
            if not ds_name:
                continue

            clean_ds = clean_dataset_name(ds_name)

            if args.dataset and args.dataset not in clean_ds:
                continue

            model_name = get_model_name(run)
            filename = f"{clean_ds}_seed_{TARGET_SEED}_{model_name}_{run.id}.npz"
            filepath = os.path.join(LOGITS_DIR, filename)

            if os.path.exists(filepath):
                print(f"Skipping {filename}, already exists.")
                continue

            print(f"Processing {run.id} for dataset {ds_name}...")
            try:
                # Download artifacts if needed
                local_path = os.path.abspath("artifacts")
                os.makedirs(local_path, exist_ok=True)
                dataset_artifact = ds_name if ":" in ds_name else f"{ds_name}:v0"

                # Check for existing qwen model and dataset
                qwen_found = any(
                    "qwen2.5_7B_bfloat16" in d for d in os.listdir(local_path)
                )
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

                env, params, rng_key = _setup_training_env(config, data, rng_key)
                print("Training env setup.")
                params, *_ = load_checkpoint_new(run, params, env.s_model)
                print("Checkpoint loaded.")
                params = env.s_model.set_lora_rho(rho_s=0.0, rho_w=0.0, params=params)

                x_test, attn_test, labels = data.get("test")
                num_test = len(labels)

                # Revert manual chunking and use setup_metrics directly with a fixed small batch size
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
