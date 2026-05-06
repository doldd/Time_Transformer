#!/usr/bin/env python3
import argparse
import os
import jax
import jax.numpy as jnp
import numpy as np
import wandb
import yaml
from transformers import AutoTokenizer
from qwen_fineTuning_PtoC_fixCps import (
    Config,
    _setup_training_env,
    _prepare_datasets,
    setup_metrics,
)

# Configure environment
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

LOGITS_DIR = "dataset/logits_backup"
os.makedirs(LOGITS_DIR, exist_ok=True)


def clean_dataset_name(ds_name):
    return "_".join(ds_name.split(".")[0].split("_")[:-1])


def load_ood_data(tokenizer_path):
    ood_yaml_path = "dataset/my_ood_data.yaml"
    with open(ood_yaml_path, "r") as f:
        ood_data = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [ex["prompt"] for ex in ood_data["examples"]]
    labels = [ex["label"] for ex in ood_data["examples"]]

    # Tokenize
    encodings = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )

    x_ood = jnp.array(encodings["input_ids"])
    attn_ood = jnp.array(encodings["attention_mask"])

    # Ensure labels are object arrays if they contain lists (multiple correct answers)
    labels_out = np.empty(len(labels), dtype=object)
    for i, j in enumerate(labels):
        labels_out[i] = j

    return x_ood, attn_ood, labels_out


def load_config_and_data(run, local_path="artifacts"):
    # Ensure local path exists
    os.makedirs(local_path, exist_ok=True)

    # Update paths to local artifact directory
    run.config["net_kwargs"]["model_path"] = f"{local_path}/qwen2.5_7B_bfloat16:v0"
    ds_name = run.config["data"]["dataset_path"].split("/")[-1]
    run.config["data"]["dataset_path"] = f"{local_path}/{ds_name}"

    # Download artifacts if missing
    api = wandb.Api()
    if not any("qwen2.5_7B_bfloat16" in d for d in os.listdir(local_path)):
        print("Downloading base model artifact...")
        art = api.artifact("ddold/qwen_PtoC_newDS/qwen2.5_7B_bfloat16:v0")
        art.download(local_path)

    if not any(ds_name.split(":")[0] in d for d in os.listdir(local_path)):
        print(f"Downloading dataset artifact {ds_name}...")
        art = api.artifact(
            f"ddold/qwen_PtoC_newDS/{ds_name if ':' in ds_name else ds_name + ':v0'}"
        )
        art.download(local_path)

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
    parser.add_argument(
        "--run-ids", nargs="+", required=True, help="WandB run IDs to select datasets"
    )
    args = parser.parse_args()

    api = wandb.Api()
    project = "qwen_PtoC_newDS"
    entity = "ddold"

    ood_computed = False

    for run_id in args.run_ids:
        print(f"\nProcessing base Qwen for run {run_id}...")
        try:
            run_path = f"{entity}/{project}/{run_id}"
            run = api.run(run_path)

            ds_name = run.config.get("data", {}).get("dataset_path", "").split("/")[-1]
            if not ds_name:
                print(f"Skipping {run_id}: No dataset path found.")
                continue

            clean_ds = clean_dataset_name(ds_name)
            save_path = os.path.join(LOGITS_DIR, f"{clean_ds}_QWEN.npz")
            ood_save_path = os.path.join(LOGITS_DIR, f"{clean_ds}_QWEN_ood.npz")

            # Check if both exist
            if os.path.exists(save_path) and (
                ood_computed or os.path.exists(ood_save_path)
            ):
                print(f"Skipping {run_id}: Both {save_path} and OOD logits exist.")
                continue

            config, data, rng_key = load_config_and_data(run)

            # Setup environment - initial params will have LoRA B=0
            env, params, rng_key = _setup_training_env(config, data, rng_key)

            # 1. Handle OOD data if needed
            if not ood_computed and not os.path.exists(ood_save_path):
                print("Computing OOD base Qwen logits...")
                x_ood, attn_ood, labels_ood = load_ood_data(
                    config.net_kwargs["model_path"]
                )

                # Ensure labels are JAX compatible for acc_fn (use -1 for OOD if needed)
                labels_for_metrics = jnp.array([-1] * len(labels_ood))

                pred_fn_ood = setup_metrics(
                    env.s_model,
                    4,  # Small batch size for memory safety
                    x_ood,
                    attn_ood,
                    labels_for_metrics,
                    n_samples=1,
                    use_linspace=True,
                )
                _, logits_ood = pred_fn_ood(jax.random.PRNGKey(0), params)

                np.savez(
                    ood_save_path,
                    logits=np.array(logits_ood),
                    labels=np.array(labels_ood),
                    dataset="ood_custom",
                )
                print(f"Successfully saved {ood_save_path}")
                ood_computed = True
                # Clear OOD specific things
                pred_fn_ood = None
                import gc

                gc.collect()

            # 2. Handle ID data if needed
            if not os.path.exists(save_path):
                x_test, attn_test, labels = data.get("test")

                print(f"Running inference on {len(labels)} examples for {clean_ds}...")
                # Use n_samples=1 for the deterministic base model
                pred_fn = setup_metrics(
                    env.s_model,
                    4,  # Small batch size for memory safety
                    x_test,
                    attn_test,
                    labels,
                    n_samples=1,
                    use_linspace=True,
                )

                _, logits = pred_fn(jax.random.PRNGKey(0), params)

                # Ensure labels are saved as object arrays if they contain lists
                labels_out = np.empty(len(labels), dtype=object)
                for i, j in enumerate(labels):
                    labels_out[i] = j

                np.savez(
                    save_path,
                    logits=np.array(logits),
                    labels=labels_out,
                    wandb_run_id=run_id,
                    dataset=clean_ds,
                )
                print(f"Successfully saved {save_path}")

            # Clear memory explicitly
            params = None
            env = None
            pred_fn = None
            import gc

            gc.collect()

        except Exception as e:
            print(f"Error processing {run_id}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
