"""Fast Geometric Ensembling (FGE) training entrypoint for Qwen text classification.

Trains a single k=0 point wˆ (locally via a normal warmup phase, or loaded
from a finished wandb run via ``load_params_path``), then runs a dedicated
short cyclic learning-rate phase — snapshotting weights at each cycle's
low-LR midpoint and ensembling their predictions (not their weights; see
``qwen_swa.py`` for weight averaging).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import wandb

from qwen_fineTuning_PtoC_fixCps import (
    Config,
    _init_optimizer,
    _prepare_datasets,
    _setup_training_env,
    entropy,
    mean_entropy,
    mutual_information,
    post_pred_performance,
    run_training,
    setup_metrics,
)
from swa_fge_common import (
    _load_pretrained_point,
    _merge_trainable,
    _run_snapshot_training_loop,
)


WANDB_PATH = "ddold/qwen_PtoC_newDS"


def _build_fge_lr_schedule(alpha1: float, alpha2: float, cycle_steps: int):
    """Piecewise linear cyclical LR used by FGE."""
    if cycle_steps <= 0 or cycle_steps % 2 != 0:
        raise ValueError("FGE cycle_steps must be a positive even integer")

    half_cycle = cycle_steps // 2

    def schedule(step):
        phase = jnp.mod(step, cycle_steps).astype(jnp.float32)
        half = jnp.array(float(half_cycle), dtype=jnp.float32)
        down = alpha1 + (alpha2 - alpha1) * (phase / half)
        up = alpha2 + (alpha1 - alpha2) * ((phase - half) / half)
        return jnp.where(phase <= half, down, up)

    return schedule


def _log_fge_ensemble_metrics(wb_run, env, data, template_params, snapshots, config):
    """Ensemble the FGE snapshots' predictions on the test split."""
    test_ids, test_mask, test_labels = data.get("test")
    if test_ids is None or test_mask is None or test_labels is None:
        raise ValueError("FGE evaluation requires test data")

    batch_size_eval = config.train_hyper.batch_size_eval

    snapshot_logits = []
    for idx, snapshot in enumerate(snapshots):
        full_snapshot = _merge_trainable(
            env.s_model, snapshot, template_params["params"]
        )
        # Zero LoRA rho noise before scoring — matches run_evaluation's
        # behavior, so FGE metrics don't include stray training-time noise.
        eval_params = env.s_model.set_lora_rho(0.0, 0.0, {"params": full_snapshot})
        metrics_fn = setup_metrics(
            env.s_model,
            batch_size_eval,
            test_ids,
            test_mask,
            test_labels,
            n_samples=1,
            use_linspace=True,
            average=False,
            key_prefix=f"snap_{idx}_",
        )
        _, logits = metrics_fn(random.PRNGKey(config.rng_seed), eval_params)
        snapshot_logits.append(logits[0])

    stacked_logits = jnp.stack(snapshot_logits, axis=0)
    fge_pp = post_pred_performance(stacked_logits, test_labels, key_prefix="fge_")
    wb_run.summary.update(fge_pp)
    wb_run.summary.update(
        {
            "fge_uniform_mutual_information": mutual_information(
                stacked_logits.astype(jnp.float32)
            ),
            "fge_uniform_mean_entropy": mean_entropy(
                stacked_logits.astype(jnp.float32)
            ),
            "fge_uniform_std_entropy_along_curve": entropy(
                stacked_logits.astype(jnp.float32)
            )
            .mean(-1)
            .std(),
            "fge_uniform_mean_entropy_along_curve": entropy(
                stacked_logits.astype(jnp.float32)
            ).mean(),
        }
    )
    return fge_pp


def train(wb_run, config_dict):
    config_dict = dict(config_dict)
    fge_cfg = dict(config_dict.get("fge", {}))
    load_params_path = config_dict.get("load_params_path", False)

    rng_key = random.PRNGKey(config_dict["rng_seed"])

    net_kwargs = dict(config_dict["net_kwargs"])
    if net_kwargs["model_path"].startswith("ddold/"):
        model_artifact = wb_run.use_artifact(net_kwargs["model_path"], type="model")
        net_kwargs["model_path"] = model_artifact.download()
    config_dict["net_kwargs"] = net_kwargs

    # warmup_steps only matters when we actually train wˆ locally below
    if not load_params_path:
        warmup_steps = int(
            fge_cfg.get("warmup_steps", config_dict["train_hyper"]["num_steps"])
        )
        config_dict["train_hyper"]["num_steps"] = warmup_steps
    config_dict["train_hyper"].setdefault("num_epochs", -1)

    data, unique_target_ids, rng_key = _prepare_datasets(
        rng_key,
        dataset_path=config_dict["data"]["dataset_path"],
        val_percentage=config_dict["data"].get("val_percentage", 0.0),
        batch_size=config_dict["train_hyper"]["batch_size"],
        smoke_test=config_dict["train_hyper"].get("smoke_test", False),
        logger=wb_run,
    )
    config_dict["net_kwargs"]["target_token_ids"] = unique_target_ids
    config = Config.from_dict(config_dict, data)

    if config.model_params.k != 0:
        raise ValueError(
            "The FGE entrypoint currently expects k=0 single-point training"
        )

    env, params, rng_key = _setup_training_env(
        config=config, data=data, rng_key=rng_key
    )

    if load_params_path:
        params = _load_pretrained_point(wb_run, load_params_path, config, env, params)
    else:
        rng_key, params, _ = run_training(
            rng_key, env, params, data, config, wb_run, "warmup_"
        )

    fge_steps = int(fge_cfg.get("steps", 0))
    if fge_steps <= 0:
        raise ValueError("fge.steps must be > 0")

    cycle_steps = int(fge_cfg.get("cycle_steps", 200))
    if cycle_steps % 2 != 0:
        raise ValueError("FGE cycle_steps must be even")
    lr_high = float(fge_cfg.get("lr_high", 1e-4))
    lr_low = float(fge_cfg.get("lr_low", lr_high / 10.0))
    lr_schedule = _build_fge_lr_schedule(lr_high, lr_low, cycle_steps)

    phase2_config = deepcopy(config)
    phase2_config.train_hyper.num_steps = fge_steps
    phase2_config.train_hyper.num_epochs = 0
    # Route the FGE cyclic schedule through the same optimizer construction
    # path as every other phase, instead of leaving the original schedule
    # from optimizer_conf in charge of the actual weight updates.
    phase2_config.optimizer_conf.kwargs["learning_rate"] = lr_schedule
    phase2_config.optimizer_conf.lr_schedule = lr_schedule

    fge_optimizer = _init_optimizer(
        phase2_config, curve_mask=env.s_model.curve_mask, curve_params=params["params"]
    )
    # Reuse the same s_model / t_sample_fn / validation helpers — only the
    # optimizer differs for the FGE phase, so no second full environment
    # (and no second base-model load) is needed.
    phase2_env = replace(env, optimizer=fge_optimizer)

    eval_every_n_batch = int(
        fge_cfg.get("eval_every_n_batch", phase2_config.train_hyper.eval_every_n_batch)
    )

    final_params, snapshots, snapshot_steps, _, _ = _run_snapshot_training_loop(
        wb_run=wb_run,
        rng_key=rng_key,
        env=phase2_env,
        params=params,
        data=data,
        config=phase2_config,
        num_steps=fge_steps,
        eval_every_n_batch=eval_every_n_batch,
        lr_schedule=lr_schedule,
        should_snapshot=lambda step: step % cycle_steps == cycle_steps // 2,
        log_prefix="fge",
    )

    wb_run.summary.update(
        {
            "fge_num_snapshots": len(snapshots),
            "fge_snapshot_steps": str(snapshot_steps),
        }
    )

    _log_fge_ensemble_metrics(
        wb_run, phase2_env, data, final_params, snapshots, phase2_config
    )

    if config.train_hyper.save_params:
        # snapshots is already a list of trainable-only pytrees (each came from
        # s_model.only_trainable_params inside _run_snapshot_training_loop) —
        # save that list directly rather than the full (frozen + trainable)
        # params for each ensemble member.
        os.makedirs("tmp_files", exist_ok=True)
        trainable_path = f"tmp_files/{wb_run.id}_trainable_params.npy"
        np.save(trainable_path, snapshots)
        time.sleep(1)
        print("Saved FGE snapshot trainable parameters to tmp_files")
        artifact = wandb.Artifact(name="params", type="pytree")
        artifact.add_file(trainable_path)
        wb_run.log_artifact(artifact)
        time.sleep(5)
        print("Logged artifact with FGE snapshot trainable parameters.")

    wb_run.finish()


def main():
    jax.config.update("jax_explain_cache_misses", True)
    wb_run = wandb.init()
    config_dict = dict(wandb.config)  # type: ignore[arg-type]
    train(wb_run, config_dict)


if __name__ == "__main__":
    wandb.require("core")  # type: ignore[attr-defined]
    os.makedirs("tmp_files", exist_ok=True)

    parser = argparse.ArgumentParser(description="Train FGE on a Qwen single point.")
    parser.add_argument(
        "--use-sweep", action="store_true", help="Use wandb sweep config."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="batch size for training. Default is 4.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a quick smoke test with reduced warmup/FGE steps, a small "
            "cycle length, and a truncated dataset."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["winogrande_s", "winogrande_m", "ARC-Challenge", "ARC-Easy"],
        default="winogrande_s",
        help="Which dataset artifact to train/eval on. Default is winogrande_s.",
    )
    args = parser.parse_args()

    print("FGE training")
    if args.use_sweep:
        wb_run = wandb.init()
        train(wb_run, wandb.config)
    else:
        smoke_test = args.smoke_test
        default_config = {
            "rng_seed": 2,
            # Set to a finished k=0 pretraining run path (e.g.
            # WANDB_PATH + "/47xexal3") to load wˆ from wandb instead of
            # training it locally below; False trains wˆ locally (warmup phase).
            "load_params_path": False,
            "data": {
                "dataset_path": WANDB_PATH + f"/{args.dataset}_dataset:v0",
                "val_percentage": 0.1,
            },
            "net_kwargs": {"model_path": "artifacts/qwen2.5_7B_bfloat16:v0"},
            "train_hyper": {
                "batch_size": args.batch_size,
                "num_epochs": -1,
                "num_steps": 10 if smoke_test else 2000,
                "eval_every_n_batch": 5 if smoke_test else 200,
                "temperature": [1.0],
                "dataset_sampling": {"minibatch": {}},
                "save_params": True,
                "smoke_test": smoke_test,
            },
            "model_params": {
                "n_samples_eval": 1,
                "num_curve_segment": 0,
                "SegDeg": 1,
                "Pretraining": True,
                "curve_sampling_mode": "combined_noBMA",
                "subspace_model": "lora_category",
                "jitter_multiplier": 0.0,
                "natural_parameterization": False,
                "weight_decay": 0.0,
                "curve_parameterization": "bezier",
                "indepent_connected": False,
                "entropy_weight": 0.0,
                "target_jsd": 0.1,
                "noise_rate": 0.05,
                "filter_masks": [
                    {"keys": ["self_attn", "q_proj", "kernel"], "op": "all"},
                    {"keys": ["self_attn", "v_proj", "kernel"], "op": "all"},
                    {"keys": ["lm_head", "kernel"], "op": "all"},
                ],
                "lora_params": {
                    "use_lora": True,
                    "r": 8,
                    "lora_dtype": "float32",
                    "lora_alpha": 16.0,
                    "lora_mode": "A(t)eB(t)",
                    "rho_scheduler_frequency": 50.0,
                    "lora_rho": 0.25,
                    "lora_rho_s": 0.0,
                    "filter_masks": [
                        {
                            "keys": ["self_attn", "q_proj", "kernel"],
                            "op": "all",
                            "dims": [1, 0],
                        },
                        {
                            "keys": ["self_attn", "v_proj", "kernel"],
                            "op": "all",
                            "dims": [1, 0],
                        },
                        {"keys": ["lm_head", "kernel"], "op": "all", "dims": [1, 0]},
                    ],
                },
            },
            "optimizer_conf": {
                "name": "adamw",
                "kwargs": {
                    "learning_rate": {
                        "name": "linear_onecycle_schedule",
                        "kwargs": {
                            "transition_steps": 2000,
                            "peak_value": 1e-4,
                            "pct_start": 0.12,
                            "pct_final": 1.0,
                            "div_factor": 300.0,
                            "final_div_factor": 300.0,
                        },
                    },
                    "weight_decay": 0.001,
                },
                "freeze_other_params": True,
            },
            "fge": {
                "warmup_steps": 10 if smoke_test else 2000,
                "steps": 10 if smoke_test else 2000,
                "cycle_steps": 4 if smoke_test else 200,
                "lr_high": 1e-4,
                "lr_low": 1e-5,
            },
        }
        wb_run = wandb.init(
            project=WANDB_PATH.split("/")[-1],
            name=f"fge_test_{args.dataset}",
            entity=WANDB_PATH.split("/")[0],
            config=default_config,
        )
        train(wb_run, default_config)
