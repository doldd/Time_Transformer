"""Stochastic Weight Averaging (SWA) training entrypoint for Qwen text classification.

Trains a single k=0 point wˆ (locally, or loaded from a finished wandb run
via ``load_params_path``) under whatever learning-rate schedule
``optimizer_conf`` already specifies — no dedicated SWA schedule — while
collecting weight snapshots equally spaced over the tail of training
(``swa.start_pct`` through the final step) and averaging them into a single
point (see ``qwen_fge.py`` for the cyclic-LR/prediction-ensembling
alternative).
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import numpy as np
import wandb
from jax import random

from qwen_fineTuning_PtoC_fixCps import (
    Config,
    _prepare_datasets,
    _setup_training_env,
    run_evaluation,
)
from swa_fge_common import (
    _load_pretrained_point,
    _merge_trainable,
    _run_snapshot_training_loop,
)


WANDB_PATH = "ddold/qwen_PtoC_newDS"


def train(wb_run, config_dict):
    config_dict = dict(config_dict)
    swa_cfg = dict(config_dict.get("swa", {}))
    load_params_path = config_dict.get("load_params_path", False)

    rng_key = random.PRNGKey(config_dict["rng_seed"])

    net_kwargs = dict(config_dict["net_kwargs"])
    if net_kwargs["model_path"].startswith("ddold/"):
        model_artifact = wb_run.use_artifact(net_kwargs["model_path"], type="model")
        net_kwargs["model_path"] = model_artifact.download()
    config_dict["net_kwargs"] = net_kwargs

    swa_steps = int(swa_cfg.get("steps", config_dict["train_hyper"]["num_steps"]))
    config_dict["train_hyper"]["num_steps"] = swa_steps
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
            "The SWA entrypoint currently expects k=0 single-point training"
        )

    env, params, rng_key = _setup_training_env(
        config=config, data=data, rng_key=rng_key
    )

    # load_params_path only replaces the starting params for the single
    # training loop below — it doesn't skip part of it. The LR schedule
    # (whatever optimizer_conf specifies) always runs its full course over
    # swa_steps starting from step 0 of this run, since a schedule like
    # linear_onecycle_schedule depends on the whole step range and can't be
    # correctly resumed mid-course after a separate warmup call.
    if load_params_path:
        params = _load_pretrained_point(wb_run, load_params_path, config, env, params)

    start_pct = float(swa_cfg.get("start_pct", 0.8))
    if not (0.0 <= start_pct < 1.0):
        raise ValueError("swa.start_pct must be in [0, 1)")

    start_step = int(start_pct * swa_steps)

    eval_every_n_batch = int(
        swa_cfg.get("eval_every_n_batch", config.train_hyper.eval_every_n_batch)
    )

    # No dedicated schedule override — env.optimizer already carries whatever
    # optimizer_conf specifies, used unmodified for this entire run.
    # Only the Welford running mean is used for weight averaging (an online
    # mean over every step from start_step onward via _welford_update) — no
    # discrete snapshot collection, so should_snapshot is always False.
    # best_params (tracked internally by _run_snapshot_training_loop) gives a
    # second point of comparison: the single best-validation checkpoint, no
    # averaging at all.
    final_params, _, _, best_params, welford_mean = _run_snapshot_training_loop(
        wb_run=wb_run,
        rng_key=rng_key,
        env=env,
        params=params,
        data=data,
        config=config,
        num_steps=swa_steps,
        eval_every_n_batch=eval_every_n_batch,
        lr_schedule=config.optimizer_conf.lr_schedule,
        should_snapshot=lambda step: False,
        log_prefix="swa",
        welford_start_step=start_step,
    )

    welford_full = _merge_trainable(env.s_model, welford_mean, final_params["params"])
    welford_eval_params = env.s_model.set_lora_rho(0.0, 0.0, {"params": welford_full})

    run_evaluation(
        rng_key=rng_key,
        env=env,
        params=welford_eval_params,
        data=data,
        config=config,
        logger=wb_run,
        logger_prefix="swa_",
    )

    # Compare against just picking the single best-validation checkpoint seen
    # during training (early-stopping style) instead of averaging at all —
    # env.get_best_params/best_params already tracks this internally.
    if best_params is not None:
        best_full = _merge_trainable(env.s_model, best_params, final_params["params"])
        best_eval_params = env.s_model.set_lora_rho(0.0, 0.0, {"params": best_full})

        run_evaluation(
            rng_key=rng_key,
            env=env,
            params=best_eval_params,
            data=data,
            config=config,
            logger=wb_run,
            logger_prefix="swa_best_",
        )

    if config.train_hyper.save_params:
        # welford_mean is already trainable-only (only_trainable_params
        # applied each update inside _run_snapshot_training_loop) — save that
        # directly rather than the full (frozen + trainable) params.
        os.makedirs("tmp_files", exist_ok=True)
        trainable_path = f"tmp_files/{wb_run.id}_trainable_params.npy"
        np.save(trainable_path, welford_mean)
        time.sleep(1)
        print("Saved SWA trainable parameters to tmp_files")
        artifact = wandb.Artifact(name="params", type="pytree")
        artifact.add_file(trainable_path)
        wb_run.log_artifact(artifact)
        time.sleep(5)
        print("Logged artifact with SWA trainable parameters.")

    wb_run.finish()


def main():
    jax.config.update("jax_explain_cache_misses", True)
    wb_run = wandb.init()
    config_dict = dict(wandb.config)  # type: ignore[arg-type]
    train(wb_run, config_dict)


if __name__ == "__main__":
    wandb.require("core")  # type: ignore[attr-defined]
    os.makedirs("tmp_files", exist_ok=True)

    parser = argparse.ArgumentParser(description="Train SWA on a Qwen single point.")
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
            "Run a quick smoke test with reduced steps, few snapshots, and a "
            "truncated dataset."
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2000,
        help=(
            "Total training steps (also sets the LR schedule's "
            "transition_steps and swa.steps to match). Default is 2000."
        ),
    )
    parser.add_argument(
        "--start-pct",
        type=float,
        default=0.8,
        help="swa.start_pct: fraction of training after which snapshots are collected.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=2,
        help="rng_seed for this run.",
    )
    args = parser.parse_args()

    print("SWA training")
    if args.use_sweep:
        wb_run = wandb.init()
        train(wb_run, wandb.config)
    else:
        smoke_test = args.smoke_test
        default_config = {
            "rng_seed": args.rng_seed,
            # Set to a finished k=0 pretraining run path (e.g.
            # WANDB_PATH + "/47xexal3") to load wˆ from wandb instead of
            # training it locally below; False trains it locally.
            "load_params_path": False,
            "data": {
                "dataset_path": WANDB_PATH + "/winogrande_s_dataset:v0",
                "val_percentage": 0.1,
            },
            "net_kwargs": {"model_path": "artifacts/qwen2.5_7B_bfloat16:v0"},

            "train_hyper": {
                "batch_size": args.batch_size,
                "num_epochs": -1,
                "num_steps": 20 if smoke_test else args.num_steps,
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
                            "transition_steps": 20 if smoke_test else args.num_steps,
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
            "swa": {
                "steps": 20 if smoke_test else args.num_steps,
                "start_pct": 0.5 if smoke_test else args.start_pct,
            },
        }
        wb_run = wandb.init(
            project=WANDB_PATH.split("/")[-1],
            name="swa_test",
            entity=WANDB_PATH.split("/")[0],
            config=default_config,
        )
        train(wb_run, default_config)
