"""SWA / FGE training entrypoint for Qwen text classification.

This script reuses the existing dataset/model/eval helpers from
`qwen_fineTuning_PtoC_fixCps.py`, but runs the final trajectory stage with a
dedicated Fast Geometric Ensembling (FGE) learning-rate schedule.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import argparse
import os

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import wandb

from qwen_fineTuning_PtoC_fixCps import (
    Config,
    _init_optimizer,
    _prepare_datasets,
    _setup_rho_scheduler,
    _setup_training_env,
    entropy,
    mean_entropy,
    mutual_information,
    post_pred_performance,
    run_evaluation,
    run_training,
    setup_metrics,
)
from src.utils import load_checkpoint_new


WANDB_PATH = "ddold/qwen_PtoC"


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


def _tree_mean(trees):
    if not trees:
        raise ValueError("Cannot average an empty snapshot list")
    return jax.tree.map(lambda *xs: jnp.mean(jnp.stack(xs, axis=0), axis=0), *trees)


def _load_pretrained_point(wb_run, load_params_path: str, config, env, params):
    """Load a converged single-point checkpoint (wˆ) from a finished wandb run.

    Replaces the local phase-1 warmup training: instead of training wˆ inside
    this script, fetch it from a previously completed
    ``qwen_fineTuning_PtoC_fixCps.py`` k=0 pretraining run.
    """
    source_run = wandb.Api().run(load_params_path)
    src_model_params = source_run.config.get("model_params", {})
    src_num_curve_segment = src_model_params.get("num_curve_segment", 1)
    src_seg_deg = src_model_params.get("SegDeg", 1)
    src_k = src_num_curve_segment * src_seg_deg
    if src_k != 0 or not src_model_params.get("Pretraining", False):
        raise ValueError(
            f"load_params_path={load_params_path!r} does not look like a k=0 "
            f"single-point pretraining run (source k={src_k}, "
            f"Pretraining={src_model_params.get('Pretraining')}). "
            "The SWA/FGE entrypoint can only load k=0 checkpoints."
        )

    src_dataset_path = source_run.config.get("data", {}).get("dataset_path")
    cur_dataset_path = config.data.get("dataset_path")
    if src_dataset_path is not None and src_dataset_path != cur_dataset_path:
        raise ValueError(
            f"load_params_path={load_params_path!r} was trained on dataset "
            f"{src_dataset_path!r} but this run uses {cur_dataset_path!r}. "
            "The source run's dataset must match the current dataset_path."
        )

    print(f"Loading pretrained point from {load_params_path}")
    params, _, _, _, _, _ = load_checkpoint_new(source_run, params, env.s_model)
    return params


def _pad_batch_indices(num_items: int, batch_size: int, step: int, permutation):
    num_batches = max(1, int(np.ceil(num_items / batch_size)))
    batch_pos = step % num_batches
    start = batch_pos * batch_size
    stop = min(start + batch_size, num_items)
    batch_idx = permutation[start:stop]
    if batch_idx.shape[0] == batch_size:
        return batch_idx
    pad_count = batch_size - batch_idx.shape[0]
    return jnp.concatenate([batch_idx, permutation[:pad_count]], axis=0)


def _validation_metrics(valid_metrics_fn, valid_key, params, empty_metric, should_eval):
    if should_eval:
        return valid_metrics_fn(valid_key, params)[0]
    return empty_metric[0]


def _run_fge_phase(
    wb_run, rng_key, env, params, data, config, fge_cfg, lr_schedule, cycle_steps
):
    """Run the FGE cyclic-LR phase and collect snapshots at low-LR midpoints.

    ``lr_schedule``/``cycle_steps`` must be the same values used to build
    ``env.optimizer`` (see ``train()``) — the schedule here is only used to
    log the current LR and to time snapshot collection, not to drive the
    actual parameter updates (those come from ``env.optimizer`` itself).
    """
    s_model = env.s_model
    optimizer = env.optimizer
    t_sample_fn = env.t_sample_fn
    valid_metrics_fn = env.valid_metrics_fn
    empty_metric = env.empty_metric
    get_best_params = env.get_best_params

    train_input_ids, train_attention_mask, train_labels = data.get("train")
    if train_input_ids is None or train_attention_mask is None or train_labels is None:
        raise ValueError("FGE requires train data")

    batch_size = config.train_hyper.batch_size
    num_items = int(train_input_ids.shape[0])
    num_steps = int(fge_cfg.get("steps", 0))
    eval_every_n_batch = int(
        fge_cfg.get("eval_every_n_batch", config.train_hyper.eval_every_n_batch)
    )

    rho_scheduler = _setup_rho_scheduler(config.model_params.lora_params, num_steps)

    only_train = s_model.only_trainable_params(params["params"])
    opt_state = optimizer.init(only_train)

    best_val_ll = -np.inf
    best_params = (
        s_model.only_trainable_params(params["params"]) if data.has_val else None
    )
    snapshots = []
    snapshot_steps = []
    permutation = jnp.arange(num_items)

    for step in range(num_steps):
        if step % max(1, num_items // batch_size) == 0:
            rng_key, perm_key = random.split(rng_key)
            permutation = random.permutation(perm_key, num_items)

        batch_idx = _pad_batch_indices(num_items, batch_size, step, permutation)
        x_batch = jnp.take(train_input_ids, batch_idx, axis=0)
        atten_batch = jnp.take(train_attention_mask, batch_idx, axis=0)
        y_batch = jnp.take(train_labels, batch_idx, axis=0)

        rng_key, step_key, valid_key = random.split(rng_key, 3)
        t = t_sample_fn(step_key)

        lora_params = config.model_params.lora_params
        if lora_params and (lora_params.lora_rho > 0.0 or lora_params.lora_rho_s > 0.0):
            current_rho = rho_scheduler(step, lora_params.lora_rho)
            current_rho_s = rho_scheduler(step, lora_params.lora_rho_s)
            params = s_model.set_lora_rho(current_rho, current_rho_s, params)  # type: ignore[attr-defined]

        loss, params, opt_state, (_, logs) = s_model.train_step(
            step_key,
            t,
            params,
            (x_batch, atten_batch),
            y_batch,
            opt_state,
            optimizer,
        )

        lr_value = lr_schedule(step)

        metrics = _validation_metrics(
            valid_metrics_fn,
            valid_key,
            params,
            empty_metric,
            (step % eval_every_n_batch) == 0,
        )
        if metrics["val_bma_ll"] > best_val_ll:
            best_params = get_best_params(params)
            best_val_ll = metrics["val_bma_ll"]

        if step % cycle_steps == cycle_steps // 2:
            snapshots.append(deepcopy(params["params"]))
            snapshot_steps.append(step)

        log_row = {
            "fge/loss": loss,
            "fge/lr": lr_value,
            "fge/step": step,
            **{
                f"fge/{k}": v
                for k, v in metrics.items()
                if not k.startswith("val_mean")
            },
            **{f"fge/{k}": v for k, v in logs.items()},
        }
        wb_run.log(log_row)

    if not snapshots:
        snapshots.append(deepcopy(params["params"]))
        snapshot_steps.append(num_steps - 1 if num_steps > 0 else 0)

    swa_params = _tree_mean(snapshots)
    return params, swa_params, snapshots, snapshot_steps, best_params


def _log_fge_ensemble_metrics(wb_run, env, data, swa_params, snapshots, config):
    """Compute SWA and FGE metrics on the test split."""
    test_ids, test_mask, test_labels = data.get("test")
    if test_ids is None or test_mask is None or test_labels is None:
        raise ValueError("FGE evaluation requires test data")

    batch_size_eval = config.train_hyper.batch_size_eval

    swa_eval_params = {"params": swa_params}
    swa_metrics = run_evaluation(
        rng_key=random.PRNGKey(config.rng_seed),
        env=env,
        params=swa_eval_params,
        data=data,
        config=config,
        logger=wb_run,
        logger_prefix="swa_",
    )

    snapshot_logits = []
    for idx, snapshot in enumerate(snapshots):
        # Zero LoRA rho noise before scoring — matches run_evaluation's/
        # valid_metrics_fn's behavior, so FGE metrics are comparable to SWA's
        # rather than including stray training-time noise.
        eval_params = env.s_model.set_lora_rho(0.0, 0.0, {"params": deepcopy(snapshot)})
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

    return swa_metrics, fge_pp


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
            "The SWA/FGE entrypoint currently expects k=0 single-point training"
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

    if not fge_cfg.get("enabled", True):
        run_evaluation(
            rng_key=rng_key,
            env=env,
            params=params,
            data=data,
            config=config,
            logger=wb_run,
        )
        return

    fge_steps = int(fge_cfg.get("steps", 0))
    if fge_steps <= 0:
        raise ValueError("fge.steps must be > 0 when FGE is enabled")

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
    # path as every other phase, instead of leaving the original one-cycle
    # schedule from optimizer_conf in charge of the actual weight updates.
    phase2_config.optimizer_conf.kwargs["learning_rate"] = lr_schedule
    phase2_config.optimizer_conf.lr_schedule = lr_schedule

    fge_optimizer = _init_optimizer(
        phase2_config, curve_mask=env.s_model.curve_mask, curve_params=params["params"]
    )
    # Reuse the same s_model / t_sample_fn / validation helpers — only the
    # optimizer differs for the FGE phase, so no second full environment
    # (and no second base-model load) is needed.
    phase2_env = replace(env, optimizer=fge_optimizer)

    params, swa_params, snapshots, snapshot_steps, _ = _run_fge_phase(
        wb_run=wb_run,
        rng_key=rng_key,
        env=phase2_env,
        params=params,
        data=data,
        config=phase2_config,
        fge_cfg=fge_cfg,
        lr_schedule=lr_schedule,
        cycle_steps=cycle_steps,
    )

    wb_run.summary.update(
        {
            "fge_num_snapshots": len(snapshots),
            "fge_snapshot_steps": str(snapshot_steps),
        }
    )

    _log_fge_ensemble_metrics(
        wb_run, phase2_env, data, swa_params, snapshots, phase2_config
    )


def main():
    jax.config.update("jax_explain_cache_misses", True)
    wb_run = wandb.init()
    config_dict = dict(wandb.config)  # type: ignore[arg-type]
    train(wb_run, config_dict)


if __name__ == "__main__":
    wandb.require("core")  # type: ignore[attr-defined]
    os.makedirs("tmp_files", exist_ok=True)

    parser = argparse.ArgumentParser(description="Train SWA / FGE on Qwen curves.")
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
    args = parser.parse_args()

    print("SWA/FGE training")
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
                "dataset_path": WANDB_PATH + "/ARC-Easy_dataset:v0",
                "val_percentage": 0.1,
            },
            "net_kwargs": {"model_path": "artifacts/qwen2.5_0.5B_bfloat16:v0"},
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
                "subspace_model": "jsd_noise_sampling_dropout_category",
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
                "enabled": True,
                "warmup_steps": 10 if smoke_test else 2000,
                "steps": 10 if smoke_test else 2000,
                "cycle_steps": 4 if smoke_test else 200,
                "lr_high": 1e-4,
                "lr_low": 1e-5,
            },
        }
        wb_run = wandb.init(
            project=WANDB_PATH.split("/")[-1],
            name="swa_fge_test",
            entity=WANDB_PATH.split("/")[0],
            config=default_config,
        )
        train(wb_run, default_config)
