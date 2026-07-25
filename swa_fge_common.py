"""Shared plumbing for the SWA (`qwen_swa.py`) and FGE (`qwen_fge.py`) entrypoints.

Both methods train a single k=0 point, collect weight snapshots during (part
of) training, and combine them — SWA by averaging weights, FGE by ensembling
predictions. This module holds the pieces that are identical between the two:
loading a pretrained starting point from wandb, the manual per-step training
loop with pluggable snapshot/logging policy, and small pytree helpers.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from jax import random

from qwen_fineTuning_PtoC_fixCps import _setup_rho_scheduler
from src.utils import load_checkpoint_new


def _tree_mean(trees):
    if not trees:
        raise ValueError("Cannot average an empty snapshot list")
    return jax.tree.map(lambda *xs: jnp.mean(jnp.stack(xs, axis=0), axis=0), *trees)


def _merge_trainable(s_model, trainable_params, template_params):
    """Merge a trainable-only pytree (from ``only_trainable_params``) back into
    a full params pytree, taking frozen leaves from ``template_params``.

    ``trainable_params``/``template_params`` must share ``s_model.train_mask``'s
    structure — frozen leaves in ``trainable_params`` are empty placeholder
    arrays (as produced by ``only_trainable_params``) and are ignored here.
    """
    return jax.tree.map(
        lambda t, f, m: t if m else f,
        trainable_params,
        template_params,
        s_model.train_mask,
    )


def _load_pretrained_point(wb_run, load_params_path: str, config, env, params):
    """Load a converged single-point checkpoint (wˆ) from a finished wandb run.

    Replaces local warmup training: instead of training wˆ inside this
    script, fetch it from a previously completed
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
            "This entrypoint can only load k=0 checkpoints."
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


def _dealias_small_leaves(tree):
    """Force distinct buffers for tiny leaves so a donating jit doesn't see the
    same physical buffer at multiple leaf positions ("donate the same buffer
    twice") — e.g. the scalar rho broadcast onto every LoRA param by
    ``set_lora_rho``, or the ``jnp.array([])`` placeholders ``only_trainable_params``
    puts at every frozen leaf. Leaves above this size are real weight matrices;
    those are left untouched, so this doesn't force an extra full-model copy.
    """
    return jax.tree.map(
        lambda x: x + 0 if isinstance(x, jnp.ndarray) and x.size <= 8 else x,
        tree,
    )


def _validation_metrics(valid_metrics_fn, valid_key, params, empty_metric, should_eval):
    if should_eval:
        return valid_metrics_fn(valid_key, params)[0]
    return empty_metric[0]


def _welford_update(mean, count, new_value):
    """One step of Welford's online mean update (the mean half of Welford's
    algorithm — no variance/M2 tracking needed here): ``mean_n = mean_{n-1} +
    (x_n - mean_{n-1}) / n``. Keeps a single running-mean pytree instead of
    storing every per-step snapshot, so averaging over every step of a long
    tail window doesn't cost O(num_steps) copies of the trainable params.
    """
    count = count + 1
    mean = jax.tree.map(lambda m, x: m + (x - m) / count, mean, new_value)
    return mean, count


def _run_snapshot_training_loop(
    wb_run,
    rng_key,
    env,
    params,
    data,
    config,
    *,
    num_steps: int,
    eval_every_n_batch: int,
    lr_schedule: Callable[[int], Any],
    should_snapshot: Callable[[int], bool],
    log_prefix: str,
    welford_start_step: int | None = None,
):
    """Manual per-step training loop with pluggable snapshot collection.

    ``env.optimizer`` drives the actual parameter updates (SWA passes it
    through unmodified; FGE passes one built with a dedicated cyclic
    schedule) — ``lr_schedule`` here is only used to log the current LR
    value under ``f"{log_prefix}/lr"``. ``should_snapshot(step)`` decides
    when to capture a weight snapshot; snapshots are the *trainable-only*
    slice (``s_model.only_trainable_params``), not the full model, so
    storing many of them doesn't scale with the (possibly multi-billion
    parameter) frozen base model.

    ``welford_start_step``, if given, additionally maintains a running
    (Welford) mean of the trainable params over every step from that step
    onward — independent of ``should_snapshot``'s discrete collection.
    Defaults to ``None`` (disabled), which leaves FGE's call site (and its
    discrete-snapshot ensembling) completely unaffected.

    Returns ``(final_params, snapshots, snapshot_steps, best_params,
    welford_mean)`` where ``final_params`` is the full (frozen + trainable)
    params at the end of the loop — callers use it as the template to merge
    trainable-only snapshots (or ``welford_mean``) back into full params
    before evaluating them (``_merge_trainable``). ``welford_mean`` is
    ``None`` if ``welford_start_step`` was not given.
    """
    s_model = env.s_model
    optimizer = env.optimizer
    t_sample_fn = env.t_sample_fn
    valid_metrics_fn = env.valid_metrics_fn
    empty_metric = env.empty_metric
    get_best_params = env.get_best_params

    train_input_ids, train_attention_mask, train_labels = data.get("train")
    if train_input_ids is None or train_attention_mask is None or train_labels is None:
        raise ValueError(f"{log_prefix} training requires train data")

    batch_size = config.train_hyper.batch_size
    num_items = int(train_input_ids.shape[0])

    rho_scheduler = _setup_rho_scheduler(config.model_params.lora_params, num_steps)

    only_train = s_model.only_trainable_params(params["params"])
    opt_state = optimizer.init(only_train)

    # ``s_model.train_step`` is jitted with ``donate_argnums`` for params and
    # opt_state so XLA can update the (possibly multi-billion parameter) frozen
    # base model in place instead of allocating a second full copy every step.
    # Calling it directly at top level from this Python loop makes that
    # donation real, which used to crash ("donate the same buffer twice")
    # because the pytrees contain aliased leaves — ``set_lora_rho`` broadcasts
    # one ``current_rho`` buffer onto every LoRA param, and the empty
    # placeholders from ``only_trainable_params`` dedupe to a single buffer in
    # opt_state. Wrapping the step in an outer non-donating ``jax.jit`` avoided
    # that crash but silently disabled donation entirely (the inner donation
    # becomes a no-op), which is affordable at 0.5B scale but OOMs a 7B model
    # on a single 40GB GPU (needs a second ~14GB copy of the frozen params
    # every step). De-aliasing just the tiny leaves below and calling
    # ``train_step`` directly keeps donation working for the actual weight
    # matrices.
    params = _dealias_small_leaves(params)

    best_val_ll = -np.inf
    best_params = (
        s_model.only_trainable_params(params["params"]) if data.has_val else None
    )
    snapshots = []
    snapshot_steps = []
    welford_mean = None
    welford_count = 0
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
            params = _dealias_small_leaves(params)

        # opt_state's frozen-leaf placeholders (empty arrays from
        # only_trainable_params) get re-deduped by XLA on every train_step's
        # output, not just the first call, so de-alias every iteration too.
        opt_state = _dealias_small_leaves(opt_state)

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

        if should_snapshot(step):
            snapshots.append(deepcopy(s_model.only_trainable_params(params["params"])))
            snapshot_steps.append(step)

        if welford_start_step is not None and step >= welford_start_step:
            current_trainable = s_model.only_trainable_params(params["params"])
            if welford_mean is None:
                # only_trainable_params returns leaves aliased to params'
                # underlying buffers (not copies); train_step donates params
                # on the next iteration, which would delete those same
                # buffers out from under welford_mean if we didn't copy here.
                welford_mean = deepcopy(current_trainable)
                welford_count = 1
            else:
                welford_mean, welford_count = _welford_update(
                    welford_mean, welford_count, current_trainable
                )

        log_row = {
            f"{log_prefix}/loss": loss,
            f"{log_prefix}/lr": lr_value,
            f"{log_prefix}/step": step,
            **{
                f"{log_prefix}/{k}": v
                for k, v in metrics.items()
                if not k.startswith("val_mean")
            },
            **{f"{log_prefix}/{k}": v for k, v in logs.items()},
        }
        wb_run.log(log_row)

    if not snapshots:
        snapshots.append(deepcopy(s_model.only_trainable_params(params["params"])))
        snapshot_steps.append(num_steps - 1 if num_steps > 0 else 0)

    return params, snapshots, snapshot_steps, best_params, welford_mean
