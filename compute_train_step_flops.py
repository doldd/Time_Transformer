"""Compare single train_step FLOPs: MLE/MAP (k=0) vs. FLC curve (SegDeg=2, N=2,3,5,9) on Qwen2.5-7B.

Uses jax.eval_shape / jax.jit(...).lower(...).cost_analysis() throughout so no
real 7B weights are ever loaded from disk and no GPU/compute is required -
only abstract shapes flow through the real SubspaceBaseModel.train_step graph.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import functools

import jax
import jax.numpy as jnp
import optax

from qwen_fineTuning_PtoC_fixCps import Config, LoraParams, ModelParams, OptimizerConf
from src.qwen_jax import QwenTextClassificationWrapper
from src.subspace_curve import get_subspace_model

MODEL_PATH = "artifacts/qwen2.5_7B_bfloat16:v0"
BATCH_SIZE = 4
SEQ_LEN = 256

CURVE_FILTER_MASKS = [
    {"keys": ["self_attn", "q_proj", "kernel"], "op": "all"},
    {"keys": ["self_attn", "v_proj", "kernel"], "op": "all"},
    {"keys": ["lm_head", "kernel"], "op": "all"},
]
LORA_FILTER_MASKS = [
    {"keys": ["self_attn", "q_proj", "kernel"], "op": "all", "dims": [1, 0]},
    {"keys": ["self_attn", "v_proj", "kernel"], "op": "all", "dims": [1, 0]},
    {"keys": ["lm_head", "kernel"], "op": "all", "dims": [1, 0]},
]


def build_config(*, num_curve_segment, seg_deg, pretraining):
    lora_params = LoraParams(
        use_lora=True,
        r=8,
        lora_dtype="float32",
        lora_alpha=16.0,
        lora_mode="A(t)eB(t)",
        filter_masks=LORA_FILTER_MASKS,
    )
    model_params = ModelParams(
        num_curve_segment=num_curve_segment,
        SegDeg=seg_deg,
        Pretraining=pretraining,
        subspace_model="lora_category",
        filter_masks=CURVE_FILTER_MASKS,
        jitter_multiplier=0.0,
        lora_params=lora_params,
    )
    return Config(
        net_kwargs={"model_path": MODEL_PATH},
        model_params=model_params,
        optimizer_conf=OptimizerConf(
            name="adamw",
            kwargs={"learning_rate": 1e-4, "weight_decay": 0.001},
            freeze_other_params=True,
        ),
    )


def build_abstract_env(config, batch_size, seq_len, target_token_ids):
    """Build s_model + fully-abstract (ShapeDtypeStruct) params/opt_state.

    Mirrors _init_subspace_params / _init_optimizer in
    qwen_fineTuning_PtoC_fixCps.py, but never touches the real weight file on
    disk - the "point" params come from the raw QwenForCausalLM.init (pure
    random init) run under jax.eval_shape instead of QwenTextClassificationWrapper.init
    (which loads jax_params.npy from disk).
    """
    wrapper = QwenTextClassificationWrapper(
        model_path=config.net_kwargs["model_path"], target_token_ids=target_token_ids
    )
    mp = config.model_params
    s_model = get_subspace_model(mp.subspace_model)(**mp.get_model_kwargs(model=wrapper))

    dummy_ids = jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32)
    dummy_mask = jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32)

    def init_point(key, ids, mask):
        return wrapper.model.init(key, ids, mask, train=False)

    abstract_point_params = jax.eval_shape(
        init_point, jax.random.PRNGKey(0), dummy_ids, dummy_mask
    )

    curve_mask, train_mask, lora_mask = config.build_masks(abstract_point_params["params"])

    def init_curve(key, point_params):
        return s_model.init_params_from_point(
            key,
            point_params,
            curve_mask,
            noise_fn=lambda p, k: p
            + mp.jitter_multiplier * jax.random.normal(k, p.shape, dtype=p.dtype),
            lora_mask=lora_mask,
        )

    abstract_params = jax.eval_shape(
        init_curve, jax.random.PRNGKey(1), abstract_point_params
    )

    lp = mp.lora_params
    if lp and lp.use_lora:
        updated_train_mask = s_model.update_mask_for_lora(
            train_mask, abstract_params["params"], b_off=False, w0_off=True
        )
    else:
        updated_train_mask = train_mask
    s_model.set_train_mask(updated_train_mask)

    optimizer = getattr(optax, config.optimizer_conf.name)(**config.optimizer_conf.kwargs)
    trainable_params = s_model.only_trainable_params(abstract_params["params"])
    opt_state = jax.eval_shape(optimizer.init, trainable_params)

    return s_model, abstract_params, opt_state, optimizer, dummy_ids, dummy_mask


def _lower_train_step(s_model, key, t, params, x, y, opt_state, optimizer):
    """Lower s_model.train_step for cost analysis.

    s_model.train_step is decorated with
    @partial(jax.jit, static_argnums=(0, 7), donate_argnums=(3, 6)) at the
    class level (self=0, optimizer=7). Calling the bound method directly
    (as the real training loop does) works fine, but `.lower()` on the bound
    accessor does not correctly re-map static_argnums after self is bound
    (verified empirically: optimizer ends up traced as an abstract array
    instead of treated as static). So re-jit explicitly against the unbound
    method with self baked in via functools.partial, shifting the original
    static/donate indices down by one to account for the removed self arg.
    """
    unbound = type(s_model).train_step
    bound_fn = jax.jit(
        functools.partial(unbound, s_model),
        static_argnums=(6,),
        donate_argnums=(2, 5),
    )
    return bound_fn.lower(key, t, params, x, y, opt_state, optimizer)


def compute_step_flops(s_model, params, opt_state, optimizer, ids, mask, batch_size):
    key = jax.random.PRNGKey(0)
    t = jnp.float32(0.5)
    y = jax.ShapeDtypeStruct((batch_size,), jnp.int32)

    lowered = _lower_train_step(s_model, key, t, params, (ids, mask), y, opt_state, optimizer)
    try:
        analysis = lowered.cost_analysis()
    except Exception:
        analysis = lowered.compile().cost_analysis()
    if isinstance(analysis, (list, tuple)):
        analysis = analysis[0]
    return analysis["flops"]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = parser.parse_args()
    batch_size, seq_len = args.batch_size, args.seq_len

    target_token_ids = jnp.array([1565, 2152])  # dummy 2-class token ids; only count matters

    rows = []

    mle_config = build_config(num_curve_segment=0, seg_deg=0, pretraining=True)
    env = build_abstract_env(mle_config, batch_size, seq_len, target_token_ids)
    mle_flops = compute_step_flops(*env, batch_size)
    rows.append(("MLE/MAP", "-", 0, mle_flops))
    print(f"MLE/MAP (k=0): {mle_flops:.3e} FLOPs/step")

    for n in (2, 3, 5, 9):
        cfg = build_config(num_curve_segment=n, seg_deg=2, pretraining=False)
        env = build_abstract_env(cfg, batch_size, seq_len, target_token_ids)
        flops = compute_step_flops(*env, batch_size)
        rows.append(("FLC", n, cfg.model_params.k, flops))
        print(f"FLC (N={n}, SegDeg=2, k={cfg.model_params.k}): {flops:.3e} FLOPs/step")

    print()
    header = f"{'mode':<8} {'N':>3} {'k':>3} {'batch':>6} {'seq_len':>8} {'FLOPs/step':>14} {'vs MLE/MAP':>11}"
    print(header)
    print("-" * len(header))
    for mode, n, k, flops in rows:
        ratio = flops / mle_flops
        print(
            f"{mode:<8} {str(n):>3} {k:>3} {batch_size:>6} {seq_len:>8} {flops:>14.3e} {ratio:>10.4f}x"
        )


if __name__ == "__main__":
    main()
