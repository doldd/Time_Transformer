### Purpose
Time Transformer research framework: concise agent guidance for Bézier curve parameterized neural networks, LoRA fine-tuning, and experiment orchestration.

### Framework overview (core concepts)
- Subspace exploration: train curves (k+1 control points) in parameter space rather than single points to enforce mode connectivity.
- Uncertainty quantification: sample parameter configurations along curves to compute predictive distributions (mutual information, entropy, ECE).
- LoRA integration: low-rank adapters mixed with control-point subspaces for efficient fine-tuning of large models.

### Essential patterns (what to change and where)
- `SubspaceBaseModel` (src/subspace_curve.py): central pattern—models expose `nll`, `__call__`, `train_step`, and provide `init_mask`, `init_params_from_point` helpers. Follow its API when adding new task models.
- Wrapper classes: external PyTorch/JAX models are wrapped with a small adapter exposing `init` and `apply` compatible with `SubspaceBaseModel` (see wrappers in training scripts).
- LoRA and masks: LoRA lives in `LoRAMixin` and related `LoRA*Subspace` classes. Use `filter_mask` / `masked_pytree_to_matrix` formats (list of dicts with `keys`/`op`/`dims`) to select parameters.

### Quick code examples (patterns to copy)
- Subspace `nll` pattern:
  ```python
  # sample Bezier coefficient
  bezier_coeff = self.bezier(t)
  # Compute one parameter set per sample
  final_params = jax.tree.map(lambda p, m: jnp.einsum(
      'k,k...->...', bezier_coeff, p).astype(p.dtype) if m else p, params, self.curve_mask)
  loss, state, logits = self.model.nll(final_params, state, x, y, train)
  ```

  ```
  # Lora apply pattern with leaf of type LoraAbstractParams
  params = jax.tree.map(lambda lp, key: lp.apply(key) if isinstance(lp, LoraAbstractParams) else lp, params, key_tree, is_leaf=lambda x: isinstance(x, LoraAbstractParams))  
  ```

### Training & config patterns
- Entry point: `train()` in `qwen_fineTuning_PtoC_fixCps.py` — it loads datasets, prepares masks, initializes curve params (from point or random), creates optimizer schedules, and runs the curve training loop.
- Sweep configs: `configs/*.yaml` use wandb sweep format; `curve_params` contains keys like `k`, `curve_parameterization`, `cp_fix`, `num_steps`, and `optimizer_conf`.

### Environment & serialization
- Expected stack: JAX/Flax + PyTorch (for source weights), numpyro, jaxopt, optax. Use GPU-enabled JAX for training.
- Serialization: `save_params_and_config` / `load_params_and_config` in `src/qwen_jax.py` — params saved as `jax_params.npy` and `jax_config.json`.

### Performance & JAX gotchas (practical tips)
- Use `jax.lax.scan` for batch/epoch loops to avoid repeated compilation.
- Keep JIT functions pure and avoid side effects; pass mutable state explicitly.
- Manage dtype carefully (bfloat16 for large models). Check `get_dtype_from_tensor` in `src/qwen_jax.py`.
- Compilation cache and XLA flags are set in training scripts (`XLA_FLAGS`, `JAX_COMPILATION_CACHE_DIR`). Verify these when debugging device issues.

### Where to look (critical files)
- Core: `src/subspace_curve.py` (subspace math, LoRA mixing)
- Model & conversion: `src/qwen_jax.py` (model implementation, weight mapping)
- Training orchestration: `qwen_fineTuning_PtoC_fixCps.py` (data loading, train loop, wandb integration)
- Configs: `configs/*.yaml`

### Run / development workflows
- Typical environment: JAX/Flax + PyTorch (for weight conversion). The repo expects a GPU/JAX-capable environment; some configs set `XLA_FLAGS` to force host device counts.
- Run training (example): activate your Python env and run the program referenced by a config. Example command used in sweep YAMLs:

  - ${env}
  - ${interpreter}
  - ${program}
  - "--use-sweep"

  Concretely (local dev):

  ```bash
  conda activate jaxTorchTrans
  JAX_COMPILATION_CACHE_DIR="jax_cache" CUDA_VISIBLE_DEVICES="2" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 python "qwen_fineTuning_PtoC_fixCps.py" --batch-size=4 --cp-fix 0 0 0
  ```

- Convert/load weights: use `load_params_and_config` and `convert_pytorch_weights_to_jax` in [src/qwen_jax.py](src/qwen_jax.py). Saved params use `jax_params.npy` plus `jax_config.json`.

If you want, I can expand this with: a short checklist for adding a new LoRA mask, a unit-test template for a new wrapper, or examples showing `pytree_to_matrix`/mask usage.
