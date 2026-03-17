# Time_Transformer

Quick start
-----------

This repository contains the Time Transformer research framework: Bézier-curve parameterized neural networks with LoRA integration and tooling for fine-tuning large models in a subspace of parameter space.

Environment
-----------

Use the project's conda environment:

```bash
conda activate jaxTorchTrans
```

Canonical training command
--------------------------

Run experiments on a single A100 GPU (this project typically uses `CUDA_VISIBLE_DEVICES="2"`):

```bash
JAX_COMPILATION_CACHE_DIR="jax_cache" \
  CUDA_VISIBLE_DEVICES="2" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 \
  python qwen_fineTuning_PtoC_fixCps.py --batch-size=4 --cp-fix 1 0 1 0 1 --smoke-test
```

Tests
-----

Unit tests live in `tests/`. Run them with:

```bash
pytest -q tests/
```

Project layout
--------------

- `src/subspace_curve.py` — Bézier subspace math and mixing helpers.
- `src/qwen_jax.py` — model implementations, (de)serialization helpers, conversion utilities.
- `qwen_fineTuning_PtoC_fixCps.py` — training entry point used by sweep configs.
- `configs/*.yaml` — example sweep configs.

See `AGENTS.md` for more detailed developer commands and style guidelines.
