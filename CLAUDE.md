# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Time Transformer fine-tunes LLMs (Qwen2.5) within a low-dimensional **subspace of parameter space**, parameterized as a Bézier curve (or segmented Bézier / B-spline) connecting control points, optionally combined with LoRA. Instead of training a single point estimate, the framework trains several fixed control points independently and then fits a continuous curve through them, enabling Bayesian model averaging (BMA) over points sampled along the curve at evaluation time. The Qwen2.5 model itself is reimplemented from scratch in JAX/Flax (not via a JAX port of `transformers`), and weights are converted from a HuggingFace PyTorch checkpoint once and cached to disk.

## Environment Setup

```bash
poetry install
```

Python 3.10/3.11, JAX with CUDA 12, PyTorch (CUDA build) only used for loading/converting HF checkpoints and for `datasets`/`transformers` tokenization. GPU selection and JAX memory behavior are controlled via env vars (see `.vscode/launch.json` for examples):

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python qwen_fineTuning_PtoC_fixCps.py ...
```

## Common Commands

Smoke test (fast, reduced steps — always run this after touching training code):

```bash
python qwen_fineTuning_PtoC_fixCps.py --batch-size=4 --cp-fix 1 0 1 0 1 --smoke-test
```

Run with a local hardcoded config (edit the `config` dict under `if __name__ == "__main__":` in `qwen_fineTuning_PtoC_fixCps.py`):

```bash
python qwen_fineTuning_PtoC_fixCps.py --batch-size 4 --cp-fix 1 0 1 --smoke-test
```

Run as a wandb sweep agent (config comes from `wandb.config`, e.g. one of `configs_newDS/*.yaml`):

```bash
wandb sweep configs_newDS/OpenFlat.yaml
wandb agent <sweep_id>
# equivalent to running the script directly with --use-sweep
```

SWA/FGE variant (reuses helpers from the main script, swaps in a cyclical FGE learning-rate schedule for the final trajectory stage):

```bash
python qwen_swa_fge.py --batch-size=4 --smoke-test
```

Offline evaluation / plotting pipeline (run after a training run has logged params to wandb):

```bash
python compute_base_qwen_logits.py   # base (non-curve) Qwen logits on OOD/eval data
python compute_curve_logits.py       # logits for points sampled along a trained curve
python plot_curve_logits.py          # visualize entropy/mutual-information/calibration
```

Export/prepare datasets and base model params as wandb artifacts:

```bash
python save_datasets_and_params.py
```

Linting/formatting (configured via `.pre-commit-config.yaml`; no ruff/mypy config lives in `pyproject.toml`):

```bash
pre-commit run --all-files
# or individually
ruff check --fix .
ruff format .
```

There is no automated test suite (`pytest` is a dependency but no `tests/` directory exists) — validate changes with `--smoke-test` runs.

## Architecture

### Core subspace framework — `src/subspace_curve.py`

- **`SubspaceBaseModel`**: base class wrapping an underlying Flax model. Holds `k` (curve degree = number of control points − 1) and a `curve_mask` pytree marking which leaves of the model's params are promoted to a curve (leading dim `k+1`) vs. kept as a single shared point. `init_params_from_point` builds the initial curve params from one pretrained point, adding per-parameter-type noise/init logic (different for plain params vs. each `LoRAParams` subclass). `__call__` collapses curve params to a single point at parameter `t` via Bézier/B-spline coefficients before calling the wrapped model.
- **Curve parameterizations**: `bezier_coeff_fn` (single Bézier over all control points), `segmented_bezier_coeff_fn` (piecewise Bézier — open curve or closed loop depending on `num_bends`/`segment_degree`), and a B-spline option (`"b-spline"` in `curve_parameterization`). Selected in `SubspaceBaseModel.__init__` based on `SegDeg`/`curve_parameterization`.
- **Model registry**: `@register_subspace_model("name")` registers a `SubspaceBaseModel` subclass into `SUBSPACE_REGISTRY`; `get_subspace_model(name)` looks it up. Base task heads: `category`, `regression`, `dist_regression`, `quantile` (each implements `nll`).
- **Mixins** (composed via multiple inheritance, not standalone): `RepulsiveMixin` (control-point repulsion regularizer with `gravity`/`energy_weakening`), `EntropyMixin`/`EntropyMixin2`, and a family of Jensen–Shannon mixins (`JensenShannonMixin`, `_v2`, `NoiseMixin`, `NoiseSamplingMixin`, `NoiseSamplingDropoutMixin`) that add JSD-based diversity regularizers between curve samples, each overriding `compute_loss_t()` to add its term. Concrete registered models combine a task class with zero or more mixins, e.g. `LoRAJSDNoiseCategorySubspace(LoRAMixin, JensenShannonNoiseMixin, CategorySubspace)`.
- **`LoRAMixin`** + `LoRA*Params` classes (`LoRAParams`, `LoRAABParam`, `LoRAScaleParams`, `LoRAScaleDiagParams`, `LoRAepsilon*`, `LoRARotaEpsilonDiagscaleRotbParams`, ...): different LoRA parameterizations (plain A/B, scale-only, epsilon-perturbed, rotation+diag-scale) each with their own init/curve logic; the active one is selected via `lora_params.lora_mode` in config.
- New subspace model: subclass `SubspaceBaseModel`, register with `@register_subspace_model("name")`, implement `nll(self, params, state, t, x, y, train, key)`. New mixin: subclass overriding `compute_loss_t()`, add its params through `ModelParams.get_model_kwargs()` (gated by `issubclass(s_model_cls, YourMixin)`).

### Qwen JAX model — `src/qwen_jax.py`

Hand-written Flax reimplementation of Qwen2 (RMSNorm, RoPE, GQA attention, SwiGLU MLP) mirroring HF's `transformers` implementation layer-for-layer, so PyTorch weights can be mapped 1:1. `create_parameter_mapping` + `convert_pytorch_weights_to_jax` convert a loaded HF `AutoModelForCausalLM` into the JAX pytree once; `save_params_and_config`/`load_params_and_config` persist the converted params (`jax_params.npy` + `jax_config.json`) so conversion isn't repeated. `QwenTextClassificationWrapper` adapts the causal-LM model to `SubspaceBaseModel`'s `model.init`/`model.apply` interface by slicing the last-token logits down to `target_token_ids` (the token IDs for the classification labels, e.g. True/False) — classification is done via next-token prediction, not a separate head.

### Training orchestration — `qwen_fineTuning_PtoC_fixCps.py`

This is the main entry point; `qwen_swa_fge.py` and the `compute_*_logits.py` scripts import helpers from it directly (`Config`, `_prepare_datasets`, `_setup_training_env`, `run_training`, `run_evaluation`, `setup_metrics`, ...), so changes here ripple into those scripts.

Config flows through nested dataclasses built from a plain dict (wandb config or the hardcoded dict in `__main__`): `Config` → `TrainHyperparams`, `ModelParams`, `LoraParams`, `OptimizerConf`. `ModelParams.k` (curve degree) is derived from `num_curve_segment * SegDeg`, and `ModelParams.cp_fix` is derived automatically from `Pretraining`/`SegDeg` rather than being set directly. `ModelParams.build_curve_mask` / `get_model_kwargs` apply `filter_masks` (list of `{keys, op}` selectors, `op` is `eval()`'d as `all`/`any` over pytree key paths) to decide which weight matrices (typically `q_proj`, `v_proj`, `lm_head`) become part of the curve/LoRA vs. staying frozen at the base point.

Two-phase training driven by `train()`:
1. **Pretraining** (`_pretrain_fixed_cps`, `Pretraining=True`, effectively `k=0` per fixed point): each fixed control point (`cp_fix[i]==1`) trained independently to a local optimum.
2. **Curve fitting** (`_train_full_curve`, runs whenever `len(cp_fix) > 1`): a Bézier/segmented-Bézier/B-spline curve is fit connecting the (now fixed) pretrained control points, training only the intermediate trainable points.

Evaluation (`run_evaluation`) always runs after training and supports several `curve_sampling_mode`s controlling how points are sampled along `t` and combined: `combined_*` (compute BMA weights on train/val, evaluate on test), `per_leave` (leave-one-out), `noBMA` (uniform averaging, no weighting). Results/artifacts (trainable params, BMA weights) are logged to Weights & Biases under `WANDB_PATH = "ddold/qwen_PtoC"`; datasets and base-model checkpoints are also pulled from wandb artifacts (`ddold/qwen_PtoC_newDS/...`) rather than being stored in the repo.

### Datasets — `dataset/`

`dataset/S2ClassDataset.py` loads classification datasets (GLUE, SuperGLUE, ARC, Winogrande, OpenBookQA, MMLU-subset) via HF `datasets`, tokenizes them for Qwen (prompt formatted as a text continuation, e.g. "Answer with only True or False..."), and is the source data later exported to numpy/wandb by `save_datasets_and_params.py`. `dataset/utils/datasetbase.py` / `dsets.py` provide the base loader interface. `S2SDataset*.py` are for sequence-to-sequence variants. Prepared datasets are consumed downstream as numpy arrays of `input_ids`/`attention_mask`/`labels`, not as live HF `Dataset` objects.

### `configs_newDS/*.yaml`

Wandb sweep configs (`method: grid` sweeps over `configs_newDS/*.yaml`) — each maps directly onto the `Config` dataclass tree above (`data`, `net_kwargs`, `train_hyper`, `model_params` incl. nested `lora_params`, `optimizer_conf`). File naming reflects the curve/LoRA topology being tested (`OpenFlat*` = open Bézier curve, `PretrainedFlat*` = pretraining-fixed control points, `SWA_FGE_*` = stochastic weight averaging / fast geometric ensembling variant, `*NoSeg` = unsegmented full-degree Bézier).
