# Time Transformer

Bézier-curve parameterized neural networks with LoRA integration for fine-tuning large models in a subspace of parameter space.
This framework enables fine-tuning of Large Language Models (LLMs) within a parameter subspace by learning weights as continuous Bezier curves rather than static points.

This repository contains the code to reproduce the experiments of *On the Construction and Implications of Low-Loss Valleys in LoRA-based Bayesian Inference* (NeurIPS 2026); see [Reproducing the paper results](#reproducing-the-paper-results).
If you want to build on LoRA-Curve rather than reproduce the paper, use [bayes_sub_inf](https://github.com/doldd/bayes_sub_inf), a lightweight standalone library extracted from this code base with fewer dependencies.

## Project Layout

```
.
├── qwen_fineTuning_PtoC_fixCps.py   # Main training entry point
├── src/
│   ├── subspace_curve.py             # Bézier math, SubspaceBaseModel, LoRA mixins
│   ├── qwen_jax.py                   # Qwen JAX model, PyTorch→JAX conversion
│   ├── utils.py                      # Utility functions (ECE, curve metrics)
├── configs_newDS/                    # Wandb sweep configurations
│   ├── OpenFlat*.yaml               # Open curve configurations
│   ├── PretrainedFlat*.yaml         # Pretrained CP configurations
│   └── *.yaml                       # Dataset-specific configs
├── dataset/
│   ├── S2ClassDataset.py            # Classification dataset loader
│   └── utils/                       # Dataset utilities
├── save_datasets_and_params.py      # Dataset/model export utilities for WandB
├── qwen_swa.py, qwen_fge.py         # SWA and FGE baselines (shared code in swa_fge_common.py)
├── compute_curve_logits.py          # Offline curve evaluation
├── plot_curve_logits.py             # Visualization utilities
├── compute_train_step_flops.py      # FLOPs per training step
├── analyse_*.ipynb                  # Analysis notebooks producing the paper tables and figures
```

### Key Files

- **`src/subspace_curve.py`**: Core subspace math, model registry, mixin classes
- **`src/qwen_jax.py`**: Qwen model wrapper, weight conversion, serialization
- **`qwen_fineTuning_PtoC_fixCps.py`**: Training orchestration, evaluation pipeline


### Environment Setup

#### Install dependencies and setup environment
poetry install

### Smoke Test

```bash
  python qwen_fineTuning_PtoC_fixCps.py --batch-size=4 --cp-fix 1 0 1 0 1 --smoke-test
```

## Training Modes

### 1. Pretraining Mode (k=0)

Train fixed control points independently before fitting the curve:

```bash
# Set Pretraining=True in config, cp_fix=[1,] for example
# Each fixed CP (value=1) is trained separately
```

### 2. Curve Fitting Mode (k>0)

Fit Bézier curve connecting the control points:

```bash
# num_curve_segment=2, SegDeg=2 → k=5 control points
# cp_fix=[1, 0, 1, 0, 1] fixes endpoints, trains intermediate points
```

### 3. Evaluation

Automatically runs after training. Supports multiple evaluation modes via `curve_sampling_mode`:

- `combined_*`: Compute BMA weights on train/val, evaluate on test
- `per_leave`: Leave-one-out evaluation
- `noBMA`: Uniform averaging without Bayesian model averaging

## Reproducing the paper results

All experiments use Qwen2.5-7B (bfloat16) on six datasets (`winogrande_s`, `winogrande_m`, `ARC-Easy`, `ARC-Challenge`, `obqa`, `boolq`) with 5 seeds per configuration.

### 1. Weights & Biases

Datasets, the converted base model and all results are stored as [Weights & Biases](https://wandb.ai) artifacts and runs. Log in to your own W&B account or server (`wandb login`; set `WANDB_BASE_URL` for a self-hosted server).
The entity/project `ddold/qwen_PtoC_newDS` is hard-coded and has to be replaced by your own:

- `dataset_path` and `model_path` in the sweep configs in `configs_newDS/`
- `WANDB_PATH` in `qwen_fineTuning_PtoC_fixCps.py`, `qwen_swa.py` and `qwen_fge.py`
- the W&B API calls in the analysis notebooks

### 2. Export datasets and base model

```bash
python save_datasets_and_params.py --mode all --qwen-models Qwen/Qwen2.5-7B --dtype bfloat16 \
  --upload-to-wandb --wandb-project qwen_PtoC_newDS
```

This creates the artifacts `<dataset>_dataset:v0` and `qwen2.5_7B_bfloat16:v0` referenced by the sweep configs.

### 3. Run the sweeps

```bash
wandb sweep "configs_newDS/<config>.yaml"
wandb agent <sweep_id>
```

| Method in the paper | Sweep config(s) |
|---|---|
| ALC(N, m), incl. DE(N) and MAP (anchors) | `PretrainedFlat_Seg(1,2,4,8).yaml`, `PretrainedBoolq.yaml` |
| FLC(N, m) | `OpenFlat.yaml`, `OpenFlat_Boolq.yaml` |
| Lin(N) (linear interpolation) | `PretrainedFlat_Linear.yaml`, `PretrainedBoolqLinear.yaml` |
| Single-segment, high-degree curves | `OpenFlat_NoSeg.yaml` |
| Without flat-minima perturbation | `OpenNoFlat.yaml`, `PretrainedNoFlat.yaml` |
| LoRA rank 40 | `PretrainedFlat_Seg(1,2,4,8)_HighRank.yaml` |
| SWA | `SWA_OpenFlat_NoSeg_round2.yaml`, `SWA_OpenFlat_NoSeg_round3_*.yaml` |
| FGE | `FGE_OpenFlat_NoSeg_round2.yaml`, `FGE_OpenFlat_NoSeg_round3_*.yaml` |

DE(N) and MAP are not run separately: they are the independently trained anchors of the ALC runs and are evaluated within them.
BLoB and ScalaBL were reproduced with the framework released by Samplawski et al. (2025) and are not part of this repository.

### 4. Tables and figures

| Notebook / script | Output |
|---|---|
| `analyse_qwen_only_new DS.ipynb` | Performance table and main performance figures |
| `analyse_qwen_swa_fge_compare.ipynb` | FGE/SWA comparison |
| `analyse_qwen_rank_compare.ipynb` | LoRA rank 8 vs. 40 |
| `analyse_pareto.ipynb` | Training-cost Pareto figure and cost table (inputs in `data/pareto/`: steps to the best validation checkpoint from the W&B run histories, and per-seed test log-likelihoods) |
| `functional_smootheness.ipynb` | Functional smoothness along the curve |
| `compute_train_step_flops.py` | FLOPs per training step |

## Configuration

### CLI Arguments

```bash
python qwen_fineTuning_PtoC_fixCps.py \
  --batch-size 4 \
  --cp-fix 1 0 1 0 1 \
  --smoke-test
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 16 | Training batch size |
| `--cp-fix` | `[1, 0, 1]` | Control point fix mask (1=fixed, 0=trainable) |
| `--smoke-test` | False | Quick test with reduced steps/epochs |
| `--use-sweep` | False | Use wandb sweep config |



### Adding a New Subspace Model

1. Create class inheriting `SubspaceBaseModel` in `src/subspace_curve.py`
2. Register with `@register_subspace_model("name")`
3. Implement `nll(self, params, state, t, x, y, train, key)`
4. Add to `get_subspace_model()` lookup

### Adding a New Regularizer/Mixin

1. Create mixin class inheriting from `SubspaceBaseModel`
2. Override `compute_loss_t()` to add regularizer term
3. Pass mixin params via `ModelParams.get_model_kwargs()`
