# Time Transformer

Bézier-curve parameterized neural networks with LoRA integration for fine-tuning large models in a subspace of parameter space.
This framework enables fine-tuning of Large Language Models (LLMs) within a parameter subspace by learning weights as continuous Bezier curves rather than static points.

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
├── compute_curve_logits.py          # Offline curve evaluation
├── plot_curve_logits.py             # Visualization utilities
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
