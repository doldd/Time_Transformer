# Time Transformer

Bézier-curve parameterized neural networks with LoRA integration for fine-tuning large models in a subspace of parameter space.

## Quick Start

### Environment Setup

```bash
conda activate jaxTorchTrans
```

Set XLA environment variables before running:

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=10"
export JAX_COMPILATION_CACHE_DIR="jax_cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
```

### Smoke Test

```bash
CUDA_VISIBLE_DEVICES="2" \
  python qwen_fineTuning_PtoC_fixCps.py --batch-size=4 --cp-fix 1 0 1 0 1 --smoke-test
```

## Training Modes

### 1. Pretraining Mode (k=0)

Train fixed control points independently before fitting the curve:

```bash
# Set Pretraining=True in config, cp_fix=[1, 0, 1] for example
# Each fixed CP (value=1) is trained separately
```

### 2. Curve Fitting Mode (k>0)

Fit Bézier curve connecting the control points:

```bash
# num_curve_segment=2, SegDeg=2 → k=4 control points
# cp_fix=[1, 0, 0, 0, 1] fixes endpoints, trains intermediate points
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

### Config Structure

Key configuration paths (in YAML sweep configs or inline dict):

**Model Parameters** (`model_params`):
- `num_curve_segment`: Number of curve segments
- `SegDeg`: Degree of each segment
- `k = num_curve_segment * SegDeg` (total control points - 1)
- `Pretraining`: Enable k=0 pretraining mode
- `curve_sampling_mode`: Evaluation mode
- `subspace_model`: Model type (e.g., `"lora_category"`, `"jsd_noise_sampling_dropout_category"`)
- `filter_masks`: Which parameters become curve parameters
- `lora_params`: LoRA configuration (r, alpha, mode, filter_masks)

**Training Hyperparameters** (`train_hyper`):
- `batch_size`: Batch size
- `num_steps`: Total training steps (-1 for epoch-based)
- `num_epochs`: Number of epochs
- `eval_every_n_batch`: Evaluation frequency
- `dataset_sampling`: `"minibatch"` or `"expert"`
- `save_params`: Save parameters as artifacts

**Optimizer** (`optimizer_conf`):
- `name`: Optimizer type (e.g., `"adamw"`)
- `kwargs.learning_rate`: Can be dict with schedule (e.g., `linear_onecycle_schedule`)
- `freeze_other_params`: Freeze non-curve parameters

### Example Config (Inline)

```python
config = {
    "model_params": {
        "num_curve_segment": 2,
        "SegDeg": 2,
        "Pretraining": False,
        "curve_sampling_mode": "combined_test",
        "subspace_model": "lora_category",
        "filter_masks": [
            {"keys": ["self_attn", "q_proj", "kernel"], "op": "all"},
            {"keys": ["self_attn", "v_proj", "kernel"], "op": "all"},
            {"keys": ["lm_head", "kernel"], "op": "all"}
        ],
        "lora_params": {
            "use_lora": True,
            "r": 8,
            "lora_alpha": 16.0,
            "lora_mode": "A(t)eB(t)",
            "filter_masks": [
                {"keys": ["self_attn", "q_proj", "kernel"], "op": "all", "dims": [1, 0]},
                {"keys": ["self_attn", "v_proj", "kernel"], "op": "all", "dims": [1, 0]},
                {"keys": ["lm_head", "kernel"], "op": "all", "dims": [1, 0]}
            ]
        }
    },
    "train_hyper": {
        "batch_size": 4,
        "num_steps": 10000,
        "dataset_sampling": {"minibatch": {}}
    },
    "optimizer_conf": {
        "name": "adamw",
        "kwargs": {
            "learning_rate": {
                "name": "linear_onecycle_schedule",
                "kwargs": {
                    "transition_steps": 10000,
                    "peak_value": 1e-4,
                    "pct_start": 0.12
                }
            },
            "weight_decay": 0.001
        },
        "freeze_other_params": True
    }
}
```

## Project Layout

```
.
├── qwen_fineTuning_PtoC_fixCps.py   # Main training entry point
├── src/
│   ├── subspace_curve.py             # Bézier math, SubspaceBaseModel, LoRA mixins
│   ├── qwen_jax.py                   # Qwen JAX model, PyTorch→JAX conversion
│   ├── utils.py                      # Utility functions (ECE, curve metrics)
│   └── weight_matching.py            # Permutation matching utilities
├── configs_newDS/                    # Wandb sweep configurations
│   ├── OpenFlat*.yaml               # Open curve configurations
│   ├── PretrainedFlat*.yaml         # Pretrained CP configurations
│   └── *.yaml                       # Dataset-specific configs
├── dataset/
│   ├── S2ClassDataset.py            # Classification dataset loader
│   └── utils/                       # Dataset utilities
├── save_datasets_and_params.py      # Dataset/model export utilities
├── compute_curve_logits.py          # Offline curve evaluation
├── plot_curve_logits.py             # Visualization utilities
└── AGENTS.md                        # Detailed agent instructions
```

## Architecture

### Subspace Curves

Instead of training a single point in parameter space, this framework trains a Bézier curve:

- **Control Points**: k+1 points in parameter space
- **Curve Parameterization**: `θ(t) = Σᵢ Bᵢ(t) · CPᵢ` where Bᵢ are Bézier basis functions
- **Segmented Curves**: Multiple connected Bézier segments for higher flexibility

### LoRA Integration

Low-rank adapters are parameterized along the curve:
- **A(t)B(t)**: Both A and B matrices vary along curve
- **A(t)eB(t)**: A varies, B is fixed (exponential form)

### Key Files

- **`src/subspace_curve.py`**: Core subspace math, model registry, mixin classes
- **`src/qwen_jax.py`**: Qwen model wrapper, weight conversion, serialization
- **`qwen_fineTuning_PtoC_fixCps.py`**: Training orchestration, evaluation pipeline

## Serialization

Parameters and config are saved as:
- `jax_params.npy`: NumPy array of unfrozen parameter dict
- `jax_config.json`: Experiment configuration

Load with:
```python
from src.qwen_jax import load_params_and_config
params, config = load_params_and_config(path)
```

Convert PyTorch weights:
```python
from src.qwen_jax import convert_pytorch_weights_to_jax
jax_params = convert_pytorch_weights_to_jax(jax_params, torch_model, config)
```

## Development

### Adding a New Subspace Model

1. Create class inheriting `SubspaceBaseModel` in `src/subspace_curve.py`
2. Register with `@register_subspace_model("name")`
3. Implement `nll(self, params, state, t, x, y, train, key)`
4. Add to `get_subspace_model()` lookup

### Adding a New Regularizer/Mixin

1. Create mixin class inheriting from `SubspaceBaseModel`
2. Override `compute_loss_t()` to add regularizer term
3. Pass mixin params via `ModelParams.get_model_kwargs()`

### Debugging Tips

- Check `config.model_params.k` to confirm curve degree
- Verify masks: `jax.tree.map(lambda x: x.shape, curve_mask)`
- Use `--smoke-test` for quick iteration
- Log `cp_fix` to verify control point configuration

## See Also

- **AGENTS.md**: Detailed agent instructions for development tasks
- **configs_newDS/**: Example sweep configurations
