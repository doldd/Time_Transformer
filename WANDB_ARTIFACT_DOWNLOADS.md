# WandB Artifact Downloads - Parameter Loading Reference

## Overview
This document documents all locations in the codebase where saved parameters and other artifacts are downloaded from WandB.

---

## 1. Core Utility Functions (`src/utils.py`)

### Function: `get_cp_w()` - Lines 202-225
**Purpose:** Load curve parameters from logged artifacts

**Key Features:**
- Downloads curve parameters (*.npy files) from pytree-type artifacts
- Supports both logged and used artifacts
- Optional `last=True` parameter to load last checkpoint

**Implementation:**
```python
def get_cp_w(run, from_path_optim_sweep=True, last=False):
    if from_path_optim_sweep:
        rrun_art = run.logged_artifacts()
    else:
        rrun_art = run.used_artifacts()
    for art in rrun_art:
        if art.type == "pytree":
            if from_path_optim_sweep:
                filename = f"{run.id}_params"
            else:
                filename = f"{art.logged_by().id}_params"
            if last:
                filename += "_last"
            filename += ".npy"
            
            file = art.get_entry(filename).download(root="./art_tmp")
            params = jnp.load(file, allow_pickle=True).item()
            return params
```

**Artifact Details:**
- Type: `"pytree"`
- File naming: `{run_id}_params.npy` or `{logged_by_id}_params.npy`
- Download root: `./art_tmp`

---

### Function: `get_data()` - Lines 228-243
**Purpose:** Load training/validation/test data from artifacts

**Key Features:**
- Downloads numpy arrays containing train/val/test splits
- Returns: x, y, x_val, y_val, x_test, y_test

**Artifact Details:**
- Type: `"numpy"`
- File naming: `{rng_seed}_{dataset}_data.npz`
- Download root: `./art_tmp`

---

### Function: `get_params_from_run()` - Lines 320-326
**Purpose:** Simple parameter loading from run artifacts

**Implementation:**
```python
def get_params_from_run(run, file_name_without_id="params.npy"):
    for art in run.logged_artifacts():
        if art.type == "pytree":
            file = art.get_entry(f"{run.id}_{file_name_without_id}").download(root="./art_tmp")
            params_loaded = jnp.load(file, allow_pickle=True).item()
            return params_loaded
```

**Artifact Details:**
- Type: `"pytree"`
- File naming: `{run_id}_{file_name_without_id}`
- Default file name: `params.npy`

---

### Function: `load_checkpoint_new()` - Lines 478-554
**Purpose:** Load complete checkpoint including trainable params, BMA weights, and pretrained params

**Key Features:**
- Comprehensive parameter loading from artifacts
- Supports multiple parameter types:
  - `init_trainable_params`
  - `last_trainable_params`
  - `trainable_params`
  - `pretrained_params`
  - `bma_weights`
- Handles LoRA parameter loading with fallback for missing `lora_rho` attributes

**Implementation:**
```python
def load_checkpoint_new(run, params, s_model):
    def load_art(load_run):
        artifact_use, artifact_weights_use = False, False
        for art_l in load_run.logged_artifacts():
            if art_l.type == "pytree" and art_l.name.startswith("params"):
                artifact_use = art_l
            if art_l.type == "npz" and art_l.name.startswith("bma_weights"):
                artifact_weights_use = art_l
        return artifact_use, artifact_weights_use
    
    artifact_use, artifact_weights_use = load_art(run)
    
    # Load files
    for f in artifact_use.files():
        if 'trainable_params' in f.name:
            if 'init' in f.name:
                init_trainable_params = np.load(artifact_use.get_entry(f.name).download(), allow_pickle=True).item()
            elif 'last' in f.name:
                last_trainable_params = np.load(artifact_use.get_entry(f.name).download(), allow_pickle=True).item()
            else:
                trainable_params = np.load(artifact_use.get_entry(f.name).download(), allow_pickle=True).item()
        elif 'pretrained_params' in f.name:
            pretrained_params = np.load(artifact_use.get_entry(f.name).download(), allow_pickle=True)
```

**Artifact Details:**
- Params Type: `"pytree"` (artifact name starts with `"params"`)
- Weights Type: `"npz"` (artifact name starts with `"bma_weights"`)
- Files searched: `trainable_params*.npy`, `pretrained_params*.npy`, `weights*.npz`

---

### Function: `load_config_and_data()` - Lines 628+
**Purpose:** Load configuration and dataset from wandb run for training

**Key Features:**
- Supports local artifact paths or direct wandb paths
- Default local path: `"artifacts"`
- Resolves model and dataset artifacts

---

## 2. CIFAR-10 Training (`cifar10_PtoC.py`)

### Location: `single_run()` function - Lines 228-235
**Purpose:** Load initial parameters from artifact for curve initialization

**Implementation:**
```python
if 'params_path' in config:
    art = logger.use_artifact(config['params_path'], type='pytree')
    artifact_dir = art.download()
    filename = os.path.join(artifact_dir, f"{art.logged_by().id}_params.npy")
    init_params = jnp.load(filename, allow_pickle=True).item()
```

**Artifact Details:**
- Type: `"pytree"`
- Config key: `config['params_path']`
- File naming: `{logged_by_id}_params.npy`

---

## 3. Qwen Fine-Tuning with Point-to-Curve (`qwen_fineTuning_PtoC_fixCps.py`)

### Location: `train()` function - Lines 1927-1935
**Purpose:** Load model artifact from wandb when using reference path

**Implementation:**
```python
artifact = wandb.Artifact(name="params", type="pytree")

# Resolve model path from wandb if a local path is not given
net_kwargs = dict(config_dict['net_kwargs'])
if net_kwargs['model_path'].startswith('ddold/'):
    model_artifact = logger.use_artifact(net_kwargs['model_path'], type='model')
    net_kwargs['model_path'] = model_artifact.download()
```

**Artifact Details:**
- Type: `"model"`
- Path pattern: `ddold/{model_path}`
- Returns downloaded directory path

### Location: Dataset loading in `_prepare_datasets()` - Lines 540-550
**Purpose:** Load dataset artifact

**Implementation:**
```python
if run:
    artifact = run.use_artifact(dataset_path, type='dataset')
    dataset_path = artifact.download()

dataset_path = os.path.join(dataset_path, "test_data.npz" if test else "train_data.npz")
```

---

## 4. Qwen Fine-Tuning with Deep Ensemble (`qwen_fineTuning_DE.py`)

### Location: `load_text_classification_data()` - Lines 110-120
**Purpose:** Load classification dataset artifact

**Implementation:**
```python
def load_text_classification_data(dataset_path: str, run=None, test=False):
    if run:
        artifact = run.use_artifact(dataset_path, type='dataset')
        artifact_dir = artifact.download()
        dataset_path = os.path.join(artifact_dir, "test_data.npz" if test else "train_data.npz")
```

**Artifact Details:**
- Type: `"dataset"`
- Split files: `train_data.npz`, `test_data.npz`

### Location: `train()` sweep parameter loading - Lines 450-500
**Purpose:** Load trained parameters from matching sweep run

**Implementation:**
```python
def load_art(run):
    artifact_use, artifact_weights_use = False, False
    for art_l in run.logged_artifacts():
        if art_l.type == "pytree" and art_l.name.startswith("params"):
            print(f"Downloading params from {art_l.name}")
            artifact_use = logger.use_artifact(art_l, type='pytree')
        if art_l.type == "npz" and art_l.name.startswith("bma_weights"):
            print(f"Downloading weights from {art_l.name}")
            artifact_weights_use = logger.use_artifact(art_l, type='npz')
    return artifact_use, artifact_weights_use

# Load trainable params
for f in artifact_use.files():
    if 'trainable_params' in f.name:
        trainable_params = np.load(artifact_use.get_entry(f.name).download(), allow_pickle=True).item()
```

**Artifact Details:**
- Params Type: `"pytree"` (artifact name starts with `"params"`)
- Weights Type: `"npz"` (artifact name starts with `"bma_weights"`)

---

## 5. Toy Regression (`toyReg_PtoC.py`)

### Similar pattern to CIFAR-10
- Downloads artifact using `logger.use_artifact()`
- Extracts directory and loads `.npy` files
- Line 162: `artifact_dir = art.download()`

---

## Common Download Patterns

### Pattern 1: Direct artifact download (returns directory)
```python
artifact = run.use_artifact(path, type='type_name')
artifact_dir = artifact.download()
file_path = os.path.join(artifact_dir, filename)
```

### Pattern 2: Get specific file entry
```python
artifact = run.use_artifact(path, type='type_name')
file = artifact.get_entry(filename).download(root="./art_tmp")
data = jnp.load(file, allow_pickle=True)
```

### Pattern 3: Iterate through artifact files
```python
artifact = run.use_artifact(path, type='type_name')
for f in artifact.files():
    if condition(f.name):
        data = np.load(artifact.get_entry(f.name).download(), allow_pickle=True)
```

---

## Artifact Type Summary

| Type | Purpose | File Patterns |
|------|---------|---------------|
| `"pytree"` | JAX/Flax parameters | `*_params.npy`, `*_init_params.npy`, `*_last_params.npy` |
| `"model"` | Complete model weights | Directory structure |
| `"dataset"` | Preprocessed data | `train_data.npz`, `test_data.npz` |
| `"npz"` | Bayesian Model Averaging weights | `*_weights*.npz` |
| `"numpy"` | Raw numpy arrays | `*_data.npz` |
| `"xarray"` | MCMC samples | `*_samples.nc` |

---

## Download Directory Conventions

- **Temporary artifact downloads:** `./art_tmp/` (used by `get_entry().download()`)
- **Model paths:** Stored as configured paths (local or `ddold/` prefix for wandb)
- **Default patterns:** `{run_id}` or `{logged_by_id}` prepended to filenames

---

## Key Config Keys

- `config['params_path']` - Artifact path for initial parameters
- `config['dataset_path']` - Artifact path or local path to dataset
- `config['net_kwargs']['model_path']` - Model artifact path (often starts with `ddold/`)
- `SWEEP_PARAM_PATH` - Path to sweep containing parameters to load

---

## Notes

1. All downloads use `allow_pickle=True` for `.npy` and `.npz` files (JAX pytrees)
2. The `art_tmp` directory is used as a cache/temporary location
3. Some functions distinguish between `logged_artifacts()` and `used_artifacts()`
4. LoRA parameters require special handling with fallback for `lora_rho` attribute
5. Most artifact paths can use either local paths or wandb reference paths (`{entity}/{project}/{artifact_name}:{version}`)
