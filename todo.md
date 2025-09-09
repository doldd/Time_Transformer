# TODO: Add Qwen2.5-7B Model to Time Transformer Pipeline

## Model Availability Assessment ✅

**Paper**: "Scalable Bayesian Low‑Rank Adaptation of LLMs" uses **Qwen2.5-7B** as the base model.

**JAX/Flax Implementation Status**: ❌ **NOT OFFICIALLY AVAILABLE**
- ✅ Tokenizer loads successfully: `Qwen/Qwen2.5-7B` (vocab: 151,643, max_len: 131,072)
- ❌ No official Flax implementation in transformers library
- ❌ `benjamin/Qwen2.5-7B-flax` models have configuration issues
- ✅ PyTorch version available: `Qwen/Qwen2.5-7B`

**Root Cause**: `Qwen2Config` not supported by `FlaxAutoModelForCausalLM`. Supported: Llama, Mistral, GPT2, etc.

**Revised Decision**: 
- **Phase 1**: ✅ Test framework with **Llama-2-7B** (has official Flax support, similar architecture)
- **Phase 2**: Manual PyTorch→Flax conversion for Qwen2.5 OR use Llama as Qwen2.5 proxy

## Integration Steps (Next Phase)

### 1. Model Integration (Phase 1: Llama-2-7B Validation)
- [x] ✅ Test loading `Qwen/Qwen2.5-7B` tokenizer (COMPLETED)
- [x] ✅ Identify Flax compatibility issues (COMPLETED)  
- [ ] Test loading `meta-llama/Llama-2-7b-hf` with Flax/JAX
- [ ] Create `LlamaWrapper` class similar to `timesfmWrapper`
- [ ] Verify Llama model forward pass compatibility with our framework
- [ ] Add to `src/models.py` or create separate `llama_models.py`

### 1b. Model Integration (Phase 2: Qwen2.5 Custom Implementation)
- [ ] Study Qwen2.5 vs Llama architecture differences
- [ ] Create custom Flax implementation of Qwen2.5 architecture
- [ ] Port PyTorch weights to Flax format
- [ ] Create `Qwen25Wrapper` class
- [ ] Verify parameter compatibility and forward pass

### 2. SubspaceBaseModel Integration  
- [ ] Create `Qwen25LanguageModelSubspace(SubspaceBaseModel)` class
- [ ] Implement task-specific `nll()` method for:
  - [ ] Text generation (next-token prediction)  
  - [ ] Question answering (if needed)
  - [ ] Classification tasks (multiple choice)
- [ ] Add LoRA parameter filtering for Qwen2.5 architecture

### 3. Configuration & Training
- [ ] Create `configs/qwen_config.yaml` for hyperparameter sweeps
- [ ] Add script `qwen25_PtoC.py` following pattern from:
  - `timeseries_timesfmToCurve.py` 
  - `cifar10_PtoC.py`
- [ ] Configure curve parameters (`k`, batch size, learning rates)
- [ ] Set up LoRA masks for attention layers (query, key, value projections)

### 4. Evaluation Pipeline
- [ ] Implement text generation metrics (perplexity, BLEU, etc.)
- [ ] Add uncertainty quantification for language tasks
- [ ] Create visualization for parameter curves in text generation context
- [ ] Test on common NLP benchmarks (matching ScalaBL paper)

### 5. Documentation & Integration
- [ ] Update README with Qwen2.5 usage examples
- [ ] Add model architecture documentation
- [ ] Create example notebooks for text generation with curves
- [ ] Integration tests with existing curve infrastructure

## Key Architecture Notes
- **Model Size**: 7B parameters (similar to TimesFM scale)
- **Architecture**: Transformer decoder (compatible with our curve framework)  
- **LoRA Targets**: Query, Key, Value projections + output head
- **Memory**: Should fit within existing GPU memory constraints
- **Datasets**: Can use common NLP benchmarks or continue with reasoning tasks from ScalaBL paper 