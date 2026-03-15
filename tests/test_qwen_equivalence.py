#!/usr/bin/env python3
"""
Test script to verify JAX implementation matches PyTorch implementation.
Tests individual components and full model outputs.
"""

import os
import torch
import jax
import jax.numpy as jnp
import numpy as np
import traceback
from flax.core import freeze, unfreeze
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import (
    rotate_half as torch_rotate_half,
    apply_rotary_pos_emb as torch_apply_rotary_pos_emb,
    repeat_kv as torch_repeat_kv,
)

# Import our JAX implementation
from src.qwen_jax import (
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_emb,
    rotate_half,
    repeat_kv,
    QwenMLP,
    QwenAttention,
    QwenDecoderLayer,
    QwenForCausalLM,
    create_qwen_config_from_pytorch,
)

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def load_pytorch_model():
    """Load PyTorch Qwen model"""
    print(f"Loading PyTorch model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=TORCH_DTYPE, device_map="auto", trust_remote_code=False
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    return model, tokenizer


def analyze_parameter_structure(pytorch_model, jax_params):
    """Comprehensive parameter structure analysis"""
    print("\n" + "=" * 60)
    print("DETAILED PARAMETER STRUCTURE ANALYSIS")
    print("=" * 60)

    # Count PyTorch parameters
    pytorch_param_count = sum(p.numel() for p in pytorch_model.parameters())
    pytorch_params = dict(pytorch_model.named_parameters())

    # Count JAX parameters
    jax_param_count = sum(p.size for p in jax.tree.leaves(jax_params))

    print(f"PyTorch Total Parameters: {pytorch_param_count:,}")
    print(f"JAX Total Parameters: {jax_param_count:,}")
    print(f"Parameter Count Match: {pytorch_param_count == jax_param_count}")

    if pytorch_param_count != jax_param_count:
        print(
            f"❌ PARAMETER MISMATCH: {abs(pytorch_param_count - jax_param_count):,} parameters difference"
        )
    else:
        print("✅ Parameter counts match perfectly!")

    print("\n" + "=" * 40)
    print("PYTORCH PARAMETER BREAKDOWN")
    print("=" * 40)

    pytorch_by_layer = {}
    for name, param in pytorch_params.items():
        layer_name = name.split(".")[0] if "." in name else name
        if layer_name not in pytorch_by_layer:
            pytorch_by_layer[layer_name] = []
        pytorch_by_layer[layer_name].append((name, param.shape, param.numel()))

    for layer_name, params in pytorch_by_layer.items():
        total_params = sum(p[2] for p in params)
        print(f"\n{layer_name}: {total_params:,} parameters")
        for name, shape, count in params:
            print(f"  {name}: {shape} = {count:,}")

    print("\n" + "=" * 40)
    print("JAX PARAMETER BREAKDOWN")
    print("=" * 40)

    jax_by_layer = {}
    for path, param in jax.tree.flatten_with_path(jax_params)[0]:
        path_str = "/".join(
            str(key.key) if hasattr(key, "key") else str(key) for key in path
        )
        layer_name = path_str.split("/")[1] if "/" in path_str else path_str
        if layer_name not in jax_by_layer:
            jax_by_layer[layer_name] = []
        jax_by_layer[layer_name].append((path_str, param.shape, param.size))

    for layer_name, params in jax_by_layer.items():
        total_params = sum(p[2] for p in params)
        print(f"\n{layer_name}: {total_params:,} parameters")
        for path, shape, count in params:
            print(f"  {path}: {shape} = {count:,}")

    # Find specific mismatches
    print("\n" + "=" * 40)
    print("PARAMETER MAPPING ANALYSIS")
    print("=" * 40)

    # Key layer comparisons
    layer_mappings = [
        ("model.embed_tokens.weight", "params/embed_tokens/embedding"),
        ("model.norm.weight", "params/model/norm/scale"),
        ("lm_head.weight", "params/lm_head/kernel"),
    ]

    for pytorch_name, jax_path in layer_mappings:
        if pytorch_name in pytorch_params:
            pytorch_param = pytorch_params[pytorch_name]
            print(f"\nComparing {pytorch_name} vs {jax_path}")
            print(
                f"  PyTorch: {pytorch_param.shape} = {pytorch_param.numel():,} params"
            )

            # Find corresponding JAX parameter
            jax_param = None
            for path, param in jax.tree.flatten_with_path(jax_params)[0]:
                path_str = "/".join(
                    str(key.key) if hasattr(key, "key") else str(key) for key in path
                )
                if jax_path in path_str:
                    jax_param = param
                    print(f"  JAX: {param.shape} = {param.size:,} params")
                    break

            if jax_param is None:
                print(f"  ❌ JAX parameter not found for {jax_path}")
            elif pytorch_param.numel() != jax_param.size:
                print(
                    f"  ❌ Size mismatch: {pytorch_param.numel()} vs {jax_param.size}"
                )
            else:
                print(f"  ✅ Match!")

    return pytorch_param_count == jax_param_count


def test_rotate_half():
    """Test rotate_half function equivalence"""
    print("\n=== Testing rotate_half function ===")

    # Create test tensor
    batch_size, seq_len, hidden_dim = 2, 8, 64
    x_torch = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.float32)
    x_jax = jnp.array(x_torch.numpy())

    # Test both implementations
    result_torch = torch_rotate_half(x_torch)
    result_jax = rotate_half(x_jax)

    # Compare results
    diff = np.abs(result_torch.numpy() - np.array(result_jax)).max()
    print(f"Max difference: {diff}")
    print(f"Test passed: {diff < 1e-6}")

    return diff < 1e-6


def test_repeat_kv():
    """Test repeat_kv function equivalence"""
    print("\n=== Testing repeat_kv function ===")

    # Create test tensor: (batch, num_key_value_heads, seq_len, head_dim)
    batch_size, num_kv_heads, seq_len, head_dim = 2, 2, 8, 64
    n_rep = 7  # 14 attention heads / 2 kv heads = 7

    x_torch = torch.randn(
        batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.float32
    )
    x_jax = jnp.array(x_torch.numpy())

    # Test both implementations
    result_torch = torch_repeat_kv(x_torch, n_rep)
    result_jax = repeat_kv(x_jax, n_rep)

    # Compare results
    diff = np.abs(result_torch.numpy() - np.array(result_jax)).max()
    print(f"Input shape: {x_torch.shape}")
    print(f"Output shape (torch): {result_torch.shape}")
    print(f"Output shape (jax): {result_jax.shape}")
    print(f"Max difference: {diff}")
    print(f"Test passed: {diff < 1e-6}")

    return diff < 1e-6


def test_rms_norm(pytorch_model):
    """Test RMSNorm equivalence"""
    print("\n=== Testing RMSNorm ===")

    # Get first RMSNorm layer from PyTorch model
    torch_rms_norm = pytorch_model.model.layers[0].input_layernorm
    hidden_size = pytorch_model.config.hidden_size
    eps = pytorch_model.config.rms_norm_eps

    # Get the device of the model
    model_device = next(pytorch_model.parameters()).device

    # Create test input
    batch_size, seq_len = 2, 8
    x_torch = torch.randn(
        batch_size, seq_len, hidden_size, dtype=torch.float32, device=model_device
    )
    x_jax = jnp.array(x_torch.cpu().numpy())

    # PyTorch forward pass
    with torch.no_grad():
        result_torch = torch_rms_norm(x_torch)

    # JAX forward pass
    jax_rms_norm = RMSNorm(eps=eps)
    rng = jax.random.PRNGKey(42)
    jax_params = jax_rms_norm.init(rng, x_jax)

    # Copy PyTorch weights to JAX
    weight_torch = torch_rms_norm.weight.detach().cpu().numpy()
    params_dict = unfreeze(jax_params)
    params_dict["params"]["weight"] = weight_torch
    params_dict = freeze(params_dict)

    result_jax = jax_rms_norm.apply(params_dict, x_jax)

    # Compare results
    diff = np.abs(result_torch.cpu().numpy() - np.array(result_jax)).max()
    print(f"Input shape: {x_torch.shape}")
    print(f"PyTorch weight shape: {weight_torch.shape}")
    print(f"Max difference: {diff}")
    print(f"Test passed: {diff < 1e-6}")

    return diff < 1e-6


def test_rotary_embedding(pytorch_model):
    """Test RotaryEmbedding equivalence"""
    print("\n=== Testing RotaryEmbedding ===")

    # Get rotary embedding from PyTorch model
    torch_rope = pytorch_model.model.rotary_emb
    config = pytorch_model.config

    # Get the device of the model
    model_device = next(pytorch_model.parameters()).device

    # Create test input and position_ids
    batch_size, seq_len = 2, 16
    head_dim = config.hidden_size // config.num_attention_heads

    x_torch = torch.randn(
        batch_size,
        seq_len,
        config.hidden_size,
        dtype=torch.float32,
        device=model_device,
    )
    position_ids_torch = (
        torch.arange(seq_len, dtype=torch.long, device=model_device)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )

    x_jax = jnp.array(x_torch.cpu().numpy())
    position_ids_jax = jnp.array(position_ids_torch.cpu().numpy())

    # PyTorch forward pass
    with torch.no_grad():
        cos_torch, sin_torch = torch_rope(x_torch, position_ids_torch)

    # JAX forward pass
    jax_rope = RotaryEmbedding(head_dim=head_dim, rope_theta=config.rope_theta)
    rng = jax.random.PRNGKey(42)
    _ = jax_rope.init(rng, x_jax, position_ids_jax)

    cos_jax, sin_jax = jax_rope.apply({}, x_jax, position_ids_jax)
    # cos_jax, sin_jax = jax_rope(x_jax, position_ids_jax)

    # Compare results
    cos_diff = np.abs(cos_torch.cpu().numpy() - np.array(cos_jax)).max()
    sin_diff = np.abs(sin_torch.cpu().numpy() - np.array(sin_jax)).max()

    print(f"PyTorch cos shape: {cos_torch.shape}")
    print(f"JAX cos shape: {cos_jax.shape}")
    print(f"PyTorch sin shape: {sin_torch.shape}")
    print(f"JAX sin shape: {sin_jax.shape}")
    print(f"Cos max difference: {cos_diff}")
    print(f"Sin max difference: {sin_diff}")
    print(f"Test passed: {cos_diff < 1e-5 and sin_diff < 1e-5}")

    return cos_diff < 1e-5 and sin_diff < 1e-5


def test_apply_rotary_pos_emb():
    """Test apply_rotary_pos_emb equivalence"""
    print("\n=== Testing apply_rotary_pos_emb ===")

    # Create test tensors
    batch_size, num_heads, seq_len, head_dim = 2, 14, 8, 64

    q_torch = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
    k_torch = torch.randn(
        batch_size, 2, seq_len, head_dim, dtype=torch.float32
    )  # 2 kv heads
    cos_torch = torch.randn(batch_size, seq_len, head_dim, dtype=torch.float32)
    sin_torch = torch.randn(batch_size, seq_len, head_dim, dtype=torch.float32)

    q_jax = jnp.array(q_torch.numpy())
    k_jax = jnp.array(k_torch.numpy())
    cos_jax = jnp.array(cos_torch.numpy())
    sin_jax = jnp.array(sin_torch.numpy())

    # PyTorch forward pass
    q_embed_torch, k_embed_torch = torch_apply_rotary_pos_emb(
        q_torch, k_torch, cos_torch, sin_torch, unsqueeze_dim=1
    )

    # JAX forward pass
    q_embed_jax = apply_rotary_emb(q_jax, cos_jax, sin_jax)
    k_embed_jax = apply_rotary_emb(k_jax, cos_jax, sin_jax)

    # Compare results
    q_diff = np.abs(q_embed_torch.numpy() - np.array(q_embed_jax)).max()
    k_diff = np.abs(k_embed_torch.numpy() - np.array(k_embed_jax)).max()

    print(f"Q max difference: {q_diff}")
    print(f"K max difference: {k_diff}")
    print(f"Test passed: {q_diff < 1e-5 and k_diff < 1e-5}")

    return q_diff < 1e-5 and k_diff < 1e-5


def test_compute_posterior_weights():
    """Test _compute_posterior_weights behaviour."""
    print("\n=== Testing _compute_posterior_weights ===")

    # Local import to avoid top-level dependency ordering
    from qwen_fineTuning_PtoC_fixCps import _compute_posterior_weights

    # Simple log-likelihoods and temperatures
    log_like = jnp.array([0.0, 1.0, 2.0])
    temperatures = [1.0, 0.5, 10.0]

    weights = _compute_posterior_weights(log_like, temperatures)
    weights_np = np.array(weights)

    # Shape: (len(temperatures), n_samples)
    expected_shape = (len(temperatures), log_like.shape[0])
    print(f"Weights shape: {weights_np.shape}, expected: {expected_shape}")
    if weights_np.shape != expected_shape:
        print("Shape mismatch")
        return False

    # Each row should sum to ~1 (softmax over samples)
    row_sums = weights_np.sum(axis=-1)
    print(f"Row sums: {row_sums}")
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        print("Rows do not sum to 1")
        return False

    # All weights must be non-negative
    if (weights_np < 0).any():
        print("Negative weights found")
        return False

    # Low temperature => more concentrated on max log-likelihood
    max_idx = int(jnp.argmax(log_like))
    low_temp_idx = 1  # temperature 0.5
    concentrated_on_max = int(np.argmax(weights_np[low_temp_idx])) == max_idx

    # High temperature => closer to uniform distribution
    high_temp_idx = 2  # temperature 10.0
    uniform_approx = np.allclose(
        weights_np[high_temp_idx],
        np.ones_like(weights_np[high_temp_idx]) / weights_np.shape[1],
        atol=5e-2,
    )

    print(f"Concentrated on max at low temp: {concentrated_on_max}")
    print(f"High-temp approx uniform: {uniform_approx}")

    # Pytest-style assertions
    assert weights_np.shape == expected_shape
    assert np.allclose(row_sums, 1.0, atol=1e-6)
    assert np.all(weights_np >= 0)
    assert concentrated_on_max
    assert uniform_approx


def test_mlp(pytorch_model):
    """Test MLP equivalence"""
    print("\n=== Testing MLP ===")

    # Get first MLP layer from PyTorch model
    torch_mlp = pytorch_model.model.layers[0].mlp
    config = pytorch_model.config

    # Get the device of the model
    model_device = next(pytorch_model.parameters()).device

    # Create test input with matching dtype
    batch_size, seq_len = 2, 8
    model_dtype = torch_mlp.gate_proj.weight.dtype
    x_torch = torch.randn(
        batch_size, seq_len, config.hidden_size, dtype=model_dtype, device=model_device
    )
    x_jax = jnp.array(x_torch.cpu().numpy())

    # PyTorch forward pass
    with torch.no_grad():
        result_torch = torch_mlp(x_torch)

    # JAX forward pass
    jax_mlp = QwenMLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
    )
    rng = jax.random.PRNGKey(42)
    jax_params = jax_mlp.init(rng, x_jax)

    # Copy PyTorch weights to JAX (need to transpose for Dense layers)
    gate_weight = torch_mlp.gate_proj.weight.detach().cpu().numpy().T
    up_weight = torch_mlp.up_proj.weight.detach().cpu().numpy().T
    down_weight = torch_mlp.down_proj.weight.detach().cpu().numpy().T

    params_dict = unfreeze(jax_params)
    params_dict["params"]["gate_proj"]["kernel"] = gate_weight
    params_dict["params"]["up_proj"]["kernel"] = up_weight
    params_dict["params"]["down_proj"]["kernel"] = down_weight
    params_dict = freeze(params_dict)

    result_jax = jax_mlp.apply(params_dict, x_jax)

    # Compare results
    diff = np.abs(result_torch.cpu().numpy() - np.array(result_jax)).max()
    print(f"Input shape: {x_torch.shape}")
    print(f"Output shape (torch): {result_torch.shape}")
    print(f"Output shape (jax): {result_jax.shape}")
    print(f"Max difference: {diff}")
    print(f"Model dtype: {model_dtype}")
    # Relax tolerance for half precision
    tolerance = 1e-2 if model_dtype == torch.float16 else 1e-4
    print(f"Using tolerance: {tolerance}")
    print(f"Test passed: {diff < tolerance}")

    return diff < tolerance


def test_attention(pytorch_model):
    """Test Qwen2Attention equivalence"""
    print("\n=== Testing Qwen2Attention ===")

    # Get first attention layer from PyTorch model
    torch_attention = pytorch_model.model.layers[0].self_attn
    ...(truncated)
