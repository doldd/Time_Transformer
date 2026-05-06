# JAX/Flax Implementation of Qwen2 Model
# Based on HuggingFace Transformers Qwen2 implementation

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import unfreeze, freeze
import json
import os


def rotate_half(x):
    """Rotates half the hidden dims of the input (JAX version)"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to input tensor (JAX version)

    Args:
        x: shape [batch, seq_len, num_heads, head_dim] or [batch, num_heads, seq_len, head_dim]
        cos: shape [batch, seq_len, head_dim] or [seq_len, head_dim]
        sin: shape [batch, seq_len, head_dim] or [seq_len, head_dim]
    """
    # Handle different input dimensions
    if x.ndim == 4:
        if cos.ndim == 2:  # [seq_len, head_dim]
            # Add batch and head dimensions for broadcasting
            if x.shape[1] == cos.shape[0]:  # [batch, seq_len, num_heads, head_dim]
                cos = cos[None, :, None, :]  # [1, seq_len, 1, head_dim]
                sin = sin[None, :, None, :]
            else:  # [batch, num_heads, seq_len, head_dim]
                cos = cos[None, None, :, :]  # [1, 1, seq_len, head_dim]
                sin = sin[None, None, :, :]
        elif cos.ndim == 3:  # [batch, seq_len, head_dim]
            if x.shape[1] == cos.shape[1]:  # [batch, seq_len, num_heads, head_dim]
                cos = cos[:, :, None, :]  # [batch, seq_len, 1, head_dim]
                sin = sin[:, :, None, :]
            else:  # [batch, num_heads, seq_len, head_dim]
                cos = cos[:, None, :, :]  # [batch, 1, seq_len, head_dim]
                sin = sin[:, None, :, :]

    return x * cos + rotate_half(x) * sin


def repeat_kv(hidden_states, n_rep):
    """
    Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep) for JAX
    hidden_states: (batch, num_key_value_heads, seq_len, head_dim)
    returns: (batch, num_attention_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return hidden_states

    # Expand and reshape
    hidden_states = jnp.repeat(hidden_states, n_rep, axis=1)
    return hidden_states


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (JAX version)"""

    eps: float = 1e-6

    @nn.compact
    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)

        weight = self.param("weight", nn.initializers.ones, (hidden_states.shape[-1],))
        return weight * hidden_states.astype(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (JAX version)"""

    head_dim: int
    rope_theta: float = 10000.0

    @nn.compact
    def __call__(self, x, position_ids):
        """Forward pass matching PyTorch implementation"""
        # Compute inverse frequencies as a parameter
        inv_freq = 1.0 / (
            self.rope_theta
            ** (jnp.arange(0, self.head_dim, 2, dtype=jnp.float32) / self.head_dim)
        )
        # inv_freq = self.param('inv_freq', lambda key, shape: inv_freq, inv_freq.shape)

        # x is used for device and dtype info
        batch_size = position_ids.shape[0]
        inv_freq_expanded = inv_freq[None, :, None].astype(jnp.float32)
        inv_freq_expanded = jnp.broadcast_to(
            inv_freq_expanded, (batch_size, inv_freq.shape[0], 1)
        )
        position_ids_expanded = position_ids[:, None, :].astype(jnp.float32)

        # Compute frequencies
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(0, 2, 1)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)

        return cos.astype(x.dtype), sin.astype(x.dtype)


class QwenMLP(nn.Module):
    """Qwen MLP with SwiGLU activation (JAX version)"""

    hidden_size: int
    intermediate_size: int
    hidden_act: str = "silu"

    @nn.compact
    def __call__(self, x):
        # Gate and Up projections (no bias)
        gate = nn.Dense(self.intermediate_size, use_bias=False, name="gate_proj")(x)
        up = nn.Dense(self.intermediate_size, use_bias=False, name="up_proj")(x)

        # Apply activation function (SiLU/Swish)
        if self.hidden_act == "silu":
            gate_activated = nn.silu(gate)
        else:
            raise ValueError(f"Unsupported activation: {self.hidden_act}")

        # Element-wise multiplication (SwiGLU)
        intermediate = gate_activated * up

        # Down projection (no bias)
        output = nn.Dense(self.hidden_size, use_bias=False, name="down_proj")(
            intermediate
        )

        return output


class QwenAttention(nn.Module):
    """Multi-head attention with grouped query attention (JAX version)"""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    # max_position_embeddings: int = 32768
    attention_dropout: float = 0.0
    rope_theta: float = 10000.0

    @nn.compact
    def __call__(
        self, hidden_states, attention_mask=None, position_ids=None, train=True
    ):
        num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        scaling = self.head_dim**-0.5
        rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            rope_theta=self.rope_theta,
            name="rotary_emb",
        )

        batch_size, seq_len, _ = hidden_states.shape

        # Linear projections (Qwen2 uses bias=True for q,k,v and bias=False for o)
        q = nn.Dense(
            self.num_attention_heads * self.head_dim, use_bias=True, name="q_proj"
        )(hidden_states)
        k = nn.Dense(
            self.num_key_value_heads * self.head_dim, use_bias=True, name="k_proj"
        )(hidden_states)
        v = nn.Dense(
            self.num_key_value_heads * self.head_dim, use_bias=True, name="v_proj"
        )(hidden_states)

        # Reshape for multi-head attention
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = q.reshape(hidden_shape).transpose(
            0, 2, 1, 3
        )  # [batch, heads, seq_len, head_dim]
        key_states = k.reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = v.reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        # Apply rotary embeddings
        if position_ids is None:
            position_ids = jnp.arange(seq_len, dtype=jnp.int32)[None, :]

        cos, sin = rotary_emb(hidden_states, position_ids)
        query_states = apply_rotary_emb(query_states, cos, sin)
        key_states = apply_rotary_emb(key_states, cos, sin)

        # Repeat k,v for grouped query attention
        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)

        # Compute attention scores
        attn_weights = (
            jnp.matmul(query_states, jnp.transpose(key_states, (0, 1, 3, 2))) * scaling
        )

        # Always apply causal mask
        causal_mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=bool), k=1)
        causal_mask = jnp.logical_not(causal_mask[None, None, :, :])

        # Broadcast causal_mask to match attn_weights shape: [1, 1, seq_len, seq_len]
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].astype(bool)
            causal_mask = jnp.logical_and(attention_mask, causal_mask)

        ## with where argument to avoid NaNs from -inf of all-masked rows in softmax
        attn_weights = nn.softmax(attn_weights, axis=-1, where=causal_mask)

        # Apply dropout
        if train and self.attention_dropout > 0.0:
            attn_weights = nn.Dropout(rate=self.attention_dropout)(
                attn_weights, deterministic=not train
            )

        # Apply attention to values
        attn_output = jnp.matmul(attn_weights, value_states)

        # Transpose back and reshape
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(*input_shape, -1)

        # Output projection (no bias)
        output = nn.Dense(self.hidden_size, use_bias=False, name="o_proj")(attn_output)

        return output


class QwenDecoderLayer(nn.Module):
    """Single transformer decoder layer (JAX version)"""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    rope_theta: float = 10000.0

    # @functools.partial(nn.remat, static_argnums=(4,))
    @nn.compact
    def __call__(
        self, hidden_states, attention_mask=None, position_ids=None, train=True
    ):
        head_dim = self.hidden_size // self.num_attention_heads

        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = RMSNorm(eps=self.rms_norm_eps, name="input_layernorm")(
            hidden_states
        )
        hidden_states = QwenAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=head_dim,
            attention_dropout=self.attention_dropout,
            rope_theta=self.rope_theta,
            name="self_attn",
        )(hidden_states, attention_mask, position_ids, train)
        hidden_states = residual + hidden_states

        # MLP with residual connection
        residual = hidden_states
        hidden_states = RMSNorm(eps=self.rms_norm_eps, name="post_attention_layernorm")(
            hidden_states
        )
        hidden_states = QwenMLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            name="mlp",
        )(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QwenForCausalLM(nn.Module):
    """Qwen model for causal language modeling (JAX version)"""

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0

    @nn.compact
    def __call__(
        self, input_ids, attention_mask=None, position_ids=None, train=True
    ) -> jnp.ndarray:
        embed_layer = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            name="embed_tokens",
        )
        hidden_states = embed_layer(input_ids)

        for i in range(self.num_hidden_layers):
            hidden_states = QwenDecoderLayer(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                intermediate_size=self.intermediate_size,
                hidden_act=self.hidden_act,
                rms_norm_eps=self.rms_norm_eps,
                attention_dropout=self.attention_dropout,
                rope_theta=self.rope_theta,
                name=f"layers_{i}",
            )(hidden_states, attention_mask, position_ids, train)

        hidden_states = RMSNorm(eps=self.rms_norm_eps, name="norm")(hidden_states)

        if self.tie_word_embeddings:
            logits = jnp.dot(hidden_states, embed_layer.embedding.T)
        else:
            logits = nn.Dense(self.vocab_size, use_bias=False, name="lm_head")(
                hidden_states
            )

        return logits


class QwenTextClassificationWrapper:
    """
    Wrapper class for Qwen2.5 model to interface with SubspaceBaseModel
    """

    def __init__(self, model_path: str, target_token_ids: jnp.ndarray):
        """
        Initialize Qwen wrapper

        Args:
            model_path: Path to the converted JAX model
            target_token_ids: Token IDs for classification targets (e.g., [36, 37] for True/False)
        """
        self.model_path = model_path
        config = load_model_config(model_path)
        # vocab_size removed; use self.config['vocab_size'] when required
        self.model = QwenForCausalLM(**config)
        self.target_token_ids = target_token_ids.flatten()

    def init(self, key, x):
        """Initialize model parameters for compatibility with SubspaceBaseModel"""
        # Return existing converted parameters
        # Note: key and x are unused but required for interface compatibility
        params = load_model_params(self.model_path)
        return params

    def apply(
        self,
        variables,
        *args,
        rngs=None,
        method=None,
        mutable=False,
        capture_intermediates=False,
        **kwargs,
    ):
        """Flexible apply wrapper matching project API.

        Signature: apply(self, variables, *args, rngs=None, method=None, mutable=False,
                         capture_intermediates=False, **kwargs)

        - `variables` may be a dict containing 'params' or a raw params pytree.
        - Positional `args` or keyword args may contain input data. Common patterns supported:
            * first positional arg is an inputs dict {'input_ids', 'attention_mask'}
            * keyword args 'input_ids' and optional 'attention_mask'
            * keyword arg 'inputs'

        The extra JAX/Flax kwargs (rngs, method, mutable, capture_intermediates) are forwarded
        to the underlying model.apply call.
        """
        # Call underlying model.apply and forward Flax-specific kwargs
        if mutable:
            out, state = self.model.apply(
                variables,
                *args,
                rngs=rngs,
                method=method,
                mutable=mutable,
                capture_intermediates=capture_intermediates,
                **kwargs,
            )
            last_token_logits = out[:, -1, self.target_token_ids].astype(
                jnp.float32
            )  # (batch, num_classes)
            return (
                last_token_logits,
                state,
            )  # Return probabilities for target classes only
        else:
            out = self.model.apply(
                variables,
                *args,
                rngs=rngs,
                method=method,
                mutable=mutable,
                capture_intermediates=capture_intermediates,
                **kwargs,
            )
            # last_token_logits = out[:, -1, self.target_token_ids]  # (batch, num_classes)
            last_token_logits = out[:, -1, self.target_token_ids].astype(
                jnp.float32
            )  # (batch, num_classes)
            return last_token_logits  # Return probabilities for target classes only


# Utility functions for testing


def create_qwen_config_from_pytorch(pytorch_config):
    """Convert PyTorch config to JAX model parameters"""
    return {
        "vocab_size": pytorch_config.vocab_size,
        "hidden_size": pytorch_config.hidden_size,
        "num_hidden_layers": pytorch_config.num_hidden_layers,
        "num_attention_heads": pytorch_config.num_attention_heads,
        "num_key_value_heads": pytorch_config.num_key_value_heads,
        "intermediate_size": pytorch_config.intermediate_size,
        "hidden_act": pytorch_config.hidden_act,
        "max_position_embeddings": pytorch_config.max_position_embeddings,
        "rms_norm_eps": pytorch_config.rms_norm_eps,
        "attention_dropout": getattr(pytorch_config, "attention_dropout", 0.0),
        "tie_word_embeddings": getattr(pytorch_config, "tie_word_embeddings", True),
        "rope_theta": getattr(pytorch_config, "rope_theta", 10000.0),
    }


def create_parameter_mapping(config):
    """
    Create a mapping dictionary that defines how JAX pytree parameters
    correspond to PyTorch parameters.

    Returns:
        tuple: With: (Mapping from JAX parameter paths to PyTorch parameter names, and transpose info)
    """
    mapping = {}
    NUM_LAYERS = config["num_hidden_layers"]

    # Embedding layer
    mapping["params/embed_tokens/embedding"] = ("model.embed_tokens.weight", (0, 1))

    # Transformer layers
    for layer_idx in range(NUM_LAYERS):
        prefix_torch = f"model.layers.{layer_idx}"
        prefix_jax = f"params/layers_{layer_idx}"

        # Attention weights
        mapping[f"{prefix_jax}/self_attn/q_proj/kernel"] = (
            f"{prefix_torch}.self_attn.q_proj.weight",
            (1, 0),
        )
        mapping[f"{prefix_jax}/self_attn/q_proj/bias"] = (
            f"{prefix_torch}.self_attn.q_proj.bias",
            (0,),
        )
        mapping[f"{prefix_jax}/self_attn/k_proj/kernel"] = (
            f"{prefix_torch}.self_attn.k_proj.weight",
            (1, 0),
        )
        mapping[f"{prefix_jax}/self_attn/k_proj/bias"] = (
            f"{prefix_torch}.self_attn.k_proj.bias",
            (0,),
        )
        mapping[f"{prefix_jax}/self_attn/v_proj/kernel"] = (
            f"{prefix_torch}.self_attn.v_proj.weight",
            (1, 0),
        )
        mapping[f"{prefix_jax}/self_attn/v_proj/bias"] = (
            f"{prefix_torch}.self_attn.v_proj.bias",
            (0,),
        )
        mapping[f"{prefix_jax}/self_attn/o_proj/kernel"] = (
            f"{prefix_torch}.self_attn.o_proj.weight",
            (1, 0),
        )

        # Layer norms
        mapping[f"{prefix_jax}/input_layernorm/weight"] = (
            f"{prefix_torch}.input_layernorm.weight",
            (0,),
        )
        mapping[f"{prefix_jax}/post_attention_layernorm/weight"] = (
            f"{prefix_torch}.post_attention_layernorm.weight",
            (0,),
        )

        # MLP weights
        mapping[f"{prefix_jax}/mlp/gate_proj/kernel"] = (
            f"{prefix_torch}.mlp.gate_proj.weight",
            (1, 0),
        )
        mapping[f"{prefix_jax}/mlp/up_proj/kernel"] = (
            f"{prefix_torch}.mlp.up_proj.weight",
            (1, 0),
        )
        mapping[f"{prefix_jax}/mlp/down_proj/kernel"] = (
            f"{prefix_torch}.mlp.down_proj.weight",
            (1, 0),
        )

    # Final layer norm
    mapping["params/norm/weight"] = ("model.norm.weight", (0,))

    # Language model head (if using tied embeddings, this might not exist)
    # if hasattr(model, 'lm_head') and model.lm_head is not None:
    if config["tie_word_embeddings"] is False:
        mapping["params/lm_head/kernel"] = ("lm_head.weight", (1, 0))

    return mapping


def get_dtype_from_tensor(torch_tensor):
    """return jax dtype from PyTorch tensor"""
    import torch

    if torch_tensor.dtype == torch.float32:
        return jnp.float32
    elif torch_tensor.dtype == torch.float16:
        return jnp.float16
    elif torch_tensor.dtype == torch.bfloat16:
        return jnp.bfloat16
    else:
        raise ValueError(f"Unsupported torch dtype: {torch_tensor.dtype}")


def convert_pytorch_weights_to_jax(jax_params, model, config, jax_dtype=None):
    """
    Loads PyTorch model weights into a JAX/Flax parameter pytree.

    Args:
        jax_params: FrozenDict of JAX model parameters (typically from Flax init).
        model: PyTorch model instance (with named_parameters()).
        config: dict with jax Qwen2 config.

    Returns:
        jax_params: FrozenDict with weights loaded from PyTorch.
    """

    param_mapping = create_parameter_mapping(config)
    pytorch_params = dict(model.named_parameters())
    jax_params = unfreeze(jax_params)

    tree_flat, tree_struct = jax.tree.flatten_with_path(jax_params)

    new_jax_params = []
    for n, p in tree_flat:
        jax_path = "/".join((nn.key for nn in n))
        pytorch_name, transpose_order = param_mapping.get(jax_path, (None, None))

        if pytorch_name is None:
            print(f"WARNING: No mapping found for JAX path '{jax_path}'")
            continue

        pp = pytorch_params.get(pytorch_name, None)
        if pp is None:
            print(f"WARNING: No PyTorch parameter found for '{pytorch_name}'")
            continue

        dtype = get_dtype_from_tensor(pp)
        try:
            pp = pp.detach().cpu().numpy().transpose(transpose_order)
        except Exception as e:
            print(f"WARNING: Error processing '{pytorch_name}': {e}")
            print(f"\tTranspose order: {transpose_order}, Original shape: {pp.shape}")
            continue
        if p.shape != pp.shape:
            print(
                f"WARNING: Shape mismatch for {pytorch_name}: JAX {p.shape} vs PyTorch {pp.shape}"
            )
            continue
        if jax_dtype:
            if jnp.issubdtype(dtype, jnp.floating):  # convert only floating point types
                dtype = jax_dtype
        new_jax_params.append(jnp.array(pp, dtype=dtype))

    del jax_params
    return freeze(jax.tree.unflatten(tree_struct, new_jax_params))


def save_params_and_config(params, config, path):
    """
    Save JAX model parameters and configuration to disk.
    """
    os.makedirs(path, exist_ok=True)
    jnp.save(f"{path}/jax_params.npy", unfreeze(params))

    with open(f"{path}/jax_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_model_config(path):
    """
    Load only the model configuration from disk.
    """
    with open(f"{path}/jax_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def load_model_params(path):
    """
    Load only the model parameters from disk.
    """
    params = jnp.load(f"{path}/jax_params.npy", allow_pickle=True).item()
    return params


def load_params_and_config(path):
    """
    Load JAX model parameters and configuration from disk.
    """
    params = load_model_params(path)
    config = load_model_config(path)
    return params, config
