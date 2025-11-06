#!/usr/bin/env python3
"""
Memory profiling demo for JAX subspace curve training.
Uses MLPModel with depth 30, synthetic data, and parameter masking similar to timeseries_timesfmToCurve.py
"""

import os
import jax
import jax.numpy as jnp
from jax import random, jit
import numpy as np
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
import time

# Set up JAX cache directory
os.environ["JAX_COMPILATION_CACHE_DIR"] = "./jax_cache"
os.makedirs("memory_profile", exist_ok=True)
os.makedirs("jax_cache", exist_ok=True)

# Import local modules
from src.models import MLPModel
from src.subspace_curve import (
    RegressionSubspace, 
    masked_pytree_to_matrix,
    filter_mask
)

def generate_synthetic_data(n_samples=10000, input_dim=5, noise_std=0.1, batch_size=64):
    """Generate synthetic regression data with dataset length divisible by batch_size."""
    # Ensure dataset length is divisible by batch_size
    n_samples = (n_samples // batch_size) * batch_size
    
    key = random.PRNGKey(42)
    key_x, key_noise = random.split(key)
    
    # Generate input features
    x = random.normal(key_x, (n_samples, input_dim))
    
    # Generate true function: polynomial with some nonlinearity
    y_true = (2.0 * x[:, 0] + 
              1.5 * x[:, 1]**2 + 
              0.8 * jnp.sin(x[:, 2]) + 
              0.3 * x[:, 3] * x[:, 4] +
              0.5 * jnp.cos(x[:, 0] + x[:, 1]))
    
    # Add noise
    noise = random.normal(key_noise, (n_samples,)) * noise_std
    y = y_true + noise
    
    print(f"Generated {n_samples} samples (divisible by batch_size={batch_size})")
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")
    
    return x, y

def create_parameter_mask(params, mask_fraction=0.3):
    """Create a mask that marks only a fraction of parameters as trainable (curve parameters)."""
    def random_mask(param_array):
        if param_array.ndim == 0:  # scalar parameter
            return True
        flat_size = param_array.size
        n_masked = int(flat_size * mask_fraction)
        mask_flat = jnp.zeros(flat_size, dtype=bool)
        # Set first n_masked elements to True (trainable)
        mask_flat = mask_flat.at[:n_masked].set(True)
        # Reshape back to original shape and tile for k+1 curve points
        return jnp.tile(mask_flat.reshape(param_array.shape), (3, 1, 1))  # k=2, so k+1=3
    
    # Apply mask to all parameters
    mask = jax.tree.map(random_mask, params['params'])
    
    # Filter to only include Dense layer kernels and biases for curve parameters
    mask = filter_mask(mask, "Dense", as_curve=True)
    
    return mask

def main():
    print("=== JAX Memory Profiling Demo: MLPModel with Subspace Curve Training ===")
    
    # Configuration
    config = {
        'input_dim': 50,
        'n_samples': 5120,  # Will be made divisible by batch_size
        'batch_size': 256,
        'k': 2,  # Number of curve parameters (k+1 control points)
        'num_epochs': 5,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'model_depth': 100,
        'model_width': 1040,
    }
    
    print(f"Configuration: {config}")
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    x, y = generate_synthetic_data(
        n_samples=config['n_samples'],
        input_dim=config['input_dim'],
        batch_size=config['batch_size']
    )
    
    # Initialize model with depth 30
    print(f"\n2. Initializing MLPModel with depth={config['model_depth']}...")
    base_model = MLPModel(
        depth=config['model_depth'], 
        width=config['model_width'], 
        activation="relu"
    )
    
    # Initialize parameters
    key = random.PRNGKey(0)
    dummy_input = x[:2]  # Single sample for initialization
    params = base_model.init(key, dummy_input)
    # # Print all parameter names with their shapes
    # def print_param_shapes(params):
    #     def _print_fn(path, param):
    #         print(f"{'/'.join(str(p) for p in path)}: shape={param.shape}")
    #     jax.tree_util.tree_map_with_path(_print_fn, params['params'])
    # print("Model parameter names and shapes:")
    # print_param_shapes(params)
    
    # Count total parameters
    total_params = sum(p.size for p in jax.tree.flatten(params)[0])
    print(f"Total model parameters: {total_params:,}")
    
    # Initialize subspace model
    print(f"\n4. Initializing RegressionSubspace model with k={config['k']}...")
    s_model = RegressionSubspace(
        model=base_model,
        k=config['k'],
        weight_decay=config['weight_decay']
    )
    
    # Initialize curve parameters
    key, subkey = random.split(key)
    mask = s_model.init_mask(dummy_input, all_curves=False)
    mask_true = s_model.init_mask(dummy_input, all_curves=True)
    mask = filter_mask(mask, "bias", as_curve=True, BoolschenOperator=all)
    curve_params = s_model.init_params(subkey, dummy_input, mask)
    # s_model.train_mask = mask_true  # Store the full mask for training
    s_model.set_train_mask(mask)  # Set the train mask for the model
    print(f"Memory Size for params: {np.sum([l.size * l.itemsize for l in jax.tree.leaves(curve_params)])/(1024**2):.4f}MB")

    # Memory profile before first training step
    jax.profiler.save_device_memory_profile('memory_profile/memory_profile_before_train.json')
    
    # Count curve parameters
    curve_param_matrix = masked_pytree_to_matrix(curve_params['params'], mask, config['k'])
    trainable_params = curve_param_matrix.shape[1]
    print(f"Trainable curve parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}% of total)")
    print(f"Curve parameter matrix shape: {curve_param_matrix.shape}")
    
    # Set up optimizer
    print(f"\n5. Setting up optimizer (lr={config['learning_rate']})...")
    optimizer = optax.adam(config['learning_rate'])
    # optimizer = optax.chain(
    #     optax.adam(config['learning_rate']),
    #     optax.masked(optax.set_to_zero(), mask=jax.tree.map(lambda x: not x, mask)),
    #     )
    train_params = jax.tree.map(lambda x, m: x if m else jnp.array([]), params['params'], s_model.train_mask)
    opt_state = optimizer.init(train_params)
    
    # Training loop with memory profiling
    print(f"\n7. Starting training loop for {config['num_epochs']} epochs...")
    loop_start_time = time.time()
    
    # Prepare batched data
    n_batches = len(x) // config['batch_size']
    x_batched = x[:n_batches * config['batch_size']].reshape(n_batches, config['batch_size'], -1)
    y_batched = y[:n_batches * config['batch_size']].reshape(n_batches, config['batch_size'])
    
    losses = []
    grad_norms = []
    print(f"test kernel {params['params']['Dense_0']['kernel']}")
    
    # Use JAX profiler trace context manager (newer API)
    for epoch in range(config['num_epochs']):
        epoch_losses = []
        epoch_grad_norms = []
        
        # Shuffle data for each epoch
        key, subkey = random.split(key)
        perm = random.permutation(subkey, n_batches)
        x_shuffled = x_batched[perm]
        y_shuffled = y_batched[perm]
        
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        
        for batch_idx in tqdm(range(n_batches), desc=f"Epoch {epoch + 1}", leave=False):
            key, subkey = random.split(key)
            
            t = random.uniform(subkey, (1,), minval=0., maxval=1.)
            # Training step
            loss, curve_params, opt_state, grad = s_model.train_step(
                subkey, t, curve_params,
                x_shuffled[batch_idx], y_shuffled[batch_idx],
                opt_state, optimizer
            )
            
            # Calculate gradient norm
            grad_matrix = masked_pytree_to_matrix(grad, mask, config['k'])
            grad_norm = jnp.linalg.norm(grad_matrix, axis=1).mean()
            
            epoch_losses.append(loss)
            epoch_grad_norms.append(grad_norm)
            
            # # Memory profile during first training step
            # if epoch == 0 and batch_idx == 0:
            #     jax.profiler.save_device_memory_profile('memory_profile/memory_profile_first_step.json')
            #     print("✓ Saved memory profile after first training step")
            
            # Check for NaN
            if jnp.isnan(loss):
                print(f"NaN loss detected at epoch {epoch + 1}, batch {batch_idx + 1}")
                break
        
        avg_loss = np.mean(epoch_losses)
        avg_grad_norm = np.mean(epoch_grad_norms)
        losses.extend(epoch_losses)
        grad_norms.extend(epoch_grad_norms)

        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.6f}, Avg Grad Norm: {avg_grad_norm:.6f}")
        
        # # Memory profile after each epoch
        # jax.profiler.save_device_memory_profile(f'memory_profile/memory_profile_epoch_{epoch + 1}.json')                if jnp.isnan(avg_loss):
    
    loss.block_until_ready()
    loop_end_time = time.time()
    training_loop_duration = loop_end_time - loop_start_time
    print(f"✓ Training loop finished in {training_loop_duration:.2f} seconds")
    
    print(f"test kernel {params['params']['Dense_0']['kernel']}")
    # Final memory profile
    jax.profiler.save_device_memory_profile('memory_profile/memory_profile_final.json')
    print("✓ Saved final memory profile")
    
    # Skip detailed evaluation for simplicity
    print(f"\n8. Training completed successfully!")
    final_loss = losses[-1] if losses else 0.0
    print(f"Final training loss: {final_loss:.6f}")
    
    # Save training plots
    print(f"\n9. Saving training plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Gradient norm plot
    ax2.plot(grad_norms)
    ax2.set_title('Gradient Norm')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Grad Norm')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('memory_profile/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n=== Memory Profiling Demo Complete ===")
    print(f"Model: MLPModel(depth={config['model_depth']}, width={config['model_width']})")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Memory profiles saved in: memory_profile/")
    print(f"Training plot saved as: memory_profile/training_curves.png")
    
    return curve_params, losses, grad_norms

if __name__ == "__main__":
    total_start_time = time.time()
    try:
        # jax.profiler.start_server(9999)
        # time.sleep(2)
        # with jax.profiler.trace("./mem_profile/profile-data"):
        main()
        # time.sleep(2)
        # jax.profiler.stop_server()
    except Exception as e:
        # jax.profiler.stop_server()
        print(f"Error during profiling: {e}")
        raise e
    finally:
        total_end_time = time.time()
        total_execution_time = total_end_time - total_start_time
        print(f"\nTotal execution time: {total_execution_time:.2f} seconds")
