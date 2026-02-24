"""
Qwen2.5 Fine-tuning with LoRA Subspace for Text Classification
Following the PtoC (Point-to-Curve) pattern from other models
"""

import os
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit
import optax
import wandb
from functools import partial
from einshape import jax_einshape as es
import matplotlib.pyplot as plt
from flax.core import freeze, unfreeze

from jax_tqdm import scan_tqdm
from tqdm import tqdm
from src.utils import calibration_error as ece_fn
from copy import deepcopy
from numpyro import distributions as dist

# Project imports
from src.qwen_jax import load_params_and_config, QwenForCausalLM
from src.subspace_curve import (
    masked_pytree_to_matrix, bezier_curve, filter_mask, bezier_coeff_fn,
    LoRARepulsiveCategorySubspace, LoRACategorySubspace, CategorySubspace, RepulsiveCategorySubspace, LoRAEntropyCategorySubspace, LoRAJSDCategorySubspace, LoRARepulsiveEntropyCategorySubspace
)
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from enum import Enum
import time

# Set environment variables
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=10"
# import sys
# sys.setrecursionlimit(200)

class ProcessState(Enum):
    PRETRAINING = 1
    CURVE_FITTING = 2
    CURVE_EVAL = 3
    ERROR = 0

# jax.config.update("jax_log_compiles", True)

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
        self.params, self.config = load_params_and_config(model_path)
        # vocab_size removed; use self.config['vocab_size'] when required
        self.model = QwenForCausalLM(**self.config)
        self.target_token_ids = target_token_ids.flatten()

    def init(self, key, x):
        """Initialize model parameters for compatibility with SubspaceBaseModel"""
        # Return existing converted parameters
        # Note: key and x are unused but required for interface compatibility
        params = self.params
        del self.params
        return params

    def apply(self, variables, *args, rngs=None, method=None, mutable=False, capture_intermediates=False, **kwargs):
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
            last_token_logits = out[:, -1, self.target_token_ids].astype(jnp.float32)  # (batch, num_classes)
            return last_token_logits, state # Return probabilities for target classes only   
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
            last_token_logits = out[:, -1, self.target_token_ids].astype(jnp.float32)  # (batch, num_classes)
            return last_token_logits # Return probabilities for target classes only     

def load_text_classification_data(dataset_path: str, run=None, test=False):
    """
    Load text classification dataset
    
    Args:
        dataset_path: Path to the dataset .npz file
        run: wandb run object for logging to load artifacts

    Returns:
        Tuple of (input_ids, attention_mask, labels, target_ids)
    """
    if run:
        artifact = run.use_artifact(dataset_path, type='dataset')
        artifact_dir = artifact.download()
        dataset_path = os.path.join(artifact_dir, "test_data.npz" if test else "train_data.npz")

    with np.load(dataset_path) as data:
        input_ids = jnp.array(data['input_ids'])
        attention_mask = jnp.array(data['attention_mask']) 
        labels = jnp.array(data['labels'])
        target_id = jnp.array(data['target_id'])
        all_target_ids = jnp.array(data['target_ids'])

    return input_ids, attention_mask, labels, target_id, all_target_ids


# def setup_metrics(model, batch_size, x, attention_mask, target, n_samples, use_linspace=False, average=True, key_prefix='', num_bins_ece=15, t_sample_fn=None):
#     if t_sample_fn is None:
#         if use_linspace:
#             get_t = lambda key: jnp.linspace(0, 1., n_samples)
#         else:
#             get_t = lambda key: random.uniform(key, (n_samples,), minval=0., maxval=1.)
#     else:
#         if use_linspace:
#             raise ValueError("Cannot use linspace sampling with custom t_sample_fn")
#         def tree_per_sample_fn(key):
#             keys = jax.random.split(key, n_samples)
#             return [t_sample_fn(k) for k in keys]
#         get_t = tree_per_sample_fn
#     ii = len(x) // batch_size
    
#     x_ = x[:(matching_limit:=(x.shape[0] // batch_size) * batch_size)]  # drop non matching batch size
#     x_last = x[matching_limit:]
#     x_ = es('(ib)...->ib...', x_, b=batch_size)

#     att_m_ = attention_mask[:matching_limit]  # drop non matching batch size
#     att_last = attention_mask[matching_limit:]
#     att_m_ = es('(ib)...->ib...', att_m_, b=batch_size)

#     y_all = target[:matching_limit]  # drop non matching batch size
#     y_last = target[matching_limit:]
#     y_ = es('(ib)...->ib...', y_all, b=batch_size)

#     def acc_fn(rng_key, params):
#         # @scan_tqdm(ii, desc='Evaluating', leave=False)
#         def pred(key, data):
#             x_batch, atten_batch, y_batch = data
#             # # turn off leading padding tokens in batch for faster evaluation
#             # x_batch = x_batch[:, (max_seqlen:=jnp.argmax(atten_batch, axis=1).min()):]
#             # atten_batch = atten_batch[:, max_seqlen:]

#             # @jax.jit
#             def eval(t):
#                 loss, _, out = model.nll(params['params'], {}, t, (x_batch, atten_batch), y_batch, train=False)  # params, batch_stats, t, x ; out shape (batch_size, output_dim)
#                 acc = jnp.sum(jnp.argmax(out, axis=-1) == y_batch)
#                 return loss*batch_size, acc, out  # sum over batch
#                 # return loss*batch_size, acc  # sum over batch
            
#             key, subkey = random.split(key)

#             # draw samples to approximate curve performance
#             t = get_t(subkey)
#             loss, acc, out = jax.lax.map(eval, t) # out shape (draws, batch_size, output_dim)
#             # loss, acc = jax.vmap(eval)(t) # out shape (draws, batch_size, output_dim)

#             if average:
#                 # mean over samples
#                 return key, (jnp.sum(loss), jnp.sum(acc), out)
#             else:
#                 return key, (loss, acc, out)  # no mean over samples

#         # shape (n_data, draws, output_dim)
#         rng_key, (loss, acc, out) = jax.lax.scan(pred, rng_key, (x_, att_m_, y_)) # out shape (len(loader), draws, batch_size, output_dim)
#         # compute loss of last irregular batch
#         if len(x_last) > 0:
#             rng_key, (loss_last, acc_last, out_last) = pred(rng_key, (x_last, att_last, y_last))
#             loss = jnp.concatenate([loss, loss_last[jnp.newaxis, ...]], axis=0)
#             acc = jnp.concatenate([acc, acc_last[jnp.newaxis, ...]], axis=0)
#             out = jnp.concatenate([out.transpose(1, 0, 2, 3).reshape(n_samples, -1, out.shape[-1]), out_last], axis=1)  # (n_samples, n_data, output_dim)
#         else:
#             out = out.transpose(1, 0, 2, 3).reshape(n_samples, -1, out.shape[-1])  # (n_samples, n_data, output_dim)

#         # # compute the expected calibration error (ECE)
#         ece = jax.vmap(ece_fn, in_axes=(0, None, None))(out, target, num_bins_ece)  # mean over samples

#         if average:
#             ece = jnp.mean(ece)
#             loss = loss.sum() / (n_samples * x.shape[0])
#             acc = acc.sum() / (n_samples * x.shape[0])
#         else:
#             loss = jnp.sum(loss, axis=0) / x.shape[0]
#             acc = jnp.sum(acc, axis=0) / x.shape[0]

#         return {key_prefix + 'loss': loss, key_prefix + 'acc': acc, key_prefix + 'ece': ece}, out  # mean over epochs
#     return acc_fn

def setup_metrics(model, batch_size, x, attention_mask, target, n_samples, use_linspace=False, average=True, key_prefix='', num_bins_ece=15, t_sample_fn=None):
    if t_sample_fn is None:
        if use_linspace:
            get_t = lambda key: jnp.linspace(0, 1., n_samples)
        else:
            get_t = lambda key: jax.random.uniform(key, (n_samples,), minval=0., maxval=1.)
    else:
        if use_linspace:
            raise ValueError("Cannot use linspace sampling with custom t_sample_fn")
        # get_t = partial(t_sample_fn, n_samples=n_samples)
        get_t = lambda k: jax.vmap(lambda key: t_sample_fn(key))(jax.random.split(k, n_samples))
    ii = len(x) // batch_size
    
    idx = jnp.arange(len(x))
    idx_ = idx[:(matching_limit:=(x.shape[0] // batch_size) * batch_size)]  # drop non matching batch size
    idx_ = es('(ib)...->ib...', idx_, b=batch_size)
    idx_last = idx[matching_limit:]

    # x_ = x[:(matching_limit:=(x.shape[0] // batch_size) * batch_size)]  # drop non matching batch size
    # x_last = x[matching_limit:]
    # x_ = es('(ib)...->ib...', x_, b=batch_size)

    # att_m_ = attention_mask[:matching_limit]  # drop non matching batch size
    # att_last = attention_mask[matching_limit:]
    # att_m_ = es('(ib)...->ib...', att_m_, b=batch_size)

    # y_all = target[:matching_limit]  # drop non matching batch size
    # y_last = target[matching_limit:]
    # y_ = es('(ib)...->ib...', y_all, b=batch_size)

    def post_pred_performance(logits, labels):
        # logits shape (n_samples, n_data, n_classes)
        post_predict_logits = jax.nn.logsumexp(logits, axis=0) - jnp.log(logits.shape[0])  # shape (n_data, n_classes)
        post_acc = jnp.mean(jnp.argmax(post_predict_logits, axis=-1) == labels)
        post_ece = ece_fn(post_predict_logits, labels, num_bins=15)
        return {f'{key_prefix}bma_ll': dist.Categorical(logits=post_predict_logits).log_prob(labels).mean(),
                f'{key_prefix}bma_acc': post_acc,
                f'{key_prefix}bma_ece': post_ece}

    def acc_fn(rng_key, params):
        # @scan_tqdm(ii, desc='Evaluating', leave=False)
        def pred(rng_key, data_idx):
            x_batch = x.at[data_idx].get()
            att_m_batch = attention_mask.at[data_idx].get()
            y_batch = target.at[data_idx].get()

            rng_key, key_ = jax.random.split(rng_key)
            t_tree = get_t(key_)  # shape (n_samples,)
            # keys = jax.random.split(key_, n_samples)
            
            def single_pred(key, t):
                # t is either scalar or tree of shape ()
                key, subkey = jax.random.split(key)
                loss, _, out = model.nll(params['params'], {}, t, (x_batch, att_m_batch), y_batch, train=False, key=subkey)  # params, batch_stats, t, x ; out shape (batch_size, output_dim) # no Flat-Lora sampling
                return key, (loss*batch_size, out)
            
            rng_key, (loss, out) = jax.lax.scan(single_pred, rng_key, t_tree)
            return rng_key, (loss, out)  # loss shape (n_samples), out shape (n_samples, batch_size, output_dim)

        rng_key, (loss, out) = jax.lax.scan(pred, rng_key, idx_) # loss shape (n_batches, n_samples), out shape (n_batches, n_samples, batch_size, output_dim)

        # shape (n_data, output_dim)
        # compute loss of last irregular batch
        if len(idx_last) > 0:
            rng_key, (loss_last, out_last) = pred(rng_key, idx_last)
            loss = jnp.concatenate([loss, loss_last[jnp.newaxis, ...]], axis=0)
            # jax.debug.print(f"out_last shape: {out_last.shape}")
            out = jnp.concatenate([out.transpose(1, 0, 2, 3).reshape(n_samples, -1, out.shape[-1]), out_last], axis=1)  # (n_samples, n_data, output_dim)
        else:
            out = out.transpose(1, 0, 2, 3).reshape(n_samples, -1, out.shape[-1])  # (n_samples, n_data, output_dim)

        loss = loss.transpose(1,0)
        bma_metrics = post_pred_performance(out, target)

        # # compute the expected calibration error (ECE)
        ece = jax.vmap(ece_fn, in_axes=(0, None, None))(out, target, num_bins_ece)  # mean over samples
        # acc = jax.vmap(lambda logits, labels: jnp.mean(jnp.argmax(logits, axis=-1) == labels), in_axes=(0, None))(out, target)  # shape (n_samples,)
        acc = jnp.argmax(out, axis=-1) == target # shape (n_samples, n_data)

        if average:
            ece = jnp.mean(ece)
            loss = loss.sum() / (n_samples * x.shape[0])
            acc = acc.sum() / (n_samples * x.shape[0])
        else:
            loss = jnp.sum(loss, axis=1) / x.shape[0]
            acc = jnp.sum(acc, axis=1) / x.shape[0]

        return {key_prefix + 'loss': loss, key_prefix + 'acc': acc, key_prefix + 'ece': ece, **bma_metrics}, out  # mean over epochs
    return acc_fn


def get_embedding(input_ids, attention_mask, model, params, batch_size = 11):
    x_ = input_ids[:(matching_limit:=(input_ids.shape[0] // batch_size) * batch_size)]  # drop non matching batch size
    x_last = input_ids[matching_limit:]
    x_ = es('(ib)...->ib...', x_, b=batch_size)

    att_m_ = attention_mask[:matching_limit]  # drop non matching batch size
    att_last = attention_mask[matching_limit:]
    att_m_ = es('(ib)...->ib...', att_m_, b=batch_size)
    # inp_last = 
    # @partial(jax.jit, donate_argnums=(0,))
    def get_embeddings_(params, data):
        input_ids, attention_mask = data
        model_output, state = model.apply(freeze(params), input_ids=input_ids, attention_mask=attention_mask, train=False,
                                capture_intermediates=True, mutable=['intermediates'])
        embedding = state['intermediates']['norm']['__call__'][0]
        return params, embedding

    params, embedding = jax.lax.scan(get_embeddings_, params, (x_, att_m_))
    embedding = es('ib...->(ib)...', embedding, b=batch_size)
    if x_last.shape[0] > 0:
        params, embedding_last = get_embeddings_(params, (x_last, att_last))
        embedding = jnp.concatenate([embedding, embedding_last], axis=0)
    return embedding


def mutual_information(logits, weights:np.ndarray|bool=False):
    """Compute mutual information from logits over samples

    Args:
        logits: jnp.ndarray of shape (n_samples, n_data, n_classes)
    Returns:
        mi: float mutual information averaged over each data point
    """
    use_weigts = True if isinstance(weights, jnp.ndarray) or isinstance(weights, np.ndarray) else False
    probs = jax.nn.softmax(logits, axis=-1)  # (n_samples, n_data, n_classes)
    if use_weigts:
        avg_probs = jnp.sum(probs * weights[:, None, None], axis=0)  # (n_data, n_classes)
    else:
        avg_probs = jnp.mean(probs, axis=0)  # (n_data, n_classes)
    entropy_pred = -jnp.sum(avg_probs * jnp.log(avg_probs + 1e-12), axis=-1)  # (n_data,)
    if use_weigts:
        expected_nentropy = jnp.sum(-jnp.sum(probs * jnp.log(probs + 1e-12), axis=-1) * weights[:, None], axis=0)  # (n_data,)
    else:
        expected_nentropy = jnp.mean(-jnp.sum(probs * jnp.log(probs + 1e-12), axis=-1), axis=0)  # (n_data,)
    mi = entropy_pred - expected_nentropy  # (n_data,)
    return jnp.mean(mi)

def mean_entropy(logits, weights:np.ndarray|bool=False):
    """Compute curve entropy from logits over samples

    Args:
        logits: jnp.ndarray of shape (n_samples, n_data, n_classes)
    Returns:
        entropy: float predictive entropy averaged over each data point
    """
    use_weigts = True if isinstance(weights, jnp.ndarray) or isinstance(weights, np.ndarray) else False
    probs = jax.nn.softmax(logits, axis=-1)  # (n_samples, n_data, n_classes)
    if use_weigts:
        avg_probs = jnp.sum(probs * weights[:, None, None], axis=0)  # (n_data, n_classes)
    else:
        avg_probs = jnp.mean(probs, axis=0)  # (n_data, n_classes)
    entropy_pred = -jnp.sum(avg_probs * jnp.log(avg_probs + 1e-12), axis=-1)  # (n_data,)
    return jnp.mean(entropy_pred)

def entropy(logits):
    probs = jax.nn.softmax(logits, axis=-1)  # (n_samples, n_data, n_classes)
    entropies = -jnp.sum(probs * jnp.log(probs + 1e-12), axis=-1)  # (n_samples, n_data)
    return entropies  # (n_samples, n_data)


def main():
    logger = wandb.init()
    config = wandb.config
    train(logger, config)


def train(logger, config):
    """
    Main training function for Qwen curve fine-tuning
    
    Args:
        logger: Wandb logger
        config: Training configuration
    """
    # jax.clear_caches()
    rng_key = random.PRNGKey(config['rng_seed'])
    EVAL_EVERY_N_BATCH = config['eval_every_n_batch']
    BATCH_SIZE_EVAL = config['curve_params']['batch_size'] * 2
    SAVE_PARAMS = config.get('save_params', False)
    SWEEP_PARAM_PATH = config.get('load_params_path', False)
    
    LOG_CURVE_METRICS = True # whether to log curve metrics like length and mutual information during training (can slow down training)

    # Create artifact for storing parameters
    # if SAVE_PARAMS:
    artifact = wandb.Artifact(name="params", type="pytree")

    input_ids, attention_mask, labels, target_ids, unique_target_ids = load_text_classification_data(config['dataset_path'], run=logger)
    # shuffle once for train/val split
    rng_key, split_key = random.split(rng_key)
    perm_idx = random.permutation(split_key, len(input_ids))
    print(f"Dataset loaded: {len(input_ids)} samples, {len(unique_target_ids)} classes")

    test_input_ids, test_attention_mask, test_labels, _, unique_target_ids = load_text_classification_data(config['dataset_path'], run=logger, test=True)
    print(f"Target token IDs: {unique_target_ids}")

    # Load Qwen model once
    if config['net_kwargs']['model_path'].startswith('ddold/'):
        model_artifact = logger.use_artifact(config['net_kwargs']['model_path'], type='model')
        config['net_kwargs']['model_path'] = model_artifact.download()

    # Configure optimizer kwargs
    if isinstance(config['curve_params']['optimizer_conf']['kwargs']['learning_rate'], dict):
        lr_schedule = getattr(optax, config['curve_params']['optimizer_conf']['kwargs']['learning_rate']['name'])(
            **config['curve_params']['optimizer_conf']['kwargs']['learning_rate']['kwargs'])
    else:
        lr_schedule = optax.constant_schedule(config['curve_params']['optimizer_conf']['kwargs']['learning_rate'])
    config['curve_params']['optimizer_conf']['kwargs']['learning_rate'] = lr_schedule

    def train_curve_inner(rng_key,
                         curve_state: ProcessState,
                         k: int,
                         num_epochs: int,
                         batch_size: int,
                         optimizer_conf: dict,
                         natural_parameterization: bool,
                         gravity: float,
                         energy_weakening: float,
                         regularizer: str|bool,
                         dataset_path: str,
                         entropy_maxim: float,
                         fixed_cps_train_params: list|bool = False,
                         filter_masks: list = None,
                         lora_params: dict = None,
                         net_kwargs: dict = None,
                         weight_decay: float = 0.0,
                         jitter_multiplier: float = 0.0,
                         smoke_test: bool = False,
                         num_steps: int = -1,  # num steps are either given or computed from num_epochs
                         temperature: list = [1.0,],
                         dataset_sampling: dict = None,
                         val_percentage: float = 0.0,
                         curve_parameterization = 'bezier',
                         curve_sampling_mode: str = 'combined', # 'combined', 'per_leave'
                         logger_prefix: str = '',
                         ):
        # Note: jitter parameter is unused but kept for interface compatibility
        rng_key, init_key = random.split(rng_key)
        # Load dataset
        assert not ((num_epochs > 0) and (num_steps > 0)), "Only one of num_epochs or num_steps should be > 0"
        val = False
        if num_epochs > 0 or num_steps > 0:
            # compute num_epochs if num_steps is given
            if num_steps > 0:
                num_epochs = int(np.ceil(1.*num_steps*batch_size / len(input_ids)))
                print(f"Setting num_epochs to {num_epochs} to match {num_steps} steps")
            
            num_steps = int(np.ceil(1.*num_epochs * len(input_ids) / batch_size))
            print(f"Setting num_steps to {num_steps} to match {num_epochs} epochs")

            # Split data (simple train/val split) # we dont use validate data
            n_train = int(np.ceil((1.-val_percentage)*len(input_ids) / batch_size) * batch_size)
            train_input_ids = input_ids[perm_idx][:n_train]
            train_attention_mask = attention_mask[perm_idx][:n_train]
            train_labels = labels[perm_idx][:n_train]
            
            val_input_ids = input_ids[perm_idx][n_train:]
            val_attention_mask = attention_mask[perm_idx][n_train:]
            val_labels = labels[perm_idx][n_train:]
            if len(val_input_ids) > 0:
                print("Validation data are used")
                val = True
            print(f"Training data: {len(train_input_ids)} samples, Validation data: {len(val_input_ids)} samples")

        # load init model parameters
        model = QwenTextClassificationWrapper(**net_kwargs, target_token_ids=unique_target_ids)
        params = unfreeze(model.init(None, None))
        dtype = jax.tree.leaves(params)[0].dtype

        #compute embeddings for sorting if expert sampling is used
        ds_sampling = list(dataset_sampling.keys())
        assert len(ds_sampling) == 1, "Only one dataset_sampling apporach could be given"
        ds_sampling = ds_sampling[0]
        if (ds_sampling == 'expert') and (num_steps > 0):
            print(f"compute embeddings for expert sort")
            sort_mode = dataset_sampling['expert'].get('sort_mode', False)
            if sort_mode and (sort_mode != 'Random'):
                embedding = get_embedding(train_input_ids, train_attention_mask, model, params, batch_size)
                print(f"Computed embeddings for expert sort with shape {embedding.shape}")
                device = embedding.device
                embedding = jax.device_put(embedding, jax.devices("cpu")[0])
                embedding = embedding.reshape(embedding.shape[0], -1).astype(jnp.float32)
                embedding /= jnp.linalg.norm(embedding, axis=1, keepdims=True)
                # compute cosine similarity matrix  
                cosine_sim = jax.device_put(jnp.dot(embedding, embedding.T), device)
                print(f"Computed cosine similarity matrix with shape {cosine_sim.shape}")
                del embedding
                
        
        curve_mask = jax.tree.map(lambda x: False, params['params'])
        # only if k > 0, otherwise no curve parameters are used
        if k > 0:
            for f in filter_masks:
                curve_mask = filter_mask(curve_mask, *f['keys'], as_curve=True, BoolschenOperator=eval(f['op']))

        if "freeze_other_params" in optimizer_conf and optimizer_conf["freeze_other_params"]:
            train_mask = jax.tree.map(lambda x: False, params['params'])
            # all curve and lora params are trainable
            for f in filter_masks:
                train_mask = filter_mask(train_mask, *f['keys'], as_curve=True, BoolschenOperator=eval(f['op']))
            if lora_params and lora_params['use_lora']:
                for f in lora_params['filter_masks']:
                    train_mask = filter_mask(train_mask, *f['keys'], as_curve=True, BoolschenOperator=eval(f['op']))
        else:
            train_mask = jax.tree.map(lambda x: True, params['params'])
        # set LoRA mask for all parameters
        lora_mask = jax.tree.map(lambda x: False, params['params'])  # LoRA mask for all parameters
        if lora_params and lora_params['use_lora']:
            for f in lora_params['filter_masks']:
                lora_mask = filter_mask(lora_mask, *f['keys'], as_curve=np.array(f['dims']), BoolschenOperator=eval(f['op']))
        # pprint.pprint(f"LoRA mask: {lora_mask}")
        # pprint.pprint(f"Curve mask: {curve_mask}")
        # pprint.pprint(f"Train mask: {train_mask}")

        perm_spec = False

        # add weight decay to optimizer if adamw is specified
        if (weight_decay > 0.0) and (optimizer_conf['name'] == 'adamw'):
            optimizer_conf['kwargs']['weight_decay'] = weight_decay
            weight_decay = 0.0  # set to 0.0 to avoid double weight decay in model
        if lora_params and lora_params['use_lora']:
            print("Using LoRA with parameters:", lora_params)
            if isinstance(regularizer, str):
                if entropy_maxim > 0.0:
                    s_model = LoRARepulsiveEntropyCategorySubspace(lora_mask=lora_mask, r=lora_params['r'], lora_dtype=lora_params['lora_dtype'], lora_alpha=lora_params['lora_alpha'], 
                                                                   lora_mode=lora_params['lora_mode'], lora_rho=lora_params.get('lora_rho', 0.0), lora_rho_s=lora_params.get('lora_rho_s', 0.0),
                                                                   model=model, k=k, weight_decay=weight_decay,
                                                                   perm_spec=perm_spec,
                                                                   natural_parameterization=natural_parameterization,
                                                                   mutuable_param_name=False,
                                                                   curve_parameterization=curve_parameterization,
                                                                   gravity=gravity, energy_weakening=energy_weakening, regularizer=regularizer,
                                                                   entropy_weight=entropy_maxim,)
                else:
                    s_model = LoRARepulsiveCategorySubspace(lora_mask=lora_mask, r=lora_params['r'], lora_dtype=lora_params['lora_dtype'], lora_alpha=lora_params['lora_alpha'], 
                                                            lora_mode=lora_params['lora_mode'], lora_rho=lora_params.get('lora_rho', 0.0), lora_rho_s=lora_params.get('lora_rho_s', 0.0),
                                                            model=model, k=k, weight_decay=weight_decay,
                                                            perm_spec=perm_spec,
                                                            natural_parameterization=natural_parameterization,
                                                            mutuable_param_name=False,
                                                            curve_parameterization=curve_parameterization,
                                                            gravity=gravity, energy_weakening=energy_weakening, regularizer=regularizer)
            else:
                if entropy_maxim > 0.0:
                    s_model = LoRAJSDCategorySubspace(lora_mask=lora_mask, r=lora_params['r'], lora_dtype=lora_params['lora_dtype'], lora_alpha=lora_params['lora_alpha'], 
                                                      lora_mode=lora_params['lora_mode'], lora_rho=lora_params.get('lora_rho', 0.0), lora_rho_s=lora_params.get('lora_rho_s', 0.0),
                                                      model=model, k=k, weight_decay=weight_decay, 
                                                      perm_spec=perm_spec,
                                                      natural_parameterization=natural_parameterization,
                                                      mutuable_param_name=False,
                                                      curve_parameterization=curve_parameterization,
                                                      entropy_weight=entropy_maxim,
                                                      )
                else:
                    s_model = LoRACategorySubspace(lora_mask=lora_mask, r=lora_params['r'], lora_dtype=lora_params['lora_dtype'], lora_alpha=lora_params['lora_alpha'], 
                                                   lora_mode=lora_params['lora_mode'], lora_rho=lora_params.get('lora_rho', 0.0),  lora_rho_s=lora_params.get('lora_rho_s', 0.0),
                                                   model=model, k=k, weight_decay=weight_decay, 
                                                   perm_spec=perm_spec,
                                                   natural_parameterization=natural_parameterization,
                                                   mutuable_param_name=False,
                                                   curve_parameterization=curve_parameterization,
                                                   )
        else:
            if regularizer:
                s_model = RepulsiveCategorySubspace(model=model, k=k, weight_decay=weight_decay,
                                                   perm_spec=perm_spec,
                                                   natural_parameterization=natural_parameterization,
                                                   mutuable_param_name=False,
                                                   curve_parameterization=curve_parameterization,
                                                   gravity=gravity, energy_weakening=energy_weakening, regularizer=regularizer)
            else:
                s_model = CategorySubspace(model=model, k=k, weight_decay=weight_decay, 
                                           perm_spec=perm_spec,
                                           natural_parameterization=natural_parameterization,
                                           curve_parameterization=curve_parameterization,
                                           mutuable_param_name=False)


        # initialize curve parameter with pretrained point parameter
        params = s_model.init_params_from_point(init_key, params, curve_mask, noise_fn=lambda p, key: p+jitter_multiplier*jax.random.normal(key, p.shape, dtype=p.dtype))

        # set train mask according to config
        # Only update mask for LoRA-capable subspaces
        if lora_params and lora_params['use_lora']:
            train_mask = s_model.update_mask_for_lora(train_mask, params['params'], b_off=False, w0_off=True) # w0 must be off for LoRA
        s_model.set_train_mask(train_mask)
        
        #*** Fix control points handling ***#
        #*** Map pretrained control points which are fixed into params vector ***#
        if fixed_cps_train_params != False:
            # if k > 0:
            #     assert jax.tree.all(jax.tree.map(lambda x, y: x == y, s_model.train_mask, s_model.curve_mask)), "Pretraind Curve fitting requieres train mask and curve mask to be the same(Lora Params also Curve param)"
            assert (len(fixed_cps_train_params) == (k+1)), "fixed_cps_train_params length must match k or be False"
            for i,p in enumerate(fixed_cps_train_params):
                if p != False:
                    def map_fn(m,p,cp_f):
                        if m:
                            p = p.at[i].set(cp_f)
                        return p
                    params['params'] = jax.tree.map(map_fn, s_model.curve_mask, params['params'], p)

        if curve_sampling_mode == 'per_leave':
            print("Using per-leave curve sampling")
            # num_leaves_parameter = len(jax.tree.leaves(s_model.curve_mask))
            # t_structure = jax.tree.structure(s_model.curve_mask)

            # def t_sample_fn(key):
            #     t_vec = jax.random.uniform(key, (num_leaves_parameter,), minval=0., maxval=1.)
            #     t_tree = jax.tree.unflatten(t_structure, t_vec)
            #     return t_tree
            only_curves = jax.tree.map(lambda m: True if m else None, s_model.curve_mask)
            num_curve_leaves = len(jax.tree.leaves(only_curves))
            struct = jax.tree.structure(only_curves)

            def t_sample_fn(key):
                t_flatt = jax.random.uniform(key, (num_curve_leaves,), minval=0., maxval=1.)
                t_tree = jax.tree.unflatten(struct, t_flatt)
                t_tree = jax.tree.map(lambda k: False if k is None else k, t_tree, is_leaf=lambda x: x is None) # clean None leaves to False for propper tree.map
                return t_tree
        else:
            def t_sample_fn(key):
                return random.uniform(key, (1,), minval=0., maxval=1.)


        artifact_weights_use = False
        if SWEEP_PARAM_PATH:
            raise NotImplementedError("Loading from sweep is not implemented in this version.")
            print(f"Load parameters from sweep {SWEEP_PARAM_PATH}")
            def prepare_config(cfg, with_dataset=True):
                # cfg_copy = json.loads(json.dumps(cfg))
                cfg_copy = deepcopy(dict(cfg))
                if with_dataset:
                    cfg_copy['dataset_path'] = "ddold/qwen_PtoC/obqa_dataset:v0"
                    cfg_copy['curve_params']['num_steps'] = 5_000 
                # cfg_copy['curve_params'].pop('num_steps', None) 
                cfg_copy['curve_params'].pop('num_epochs', None)
                cfg_copy['curve_params'].pop('temperature', None)
                cfg_copy.pop('save_params', None)
                cfg_copy.pop('smoke_test', None)
                cfg_copy.pop('eval_every_n_batch', None)
                cfg_copy.pop('load_params_path', None)
                cfg_copy['net_kwargs']['model_path'] = cfg_copy['net_kwargs']['model_path'].split('/')[-1]
                return cfg_copy
            print("config:", config)
            config_copy = prepare_config(config, with_dataset=True)
            artifact_use = False
            trainable_params = False
            api = wandb.Api()

            def load_art(run):
                artifact_use, artifact_weights_use = False, False
                print(f"Found config with obqa dataset with name: {run.name}, id: {run.id}")
                for art_l in run.logged_artifacts():
                    if art_l.type == "pytree" and art_l.name.startswith("params"):
                        print(f"Downloading params from {art_l.name}")
                        artifact_use = logger.use_artifact(art_l, type='pytree')
                    if art_l.type == "npz" and art_l.name.startswith("bma_weights"):
                        print(f"Downloading weights from {art_l.name}")
                        artifact_weights_use = logger.use_artifact(art_l, type='npz')
                return artifact_use, artifact_weights_use
            # search for matching run in sweep or in run if run path was given
            try:
                print(f"Searching sweep {SWEEP_PARAM_PATH} for matching config")   
                for run in api.sweep(SWEEP_PARAM_PATH).runs:
                    # check if config matches
                    matching_config = prepare_config(run.config, with_dataset=False)
                    if matching_config == config_copy:
                        if run.state != "finished":
                            print(f"Found Run {run.name} but is not finished, skipping...")
                            continue
                        artifact_use, artifact_weights_use = load_art(run)
            except wandb.errors.CommError as e:
                if e.message.startswith("Could not find sweep"):
                    try:
                        print(f"Searching run {SWEEP_PARAM_PATH} for matching config")
                        run = api.run(SWEEP_PARAM_PATH)
                        matching_config = prepare_config(run.config, with_dataset=False)
                        if matching_config == prepare_config(config, with_dataset=False):
                            if run.state != "finished":
                                print(f"Found Run {run.name} but is not finished, skipping...")
                                raise KeyError("Run not finished")
                            artifact_use, artifact_weights_use = load_art(run)
                    except Exception as e:
                        print("No sweep or run found")
                        raise e
                else:
                    raise e
            assert artifact_use, f"No matching run found in sweep {SWEEP_PARAM_PATH}"
            # assert artifact_weights_use, f"No matching weights found in sweep {SWEEP_PARAM_PATH}"
            
            # load trainable params
            for f in artifact_use.files():
                if 'trainable_params' in f.name:
                    trainable_params = np.load(artifact_use.get_entry(f.name).download(), allow_pickle=True).item()
                    print(f"Loaded trainable_params from {f.name}")
                    assert len(set(trainable_params.keys()) - set(params['params'].keys())) == 0, "Loaded trainable_params do not match model params"
            assert trainable_params, f"trainable_params not found in artifact {artifact_use.name}"

            # merge loaded trainable params with current params
            def load_fn(m, p, pl):
                if m:
                    assert p.shape == pl.shape, f"Shape mismatch for parameter with shape {p.shape} and loaded shape {pl.shape}"
                    return pl
                else:
                    return p
            params['params'] = jax.tree.map(load_fn, s_model.train_mask, params['params'], trainable_params)
        
            if (num_epochs <= 0) and (num_steps <= 0) and (artifact_weights_use == False):
                raise ValueError("Either num_epochs or num_steps must be > 0 for training, or a sweep path must be provided to load weights for BMA.")

        # compute number of parameters for last curve fitting run
        if k > 0:
            n_params = sum(p.size if isinstance(p, jnp.ndarray) else 0 for p in jax.tree.flatten(params)[0])
            param_memory = sum(p.size*p.dtype.itemsize if isinstance(p, jnp.ndarray) else 0 for p in jax.tree.flatten(params)[0])
            logger.summary.update({'n_params': n_params, 'memory_params': param_memory})
            print(f"Number of parameters: {n_params:_}\nMemory of parameters: {param_memory/(1024**2):_}MB")

        # if SAVE_PARAMS:
        #     jnp.save(f'tmp_files/{logger.id}_init_params.npy', params)
        #     artifact.add_file(f'tmp_files/{logger.id}_init_params.npy')
        # if SAVE_PARAMS:
        #     print("Save init params")
        #     # jnp.save(f'tmp_files/{logger.id}_best_params.npy', best_params)
        #     trainable_params = s_model.only_trainable_params(params['params'])
        #     jnp.save(f'tmp_files/{logger.id}_trainable_params_init.npy', trainable_params)
                #     import time
                # time.sleep(0.5)
        #     artifact.add_file(f'tmp_files/{logger.id}_trainable_params_init.npy')

        # Define optimizer
        # ToDo check sam args must be preprocessed and not in this function
        if optimizer_conf['name'].startswith('sam'):
            raise NotImplementedError("SAM optimizer is not yet implemented for text classification")
            # lr = optimizer_conf['kwargs'].pop('adv_lr')
            rho = optimizer_conf['kwargs'].pop('adv_rho')
            adv_opt = optax.chain(optax.contrib.normalize(), optax.sgd(rho))
            optim_name = optimizer_conf['name'].split('_')[1]
            base_opt = getattr(optax, optim_name)(**optimizer_conf['kwargs'])
            optimizer = optax.contrib.sam(base_opt, adv_optimizer=adv_opt, sync_period=2, opaque_mode=True)
            
            # @partial(jit, static_argnums=(6), donate_argnums=(2,5))
            # @profile
            def train_step_sam(key, t, params, x, y, opt_state, optimizer):
                """
                Performs a single training step.

                Parameters:
                - key: The random key for generating random numbers.
                - params: The parameters of the model.
                - x: The input data.
                - y: The target output.
                - opt_state: The optimizer state.
                - optimizer: The optimizer.

                Returns:
                - The loss, updated parameters, and updated optimizer state.

                """
                self = s_model 
                trainable_params = self.only_trainable_params(params['params'])
                freezed_params = freeze(jax.tree.map(lambda x, m: x if not m else jnp.array([]), params['params'], self.train_mask))
                (loss, (stats_params, logs)), grads = jax.value_and_grad(
                    self.compute_loss, argnums=2, has_aux=True)(key, t, trainable_params, params[self.mutable_name] if self.mutable_name else {}, freezed_params, x, y)
                # update params
                updates, opt_state = optimizer.update(
                    grads, opt_state, trainable_params, grad_fn=lambda pp, _ : jax.grad(lambda p: self.compute_loss(key, t, p, params[self.mutable_name] if self.mutable_name else {}, freezed_params, x, y)[0])(pp))
                trainable_params = optax.apply_updates(trainable_params, updates)
                params = jax.tree.map(lambda x, y, m: x if m else y, trainable_params, unfreeze(freezed_params), self.train_mask)
                params = {'params': params}
                if self.mutable_name:
                    params[self.mutable_name] = stats_params
                return loss, params, opt_state, (grads, logs)

            s_model.train_step = train_step_sam  # use SAM train step
        else:
            optimizer = getattr(optax, optimizer_conf['name'])(**optimizer_conf['kwargs'])
        # Add gradient clipping to the optimizer using optax.chain
        opt_chain_list = []
        if "grad_clip_norm" in optimizer_conf:
            opt_chain_list.append(optax.clip_by_global_norm(optimizer_conf["grad_clip_norm"]))
        
        #*** Fix control points handling ***#
        #*** Fix curve parameters during training if specified ***#
        def element_wise_mask(element_freeze_mask):
            """
            Multiplies the updates by a element_freeze_mask.
            element_freeze_mask leaves can be:
            - 0.0 (or False): Allow update
            - 1.0 (or True): Freeze parameter
            - Array of 0s/1s: Element-wise freeze/update
            """
            def init_fn(params):
                return optax.EmptyState()

            def update_fn(updates, state, params=None):
                # This performs element-wise multiplication (u * m)
                # It handles both scalar masks (broadcasting) and array masks
                new_updates = jax.tree.map(lambda f, u: u * (1 - f) if type(f) != bool else u, element_freeze_mask, updates)
                return new_updates, state

            return optax.GradientTransformation(init_fn, update_fn)
        if fixed_cps_train_params != False:
            def get_dims(param):
                return (-1,) + (1,)* (param.ndim -1)
            # fixed_cp_mask e.g [1,0.1] for k=2 with prodcasting to param shape such that only first dimension is selectively freezed
            cps_fixed = [True if p != False else False for p in fixed_cps_train_params]
            fixed_cp_mask = jax.tree.map(lambda m, p: jnp.array(cps_fixed).reshape(get_dims(p)) if m else False, s_model.curve_mask, params['params'])
            # clears updates for fixed control points
            opt_chain_list.append(element_wise_mask(fixed_cp_mask))

        opt_chain_list.append(optimizer)
        # if "freeze_non_curve_params" in optimizer_conf and optimizer_conf["freeze_non_curve_params"]:
        #     opt_chain_list.append(optax.masked(optax.set_to_zero(), jax.tree.map(lambda x:not x, mask)))
        optimizer = optax.chain(*opt_chain_list)

        only_train_params = s_model.only_trainable_params(params['params'])
        n_params = sum(p.size for p in jax.tree.flatten(only_train_params)[0])
        print(f"Number of trainable parameters: {n_params:_}")
        if k > 0:
            logger.summary.update({'n_params_trainable': n_params})
        opt_state = optimizer.init(only_train_params)  # only trainable parameters according mask

        best_val_ll = -np.inf
        empty_metric = {'val_loss':jnp.array(jnp.inf, dtype=jnp.float32), 'val_acc':jnp.array(-jnp.inf, dtype=jnp.float32), 'val_ece': jnp.array(jnp.inf, dtype=jnp.float32),
                        'val_bma_ll':jnp.array(-jnp.inf, dtype=jnp.float32), 'val_bma_acc':jnp.array(-jnp.inf, dtype=jnp.float32), 'val_bma_ece': jnp.array(jnp.inf, dtype=jnp.float32)}, None
        if val:
            print("save best parameters")
            # best_params = jax.device_get(params)  # best params on cpu to save memory
            best_params = s_model.only_trainable_params(params['params'])  # best params on cpu to save memory
            # best_params = jax.device_put(params, jax.devices('cpu')[0])  # best params on cpu to save memory
            valid_metrics_fn_tmp = setup_metrics(s_model, BATCH_SIZE_EVAL, val_input_ids, val_attention_mask, val_labels, n_samples=10 if k > 0 else 1, key_prefix='val_')
            def valid_metrics_fn(rng, params):
                # always compute metrics with full parameter noise
                if lora_params.get('lora_rho', 0.0) > 0.0 or lora_params.get('lora_rho_s', 0.0) > 0.0:
                    # params = s_model.set_lora_rho(rho_w=lora_params.get('lora_rho', 0.0),  rho_s=lora_params.get('lora_rho_s', 0.0), params=params)
                    params = s_model.set_lora_rho(rho_w=0.0,  rho_s=0.0, params=params)
                    # params = s_model.set_lora_rho(rho_w=0.0,  rho_s=lora_params.get('lora_rho_s', 0.0), params=params)
                return valid_metrics_fn_tmp(rng, params)
            
            def get_best_params(params):
                # return jax.device_get(params)
                # return params
                return s_model.only_trainable_params(params['params'])
        else:
            best_params = None
            valid_metrics_fn = lambda rng, params: empty_metric
            def get_best_params(params):
                return None
            

        # Bezier curve length
        # define function to compute bezier curve length
        def get_d_bezier(t):
            bezier_coeff_inv = bezier_coeff_fn(k)
            coeff = jax.vmap(bezier_coeff_inv)(t)

            def clousure(cp):
                n = cp.shape[0] - 1
                cp_diff = cp[1:] - cp[:-1]
                # improved version to reduce memory usage
                def d_length_t(cp_diff, c):
                    d_t = jnp.linalg.norm(jnp.einsum('j,j...->...', c, cp_diff) * n)
                    return cp_diff, d_t
                return jax.lax.scan(d_length_t, cp_diff, coeff)
            return clousure

        def bezier_length(cp):
            t = jnp.linspace(0, 1, 1000)
            d_norm_bezier = get_d_bezier(t)
            _, d_t = d_norm_bezier(cp)
            length = jax.scipy.integrate.trapezoid(d_t, t)
            return length
        
        def lower_bound(cp):
            # get the length of the line between the first and last control
            return jnp.linalg.norm(cp[-1]-cp[0])
        
        def upper_bound(cp):
            # get the length of the line between each control point
            return jnp.linalg.norm(jnp.diff(cp, axis=0), axis=1).sum()

        def mass_center(cp):
            return jnp.linalg.norm(cp.mean(axis=0))
        
        def gyration(cp):
            mc = cp.mean(0)
            return jnp.linalg.norm(cp - mc, axis=1).mean()
        
        def mean_curvature2(cp):
            tt = jnp.linspace(0, 1, 10)
            _, d_bezier = bezier_curve(cp.shape[0], cp)

            def single(t):
                grad_ = d_bezier(t)
                second_dev = jax.jacrev(d_bezier)(t).squeeze()

                norm_r_prime_sq = jnp.dot(grad_, grad_)  # |r'(t)|^2
                norm_r_double_prime_sq = jnp.dot(
                    second_dev, second_dev)  # |r''(t)|^2
                dot_r_prime_double_prime = jnp.dot(
                    grad_, second_dev)  # r'(t) · r''(t)

                # Gram determinant equivalent for curvature
                numerator = jnp.sqrt(
                    norm_r_prime_sq * norm_r_double_prime_sq - dot_r_prime_double_prime**2)
                denominator = norm_r_prime_sq ** (3/2)
                return numerator / denominator
            
            curvature = jax.vmap(single)(tt)
            return jax.scipy.integrate.trapezoid(curvature, tt)
        
        def init_rel_center(cp):
            mean_center_t0 = cp.mean(axis=0)
            def rel_center(cp):
                return jnp.linalg.norm(cp.mean(axis=0) - mean_center_t0)
            return rel_center
        
        # init metric
        cp_w = masked_pytree_to_matrix(params['params'], s_model.curve_mask, k)
        rel_center_fn = init_rel_center(cp_w)

        if lora_params.get('rho_scheduler_frequency', 0.0) > 0.0:
            def setup_rho_schedule(n_total_steps, frequency=0.5):
                print(f"Using cosine rho schedule with total steps {n_total_steps}")
                base_frequency = n_total_steps / (2*frequency)
                def rho_sched(x, base):
                    # return base * 0.5 * (1 - jnp.cos(x / base_frequency * jnp.pi))
                    return base * (1 - jnp.cos(x / base_frequency * jnp.pi))
                return rho_sched
            rho_scheduler = setup_rho_schedule(num_steps, lora_params.get('rho_scheduler_frequency', 0.0))
        else:
            rho_scheduler = lambda step, base: base

        def inf_mean(x):
            mask = jnp.isfinite(x)
            sum = jnp.where(mask, x, 0).sum()
            count = mask.sum()
            return sum / count


        # @partial(jit, donate_argnums=(0,))
        def train_batch(carry, batch_idx):
            rng_key, params, opt_state, best_val_ll, best_params, update_idx = carry

            x_batch = jnp.take(train_input_ids, batch_idx, axis=0)
            atten_batch = jnp.take(train_attention_mask, batch_idx, axis=0)
            y_ = jnp.take(train_labels, batch_idx, axis=0)

            rng_key, subkey = random.split(rng_key)
            t = t_sample_fn(subkey)
            # t = random.uniform(subkey, (1,), minval=0., maxval=1.)
            if lora_params.get('lora_rho', 0.0) > 0.0 or lora_params.get('lora_rho_s', 0.0) > 0.0:
                current_rho = rho_scheduler(update_idx, lora_params.get('lora_rho', 0.0))
                current_rho_s = rho_scheduler(update_idx, lora_params.get('lora_rho_s', 0.0))
                params = s_model.set_lora_rho(current_rho, current_rho_s, params)
            loss, params, opt_state, (grad_, logs) = s_model.train_step(
                subkey, t, params, (x_batch, atten_batch), y_, opt_state, optimizer)
            # because of optax.chain [-1] gets state of optimizer and next [-1] gets state of learning rate scheduler
            current_lr = lr_schedule(update_idx)
            # validate
            rng_key, valid_key = random.split(rng_key)

            metrics = jax.lax.cond((update_idx % EVAL_EVERY_N_BATCH) == 0, lambda x: valid_metrics_fn(
                valid_key, params)[0], lambda x: empty_metric[0], None)
            best_params = jax.lax.cond(
                metrics['val_bma_ll'] > best_val_ll, lambda x: get_best_params(params), lambda x: best_params, None)
            best_val_ll = jnp.maximum(best_val_ll, metrics['val_bma_ll'])

            # grad_mean = jax.tree.map(lambda p, m: p.mean() if m else None, grad_, s_model.train_mask)
            # grad_metric = {"/".join([str(kk.key) for kk in k]):v for k,v in jax.tree.leaves_with_path(grad_mean)}

            if k>0 and LOG_CURVE_METRICS:
                cp = masked_pytree_to_matrix(params['params'], s_model.curve_mask, k)
                upper_length = upper_bound(cp)
                lower_length = lower_bound(cp)
                mean_center = mass_center(cp)
                gyration_radius = gyration(cp)
                rel_center = rel_center_fn(cp)

                return (rng_key, params, opt_state, best_val_ll, best_params, update_idx+1), {'loss': loss, 'current_lr': current_lr, 'upper_length': upper_length, 'lower_length': lower_length, 'mean_center': mean_center, 'gyration_radius': gyration_radius, 'rel_center': rel_center, **metrics, **logs}
            else:
                return (rng_key, params, opt_state, best_val_ll, best_params, update_idx+1), {'loss': loss, 'current_lr': current_lr, **metrics, **logs}

        @scan_tqdm(num_epochs)
        def train_epoch(carry, epoch):
            rng_key = carry[0]
            rng_key, subkey = random.split(rng_key)
            shuffel_idx = jax.random.permutation(subkey, train_labels.shape[0])
            shuffel_idx = es('(ib)->ib', shuffel_idx, b=batch_size)
            # x_ = es('(ib)...->ib...', train_input_ids[shuffel], b=batch_size)
            # x_atten_mask = es('(ib)...->ib...', train_attention_mask[shuffel], b=batch_size)
            # y_ = es('(ib)...->ib...', train_labels[shuffel], b=batch_size)
            
            # Limit number of batches for smoke test
            if smoke_test:
                smoke_batch_limit = 3
                n_batches = min(smoke_batch_limit, shuffel_idx.shape[0])
                shuffel_idx = shuffel_idx[:n_batches]
                jax.debug.print(f"Smoke test: Limited to {n_batches} batches per epoch")
                
            # train on batches
            carry, metrics = jax.lax.scan(
                train_batch, carry, shuffel_idx)
            # jax.clear_caches()
            # jax.debug.print(f"Epoch {epoch+1}/{num_epochs}, Loss: {metrics['loss'][-1].item()}{metrics['loss'].shape}, Val Loss: {metrics['val_loss'][-1]}, Val Acc: {metrics['val_acc'][-1]}")
            val_loss_ = inf_mean(metrics['val_loss'])
            val_acc_ = inf_mean(metrics['val_acc'])
            jax.debug.print("Epoch {}/{}: Loss: {}, Val Loss: {}, Val Acc: {}", epoch+1,num_epochs, metrics['loss'][-1], val_loss_, val_acc_)
            return carry, metrics


        def init_train_expert(window_size):
            @scan_tqdm(num_steps)
            # @partial(jit, donate_argnums=(0,))
            def _train_expert_step(carry, step):
                # x_all, atten_all, y_all, step = data
                # # skip leading padding tokens in batch for faster training
                # x_batch = x_batch[:, (max_seqlen:=jnp.argmax(atten_batch, axis=1).min()):]
                # atten_batch = atten_batch[:, max_seqlen:]

                # get curve parameter
                rng_key, params, opt_state, best_val_ll, best_params, (x_all, atten_all, y_all) = carry
                rng_key, subkey = random.split(rng_key)

                t = random.uniform(subkey, (1,), minval=0., maxval=1.)

                # get batch from data
                pos = jnp.round(t*x_all.shape[0]).squeeze().astype(int)
                rng_key, subkey = random.split(rng_key)
                idx = jax.random.choice(subkey, window_size, shape=(batch_size,), replace=False) + pos - window_size//2
                idx %= x_all.shape[0] # wrap around
                x_batch = jnp.take(x_all, idx, axis=0, mode='clip')
                atten_batch = jnp.take(atten_all, idx, axis=0, mode='clip')
                y_ = jnp.take(y_all, idx, axis=0, mode='clip')

                t = t_sample_fn(subkey)
                loss, params, opt_state, (grad_, logs) = s_model.train_step(
                    subkey, t, params, (x_batch, atten_batch), y_, opt_state, optimizer)
                # because of optax.chain [-1] gets state of optimizer and next [-1] gets state of learning rate scheduler
                current_lr = lr_schedule(step)
                
                # validate
                # rng_key, valid_key = random.split(rng_key)
                # # t = random.uniform(subkey, (1,), minval=0., maxval=1.)
                # idx += window_size  # move window
                # idx = jnp.arange(0, batch_size*3) + pos + x_all.shape[0]//2 # move halfway around the dataset
                # idx %= x_all.shape[0]  # wrap around
                # x_val = jnp.take(x_all, idx, axis=0, mode='clip')
                # atten_val = jnp.take(atten_all, idx, axis=0, mode='clip')
                # y_val = jnp.take(y_all, idx, axis=0, mode='clip')
                # val_loss, stat, val_out = s_model.nll(params['params'], params[s_model.mutable_name] if s_model.mutable_name else {}, t, (x_val, atten_val), y_val, False)
                # val_acc = (jnp.argmax(val_out, axis=-1) == y_val).mean()
                # return (rng_key, params, opt_state, best_val_ll, best_params, (x_all, atten_all, y_all)), {'loss': loss, 'current_lr': current_lr, 'val_loss': val_loss, 'val_acc': val_acc, **logs}

                # validate
                rng_key, valid_key = random.split(rng_key)
                metrics = jax.lax.cond((step % EVAL_EVERY_N_BATCH) == 0, lambda x: valid_metrics_fn(
                    valid_key, params)[0], lambda x: empty_metric[0], None)
                best_params = jax.lax.cond(
                    metrics['val_bma_ll'] > best_val_ll, lambda x: get_best_params(params), lambda x: best_params, None)
                best_val_ll = jnp.maximum(best_val_ll, metrics['val_bma_ll'])

                return (rng_key, params, opt_state, best_val_ll, best_params, (x_all, atten_all, y_all)), {'loss': loss, 'current_lr': current_lr, **metrics, **logs}


            return _train_expert_step

        # print memory stats
        # memory_stats = jax.devices()[0].memory_stats()
        # for stats_name, v in memory_stats.items():
        #     print(f"{stats_name}: {v / (1024 * 1024):.2f} MB" if "bytes" in stats_name else f"{stats_name}: {v}")
        
        # train loop if num_steps or num_epochs > 0 and k = 0 => single point fine-tuning or 
        # if num_steps > 0 and ((k==0) or (k > 1) and (np.sum([0. if type(p) == bool else 1. for p in fixed_cps_train_params]) <= k)): # only if there are free control points left to train
        if num_steps > 0 and ((curve_state == ProcessState.CURVE_FITTING) or (curve_state == ProcessState.PRETRAINING)): # only if there are free control points left to train
            if ds_sampling == 'expert':
                print(f"Start sorted expert training for {num_steps}")
                if sort_mode == 'Random':
                    order = jax.random.permutation(rng_key, train_input_ids.shape[0])
                elif sort_mode:
                    distance = 1 - cosine_sim
                    if sort_mode == 'Hierarchical':
                        # convert to condensed vector form
                        # distance = distance.at[jnp.diag_indices(distance.shape[0])].set(0.0)
                        # distance = 0.5 * (distance + distance.T)  # ensure symmetry
                        condensed = squareform(distance, checks=False)
                        Z = hierarchy.linkage(condensed, method='average')
                        Z_opt = hierarchy.optimal_leaf_ordering(Z, condensed)
                        order = hierarchy.dendrogram(Z_opt.astype(np.float64), no_plot=True)['leaves']
                        order = jnp.array(order)
                    elif sort_mode == 'Spectral':
                        distance = distance.at[jnp.diag_indices(distance.shape[0])].set(0.0)
                        # Schritt 3: Gradientenmatrix D
                        D = jnp.diag(distance.sum(axis=1))
                        # Schritt 4: Laplacian L
                        L = D - distance
                        # Schritt 5: Eigenvektoren holen
                        vals, vecs = jnp.linalg.eigh(L)
                        # Schritt 6: Fiedler vector (zweiter kleinster Eigenwert)
                        fiedler = vecs[:, -2]
                        # Schritt 7: Sortierung nach Fiedler-Vektor
                        order = jnp.argsort(fiedler)
                    elif sort_mode == 'First_data':
                        order = jnp.argsort(distance[0])
                    else:
                        raise ValueError(f"Unknown sort_mode {sort_mode} for expert sampling.")
                    train_input_ids = train_input_ids[order]
                    train_attention_mask = train_attention_mask[order]
                    train_labels = train_labels[order]
                    print(f"Data sorted for expert sampling using {sort_mode} method")

                # init expert training loop
                window_size = dataset_sampling['expert']['window_size']  # context window size for sampling
                window_size = max(batch_size, window_size) # at least half batch size
                train_expert_step = init_train_expert(window_size)
                # start expert training loop
                carry = (rng_key, params, opt_state, best_val_ll, best_params, (train_input_ids, train_attention_mask, train_labels))
                carry, metrics = jax.lax.scan(
                    train_expert_step, carry, jnp.arange(num_steps))
            elif ds_sampling == 'minibatch':
                print(f"Start minibatch training for {num_epochs} epochs")
                carry = (rng_key, params, opt_state, best_val_ll, best_params, 0) # 0 is used for update counts
                carry, metrics = jax.lax.scan(
                    train_epoch, carry, jnp.arange(num_epochs))
                window_size = 0  # not used in minibatch training
            else:
                raise ValueError(f"Unknown dataset_sampling method {ds_sampling} given. Choose either 'expert' or 'minibatch'.")

            # clean up inf values because of lazy validation
            for key, value in metrics.items():
                if key.startswith('val_'):
                    metrics[key] = jnp.array(value).flatten()[::EVAL_EVERY_N_BATCH].repeat(
                        EVAL_EVERY_N_BATCH, axis=0)[:num_steps]
                else:
                    metrics[key] = jnp.array(value).flatten()
            
            # log all metrics
            _ = [logger.log(dict(zip(metrics.keys(), m))) for m in zip(*metrics.values())]

            # # print memory stats
            # memory_stats = jax.devices()[0].memory_stats()
            # for stats_name, v in memory_stats.items():
            #     print(f"{stats_name}: {v / (1024 * 1024):.2f} MB" if "bytes" in stats_name else f"{stats_name}: {v}")

            # log metrics as artifact
            epochs = np.repeat(np.arange(num_epochs), num_steps//batch_size)
            jnp.savez(f'tmp_files/{logger.id}_{logger_prefix}metrics.npz', **metrics, epochs=epochs)
            art = wandb.Artifact(name="metrics", type="npz")
            art.add_file(f'tmp_files/{logger.id}_{logger_prefix}metrics.npz')
            logger.log_artifact(art)

            rng_key = carry[0]
            params = carry[1]  # last params on device
            last_trainable_params = None
            if val:
                # params = carry[1]  # last params on device
                last_trainable_params = s_model.only_trainable_params(params['params'])
                best_curve_params = carry[4]  # best params (only trainable params)
                # update last params with best curve params (only trainable params were stored)
                params['params'] = jax.tree.map(lambda x, y, m: x if m else y, best_curve_params, params['params'], s_model.train_mask)
            

            if k > 0:
                # Bezier curve length        
                cp = masked_pytree_to_matrix(params['params'], s_model.curve_mask, k)
                length = bezier_length(cp)
                logger.summary.update({'Curve length': length})

        #*** compute test metrics with BMA ***#
        def post_pred_performance(logits, labels, weights:bool|jnp.ndarray=False):
            # logits shape (n_samples, n_data, n_classes)
            print(f"weights type {type(weights)} for BMA prediction {type(weights) == bool}")
            if type(weights) == bool:
                post_predict_logits = jax.nn.logsumexp(logits, axis=0) - jnp.log(logits.shape[0])  # shape (n_data, n_classes)
                # post_predict_logits = (logits * weights[:, None, None]).sum(axis=0)  # shape (n_data, n_classes) #false because of log multiply
            else:
                print("Using provided weights for BMA prediction")
                post_predict_logits = jax.nn.logsumexp(logits, b=weights[:, None, None], axis=0)
            post_acc = jnp.mean(jnp.argmax(post_predict_logits, axis=-1) == labels)
            post_ece = ece_fn(post_predict_logits, labels, num_bins=15)
            return {'ll': dist.Categorical(logits=post_predict_logits).log_prob(labels).mean(),
                           'acc': post_acc,
                           'ece': post_ece}
 
        # plot likelihood and acc along curve
        t_space = jnp.linspace(0, 1, 20)
        if smoke_test or k == 0:
            t_space = jnp.linspace(0, 1, 3)
        test_metrics_fn_final = setup_metrics(s_model, max(BATCH_SIZE_EVAL, 1), test_input_ids, test_attention_mask, test_labels, n_samples=len(t_space), use_linspace=True, average=False, key_prefix='final_')
        # valid
        # set lora noise off for evaluation if specified
        if lora_params.get('lora_rho', 0.0) > 0.0 or lora_params.get('lora_rho_s', 0.0) > 0.0:
            # params = s_model.set_lora_rho(rho_s=lora_params.get('lora_rho_s', 0.0), rho_w=lora_params.get('lora_rho', 0.0), params=params)
            params = s_model.set_lora_rho(rho_s=0.0, rho_w=0.0, params=params)
            # params = s_model.set_lora_rho(rho_s=lora_params.get('lora_rho_s', 0.0), rho_w=0.0, params=params)
        curve_metric, logits_test = test_metrics_fn_final(rng_key, params) # logits test with shape (n_samples, n_data, n_classes)
        # post_performance = post_pred_performance(logits_test, test_labels)
        logger.summary.update({f'{logger_prefix}test_mean_{k}':v.mean() for k,v in curve_metric.items()})

        if (curve_state == ProcessState.CURVE_FITTING) or (curve_state == ProcessState.CURVE_EVAL):
            fig, ax = plt.subplots(1, 1, figsize=(4, 3), sharey=True)
            ax.plot(t_space, -curve_metric['final_loss'], label="test_ll", c=plt.get_cmap('tab10')(0))
            ax2 = ax.twinx()
            ax2.plot(t_space, curve_metric['final_acc'], label="test acc",
                    c=plt.get_cmap('tab10')(1), linestyle="--")

            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("outward", 40))
            ax3.set_frame_on(True)
            ax3.patch.set_visible(False)
            ax3.plot(t_space, curve_metric['final_ece'], label="test ece",
                    c=plt.get_cmap('tab10')(2), linestyle=":")
            ax.set_xlabel("t")
            ax.set_ylabel("mean log likelihood")
            ax2.set_ylabel("Accuracy")
            ax3.set_ylabel("ECE")
            ax.set_title("Final testset")
            # combine legends from primary and twin axes
            lines, legend_labels1 = ax.get_legend_handles_labels()
            lines2, legend_labels2 = ax2.get_legend_handles_labels()
            lines3, legend_labels3 = ax3.get_legend_handles_labels()
            ax.legend(lines + lines2 + lines3, legend_labels1 + legend_labels2 + legend_labels3, loc='best', fontsize='small')
            plt.tight_layout()
            logger.log(
                {'Curve Predictive': wandb.Image(fig)})

            # compute and log final curve metrics
            # compute log likelihood for t grid on train data to get posterior weightings
            if curve_sampling_mode == "combined":
                weights = False
                print("Temperature:", temperature)
                def compute_weights(log_like, t):
                    log_post = log_like / t
                    log_post -= jax.scipy.special.logsumexp(log_post)
                    return jnp.exp(log_post)
                if num_epochs > 0 or num_steps > 0: # train ds must be defined
                    print("Compute posterior weights on train set")
                    train_metrics_fn = setup_metrics(s_model, max(BATCH_SIZE_EVAL, 1), train_input_ids, train_attention_mask, train_labels, n_samples=len(t_space), use_linspace=True, average=False, key_prefix='f_train_')
                    # compute weigths without noise
                    if lora_params.get('lora_rho', 0.0) > 0.0 or lora_params.get('lora_rho_s', 0.0) > 0.0:
                        params = s_model.set_lora_rho(0., 0., params)
                    train_curve_metric, logits_train = train_metrics_fn(rng_key, params)

                    post_performance = post_pred_performance(logits_train, train_labels)
                    logger.summary.update({f'train_mean_{k}':v.mean() for k,v in post_performance.items()})
                    mi = mutual_information(logits_train.astype(jnp.float32))
                    logger.summary.update({'train_mutual_information': mi})
                    entrop = entropy(logits_train.astype(jnp.float32))
                    logger.summary.update({'train_std_entropy_along_curve': entrop.mean(-1).std(), 'train_mean_entropy_along_curve': entrop.mean()}) # average data/ mean or std along curve

                    # plot curve of train log likelihood
                    fig, ax = plt.subplots(1, 1, figsize=(4, 3), sharey=True)
                    ax.plot(t_space, -train_curve_metric['f_train_loss'], label="test_ll", c=plt.get_cmap('tab10')(0))
                    ax2 = ax.twinx()
                    ax2.plot(t_space, train_curve_metric['f_train_acc'], label="test acc",
                            c=plt.get_cmap('tab10')(1), linestyle="--")
                    ax3 = ax.twinx()
                    ax3.spines["right"].set_position(("outward", 40))
                    ax3.set_frame_on(True)
                    ax3.patch.set_visible(False)
                    ax3.plot(t_space, train_curve_metric['f_train_ece'], label="test ece",
                            c=plt.get_cmap('tab10')(2), linestyle=":")
                    ax.set_xlabel("t")
                    ax.set_ylabel("mean log likelihood")
                    ax2.set_ylabel("Accuracy")
                    ax3.set_ylabel("ECE")
                    ax.set_title("Final trainset")
                    lines, legend_labels1 = ax.get_legend_handles_labels()
                    lines2, legend_labels2 = ax2.get_legend_handles_labels()
                    lines3, legend_labels3 = ax3.get_legend_handles_labels()
                    ax.legend(lines + lines2 + lines3, legend_labels1 + legend_labels2 + legend_labels3, loc='best', fontsize='small')
                    plt.tight_layout()
                    logger.log(
                        {'Curve Predictive Train': wandb.Image(fig)})
                    
                    # loss is mean and without correct likelihood constant. In addition Uniform prior is dropped to unnormalized log posterior computation
                    weights = jax.vmap(compute_weights, in_axes=(None, 0))(-train_curve_metric['f_train_loss'], jnp.array(temperature))  # shape (#temperature, #samples)
                    art_w = wandb.Artifact(name="bma_weights", type="npz")
                    jnp.savez(f'tmp_files/{logger.id}_weights.npz', weights=weights, t_space=t_space, log_like=-train_curve_metric['f_train_loss'], temperature=temperature)
                    art_w.add_file(f'tmp_files/{logger.id}_weights.npz')
                    logger.log_artifact(art_w)
                    print("Saved posterior weights artifact")
                else:
                    if artifact_weights_use:
                        for f in artifact_weights_use.files():
                            if 'weights' in f.name:
                                weight_file = np.load(artifact_weights_use.get_entry(f.name).download(), allow_pickle=True)
                                weights = jax.vmap(compute_weights, in_axes=(None, 0))(weight_file['log_like'], jnp.array(temperature))
                                # weights = weight_file['weights']
                                t_space = weight_file['t_space']
                                print("Loaded posterior weights from artifact")
                    assert type(weights) != bool, f"Weights for posterior predictive could be computed or are not saved in {SWEEP_PARAM_PATH}."

                if val:            
                    valid_metrics_fn_final = setup_metrics(s_model, max(BATCH_SIZE_EVAL, 1), val_input_ids, val_attention_mask, val_labels, n_samples=len(t_space), use_linspace=True, average=False, key_prefix='final_valid_')
                    curve_metric_val, logits_val = valid_metrics_fn_final(rng_key, params) # logits test with shape (n_samples, n_data, n_classes)
                    log_like_val_best = {'ll':-jnp.inf, 'temp': None, 'weight': None}
                    for weight, temp in zip(weights, temperature):
                        post_performance = post_pred_performance(logits_val, val_labels, weight)
                        if log_like_val_best['ll'] < post_performance['ll']:
                            log_like_val_best = {'ll': post_performance['ll'], 'temp': temp, 'weight': weight}
                        logger.summary.update({f'post_pred_val_ppd_t{temp}_{k}':v for k,v in post_performance.items()})
                    logger.summary.update({'best_val_log_like_temp': log_like_val_best['temp'], 'best_val_log_like': log_like_val_best['ll']})

                    post_performance = post_pred_performance(logits_test, test_labels, log_like_val_best['weight'])
                    logger.summary.update({f'post_pred_test_ppd_{k}':v for k,v in post_performance.items()})
                    mi = mutual_information(logits_test.astype(jnp.float32), weights=log_like_val_best['weight'])
                    m_entropy = mean_entropy(logits_test.astype(jnp.float32), weights=log_like_val_best['weight'])
                    logger.summary.update({f'test_mutual_information_ppd': mi, f'test_mean_entropy_ppd': m_entropy})
                else: 
                    for weight, temp in zip(weights, temperature):
                        post_performance = post_pred_performance(logits_test, test_labels, weight)
                        logger.summary.update({f'post_pred_test_ppd_t{temp}_{k}':v for k,v in post_performance.items()})
                        mi = mutual_information(logits_test.astype(jnp.float32), weights=weight)
                        m_entropy = mean_entropy(logits_test.astype(jnp.float32), weights=weight)
                        logger.summary.update({f'test_mutual_information_ppd_t{temp}': mi, f'test_mean_entropy_ppd_t{temp}': m_entropy})

                entrop = entropy(logits_test.astype(jnp.float32))
                logger.summary.update({'std_entropy_along_curve': entrop.mean(-1).std(), 'mean_entropy_along_curve': entrop.mean()})
            else:
                print("Using random sampling to approx. independent curves")
                if "combined" not in curve_sampling_mode:
                    print("Using random sampling to approx. independent curves")
                    test_metrics_fn_final = setup_metrics(s_model, max(BATCH_SIZE_EVAL, 1), test_input_ids, test_attention_mask, test_labels, n_samples=len(t_space), use_linspace=False, average=False, key_prefix='approx_', t_sample_fn=t_sample_fn)
                    # valid
                    curve_metric, logits_test = test_metrics_fn_final(rng_key, params) # logits test with shape (n_samples, n_data, n_classes)
                post_performance = post_pred_performance(logits_test, test_labels)
                logger.summary.update({f'{logger_prefix}approx_pred_test_mean_{k}':v for k,v in post_performance.items()})

                mi = mutual_information(logits_test.astype(jnp.float32))
                m_entropy = mean_entropy(logits_test.astype(jnp.float32))
                entrop = entropy(logits_test.astype(jnp.float32))
                logger.summary.update({f'test_mutual_information_approx': mi, f'test_mean_entropy_approx': m_entropy, 'std_entropy_approx': entrop.mean(-1).std(), 'mean_entropy_approx': entrop.mean()})

            # reset lora noise if specified
            # if lora_params.get('lora_rho', 0.0) > 0.0 or lora_params.get('lora_rho_s', 0.0) > 0.0:
            #     params = s_model.set_lora_rho(rho_s=lora_params.get('lora_rho_s', 0.0), rho_w=lora_params.get('lora_rho', 0.0), params=params)
            #     curve_metric, logits_test = test_metrics_fn_final(rng_key, params) # logits test with shape (n_samples, n_data, n_classes)
            #     post_performance = post_pred_performance(logits_test, test_labels)
            #     logger.summary.update({f'{logger_prefix}approx_pred_test_mean_wRho_{k}':v for k,v in post_performance.items()})

            #     mi = mutual_information(logits_test.astype(jnp.float32))
            #     m_entropy = mean_entropy(logits_test.astype(jnp.float32))
            #     entrop = entropy(logits_test.astype(jnp.float32))
            #     logger.summary.update({f'test_mutual_information_approx_wRho_': mi, f'test_mean_entropy_approx_wRho_': m_entropy, 'std_entropy_approx_wRho_': entrop.mean(-1).std(), 'mean_entropy_approx_wRho_': entrop.mean()})



        # post_predict_logits = (logits_test * weights[:, None, None]).sum(axis=0)  # shape (n_data, n_classes)
        # post_acc = jnp.mean(jnp.argmax(post_predict_logits, axis=-1) == test_labels)
        # post_ece = ece_fn(post_predict_logits, test_labels, num_bins=15)
        # logger.log({'post_predictive': (-curve_metric['final_loss'] * weights).sum(),
        #             'post_pred_acc': post_acc,
        #             'post_pred_ece': post_ece,
        #             })
        return rng_key, s_model, params, last_trainable_params



    params = None
    cp_fix = config['curve_params'].pop('cp_fix')

    # pretrain fixed control points 
    if np.sum(cp_fix) > 0:
        curve_state = ProcessState.PRETRAINING
        if config.get('lora_params', None) is not None and np.sum(cp_fix) < len(cp_fix):
            assert (config['lora_params']['lora_mode'] == 'A(t)B(t)') or config['lora_params']['lora_mode'] == 'A(t)eB(t)', "Only A(t)B(t) or A(t)eB(t) LoRA mode makes sense for subspace curves with fixed control points" 
        train_params = []
        print("Pretraining fixed control points")
        for i, fixed in enumerate(cp_fix):
            if fixed:
                print(f"Training cp member {i}")
                config['curve_params']['k'] = 0 

                if params is not None:
                    del params  # free memory
                rng_key, s_model, params, _ = train_curve_inner(
                    rng_key, curve_state, **config['curve_params'], net_kwargs=config['net_kwargs'], 
                    lora_params=config.get('lora_params', None), dataset_path=config['dataset_path'],
                    smoke_test=config.get('smoke_test', False), dataset_sampling=config.get('dataset_sampling', {}),
                    logger_prefix=f'cp_fixed_{i}_')
                train_params.append(s_model.only_trainable_params(params['params']))
                rng_key, _ = random.split(rng_key)
            else:
                train_params.append(False)

        if SAVE_PARAMS:
            print("Save pretrained trainable params ...")
            np.save(f'tmp_files/{logger.id}_pretrained_params.npy', train_params)
            # artifact.add_file(f'tmp_files/{logger.id}_best_params.npy')
            time.sleep(1)  # ensure file is written
            print("Saved pretrained parameters to tmp_files")
            artifact.add_file(f'tmp_files/{logger.id}_pretrained_params.npy')
    else:
        train_params = False
    if params is not None:
        del params  # free memory
    

    # train curve model
    if len(cp_fix) > 1:
        print("Training curve model")
        config['curve_params']['k'] = len(cp_fix)-1
        if len(cp_fix) == np.sum(cp_fix): # all control points are fixed e.g. [1,1] but we check linear mode connection
            curve_state = ProcessState.CURVE_EVAL
        else:
            curve_state = ProcessState.CURVE_FITTING
        rng_key, s_model, params, last_trainable_params = train_curve_inner(
                rng_key, curve_state, **config['curve_params'], net_kwargs=config['net_kwargs'], 
                lora_params=config.get('lora_params', None), dataset_path=config['dataset_path'],
                smoke_test=config.get('smoke_test', False), dataset_sampling=config.get('dataset_sampling', {}), 
                fixed_cps_train_params=train_params)
    
        if SAVE_PARAMS:
            print("Save final trainable params ...")
            np.save(f'tmp_files/{logger.id}_trainable_params.npy', s_model.only_trainable_params(params['params']))
            if last_trainable_params is not None:
                np.save(f'tmp_files/{logger.id}_last_trainable_params.npy', last_trainable_params)
            # artifact.add_file(f'tmp_files/{logger.id}_best_params.npy')
            time.sleep(1)  # ensure file is written
            print("Saved trainable parameters to tmp_files")
            artifact.add_file(f'tmp_files/{logger.id}_trainable_params.npy')
            if last_trainable_params is not None:
                artifact.add_file(f'tmp_files/{logger.id}_last_trainable_params.npy')

    if SAVE_PARAMS:    
        logger.log_artifact(artifact)
        time.sleep(5)  # ensure artifact is logged
        print("Logged artifact with parameters.")
    logger.finish()


if __name__ == '__main__':
        # Add requirement for wandb core
    wandb.require("core")
    os.makedirs('tmp_files', exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Train a single model or a curve model.")
    # parser.add_argument("--train", choices=["single", "curve"], required=True, help="Specify the training mode: 'single' for single model training, 'curve' for curve model training.")
    parser.add_argument("--use-sweep", action="store_true", help="Use sweep configuration.")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size for training. Default is 16.")
    parser.add_argument("--cp-fix", type=int, nargs='+', default=[1,0,1], help=" number of control points as list where 1 means p is fixed and 0 means p is trainable as curve. Default is [1,0,1].")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test with reduced epochs (2) and batch iterations (3).")

    args = parser.parse_args()
    print(f"Fixed control points: {args.cp_fix}")

    
    print("Curve training")
    if args.use_sweep:
        print("Using wandb config")
        logger = wandb.init()
        config = wandb.config
        train(logger, config)
    else:
        print("Using perdefined config")
        
        # Apply smoke test settings if enabled
        num_epochs = 2 if args.smoke_test else 5  # Keep original value of 2
        
        config = {
            'rng_seed': 1,
            # 'dataset_path': "datasets/text_classification/boolq/train_data.npz",
            'dataset_path': "ddold/qwen_PtoC/winogrande_m_dataset:v0",
            # 'dataset_path': "ddold/qwen_PtoC/ARC-Easy_dataset:v0",
            # 'dataset_path': "ddold/qwen_PtoC/MMLU_chem_dataset:v0",
            # 'dataset_path': "ddold/qwen_PtoC/obqa_dataset:v0",
            'net_kwargs':{'model_path': "artifacts/qwen2.5_7B_bfloat16:v0",},
            # 'net_kwargs':{'model_path': "ddold/qwen_PtoC/qwen2.5_7B_bfloat16:v0",},
            # 'load_params_path': "ddold/qwen_PtoC/47xexal3", # path to pretrained trainable parameters
            'load_params_path': False, # path to pretrained trainable parameters
            # 'load_params_path': "ddold/qwen_PtoC/ywepcmp3", # path to pretrained trainable parameters
            'save_params': True,
            'eval_every_n_batch': 20,
            'smoke_test': args.smoke_test,
            'dataset_sampling': {'minibatch': {}},  # for minibatch sampling
            # 'dataset_sampling': {'expert': {'window_size': 4, 'sort_mode': False}},  # 'balanced' or 'uniform' or None
            'lora_params': {'use_lora': True, 'r': 8, 
                            'lora_dtype': 'float32',
                            'lora_alpha': 16.0,
                            # 'lora_mode': 'A(t)B(t)',
                            'lora_mode': 'Asd(t)eB',
                            # 'lora_mode': 'Arotsd(t)eB',
                            # 'lora_mode': 'Arot(t)sdeBrot(t)',
                            'lora_rho': .25,
                            'lora_rho_s': .5,
                            'rho_scheduler_frequency': 100.,
                            'filter_masks': 
                                [
                                    {'keys': ["self_attn", "q_proj", "kernel"], 'op': 'all', 'dims': [1, 0]},
                                    {'keys': ["self_attn", "v_proj", "kernel"], 'op': 'all', 'dims': [1, 0]},
                                    {'keys': ["lm_head", "kernel"], 'op': 'all', 'dims': [1, 0]},
                                ]},  # use lora for query projection
                                # [
                                #     {'keys': ["self_attn", "q_proj", "kernel"], 'op': 'all', 'dims': [0, 1]},
                                #     {'keys': ["self_attn", "v_proj", "kernel"], 'op': 'all', 'dims': [0, 1]},
                                #     {'keys': ["lm_head", "kernel"], 'op': 'all', 'dims': [0, 1]},
                                # ]},  # use lora for query projection
            'curve_params': {'cp_fix': args.cp_fix,
                            'num_epochs': -1,
                            'num_steps': 10 if args.smoke_test else 5_000,
                            # 'num_steps': 5000,
                            # 'num_steps': -1,
                            'batch_size': args.batch_size,
                            'natural_parameterization': False,
                            'gravity': 5.0,
                            'energy_weakening': 4., 
                            'regularizer': False, # False, 'cosine', 'energy', 'force', 
                            'entropy_maxim': 0.,
                            'weight_decay': 0.,
                            'jitter_multiplier': 0.,  # np.sqrt(2./fan_in, dtype=np.float32) fan_in = 1280 * 16 or 1280 = 1e-2
                            'temperature': [1.0,],
                            'val_percentage' : 0.1,
                            'curve_parameterization': 'bezier',
                            'curve_sampling_mode': 'combined_noBMA', # 'per_leave' or 'combined'
                            # 'curve_sampling_mode': 'per_leave', # 'per_leave' or 'combined'
                            # 'filter_masks':[{'keys': ("self_attn", "q_proj", "kernel"), 'op': 'all'},
                            #                 {'keys': ("self_attn", "v_proj", "kernel"), 'op': 'all'}],
                            'filter_masks':[
                                            {'keys': ["self_attn", "q_proj", "kernel"], 'op': 'all'},
                                            {'keys': ["self_attn", "v_proj", "kernel"], 'op': 'all'},
                                            {'keys': ["lm_head", "kernel"], 'op': 'all'},
                                            ],
                            #  'optimizer_conf': {'name': "adamw", 'kwargs': {'learning_rate': 0.05, 'weight_decay': 0.0001}},
                            #  'optimizer_conf': {'name': "adamw", 'learning_rate': 0.05, 'momentum': 0.9, 'nesterov': True},
                            # 'optimizer_conf': {'name': "sgd", 
                            #                     'kwargs': {'learning_rate': 0.0001, 'momentum': 0.9, 'nesterov': True},
                            #                     'grad_clip_norm': 100.0,
                            # #                     'freeze_other_params': True},
                            # 'optimizer_conf': {'name': "sgd", 'kwargs': {'learning_rate': {'name': "linear_onecycle_schedule", 
                            #                                                                 'kwargs': {'transition_steps': 490_000, 'peak_value': 0.1, 'pct_start': 0.2, 'pct_final': 0.6, 'div_factor': 5.0, 'final_div_factor': 10.0}},
                            #                                                 'momentum': 0.9, 
                            #                                                 'nesterov': True},},
                            'optimizer_conf': {'name': "adamw", 'kwargs': {'learning_rate': {'name': "linear_onecycle_schedule", 
                                                                                            'kwargs': {'transition_steps': 5_000, 'peak_value': 1e-4, 'pct_start': 0.12, 'pct_final': 1., 'div_factor': 300.0, 'final_div_factor': 300.0}},
                                                                            'weight_decay': 0.0,
                                                                            },
                                                                    'freeze_other_params': True,
                                                },
                            # 'optimizer_conf': {'name': "sam_adamw", 'kwargs': {'learning_rate': {'name': "linear_onecycle_schedule", 
                            #                                                                 'kwargs': {'transition_steps': 12_000, 'peak_value': 1e-4, 'pct_start': 0.06, 'pct_final': 1., 'div_factor': 300.0, 'final_div_factor': 300.0}},
                            #                                                 'weight_decay': 0.01,
                            #                                                 'adv_rho': 0.05,
                            #                                                 },
                            #                                         'freeze_other_params': True,
                            #                    },
                        }}
        logger = wandb.init(project="qwen_PtoC", name="test_noWD on_curve + epsilon_s 05", entity="ddold", config=config)
        train(logger, config)

