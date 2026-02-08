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
    LoRARepulsiveCategorySubspace, LoRACategorySubspace, CategorySubspace, RepulsiveCategorySubspace, LoRAEntropyCategorySubspace, LoRAJSDCategorySubspace
)
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Set environment variables
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=10"
# import sys
# sys.setrecursionlimit(200)

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

        return {key_prefix + 'loss': loss, key_prefix + 'acc': acc, key_prefix + 'ece': ece}, out  # mean over epochs
    return acc_fn


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
    rng_key = random.PRNGKey(config['rng_seed'])
    EVAL_EVERY_N_BATCH = config['eval_every_n_batch']
    BATCH_SIZE_EVAL = config['curve_params']['batch_size'] * 2
    SAVE_PARAMS = config.get('save_params', False)
    SWEEP_PARAM_PATH = config.get('load_params_path', False)
    

    input_ids, attention_mask, labels, target_ids, unique_target_ids = load_text_classification_data(config['dataset_path'], run=logger)
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
                         k: int,
                         num_epochs: int,
                         batch_size: int,
                         optimizer_conf: dict,
                         natural_parameterization: bool,
                         gravity: float,
                         energy_weakening: float,
                         regularizer: str,
                         dataset_path: str,
                         filter_masks: list = None,
                         lora_params: dict = None,
                         net_kwargs: dict = None,
                         weight_decay: float = 0.0,
                         jitter_multiplier: float = 0.0,
                         smoke_test: bool = False,
                         num_steps: int = -1,  # num steps are either given or computed from num_epochs
                         temperature: list = [1.0,],
                         val_percentage: float = 0.0,
                         ):
        # Note: jitter parameter is unused but kept for interface compatibility
        rng_key, init_key = random.split(rng_key)
        rng_key, split_key = random.split(rng_key)
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
            perm_idx = random.permutation(split_key, len(input_ids))
            train_input_ids = input_ids[perm_idx][:n_train]
            train_attention_mask = attention_mask[perm_idx][:n_train]
            train_labels = labels[perm_idx][:n_train]
            
            val_input_ids = input_ids[perm_idx][n_train:]
            val_attention_mask = attention_mask[perm_idx][n_train:]
            val_labels = labels[perm_idx][n_train:]
            if len(val_input_ids) > 0:
                print("Validation data are used")
                val = True

        # load init model parameters
        model = QwenTextClassificationWrapper(**net_kwargs, target_token_ids=unique_target_ids)
        params = unfreeze(model.init(None, None))
        dtype = jax.tree.leaves(params)[0].dtype
        
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
            if regularizer:
                print("Using LoRARepulsiveCategorySubspace with regularizer")
                s_model = LoRARepulsiveCategorySubspace(lora_mask=lora_mask, r=lora_params['r'], lora_dtype=lora_params['lora_dtype'], lora_alpha=lora_params['lora_alpha'], lora_mode=lora_params['lora_mode'],
                                                        lora_rho=lora_params.get('lora_rho', 0.0), lora_rho_s=lora_params.get('lora_rho_s', 0.0),
                                                        model=model, k=k, weight_decay=weight_decay,
                                                        perm_spec=perm_spec,
                                                        natural_parameterization=natural_parameterization,
                                                        mutuable_param_name=False,
                                                        gravity=gravity, energy_weakening=energy_weakening, regularizer=regularizer)
            else:
                print("Using QuantileSubspace without repulsive force")
                s_model = LoRACategorySubspace(lora_mask=lora_mask, r=lora_params['r'], lora_dtype=lora_params['lora_dtype'], lora_alpha=lora_params['lora_alpha'], lora_mode=lora_params['lora_mode'],
                                                lora_rho=lora_params.get('lora_rho', 0.0), lora_rho_s=lora_params.get('lora_rho_s', 0.0),
                                                model=model, k=k, weight_decay=weight_decay, 
                                                perm_spec=perm_spec,
                                                natural_parameterization=natural_parameterization,
                                                mutuable_param_name=False,
                                                )
        else:
            if regularizer:
                print("Using RepulsiveCategorySubspace with regularizer")
                s_model = RepulsiveCategorySubspace(model=model, k=k, weight_decay=weight_decay,
                                                   perm_spec=perm_spec,
                                                   natural_parameterization=natural_parameterization,
                                                   mutuable_param_name=False,
                                                   gravity=gravity, energy_weakening=energy_weakening, regularizer=regularizer)
            else:
                print("Using CategorySubspace without repulsive force")
                s_model = CategorySubspace(model=model, k=k, weight_decay=weight_decay, 
                                           perm_spec=perm_spec,
                                           natural_parameterization=natural_parameterization,
                                           mutuable_param_name=False)


        # initialize curve parameter with pretrained point parameter
        params = s_model.init_params_from_point(init_key, params, curve_mask, noise_fn=lambda p, key: p+jitter_multiplier*jax.random.normal(key, p.shape, dtype=p.dtype))

        # set train mask according to config
        # Only update mask for LoRA-capable subspaces
        if lora_params and lora_params['use_lora']:
            train_mask = s_model.update_mask_for_lora(train_mask, params['params'], b_off=False, w0_off=True) # w0 must be off for LoRA
        s_model.set_train_mask(train_mask)

        artifact_weights_use = False
        if SWEEP_PARAM_PATH:
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

        # compute number of parameters
        n_params = sum(p.size if isinstance(p, jnp.ndarray) else 0 for p in jax.tree.flatten(params)[0])
        param_memory = sum(p.size*p.dtype.itemsize if isinstance(p, jnp.ndarray) else 0 for p in jax.tree.flatten(params)[0])
        logger.summary.update({'n_params': n_params, 'memory_params': param_memory})
        print(f"Number of parameters: {n_params:_}\nMemory of parameters: {param_memory/(1024**2):_}MB")

        # DE member run   
        # print("Computing initial metrics...")
        # init_metrics_fn = setup_metrics(s_model, BATCH_SIZE_EVAL, test_input_ids, test_attention_mask, test_labels, n_samples=1, use_linspace=True, average=True, key_prefix='init_')
        # init_metrics, init_out = init_metrics_fn(init_key, params) # returns a dict
        # logger.summary.update(init_metrics)
        # jax.debug.print("Initial metrics: {}", init_metrics)

        # # Define optimizer
        # if isinstance(optimizer_conf['kwargs']['learning_rate'], dict):
        #     lr_schedule = getattr(optax, optimizer_conf['kwargs']['learning_rate']['name'])(
        #         **optimizer_conf['kwargs']['learning_rate']['kwargs'])
        # else:
        #     lr_schedule = optax.constant_schedule(optimizer_conf['kwargs']['learning_rate'])
        # optimizer_conf['kwargs']['learning_rate'] = lr_schedule
        optimizer = getattr(optax, optimizer_conf['name'])(**optimizer_conf['kwargs'])
        # Add gradient clipping to the optimizer using optax.chain
        opt_chain_list = []
        if "grad_clip_norm" in optimizer_conf:
            opt_chain_list.append(optax.clip_by_global_norm(optimizer_conf["grad_clip_norm"]))
        opt_chain_list.append(optimizer)
        # if "freeze_non_curve_params" in optimizer_conf and optimizer_conf["freeze_non_curve_params"]:
        #     opt_chain_list.append(optax.masked(optax.set_to_zero(), jax.tree.map(lambda x:not x, mask)))
        optimizer = optax.chain(*opt_chain_list)

        only_train_params = s_model.only_trainable_params(params['params'])
        n_params = sum(p.size for p in jax.tree.flatten(only_train_params)[0])
        print(f"Number of trainable parameters: {n_params:_}")
        logger.summary.update({'n_params_trainable': n_params})
        opt_state = optimizer.init(only_train_params)  # only trainable parameters according mask

        # loss, params, opt_state, grad_ = s_model.train_step(
        #             random.PRNGKey(0), params, x[:32], y[:32], opt_state, optimizer)
        # print(f"Initial loss: {loss}, grad norm: {jnp.linalg.norm(masked_pytree_to_matrix(grad_, mask, k))}")

        best_loss = np.inf
        empty_metric = {'val_loss':jnp.array(jnp.inf, dtype=jnp.float32), 'val_acc':jnp.array(-jnp.inf, dtype=jnp.float32), 'val_ece': jnp.array(jnp.inf, dtype=jnp.float32)}, None
        if val:
            print("save best parameters")
            # best_params = jax.device_get(params)  # best params on cpu to save memory
            best_params = s_model.only_trainable_params(params['params'])  # best params on cpu to save memory
            # best_params = jax.device_put(params, jax.devices('cpu')[0])  # best params on cpu to save memory
            valid_metrics_fn = setup_metrics(s_model, BATCH_SIZE_EVAL, val_input_ids, val_attention_mask, val_labels, n_samples=10 if k > 0 else 1, key_prefix='val_')
            def get_best_params(params):
                # return jax.device_get(params)
                # return params
                return s_model.only_trainable_params(params['params'])
        else:
            best_params = None
            valid_metrics_fn = lambda rng, params: empty_metric
            def get_best_params(params):
                return None
            

        @partial(jit, donate_argnums=(0,))
        def train_batch(carry, batch):
            x_batch, atten_batch, y_, batch_idx = batch
            # # skip leading padding tokens in batch for faster training
            # x_batch = x_batch[:, (max_seqlen:=jnp.argmax(atten_batch, axis=1).min()):]
            # atten_batch = atten_batch[:, max_seqlen:]

            rng_key, params, opt_state, best_loss, best_params, update_idx = carry
            rng_key, subkey = random.split(rng_key)
            t = random.uniform(subkey, (1,), minval=0., maxval=1.)
            loss, params, opt_state, (grad_, logs) = s_model.train_step(
                subkey, t, params, (x_batch, atten_batch), y_, opt_state, optimizer)
            # because of optax.chain [-1] gets state of optimizer and next [-1] gets state of learning rate scheduler
            current_lr = lr_schedule(opt_state[-1][-1].count)
            # validate
            rng_key, valid_key = random.split(rng_key)

            metrics = jax.lax.cond((update_idx % EVAL_EVERY_N_BATCH) == 0, lambda x: valid_metrics_fn(
                valid_key, params)[0], lambda x: empty_metric[0], None)
            best_params = jax.lax.cond(
                metrics['val_loss'] < best_loss, lambda x: get_best_params(params), lambda x: best_params, None)
            best_loss = jnp.minimum(best_loss, metrics['val_loss'])

            return (rng_key, params, opt_state, best_loss, best_params, update_idx+1), {'loss': loss, 'current_lr': current_lr, **metrics}

        @scan_tqdm(num_epochs)
        def train_epoch(carry, epoch):
            rng_key = carry[0]
            rng_key, subkey = random.split(rng_key)
            shuffel = jax.random.permutation(subkey, jnp.arange(train_labels.shape[0]))
            x_ = es('(ib)...->ib...', train_input_ids[shuffel], b=batch_size)
            x_atten_mask = es('(ib)...->ib...', train_attention_mask[shuffel], b=batch_size)
            y_ = es('(ib)...->ib...', train_labels[shuffel], b=batch_size)
            
            # Limit number of batches for smoke test
            if smoke_test:
                smoke_batch_limit = 3
                n_batches = min(smoke_batch_limit, x_.shape[0])
                x_ = x_[:n_batches]
                x_atten_mask = x_atten_mask[:n_batches]
                y_ = y_[:n_batches]
                jax.debug.print(f"Smoke test: Limited to {n_batches} batches per epoch")
                
            # train on batches
            carry, metrics = jax.lax.scan(
                train_batch, carry, (x_, x_atten_mask, y_, jnp.arange(x_.shape[0])))
            # jax.clear_caches()
            # jax.debug.print(f"Epoch {epoch+1}/{num_epochs}, Loss: {metrics['loss'][-1].item()}{metrics['loss'].shape}, Val Loss: {metrics['val_loss'][-1]}, Val Acc: {metrics['val_acc'][-1]}")
            jax.debug.print("Epoch {}/{}: Loss: {}, Val Loss: {}, Val Acc: {}", epoch+1,num_epochs, metrics['loss'][-1], metrics['val_loss'][::EVAL_EVERY_N_BATCH].mean(), metrics['val_acc'][::EVAL_EVERY_N_BATCH].mean())
            return carry, metrics

        # print memory stats
        memory_stats = jax.devices()[0].memory_stats()
        for stats_name, v in memory_stats.items():
            print(f"{stats_name}: {v / (1024 * 1024):.2f} MB" if "bytes" in stats_name else f"{stats_name}: {v}")
        

        
        if num_steps > 0:
            print(f"Start minibatch training for {num_epochs} epochs")
            carry = (rng_key, params, opt_state, best_loss, best_params, 0) # 0 is used for update counts
            carry, metrics = jax.lax.scan(
                train_epoch, carry, jnp.arange(num_epochs))


            # clean up inf values because of lazy validation
            for key, value in metrics.items():
                if key.startswith('val_'):
                    metrics[key] = jnp.array(value).flatten()[::EVAL_EVERY_N_BATCH].repeat(
                        EVAL_EVERY_N_BATCH, axis=0)[:num_steps]
                else:
                    metrics[key] = jnp.array(value).flatten()
            
            # log all metrics
            _ = [logger.log(dict(zip(metrics.keys(), m))) for m in zip(*metrics.values())]

            # print memory stats
            memory_stats = jax.devices()[0].memory_stats()
            for stats_name, v in memory_stats.items():
                print(f"{stats_name}: {v / (1024 * 1024):.2f} MB" if "bytes" in stats_name else f"{stats_name}: {v}")

            # log metrics as artifact
            epochs = np.repeat(np.arange(num_epochs), num_steps//batch_size)
            jnp.savez(f'tmp_files/{logger.id}_metrics.npz', **metrics, epochs=epochs)
            art = wandb.Artifact(name="metrics", type="npz")
            art.add_file(f'tmp_files/{logger.id}_metrics.npz')
            logger.log_artifact(art)

            if val:
                params = carry[1]  # last params on device
                best_curve_params = carry[4]  # best params (only trainable params)
                # update last params with best curve params (only trainable params were stored)
                params['params'] = jax.tree.map(lambda x, y, m: x if m else y, best_curve_params, params['params'], s_model.train_mask)
            else:
                params = carry[1]  # last params on device
            rng_key = carry[0]

        return rng_key, s_model, params

    train_params = []
    params = None
    num_ensembles = config['curve_params']['k']
    for k in range(num_ensembles):
        print(f"Training DE member {k}")
        config['curve_params']['k'] = 0 

        if params is not None:
            del params  # free memory
            # jax.clear_caches()
        rng_key, s_model, params = train_curve_inner(
            rng_key, **config['curve_params'], net_kwargs=config['net_kwargs'], 
            lora_params=config.get('lora_params', None), dataset_path=config['dataset_path'],
            smoke_test=config.get('smoke_test', False))
        train_params.append(s_model.only_trainable_params(params['params']))
        rng_key, _ = random.split(rng_key)

    # # DE evaluation on test set
    # print("Evaluating DE on test set...")
    # def merge_param(t_params):
    #     return jax.tree.map(lambda tp, p, m: tp if m else p, t_params, params, s_model.train_mask)
    
    
    # def single_pred(carry, t_params):
    #     param = merge_param(t_params)
    #     out, _ = s_model._predict(param, {}, t=0. (test_input_ids, test_attention_mask), train=False)
    #     return carry, out.astype(jnp.float32)

    # _, logits = jax.lax.scan(single_pred, None, jnp.array(train_params))
    # probs = jax.nn.softmax(logits, axis=-1)
    # avg_probs = jnp.mean(probs, axis=0)
    # preds = jnp.argmax(avg_probs, axis=-1)
    # accuracy = jnp.mean(preds == test_labels)

    if SAVE_PARAMS:
        artifact = wandb.Artifact(name="params", type="pytree")
        print("Save artifacts")
        jnp.save(f'tmp_files/{logger.id}_trainable_params.npy', train_params)
        print("add files to artifact")
        # artifact.add_file(f'tmp_files/{logger.id}_best_params.npy')
        artifact.add_file(f'tmp_files/{logger.id}_trainable_params.npy')
        print("Log artifact")
        logger.log_artifact(artifact)

    # def setup_de_metrics(model, static_params, batch_size, x, attention_mask, target, average=True, key_prefix='', num_bins_ece=15):
    #     ii = len(x) // batch_size
        
    #     def acc_fn(rng_key, ensemble_params):
    #         n_samples = len(ensemble_params)
    #         x_ = x[:(matching_limit:=(x.shape[0] // batch_size) * batch_size)]  # drop non matching batch size
    #         x_last = x[matching_limit:]
    #         x_ = es('(ib)...->ib...', x_, b=batch_size)

    #         att_m_ = attention_mask[:matching_limit]  # drop non matching batch size
    #         att_last = attention_mask[matching_limit:]
    #         att_m_ = es('(ib)...->ib...', att_m_, b=batch_size)

    #         y_all = target[:matching_limit]  # drop non matching batch size
    #         y_last = target[matching_limit:]
    #         y_ = es('(ib)...->ib...', y_all, b=batch_size)

    #         #@scan_tqdm(ii, desc='Evaluating', leave=False)
    #         def pred(key, data):
    #             x_batch, atten_batch, y_batch = data

    #             # @jax.jit
    #             def eval(e_params):
    #                 param = jax.tree.map(lambda tp, p, m: tp if m else p, e_params, static_params, model.train_mask)
    #                 loss, _, out = model.nll(param, {}, 0, (x_batch, atten_batch), y_batch, train=False)  # params, batch_stats, t, x ; out shape (batch_size, output_dim)
    #                 acc = jnp.sum(jnp.argmax(out, axis=-1) == y_batch)
    #                 return loss*batch_size, acc, out  # sum over batch

    #             oo = jax.tree.map(eval, ensemble_params, is_leaf=lambda x: isinstance(x, dict))
    #             loss, acc, out = map(list, zip(*oo))
    #             # loss, acc, out = jax.lax.map(eval, ensemble_params)
    #             loss = jnp.array(loss)  # shape (n_samples,)
    #             acc = jnp.array(acc)  # shape (n_samples,)
    #             out = jnp.stack(out, axis=0)  # shape (n_samples, batch_size, output_dim)

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

    test_metrics_fn = setup_metrics(s_model, BATCH_SIZE_EVAL, test_input_ids, test_attention_mask, test_labels, n_samples=1, key_prefix='test_')
    logits_test = []
    for i, t_param in enumerate(train_params):
        param = jax.tree.map(lambda tp, p, m: tp if m else p, t_param, params['params'], s_model.train_mask)
        metric, logits_test_m = test_metrics_fn(rng_key, {'params': param})
        print(f"DE member {i} test loss: {metric['test_loss']}, test acc: {metric['test_acc']}, test ece: {metric['test_ece']}")
        print(f"Shape member logit {logits_test_m.shape}")
        logits_test.append(logits_test_m.astype(jnp.float32))
         # log individual member metrics
        logger.summary.update({f'test_loss_member_{i}': metric['test_loss'],
                               f'test_acc_member_{i}': metric['test_acc'],
                               f'test_ece_member_{i}': metric['test_ece'],})

    logits_test = jnp.stack(logits_test, axis=0)  # shape (n_ensembles, n_data, output_dim)
    print(f"Shape stacked logits {logits_test.shape}")

    # test_metrics_fn = setup_de_metrics(s_model, params['params'], BATCH_SIZE_EVAL, test_input_ids, test_attention_mask, test_labels, average=True, key_prefix='test_')
    # curve_metric, logits_test = test_metrics_fn(rng_key, train_params)
    # compute accuracy
    log_probs = jax.nn.log_softmax(logits_test, axis=-1)
    post_predict_logits = jax.nn.logsumexp(log_probs, axis=0) - jnp.log(log_probs.shape[0])

    # probs = jax.nn.softmax(logits_test, axis=-1)
    # avg_probs = jnp.mean(probs, axis=0)
    # post_predict_logits = jnp.log(avg_probs)

    preds = jnp.argmax(post_predict_logits, axis=-1)
    accuracy = jnp.mean(preds == test_labels)
    # compute ece
    ece = ece_fn(post_predict_logits, test_labels, num_bins=15)
    # compute nll for averaged probs
    ll = dist.Categorical(logits=post_predict_logits).log_prob(test_labels).mean()
    logger.summary.update({'test_de_acc': accuracy,
                           'test_de_ece': ece,
                           'test_de_ll': ll,})
    


    logger.finish()


if __name__ == '__main__':
        # Add requirement for wandb core
    wandb.require("core")
    os.makedirs('tmp_files', exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Train a single model or a curve model.")
    # parser.add_argument("--train", choices=["single", "curve"], required=True, help="Specify the training mode: 'single' for single model training, 'curve' for curve model training.")
    parser.add_argument("--use-sweep", action="store_true", help="Use sweep configuration.")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size for training. Default is 16.")
    parser.add_argument("--curve-k", type=int, default=2, help="curve k parameter. Default is 2.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test with reduced epochs (2) and batch iterations (3).")

    args = parser.parse_args()

    
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
            # 'dataset_path': "ddold/qwen_PtoC/winogrande_s_dataset:v0",
            # 'dataset_path': "ddold/qwen_PtoC/ARC-Easy_dataset:v0",
            # 'dataset_path': "ddold/qwen_PtoC/MMLU_chem_dataset:v0",
            'dataset_path': "ddold/qwen_PtoC/obqa_dataset:v0",
            'net_kwargs':{'model_path': "artifacts/qwen2.5_7B_bfloat16:v0",},
            # 'net_kwargs':{'model_path': "ddold/qwen_PtoC/qwen2.5_7B_bfloat16:v0",},
            # 'load_params_path': "ddold/qwen_PtoC/47xexal3", # path to pretrained trainable parameters
            'load_params_path': False, # path to pretrained trainable parameters
            # 'load_params_path': "ddold/qwen_PtoC/ywepcmp3", # path to pretrained trainable parameters
            'save_params': True,
            'eval_every_n_batch': 2,
            'smoke_test': args.smoke_test,
            'lora_params': {'use_lora': True, 'r': 8, 
                            'lora_dtype': 'float32',
                            'lora_alpha': 16.0,
                            # 'lora_mode': 'A(t)B(t)',
                            'lora_mode': 'A(t)B',
                            'filter_masks': 
                                [{'keys': ["self_attn", "q_proj", "kernel"], 'op': 'all', 'dims': [1, 0]},
                                    {'keys': ["self_attn", "v_proj", "kernel"], 'op': 'all', 'dims': [1, 0]},
                                    {'keys': ["lm_head", "kernel"], 'op': 'all', 'dims': [1, 0]},
                                ]},  # use lora for query projection
            'curve_params': {'k': args.curve_k,
                            'num_epochs': -1,
                            'num_steps': 10 if args.smoke_test else 5_000,
                            # 'num_steps': 5000,
                            # 'num_steps': -1,
                            'batch_size': args.batch_size,
                            'natural_parameterization': False,
                            'gravity': 0.0,
                            'energy_weakening': 4., 
                            'regularizer': False,
                            'weight_decay': 0.,
                            'jitter_multiplier': 0.,  # np.sqrt(2./fan_in, dtype=np.float32) fan_in = 1280 * 16 or 1280 = 1e-2
                            'temperature': [1.0, 2.0, 4.0,],
                            'val_percentage' : 0.1,
                            # 'filter_masks':[{'keys': ("self_attn", "q_proj", "kernel"), 'op': 'all'},
                            #                 {'keys': ("self_attn", "v_proj", "kernel"), 'op': 'all'}],
                            'filter_masks':[{'keys': ["self_attn", "q_proj", "kernel"], 'op': 'all'},
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
                                                                                            'kwargs': {'transition_steps': 12_000, 'peak_value': 1e-4, 'pct_start': 0.06, 'pct_final': 1., 'div_factor': 300.0, 'final_div_factor': 300.0}},
                                                                            'weight_decay': 0.01,
                                                                            },
                                                                    'freeze_other_params': True,
                                                },
                        }}
        logger = wandb.init(project="qwen_PtoC", name="DE test", entity="ddold", config=config)
        train(logger, config)

