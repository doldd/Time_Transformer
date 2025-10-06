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

# Project imports
from src.qwen_jax import load_params_and_config, QwenForCausalLM
from src.subspace_curve import (
    masked_pytree_to_matrix, bezier_curve, filter_mask, bezier_coeff_fn,
    LoRARepulsiveCategorySubspace, LoRACategorySubspace, CategorySubspace, RepulsiveCategorySubspace
)

# Set environment variables
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=10"

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
        out = self.model.apply(
            variables,
            *args,
            rngs=rngs,
            method=method,
            mutable=mutable,
            capture_intermediates=capture_intermediates,
            **kwargs,
        )
        last_token_logits = out[:, -1, self.target_token_ids]  # (batch, num_classes)

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


def setup_metrics(model, batch_size, x, attention_mask, target, n_samples, use_linspace=False, average=True, key_prefix=''):
    if use_linspace:
        get_t = lambda key: jnp.linspace(0, 1., n_samples)
    else:
        get_t = lambda key: random.uniform(key, (n_samples,), minval=0., maxval=1.)

    def acc_fn(rng_key, params):
        x_ = x[:(matching_limit:=(x.shape[0] // batch_size) * batch_size)]  # drop non matching batch size
        x_last = x[matching_limit:]
        x_ = es('(ib)...->ib...', x_, b=batch_size)

        att_m_ = attention_mask[:matching_limit]  # drop non matching batch size
        att_last = attention_mask[matching_limit:]
        att_m_ = es('(ib)...->ib...', att_m_, b=batch_size)

        y_ = target[:matching_limit]  # drop non matching batch size
        y_last = target[matching_limit:]
        y_ = es('(ib)...->ib...', y_, b=batch_size)

        def pred(key, data):
            x_batch, atten_batch, y_batch = data
            # # turn off leading padding tokens in batch for faster evaluation
            # x_batch = x_batch[:, (max_seqlen:=jnp.argmax(atten_batch, axis=1).min()):]
            # atten_batch = atten_batch[:, max_seqlen:]

            def eval(t):
                loss, _, out = model.nll(params['params'], {}, t, (x_batch, atten_batch), y_batch, train=False)  # params, batch_stats, t, x ; out shape (batch_size, output_dim)
                acc = jnp.sum(jnp.argmax(out, axis=-1) == y_batch)
                return loss*batch_size, acc  # sum over batch
            
            key, subkey = random.split(key)

            # draw samples to approximate curve performance
            t = get_t(subkey)
            loss, acc = jax.lax.map(eval, t)

            if average:
                # mean over samples
                return key, (jnp.sum(loss), jnp.sum(acc))
            else:
                return key, (loss, acc)  # no mean over samples

        # shape (n_data, draws, output_dim)
        rng_key, (loss, acc) = jax.lax.scan(pred, rng_key, (x_, att_m_, y_))
        # compute loss of last irregular batch
        if len(x_last) > 0:
            rng_key, (loss_last, acc_last) = pred(rng_key, (x_last, att_last, y_last))
            loss = jnp.concatenate([loss, loss_last[jnp.newaxis, ...]], axis=0)
            acc = jnp.concatenate([acc, acc_last[jnp.newaxis, ...]], axis=0)
        if average:
            loss = loss.sum() / (n_samples * x.shape[0])
            acc = acc.sum() / (n_samples * x.shape[0])
        else:
            loss = jnp.sum(loss, axis=0) / x.shape[0]
            acc = jnp.sum(acc, axis=0) / x.shape[0]
        return {key_prefix + 'loss': loss, key_prefix + 'acc': acc}  # mean over epochs
    return acc_fn


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
    BATCH_SIZE_EVAL = config['curve_params']['batch_size']
    SAVE_PARAMS = config.get('save_params', False)

    # Create artifact for storing parameters
    if SAVE_PARAMS:
        artifact = wandb.Artifact(name="params", type="pytree")

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
                         jitter: float = 0.0,
                         smoke_test: bool = False,
                         num_steps: int = -1,
                         ):
        # Note: jitter parameter is unused but kept for interface compatibility
        rng_key, init_key = random.split(rng_key)

        # Load dataset
        input_ids, attention_mask, labels, target_ids, unique_target_ids = load_text_classification_data(dataset_path, run=logger)
        test_input_ids, test_attention_mask, test_labels, _, _ = load_text_classification_data(dataset_path, run=logger, test=True)
        print(f"Dataset loaded: {len(input_ids)} samples, {len(unique_target_ids)} classes")
        print(f"Target token IDs: {unique_target_ids}")

        # compute num_epochs if num_steps is given
        assert (num_epochs > 0) ^ (num_steps > 0), "Only one of num_epochs or num_steps should be > 0"
        if num_steps > 0:
            num_epochs = int(np.ceil(1.*num_steps*batch_size / len(input_ids)))
            print(f"Setting num_epochs to {num_epochs} to match {num_steps} steps")
        
        # Load Qwen model
        if net_kwargs['model_path'].startswith('ddold/'):
            model_artifact = logger.use_artifact(net_kwargs['model_path'], type='model')
            net_kwargs['model_path'] = model_artifact.download()
        model = QwenTextClassificationWrapper(**net_kwargs, target_token_ids=unique_target_ids)
        params = unfreeze(model.init(None, None))
        dtype = jax.tree.leaves(params)[0].dtype
        # Split data (simple train/val split) # we dont use validate data
        n_train = int(np.ceil(1.*len(input_ids) / batch_size) * batch_size)
        train_input_ids = input_ids[:n_train]
        train_attention_mask = attention_mask[:n_train]
        train_labels = labels[:n_train]
        
        val_input_ids = input_ids[n_train:]
        val_attention_mask = attention_mask[n_train:]
        val_labels = labels[n_train:]
        val = True
        if len(val_input_ids) == 0:
            print("No validation data are used")
            val = False
        
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
                s_model = LoRARepulsiveCategorySubspace(lora_mask=lora_mask, r=lora_params['r'],
                                                        model=model, k=k, weight_decay=weight_decay,
                                                        perm_spec=perm_spec,
                                                        natural_parameterization=natural_parameterization,
                                                        mutuable_param_name=False,
                                                        gravity=gravity, energy_weakening=energy_weakening, regularizer=regularizer)
            else:
                print("Using QuantileSubspace without repulsive force")
                s_model = LoRACategorySubspace(lora_mask=lora_mask, r=lora_params['r'],
                                            model=model, k=k, weight_decay=weight_decay, 
                                            perm_spec=perm_spec,
                                            natural_parameterization=natural_parameterization,
                                            mutuable_param_name=False)
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
        params = s_model.init_params_from_point(init_key, params, curve_mask, jitter=jitter)
        

        init_metrics_fn = setup_metrics(s_model, BATCH_SIZE_EVAL, val_input_ids, val_attention_mask, val_labels, 1, use_linspace=True, average=True, key_prefix='init_')
        init_metrics = init_metrics_fn(init_key, params) # returns a dict
        logger.summary.update(init_metrics)
        jax.debug.print("Initial metrics: {}", init_metrics)

        # set train mask according to config
        # Only update mask for LoRA-capable subspaces
        if lora_params and lora_params['use_lora']:
            train_mask = s_model.update_mask_for_lora(train_mask, params['params'], b_off=False, w0_off=True) # w0 must be off for LoRA
        s_model.set_train_mask(train_mask)

        # compute number of parameters
        n_params = sum(p.size for p in jax.tree.flatten(params)[0])
        logger.summary.update({'n_params': n_params})
        print(f"Number of parameters: {n_params:_}")

        if SAVE_PARAMS:
            jnp.save(f'tmp_files/{logger.id}_init_params.npy', params)
            artifact.add_file(f'tmp_files/{logger.id}_init_params.npy')

        # Define optimizer
        if isinstance(optimizer_conf['kwargs']['learning_rate'], dict):
            lr_schedule = getattr(optax, optimizer_conf['kwargs']['learning_rate']['name'])(
                **optimizer_conf['kwargs']['learning_rate']['kwargs'])
        else:
            lr_schedule = optax.constant_schedule(optimizer_conf['kwargs']['learning_rate'])
        optimizer_conf['kwargs']['learning_rate'] = lr_schedule
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

        cp_w = masked_pytree_to_matrix(params['params'], s_model.curve_mask, k)
        print(f"Control points shape: {cp_w.shape}")
        del cp_w
        # loss, params, opt_state, grad_ = s_model.train_step(
        #             random.PRNGKey(0), params, x[:32], y[:32], opt_state, optimizer)
        # print(f"Initial loss: {loss}, grad norm: {jnp.linalg.norm(masked_pytree_to_matrix(grad_, mask, k))}")

        best_loss = np.inf
        empty_metric = {'val_loss':jnp.array(jnp.inf, dtype=dtype), 'val_acc':jnp.array(-jnp.inf, dtype=jnp.float32)}
        if val:
            print("Put best parameters to cpu device")
            # best_params = jax.device_get(params)  # best params on cpu to save memory
            best_params = params  # best params on cpu to save memory
            # best_params = jax.device_put(params, jax.devices('cpu')[0])  # best params on cpu to save memory
            print("best parameters to cpu device")
            valid_metrics_fn = setup_metrics(s_model, BATCH_SIZE_EVAL, val_input_ids, val_attention_mask, val_labels, n_samples=10 if k > 0 else 1, key_prefix='val_')
            def get_best_params(params):
                # return jax.device_get(params)
                return params
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

        @jit
        def bezier_length(cp):
            t = jnp.linspace(0, 1, 1000)
            d_norm_bezier = get_d_bezier(t)
            _, d_t = d_norm_bezier(cp)
            length = jax.scipy.integrate.trapezoid(d_t, t)
            return length
        
        @jit
        def lower_bound(cp):
            # get the length of the line between the first and last control
            return jnp.linalg.norm(cp[-1]-cp[0])
        
        @jit
        def upper_bound(cp):
            # get the length of the line between each control point
            return jnp.linalg.norm(jnp.diff(cp, axis=0), axis=1).sum()

        @jit 
        def mass_center(cp):
            return jnp.linalg.norm(cp.mean(axis=0))
        
        @jit
        def gyration(cp):
            mc = cp.mean(0)
            return jnp.linalg.norm(cp - mc, axis=1).mean()
        
        @jit
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
            @jit
            def rel_center(cp):
                return jnp.linalg.norm(cp.mean(axis=0) - mean_center_t0)
            return rel_center
        
        # init metric
        # cp_w = masked_pytree_to_matrix(params['params'], s_model.curve_mask, k)
        # rel_center_fn = init_rel_center(cp_w)

        if k > 0:
            @partial(jit, donate_argnums=(0,))
            def train_batch(carry, batch):
                x_batch, atten_batch, y_, batch_idx = batch
                # # skip leading padding tokens in batch for faster training
                # x_batch = x_batch[:, (max_seqlen:=jnp.argmax(atten_batch, axis=1).min()):]
                # atten_batch = atten_batch[:, max_seqlen:]

                rng_key, params, opt_state, best_loss, best_params = carry
                rng_key, subkey = random.split(rng_key)
                loss, params, opt_state, grad_ = s_model.train_step(
                    subkey, params, (x_batch, atten_batch), y_, opt_state, optimizer)
                # because of optax.chain [-1] gets state of optimizer and next [-1] gets state of learning rate scheduler
                current_lr = lr_schedule(opt_state[-1][-1].count)
                # validate
                rng_key, valid_key = random.split(rng_key)


                # metrics = {'val_loss':jnp.array(jnp.inf, dtype=dtype), 'val_acc':jnp.array(-jnp.inf, dtype=jnp.float32)}
                # mse, val_loss = valid_metrics_fn(valid_key, params)
                metrics = jax.lax.cond((batch_idx % EVAL_EVERY_N_BATCH) == 0, lambda x: valid_metrics_fn(
                    valid_key, params), lambda x: empty_metric, None)
                best_params = jax.lax.cond(
                    metrics['val_loss'] < best_loss, lambda x: get_best_params(params), lambda x: best_params, None)
                best_loss = jnp.minimum(best_loss, metrics['val_loss'])

                # cp_w = masked_pytree_to_matrix(params['params'], s_model.curve_mask, k)
                # length = bezier_length(cp_w)
                # l_bound = lower_bound(cp_w)
                # u_bound = upper_bound(cp_w)
                # gyra = gyration(cp_w)
                # center = mass_center(cp_w)
                # mean_curv = mean_curvature2(cp_w)
                # rel_center = rel_center_fn(cp_w)
                # cos_sim = s_model.summed_cosine_similarity(params['params'])
                # cos_sim /= ((k+1)*k/2)
                # grad_norm = 0.
                # grad_norm = jnp.linalg.norm(masked_pytree_to_matrix(grad_, s_model.curve_mask, k), axis=1).mean()
                # return (rng_key, params, opt_state, best_loss, best_params), {'loss': loss, **metrics, 'grad_norm': grad_norm}
                return (rng_key, params, opt_state, best_loss, best_params), {'loss': loss, 'current_lr': current_lr, **metrics}
        else:
            @partial(jit, donate_argnums=(0,))
            def train_batch(carry, batch):
                x_batch, atten_batch, y_, batch_idx = batch
                # jax.debug.print("Batch idx: {}", batch_idx)
                rng_key, params, opt_state, best_loss, best_params = carry
                rng_key, subkey = random.split(rng_key)
                loss, params, opt_state, grad_ = s_model.train_step(
                    subkey, params, (x_batch, atten_batch), y_, opt_state, optimizer)
                # because of optax.chain [-1] gets state of optimizer and next [-1] gets state of learning rate scheduler
                current_lr = lr_schedule(opt_state[-1][-1].count)  
                
                # jax.debug.print("Training loss: {}", loss)
                # validate
                rng_key, valid_key = random.split(rng_key)

                metrics = jax.lax.cond((batch_idx % EVAL_EVERY_N_BATCH) == 0, lambda x: valid_metrics_fn(
                    valid_key, params), lambda x: empty_metric, None)
                best_params = jax.lax.cond(
                    metrics['val_loss'] < best_loss, lambda x: get_best_params(params), lambda x: best_params, None)
                best_loss = jnp.minimum(best_loss, metrics['val_loss'])

                return (rng_key, params, opt_state, best_loss, best_params), {'loss': loss, 'current_lr': current_lr, **metrics}

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
            # jax.debug.print(f"Epoch {epoch+1}/{num_epochs}, Loss: {metrics['loss'][-1].item()}{metrics['loss'].shape}, Val Loss: {metrics['val_loss'][-1]}, Val Acc: {metrics['val_acc'][-1]}")
            jax.debug.print("Epoch {}/{}: Loss: {}, Val Loss: {}, Val Acc: {}", epoch+1,num_epochs, metrics['loss'][-1], metrics['val_loss'][-1], metrics['val_acc'][-1])
            return carry, metrics

        # print memory stats
        memory_stats = jax.devices()[0].memory_stats()
        for stats_name, v in memory_stats.items():
            print(f"{stats_name}: {v / (1024 * 1024):.2f} MB" if "bytes" in stats_name else f"{stats_name}: {v}")
        

        # # # # jax.profiler.save_device_memory_profile(f'memory_profile/memory_profile_0.json')
        # for e in tqdm(range(num_epochs)):
        #     print(f"Epoch {e+1}/{num_epochs}")
        #     rng_key, subkey = random.split(rng_key)
        #     shuffel = jax.random.permutation(subkey, jnp.arange(train_labels.shape[0]))
        #     x_ = es('(ib)...->ib...', train_input_ids[shuffel], b=batch_size)
        #     atten_mask = es('(ib)...->ib...', train_attention_mask[shuffel], b=batch_size)
        #     y_ = es('(ib)...->ib...', train_labels[shuffel], b=batch_size)

        #     # train on batches
        #     batch_loop = tqdm(enumerate(zip(x_,atten_mask, y_)), leave=False)
        #     for b, (xb_, amb_, yb_) in batch_loop:
        #         # skip leading padding tokens in batch for faster training
        #         xb_ = xb_[:, (max_seqlen:=jnp.argmax(amb_, axis=1).min()):]
        #         amb_ = amb_[:, max_seqlen:]

        #         rng_key, subkey = random.split(rng_key)
        #         # jax.profiler.save_device_memory_profile(f'mem_profile/memory_profile_{e}_{b}.json')
        #         loss, params, opt_state, grad_ = s_model.train_step(
        #             subkey, params, (xb_, amb_), yb_, opt_state, optimizer)
        #         # grad_norm = jnp.linalg.norm(masked_pytree_to_matrix(grad_, curve_mask, k), axis=1).mean()

        #         batch_loop.set_postfix({"Loss": loss})
        #         logs = {
        #             'epoch': e+1, 'batch': b+1, 'stepp': e*x_.shape[0]+b+1,
        #             'train_loss': loss,
        #             # 'grad_mean': grad_norm
        #         }
        #         if jnp.isnan(loss):
        #             print(f"Loss is NaN at epoch {e+1}, batch {b+1}. Stopping training.")
        #             break
                
        #         # validate
        #         rng_key, valid_key = random.split(rng_key)
        #         # if b % EVAL_EVERY_N_BATCH == 0:
        #         #     logs =  {**logs, **valid_metrics_fn(valid_key, params)}
        #         #     if logs['val_loss'] < best_loss:
        #         #         best_params = jax.device_put(params, jax.devices('cpu')[0])
        #         #         best_loss = logs['val_loss']
        #         logger.log(logs)
                
        #     batch_loop.close()
        #     if jnp.isnan(loss):
        #         print(f"ERROR: Stopping training at epoch {e+1} due to NaN loss.")
        #         break

        # jax.profiler.save_device_memory_profile(f'memory_profile/memory_profile_1.json')

        carry = (rng_key, params, opt_state, best_loss, best_params)
        carry, metrics = jax.lax.scan(
            train_epoch, carry, jnp.arange(num_epochs))

        # log losses for wandb
        n_batches = np.array(metrics['loss']).shape[-1]
        epochs = np.repeat(np.arange(num_epochs), n_batches)

        # clean up inf values because of lazy validation
        for key, value in metrics.items():
            if key.startswith('val_'):
                metrics[key] = jnp.array(value)[:, ::EVAL_EVERY_N_BATCH].repeat(
                    EVAL_EVERY_N_BATCH, axis=1)[:,:value.shape[1]].flatten()
            else:
                metrics[key] = jnp.array(value).flatten()

        
        # log all metrics
        _ = [logger.log(dict(zip(metrics.keys(), m))) for m in zip(*metrics.values())]

        # print memory stats
        memory_stats = jax.devices()[0].memory_stats()
        for stats_name, v in memory_stats.items():
            print(f"{stats_name}: {v / (1024 * 1024):.2f} MB" if "bytes" in stats_name else f"{stats_name}: {v}")

        # log metrics as artifact
        jnp.savez(f'tmp_files/{logger.id}_metrics.npz', **metrics, epochs=epochs)
        art = wandb.Artifact(name="metrics", type="npz")
        art.add_file(f'tmp_files/{logger.id}_metrics.npz')
        logger.log_artifact(art)

        if val:
            params = jax.device_put(carry[4])  # best params on cpu
        else:
            params = carry[1]  # last params on device
        rng_key = carry[0]

        if k > 0:
            # Bezier curve length        
            cp = masked_pytree_to_matrix(params['params'], s_model.curve_mask, k)
            length = bezier_length(cp)
            logger.log({'Curve length': length})

        # plot likelihood and acc along curve
        fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
        t_space = jnp.linspace(0, 1, 25)
        if smoke_test:
            t_space = jnp.linspace(0, 1, 3)
        test_metrics_fn_final = setup_metrics(s_model, max(BATCH_SIZE_EVAL, 1), test_input_ids, test_attention_mask, test_labels, n_samples=len(t_space), use_linspace=True, average=False, key_prefix='final_')
        valid_metrics_fn_final = setup_metrics(s_model, max(BATCH_SIZE_EVAL, 1), val_input_ids, val_attention_mask, val_labels, n_samples=len(t_space), use_linspace=True, average=False, key_prefix='final_')
        # ood_metrics_fn_final = setup_metrics(s_model, BATCH_SIZE_EVAL, x_ood, y_ood, n_samples=len(t_space), use_linspace=True, average=False, rescale_fn=rescale_fn)
    
        # # # plot best params
        # # # valid
        # curve_metric = test_metrics_fn_final(rng_key, best_params)
        # logger.log({k+'_best_mean':v.mean() for k,v in curve_metric.items()})
        # logger.log({k+'_best_max':v.max() for k,v in curve_metric.items()})
        # logger.log({k+'_best_min':v.min() for k,v in curve_metric.items()})
        # ax[0].plot(t_space, -curve_metric['final_loss'], label="valid_ll", c=plt.get_cmap('tab10')(1))
        # ax2 = ax[0].twinx()
        # ax2.plot(t_space, curve_metric['final_acc'], label="valid acc",
        #          c=plt.get_cmap('tab10')(1), linestyle="--")
        # # nll, mse = valid_metrics_fn_final(rng_key, best_params)
        # # logger.log({'mse_val_curve': mse.mean(), 'nll_val_curve': nll.mean(), 'mse_val_curve_min': mse.min(), 'nll_val_curve_min': nll.min()})
        # # ax[0].plot(t_space, -nll, label="valid ll", c=plt.get_cmap('tab10')(1))
        # # ax2 = ax[0].twinx()
        # # ax2.plot(t_space, mse, label="valid mse",
        # #          c=plt.get_cmap('tab10')(1), linestyle="--")
        # # # train
        # # nll, mse = train_metrics_fn_final(rng_key, best_params)
        # # ax[0].plot(t_space, -nll, label="train", c=plt.get_cmap('tab10')(0))
        # # ax2.plot(t_space, mse, label="train mse",
        # #          c=plt.get_cmap('tab10')(0), linestyle="--")
        # # # test
        # # nll, mse = test_metrics_fn_final(rng_key, best_params)
        # # logger.log({'mse_test_curve': mse.mean(), 'nll_test_curve': nll.mean(), 'mse_test_curve_min': mse.min(), 'nll_test_curve_min': nll.min()})
        # # ax[0].plot(t_space, -nll, label="test", c=plt.get_cmap('tab10')(2))
        # # ax2.plot(t_space, mse, label="test mse",
        # #          c=plt.get_cmap('tab10')(2), linestyle="--")
        # # ax[0].set_xlabel("t")
        # # ax[0].set_ylabel("mean log likelihood")
        # # ax[0].set_title("Best params")
        
        # plot last params
        # valid
        curve_metric = test_metrics_fn_final(rng_key, params)
        logger.log({k+'_mean':v.mean() for k,v in curve_metric.items()})
        logger.log({k+'_max':v.max() for k,v in curve_metric.items()})
        logger.log({k+'_min':v.min() for k,v in curve_metric.items()})
        ax[1].plot(t_space, -curve_metric['final_loss'], label="test_ll", c=plt.get_cmap('tab10')(1))
        ax2 = ax[1].twinx()
        ax2.plot(t_space, curve_metric['final_acc'], label="test acc",
                 c=plt.get_cmap('tab10')(1), linestyle="--")
        # # train
        # nll, mse = train_metrics_fn_final(rng_key, params)
        # ax[1].plot(t_space, -nll, label="train", c=plt.get_cmap('tab10')(0))
        # ax2.plot(t_space, mse, label="train mse",
        #          c=plt.get_cmap('tab10')(0), linestyle="--")

        
        # # plot predictive performance
        # fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
        # t_space = jnp.linspace(0, 1, 100)
        # # with best params
        # out, _ = jax.vmap(s_model, in_axes=(None,None,0,None,None))(params['params'], {}, t_space, x_val[:2], False)
        # out = out[0][:,0]  # shape (n_samples, n_data)
        # ax[0].plot(x, y, 'o', label='train')
        # # plt.fill_between(x.squeeze(), out.mean(axis=0) - out.std(axis=0), out.mean(axis=0) + out.std(axis=0), alpha=0.5)
        # # plot lines using viridis color map
        # colors = plt.cm.viridis(t_space)
        # for o, c in zip(out, colors):
        #     ax[0].plot(o, color=c, alpha=0.3)
        # ax[0].plot(out.mean(axis=0), label='mean',
        #         c='red', linewidth=2, alpha=0.8)
        # ax[0].set_ylabel("y")
        # ax[0].set_xlabel("x")
        # ax[0].set_title("Best params")
        # # compute functional diversity
        # functional_diff = {"Sample Std best":out.std(axis=0).mean()}  # std over samples
        
        # # with last params
        # out, _ = jax.vmap(s_model, in_axes=(None,None,0,None,None))(params['params'], {}, t_space, x_val[:2], False)
        # out = out[0][:, 0]  # shape (n_samples, n_data)
        # ax[1].plot(x, y, 'o', label='train')
        # # plt.fill_between(x.squeeze(), out.mean(axis=0) - out.std(axis=0), out.mean(axis=0) + out.std(axis=0), alpha=0.5)
        # # plot lines using viridis color map
        # colors = plt.cm.viridis(t_space)
        # for o, c in zip(out, colors):
        #     ax[1].plot(o, color=c, alpha=0.3)
        # ax[1].plot(out.mean(axis=0), label='mean',
        #         c='red', linewidth=2, alpha=0.8)
        # ax[1].set_ylabel("y")
        # ax[1].set_xlabel("x")
        # ax[1].set_title("Last params")
        # functional_diff["Sample Std last"] = out.std(axis=0).mean()  # std over samples
        # logger.log(functional_diff)

        # plt.tight_layout()
        # finalize plot: labels, legend and layout
        ax[0].set_xlabel("t")
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("mean log likelihood")
        ax2.set_ylabel("Accuracy")
        # ax[0].set_title("Best params (validation)")
        ax[1].set_title("Final testset")
        # combine legends from primary and twin axes
        lines, labels = ax[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax[1].legend(lines + lines2, labels + labels2, loc='best', fontsize='small')
        logger.log(
            {'Curve Predictive': wandb.Image(fig)})
        # plt.tight_layout()
        # logger.log(
        #     {'Curve Predictive for x_val[0]': wandb.Image(fig)})


        return rng_key, best_params, model, params

    rng_key, best_params, model, params = train_curve_inner(
        rng_key, **config['curve_params'], net_kwargs=config['net_kwargs'], 
        lora_params=config.get('lora_params', None), dataset_path=config['dataset_path'],
        smoke_test=config.get('smoke_test', False))

    if SAVE_PARAMS:
        print("Save artifacts")
        jnp.save(f'tmp_files/{logger.id}_best_params.npy', best_params)
        jnp.save(f'tmp_files/{logger.id}_last_params.npy', params)
        print("add files to artifact")
        artifact.add_file(f'tmp_files/{logger.id}_best_params.npy')
        artifact.add_file(f'tmp_files/{logger.id}_last_params.npy')
        print("Log artifact")
        logger.log_artifact(artifact)

    logger.finish()


if __name__ == '__main__':
        # Add requirement for wandb core
    wandb.require("core")
    os.makedirs('tmp_files', exist_ok=True)
    
    parser = argparse.ArgumentParser(description="Train a single model or a curve model.")
    parser.add_argument("--train", choices=["single", "curve"], required=True, help="Specify the training mode: 'single' for single model training, 'curve' for curve model training.")
    parser.add_argument("--use-sweep", action="store_true", help="Use sweep configuration.")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size for training. Default is 16.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test with reduced epochs (2) and batch iterations (3).")

    args = parser.parse_args()

    if args.train == "single":
        print("Single model training")
        
        # Apply smoke test settings if enabled
        num_epochs = 2 if args.smoke_test else 5  # Keep original value of 2
        
        config = {
            'rng_seed': 5,
            'dataset_path': "ddold/qwen_PtoC/winogrande_m_dataset:v0",
            'save_params': False,
            'net_kwargs':{'model_path': "artifacts/qwen2.5_7B_bfloat16:v0",},
            'eval_every_n_batch': 20,
            'smoke_test': args.smoke_test,
            'lora_params': {'use_lora': True, 'r': 8, 
                            'filter_masks': 
                                [{'keys': ("self_attn", "q_proj", "kernel"), 'op': 'all', 'dims': [0, 1]}]},  # use lora for query projection
            'curve_params': {'k': 0,
                            # 'num_epochs': num_epochs,
                            'num_epochs': -1,
                            'num_steps': 10,
                            'batch_size': args.batch_size,
                            'natural_parameterization': False,
                            'gravity': 0.0,
                            'energy_weakening': 1., 
                            'regularizer': False,
                            'weight_decay': 0.,
                            'jitter': 1e-2,
                            'filter_masks': [{'keys': ("self_attn", "q_proj", "kernel"), 'op': 'all'},],  # use lora for query projection
                            #  'optimizer_conf': {'name': "adamw", 'kwargs': {'learning_rate': 0.05, 'weight_decay': 0.0001}},
                            #  'optimizer_conf': {'name': "adamw", 'learning_rate': 0.05, 'momentum': 0.9, 'nesterov': True},
                                'optimizer_conf': {'name': "sgd", 
                                                   'kwargs': {'learning_rate': 1e-4, 'momentum': 0.9, 'nesterov': True},
                                                #    'grad_clip_norm': 1.0,
                                                   'freeze_other_params': True},
                            # 'optimizer_conf': {'name': "sgd", 'kwargs': {'learning_rate': {'name': "linear_onecycle_schedule", 
                                                                                            # 'kwargs': {'transition_steps': 490_000, 'peak_value': 0.1, 'pct_start': 0.2, 'pct_final': 0.6, 'div_factor': 5.0, 'final_div_factor': 10.0}},
                                                                            # 'momentum': 0.9, 
                                                                            # 'nesterov': True},},
                        }}
        logger = wandb.init(project="qwen_PtoC", name="qwen_single", entity="ddold", config=config)
        train(logger, config)
    elif args.train == "curve":
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
                'rng_seed': 5,
                # 'dataset_path': "datasets/text_classification/boolq/train_data.npz",
                'dataset_path': "ddold/qwen_PtoC/winogrande_m_dataset:v0",
                # 'net_kwargs':{'model_path': "ddold/qwen_PtoC/qwen2.5_7B_bfloat16:v0",},
                'net_kwargs':{'model_path': "artifacts/qwen2.5_7B_bfloat16:v0",},
                'eval_every_n_batch': 20,
                'smoke_test': args.smoke_test,
                'lora_params': {'use_lora': True, 'r': 8, 
                                'filter_masks': 
                                    [{'keys': ("self_attn", "q_proj", "kernel"), 'op': 'all', 'dims': [0, 1]}]},  # use lora for query projection
                'curve_params': {'k': 1,
                                'num_epochs': num_epochs,
                                'batch_size': args.batch_size,
                                'natural_parameterization': False,
                                'gravity': 0.0,
                                'energy_weakening': 1., 
                                'regularizer': False,
                                'weight_decay': 0.,
                                'jitter': 1e-2,  # np.sqrt(2./fan_in, dtype=np.float32) fan_in = 1280 * 16 or 1280 = 1e-2
                                # 'filter_masks':[{'keys': ("self_attn", "q_proj", "kernel"), 'op': 'all'},
                                #                 {'keys': ("self_attn", "v_proj", "kernel"), 'op': 'all'}],
                                'filter_masks':[{'keys': ("self_attn", "q_proj", "kernel"), 'op': 'all'},],
                                #  'optimizer_conf': {'name': "adamw", 'kwargs': {'learning_rate': 0.05, 'weight_decay': 0.0001}},
                                #  'optimizer_conf': {'name': "adamw", 'learning_rate': 0.05, 'momentum': 0.9, 'nesterov': True},
                                'optimizer_conf': {'name': "sgd", 
                                                    'kwargs': {'learning_rate': 0.0001, 'momentum': 0.9, 'nesterov': True},
                                                    'grad_clip_norm': 100.0,
                                                    'freeze_other_params': True},
                                # 'optimizer_conf': {'name': "sgd", 'kwargs': {'learning_rate': {'name': "linear_onecycle_schedule", 
                                                                                                # 'kwargs': {'transition_steps': 490_000, 'peak_value': 0.1, 'pct_start': 0.2, 'pct_final': 0.6, 'div_factor': 5.0, 'final_div_factor': 10.0}},
                                                                                # 'momentum': 0.9, 
                                                                                # 'nesterov': True},},
                            }}
            logger = wandb.init(project="qwen_PtoC", name="test_qwen_curve", entity="ddold", config=config)
            train(logger, config)

