"""
Qwen2.5 Fine-tuning with LoRA Subspace for Text Classification
Following the PtoC (Point-to-Curve) pattern from other models
"""

import os
import logging
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
from src.utils import calibration_error as ece_fn, post_pred_performance
from copy import deepcopy
from numpyro import distributions as dist

# Project imports
from src.qwen_jax import load_params_and_config, QwenForCausalLM
from src.subspace_curve import (
    masked_pytree_to_matrix, bezier_curve, filter_mask, bezier_coeff_fn, get_subspace_model
)
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from dataclasses import dataclass
import time
from contextlib import contextmanager

# Set environment variables
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=10"
# import sys
# sys.setrecursionlimit(200)

@dataclass
class DataSplits:
    """All dataset arrays for train / val / test splits.

    ``has_train`` and ``has_val`` are derived automatically so they never
    go out of sync with the actual data arrays.
    """
    train_input_ids: jnp.ndarray | None = None
    train_attention_mask: jnp.ndarray | None = None
    train_labels: jnp.ndarray | None = None
    val_input_ids: jnp.ndarray | None = None
    val_attention_mask: jnp.ndarray | None = None
    val_labels: jnp.ndarray | None = None
    test_input_ids: jnp.ndarray | None = None
    test_attention_mask: jnp.ndarray | None = None
    test_labels: jnp.ndarray | None = None

    @property
    def has_train(self) -> bool:
        return self.train_input_ids is not None

    @property
    def has_val(self) -> bool:
        return self.val_input_ids is not None

    def get(self, name: str) -> tuple[jnp.ndarray | None, jnp.ndarray | None, jnp.ndarray | None]:
        """Return ``(input_ids, attention_mask, labels)`` for *name* or ``None``."""
        mapping = {
            'train': (self.train_input_ids, self.train_attention_mask, self.train_labels),
            'val':   (self.val_input_ids,   self.val_attention_mask,   self.val_labels),
            'test':  (self.test_input_ids,  self.test_attention_mask,  self.test_labels),
        }
        return mapping.get(name, (None, None, None))


def _empty_val_metric():
    """Sentinel validation metrics used when no validation data is available."""
    return ({
        'val_loss': jnp.array(jnp.inf, dtype=jnp.float32),
        'val_acc': jnp.array(-jnp.inf, dtype=jnp.float32),
        'val_ece': jnp.array(jnp.inf, dtype=jnp.float32),
        'val_bma_ll': jnp.array(-jnp.inf, dtype=jnp.float32),
        'val_bma_acc': jnp.array(-jnp.inf, dtype=jnp.float32),
        'val_bma_ece': jnp.array(jnp.inf, dtype=jnp.float32),
    }, None)


@dataclass
class TrainHyperparams:
    """Pure training hyperparameters for the pipeline.

    All stored fields map 1-to-1 to keys inside ``config['train_hyper']``.
    Construct via ``TrainHyperparams.from_config(config, data)``; ``data``
    is only needed to resolve epoch / step counts from data size.

    Object-construction configs (optimizer_conf, model_params, curve_params,
    lora_params, net_kwargs) are intentionally excluded — those live in their
    own top-level config groups and are passed where needed.
    """
    cp_fix: list
    batch_size: int
    eval_every_n_batch: int
    num_epochs: int
    num_steps: int
    save_params: bool
    smoke_test: bool
    val_percentage: float
    dataset_sampling: dict
    curve_sampling_mode: str
    temperature: list

    # --- computed properties (not stored; derived from above fields) ------

    @property
    def batch_size_eval(self) -> int:
        return self.batch_size * 2

    @property
    def ds_sampling(self) -> str:
        return list(self.dataset_sampling.keys())[0]

    @property
    def eval_common(self) -> dict:
        """Shared kwargs for ``_run_eval_phase`` (excludes per-call args)."""
        return dict(
            batch_size_eval=self.batch_size_eval,
            curve_sampling_mode=self.curve_sampling_mode,
            temperature=self.temperature,
            smoke_test=self.smoke_test,
        )

    @classmethod
    def from_config(cls, config: dict, data: 'DataSplits') -> 'TrainHyperparams':
        """Construct from a wandb / inline config dict plus loaded data.

        Reads only ``config['train_hyper']``; no other config groups are
        touched.  ``data`` is needed solely to resolve epoch / step counts
        from the training-set size.
        """
        th = config['train_hyper']
        batch_size = th['batch_size']
        dataset_sampling = th.get('dataset_sampling', {})

        cp_fix = th['cp_fix']
        print(f"Control point fixed list: {cp_fix}")

        # Compute num_epochs / num_steps from data size
        num_epochs_cfg = th.get('num_epochs', -1)
        num_steps_cfg = th.get('num_steps', -1)
        assert not ((num_epochs_cfg > 0) and (num_steps_cfg > 0)), \
            "Only one of num_epochs or num_steps should be > 0"
        has_train_data = (num_epochs_cfg > 0 or num_steps_cfg > 0)
        if has_train_data:
            n_train = len(data.train_input_ids)
            num_epochs = num_epochs_cfg
            num_steps = num_steps_cfg
            if num_steps > 0:
                num_epochs = int(np.ceil(1. * num_steps * batch_size / n_train))
                print(f"Setting num_epochs to {num_epochs} to match {num_steps} steps")
            num_steps = int(np.ceil(1. * num_epochs * n_train / batch_size))
            print(f"Setting num_steps to {num_steps} to match {num_epochs} epochs")
        else:
            num_epochs = num_steps = 0

        return cls(
            cp_fix=cp_fix,
            batch_size=batch_size,
            eval_every_n_batch=th.get('eval_every_n_batch', 20),
            num_epochs=num_epochs,
            num_steps=num_steps,
            save_params=th.get('save_params', False),
            smoke_test=th.get('smoke_test', False),
            val_percentage=th.get('val_percentage', 0.0),
            dataset_sampling=dataset_sampling,
            curve_sampling_mode=th.get('curve_sampling_mode', 'combined'),
            temperature=th.get('temperature', [1.0]),
        )


@dataclass
class TrainingEnv:
    """Reusable training environment objects created by ``_setup_training_env``.

    For k=0 pretraining ``base_params`` holds a pristine parameter copy so
    each control-point iteration can re-initialise from it.  For k>0 it is
    ``None``.
    """
    s_model: object
    params: dict
    base_params: dict | None
    curve_mask: object
    train_mask: object
    optimizer: object
    opt_state: object
    t_sample_fn: object
    rho_scheduler: object
    valid_metrics_fn: object
    empty_metric: tuple
    get_best_params: object
    cosine_sim: jnp.ndarray | None
    n_trainable: int
    lr_schedule: object
    lora_rho: float
    lora_rho_s: float


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

def setup_metrics(model, batch_size, x, attention_mask, target, n_samples, use_linspace=False, average=True, key_prefix='', num_bins_ece=15, t_sample_fn=None, t_max=None):
    if t_max is None:
        t_max = getattr(model, 't_max', 1.0)
    if t_sample_fn is None:
        if use_linspace:
            get_t = lambda key: jnp.linspace(0, t_max, n_samples)
        else:
            get_t = lambda key: random.uniform(key, (n_samples,), minval=0., maxval=t_max)
    else:
        if use_linspace:
            raise ValueError("Cannot use linspace sampling with custom t_sample_fn")
        # get_t = partial(t_sample_fn, n_samples=n_samples)
        get_t = lambda k: jax.vmap(lambda key: t_sample_fn(key))(random.split(k, n_samples))
    ii = len(x) // batch_size
    
    idx = jnp.arange(len(x))
    idx_ = idx[:(matching_limit:=(x.shape[0] // batch_size) * batch_size)]  # drop non matching batch size
    idx_ = es('(ib)...->ib...', idx_, b=batch_size)
    idx_last = idx[matching_limit:]

    # Use shared post_pred_performance from src.utils (returns prefixed keys when key_prefix provided)

    def acc_fn(rng_key, params):
        # @scan_tqdm(ii, desc='Evaluating', leave=False)
        def pred(rng_key, data_idx):
            x_batch = x.at[data_idx].get()
            att_m_batch = attention_mask.at[data_idx].get()
            y_batch = target.at[data_idx].get()

            rng_key, key_ = random.split(rng_key)
            t_tree = get_t(key_)  # shape (n_samples,)
            # keys = random.split(key_, n_samples)
            
            def single_pred(key, t):
                # t is either scalar or tree of shape ()
                key, subkey = random.split(key)
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
        bma_metrics = post_pred_performance(out, target, key_prefix=key_prefix + "bma_", num_bins=num_bins_ece)

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
    use_weights = True if isinstance(weights, jnp.ndarray) or isinstance(weights, np.ndarray) else False
    probs = jax.nn.softmax(logits, axis=-1)  # (n_samples, n_data, n_classes)
    if use_weights:
        avg_probs = jnp.sum(probs * weights[:, None, None], axis=0)  # (n_data, n_classes)
    else:
        avg_probs = jnp.mean(probs, axis=0)  # (n_data, n_classes)
    entropy_pred = -jnp.sum(avg_probs * jnp.log(avg_probs + 1e-12), axis=-1)  # (n_data,)
    if use_weights:
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
    use_weights = True if isinstance(weights, jnp.ndarray) or isinstance(weights, np.ndarray) else False
    probs = jax.nn.softmax(logits, axis=-1)  # (n_samples, n_data, n_classes)
    if use_weights:
        avg_probs = jnp.sum(probs * weights[:, None, None], axis=0)  # (n_data, n_classes)
    else:
        avg_probs = jnp.mean(probs, axis=0)  # (n_data, n_classes)
    entropy_pred = -jnp.sum(avg_probs * jnp.log(avg_probs + 1e-12), axis=-1)  # (n_data,)
    return jnp.mean(entropy_pred)

def entropy(logits):
    probs = jax.nn.softmax(logits, axis=-1)  # (n_samples, n_data, n_classes)
    entropies = -jnp.sum(probs * jnp.log(probs + 1e-12), axis=-1)  # (n_samples, n_data)
    return entropies  # (n_samples, n_data)


# ---------------------------------------------------------------------------
# Curve geometry helpers (pure functions used by the training loop)
# ---------------------------------------------------------------------------

def _bezier_length(cp):
    """Compute arc length of a Bézier curve defined by control points *cp*."""
    k = cp.shape[0] - 1
    t = jnp.linspace(0, 1, 1000)
    bezier_coeff_inv = bezier_coeff_fn(k)
    coeff = jax.vmap(bezier_coeff_inv)(t)
    cp_diff = cp[1:] - cp[:-1]

    def _d_length_t(cp_diff, c):
        d_t = jnp.linalg.norm(jnp.einsum('j,j...->...', c, cp_diff) * k)
        return cp_diff, d_t

    _, d_t = jax.lax.scan(_d_length_t, cp_diff, coeff)
    return jax.scipy.integrate.trapezoid(d_t, t)


def _bezier_lower_bound(cp):
    """Chord length between first and last control point."""
    return jnp.linalg.norm(cp[-1] - cp[0])


def _bezier_upper_bound(cp):
    """Sum of segment lengths of the control polygon."""
    return jnp.linalg.norm(jnp.diff(cp, axis=0), axis=1).sum()


def _bezier_mass_center(cp):
    """Norm of the mass centre of control points."""
    return jnp.linalg.norm(cp.mean(axis=0))


def _bezier_gyration(cp):
    """Mean distance of control points from their centre of mass."""
    mc = cp.mean(0)
    return jnp.linalg.norm(cp - mc, axis=1).mean()


def _bezier_mean_curvature(cp):
    """Integrated mean curvature of the Bézier curve defined by *cp*."""
    tt = jnp.linspace(0, 1, 10)
    _, d_bezier = bezier_curve(cp.shape[0], cp)

    def _single(t):
        grad_ = d_bezier(t)
        second_dev = jax.jacrev(d_bezier)(t).squeeze()
        norm_r_prime_sq = jnp.dot(grad_, grad_)
        norm_r_double_prime_sq = jnp.dot(second_dev, second_dev)
        dot_r_prime_double_prime = jnp.dot(grad_, second_dev)
        numerator = jnp.sqrt(
            norm_r_prime_sq * norm_r_double_prime_sq - dot_r_prime_double_prime**2)
        denominator = norm_r_prime_sq ** (3 / 2)
        return numerator / denominator

    curvature = jax.vmap(_single)(tt)
    return jax.scipy.integrate.trapezoid(curvature, tt)


def _bezier_rel_center(cp, initial_center):
    """Distance of the current mass centre from *initial_center*."""
    return jnp.linalg.norm(cp.mean(axis=0) - initial_center)


def _plot_curve_along_t(t_space, loss, acc, ece, title='Curve Predictive',
                        loss_label='test_ll', acc_label='test acc',
                        ece_label='test ece'):
    """Create a triple-axis plot of loss / accuracy / ECE along curve parameter *t*.

    Returns the matplotlib *Figure*.
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), sharey=True)
    ax.plot(t_space, -loss, label=loss_label, c=plt.get_cmap('tab10')(0))
    ax2 = ax.twinx()
    ax2.plot(t_space, acc, label=acc_label,
             c=plt.get_cmap('tab10')(1), linestyle='--')
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 40))
    ax3.set_frame_on(True)
    ax3.patch.set_visible(False)
    ax3.plot(t_space, ece, label=ece_label,
             c=plt.get_cmap('tab10')(2), linestyle=':')
    ax.set_xlabel('t')
    ax.set_ylabel('mean log likelihood')
    ax2.set_ylabel('Accuracy')
    ax3.set_ylabel('ECE')
    ax.set_title(title)
    lines, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    lines3, lab3 = ax3.get_legend_handles_labels()
    ax.legend(lines + lines2 + lines3, lab1 + lab2 + lab3,
              loc='best', fontsize='small')
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Evaluation pipeline – compute logits per dataset, then apply metrics
# ---------------------------------------------------------------------------

# --- Individual metric functions ---------------------------------------------------
# Each receives keyword arguments from the orchestrator and may read/write *ctx*.
# Defined here as plain functions; grouped into preset lists at the end of this
# section (PRETRAIN_EVAL_FNS, CURVE_EVAL_FNS).

def _eval_summary(*, base_metrics, logger, logger_prefix='', **_kw):
    """Log the mean of every base metric to ``logger.summary``."""
    logger.summary.update({
        f'{logger_prefix}{mk}_mean': v.mean()
        for mk, v in base_metrics.items()
    })


def _eval_post_pred(*, logits, labels, logger, key_prefix, ctx, weight_key=None, **_kw):
    """BMA post-predictive performance (uniform or weighted via *weight_key*)."""
    weights = ctx.get(weight_key, False) if weight_key else False
    pp = post_pred_performance(logits, labels, weights=weights)
    logger.summary.update({f'{key_prefix}{mk}': v for mk, v in pp.items()})


def _eval_plot_curve(*, base_metrics, key_prefix, t_space, logger,
                     plot_title=None, **_kw):
    """Plot loss / accuracy / ECE along the curve parameter *t*."""
    if t_space is None:
        logging.warning("plot_curve requires linspace sampling — skipping.")
        return
    fig = _plot_curve_along_t(
        t_space,
        base_metrics[key_prefix + 'loss'],
        base_metrics[key_prefix + 'acc'],
        base_metrics[key_prefix + 'ece'],
        title=plot_title or f'{key_prefix}set')
    logger.log({f'Curve Predictive {plot_title or key_prefix}': wandb.Image(fig)})


def _eval_entropy_along_curve(*, logits, logger, key_prefix, **_kw):
    """Compute std / mean of per-sample entropy along the curve."""
    entrop = entropy(logits.astype(jnp.float32))
    logger.summary.update({
        f'{key_prefix}std_entropy_along_curve': entrop.mean(-1).std(),
        f'{key_prefix}mean_entropy_along_curve': entrop.mean(),
    })


def _eval_mutual_info(*, logits, logger, key_prefix, ctx, weight_key=None, **_kw):
    """Mutual information across curve samples (optionally weighted)."""
    weights = ctx.get(weight_key, False) if weight_key else False
    mi = mutual_information(logits.astype(jnp.float32), weights=weights)
    logger.summary.update({f'{key_prefix}mutual_information': mi})


def _eval_mean_entropy(*, logits, logger, key_prefix, ctx, weight_key=None, **_kw):
    """Predictive entropy of the BMA mixture (optionally weighted)."""
    weights = ctx.get(weight_key, False) if weight_key else False
    me = mean_entropy(logits.astype(jnp.float32), weights=weights)
    logger.summary.update({f'{key_prefix}mean_entropy': me})


def _eval_compute_posterior_weights(*, base_metrics, key_prefix, logger,
                                    temperature, t_space, ctx, **_kw):
    """Derive posterior weights from the per-sample loss and cache in *ctx*.

    Also saves a wandb ``bma_weights`` artifact.
    """
    print("Temperature:", temperature)

    def _compute_weights(log_like, t):
        log_post = log_like / t
        log_post -= jax.scipy.special.logsumexp(log_post)
        return jnp.exp(log_post)

    log_like = -base_metrics[key_prefix + 'loss']
    weights = jax.vmap(_compute_weights, in_axes=(None, 0))(
        log_like, jnp.array(temperature))
    ctx['posterior_weights'] = weights

    # Save artifact
    art_w = wandb.Artifact(name="bma_weights", type="npz")
    jnp.savez(f'tmp_files/{logger.id}_weights.npz',
              weights=weights, t_space=t_space,
              log_like=log_like, temperature=temperature)
    art_w.add_file(f'tmp_files/{logger.id}_weights.npz')
    logger.log_artifact(art_w)
    print("Saved posterior weights artifact")


def _eval_load_posterior_weights(*, logger, temperature, ctx, **_kw):
    """Load posterior weights from a wandb artifact stored in ``ctx['_artifact_weights']``."""
    artifact = ctx.get('_artifact_weights')
    if artifact is None:
        raise RuntimeError("'load_posterior_weights' requires artifact_weights in run_evaluation")

    def _compute_weights(log_like, t):
        log_post = log_like / t
        log_post -= jax.scipy.special.logsumexp(log_post)
        return jnp.exp(log_post)

    for f in artifact.files():
        if 'weights' in f.name:
            wf = np.load(artifact.get_entry(f.name).download(), allow_pickle=True)
            weights = jax.vmap(_compute_weights, in_axes=(None, 0))(
                wf['log_like'], jnp.array(temperature))
            ctx['posterior_weights'] = weights
            print("Loaded posterior weights from artifact")
            return
    raise RuntimeError("No weights file found in artifact")


def _eval_post_pred_per_temp(*, logits, labels, logger, key_prefix,
                              temperature, ctx, **_kw):
    """BMA post-predictive for each temperature using posterior weights."""
    weights_all = ctx.get('posterior_weights')
    if weights_all is None:
        raise RuntimeError("'post_pred_per_temp' requires 'compute_posterior_weights' first")
    for weight, temp in zip(weights_all, temperature):
        pp = post_pred_performance(logits, labels, weight)
        logger.summary.update({
            f'{key_prefix}ppd_t{temp}_{mk}': v for mk, v in pp.items()
        })


def _eval_mutual_info_per_temp(*, logits, logger, key_prefix,
                                temperature, ctx, **_kw):
    """Mutual information for each temperature using posterior weights."""
    weights_all = ctx.get('posterior_weights')
    if weights_all is None:
        raise RuntimeError("'mutual_info_per_temp' requires 'compute_posterior_weights' first")
    for weight, temp in zip(weights_all, temperature):
        mi = mutual_information(logits.astype(jnp.float32), weights=weight)
        logger.summary.update({f'{key_prefix}mutual_information_ppd_t{temp}': mi})


def _eval_mean_entropy_per_temp(*, logits, logger, key_prefix,
                                 temperature, ctx, **_kw):
    """Mean predictive entropy for each temperature using posterior weights."""
    weights_all = ctx.get('posterior_weights')
    if weights_all is None:
        raise RuntimeError("'mean_entropy_per_temp' requires 'compute_posterior_weights' first")
    for weight, temp in zip(weights_all, temperature):
        me = mean_entropy(logits.astype(jnp.float32), weights=weight)
        logger.summary.update({f'{key_prefix}mean_entropy_ppd_t{temp}': me})


def _eval_best_temperature(*, logits, labels, logger, key_prefix,
                            temperature, ctx, **_kw):
    """Find the best temperature on this dataset and store the best weight in *ctx*.

    Iterates over temperatures, computing BMA post-predictive performance
    for each, and caches the weight with the highest log-likelihood as
    ``ctx['best_weight']`` and ``ctx['best_temperature']``.
    """
    weights_all = ctx.get('posterior_weights')
    if weights_all is None:
        raise RuntimeError("'best_temperature' requires 'compute_posterior_weights' first")
    best = {'ll': -jnp.inf, 'temp': None, 'weight': None}
    for weight, temp in zip(weights_all, temperature):
        pp = post_pred_performance(logits, labels, weight)
        if pp['ll'] > best['ll']:
            best = {'ll': pp['ll'], 'temp': temp, 'weight': weight}
        logger.summary.update({
            f'{key_prefix}ppd_t{temp}_{mk}': v for mk, v in pp.items()
        })
    ctx['best_weight'] = best['weight']
    ctx['best_temperature'] = best['temp']
    logger.summary.update({
        f'{key_prefix}best_log_like_temp': best['temp'],
        f'{key_prefix}best_log_like': best['ll'],
    })


# ---------------------------------------------------------------------------
# Preset eval-function lists – pass these into _run_eval_phase / run_evaluation
# to declare which metrics to compute without touching run_evaluation internals.
# ---------------------------------------------------------------------------

PRETRAIN_EVAL_FNS = [
    _eval_summary,
    _eval_post_pred,
]

CURVE_EVAL_FNS = [
    _eval_post_pred,
    _eval_mutual_info,
    _eval_mean_entropy,
    _eval_entropy_along_curve,
]


def run_evaluation(*, rng_key, s_model, params, data: 'DataSplits', eval_specs,
                   batch_size_eval, t_sample_fn=None, temperature=(1.0,),
                   logger, logger_prefix='', smoke_test=False, k=0,
                   artifact_weights=None):
    """Run the modular evaluation pipeline.

    **Two-phase design**: for each entry in *eval_specs* the pipeline first
    computes logits on the specified dataset, then runs the requested metrics
    on those logits.  A shared *ctx* dict enables cross-dataset communication
    (e.g. posterior weights computed on train used by metrics on test).

    Args:
        s_model: Subspace model instance.
        params: Model parameters pytree.
        rng_key: JAX random key.
        data: ``DataSplits`` instance; splits are looked up by name via
            ``data.get(name)``.
        eval_specs: Ordered list of eval specification dicts.  Each dict may
            contain:

            - ``name`` (str): Split name passed to ``data.get()`` (``'train'``, ``'val'``, ``'test'``).
            - ``sampling`` (str): ``'linspace'`` | ``'t_sample'``
              (default ``'linspace'``).
            - ``n_samples`` (int): Number of *t* samples
              (default 20; clipped to 3 for smoke tests / k==0).
            - ``key_prefix`` (str): Prefix for metric keys
              (default ``'{name}_'``).
            - ``metrics`` (list): Callables or dicts ``{'fn': callable, **extra}``
              to run on this data split.
            - ``store_logits`` (bool): If True, cache logits in *ctx*.

        batch_size_eval: Batch size for evaluation forward passes.
        t_sample_fn: Callable for ``'t_sample'`` sampling mode.
        temperature: Sequence of temperature values for posterior weighting.
        logger: wandb run object.
        logger_prefix: Global prefix prepended to logged keys.
        smoke_test: Limit *n_samples* to 3.
        k: Bézier curve degree.
        artifact_weights: Optional wandb artifact with pre-computed weights.

    Returns:
        ctx: Context dict accumulated during evaluation.
    """
    t_max = getattr(s_model, 't_max', 1.0)
    ctx = {}
    if artifact_weights is not None:
        ctx['_artifact_weights'] = artifact_weights

    for spec in eval_specs:
        ds_name = spec['name']
        split = data.get(ds_name)
        if split is None:
            logging.warning(
                "Eval spec references dataset '%s' not available — skipping.",
                ds_name)
            continue

        input_ids, attention_mask, labels_ = split
        sampling = spec.get('sampling', 'linspace')
        n_samples = spec.get('n_samples', 20)
        if smoke_test or k == 0:
            n_samples = min(n_samples, 3)
        key_prefix = spec.get('key_prefix', f'{ds_name}_')

        t_space = (jnp.linspace(0, t_max, n_samples)
                   if sampling == 'linspace' else None)

        # Phase 1 – compute logits
        print(f"Evaluating '{ds_name}' ({sampling}, {n_samples} samples)")
        use_linspace = (sampling == 'linspace')
        metrics_fn = setup_metrics(
            s_model, max(batch_size_eval, 1), input_ids, attention_mask, labels_,
            n_samples=n_samples, use_linspace=use_linspace, average=False,
            key_prefix=key_prefix,
            t_sample_fn=None if use_linspace else t_sample_fn)
        base_metrics, logits = metrics_fn(rng_key, params)

        # Optionally cache logits in ctx
        if spec.get('store_logits'):
            ctx[f'{ds_name}_logits'] = logits
            ctx[f'{ds_name}_base_metrics'] = base_metrics

        # Phase 2 – run requested metrics
        metric_kwargs = dict(
            logits=logits, labels=labels_, base_metrics=base_metrics,
            key_prefix=key_prefix, t_space=t_space, logger=logger,
            logger_prefix=logger_prefix, temperature=temperature,
            ctx=ctx, k=k)

        for metric_spec in spec.get('metrics', []):
            if callable(metric_spec):
                metric_spec(**metric_kwargs)
            elif isinstance(metric_spec, dict):
                extra = dict(metric_spec)
                fn = extra.pop('fn')
                fn(**metric_kwargs, **extra)
            else:
                raise ValueError(
                    f"Invalid metric spec {metric_spec!r}: "
                    "must be a callable or a dict with a 'fn' key.")

    return ctx


def _build_masks(params_tree, k, filter_masks, optimizer_conf, lora_params):
    """Build the curve / train / LoRA masks from configuration.

    Returns ``(curve_mask, train_mask, lora_mask)``.
    """
    curve_mask = jax.tree.map(lambda x: False, params_tree)
    if k > 0:
        for f in filter_masks:
            curve_mask = filter_mask(curve_mask, *f['keys'], as_curve=True,
                                     BoolschenOperator=eval(f['op']))

    if 'freeze_other_params' in optimizer_conf and optimizer_conf['freeze_other_params']:
        train_mask = jax.tree.map(lambda x: False, params_tree)
        for f in filter_masks:
            train_mask = filter_mask(train_mask, *f['keys'], as_curve=True,
                                     BoolschenOperator=eval(f['op']))
        if lora_params and lora_params['use_lora']:
            for f in lora_params['filter_masks']:
                train_mask = filter_mask(train_mask, *f['keys'], as_curve=True,
                                         BoolschenOperator=eval(f['op']))
    else:
        train_mask = jax.tree.map(lambda x: True, params_tree)

    lora_mask = jax.tree.map(lambda x: False, params_tree)
    if lora_params and lora_params['use_lora']:
        for f in lora_params['filter_masks']:
            lora_mask = filter_mask(lora_mask, *f['keys'],
                                    as_curve=np.array(f['dims']),
                                    BoolschenOperator=eval(f['op']))
    return curve_mask, train_mask, lora_mask


# ---------------------------------------------------------------------------
# Setup helpers – model construction, optimizer, masking
# ---------------------------------------------------------------------------

def _element_wise_mask(element_freeze_mask):
    """Optax GradientTransformation that zeros updates for frozen parameters.

    ``element_freeze_mask`` leaves can be:
    - 0.0 (or False): Allow update
    - 1.0 (or True): Freeze parameter
    - Array of 0s/1s: Element-wise freeze/update
    """
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        new_updates = jax.tree.map(
            lambda f, u: u * (1 - f) if type(f) != bool else u,
            element_freeze_mask, updates)
        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def _create_subspace_model(*, model, params_tree, k, filter_masks, optimizer_conf,
                            lora_params, model_params, subspace_model):
    """Create masks and instantiate a subspace model (no parameter initialization).

    Also handles weight_decay migration from model_params to optimizer_conf for
    adamw optimizers.

    Returns:
        ``(s_model, curve_mask, train_mask, _model_params)``
    """
    curve_mask, train_mask, lora_mask = _build_masks(
        params_tree, k, filter_masks, optimizer_conf, lora_params)

    # Copy model_params to allow mutation (e.g. zeroing weight_decay passed to optimizer)
    _model_params = dict(model_params) if model_params else {}
    weight_decay = _model_params.get('weight_decay', 0.0)
    if (weight_decay > 0.0) and (optimizer_conf['name'] == 'adamw'):
        optimizer_conf['kwargs']['weight_decay'] = weight_decay
        _model_params['weight_decay'] = 0.0  # zero out to avoid double weight decay in model

    logging.info("Using subspace model: %s", subspace_model)
    _runtime_kwargs = dict(model=model, k=k, perm_spec=False, mutuable_param_name=False)
    if lora_params and lora_params.get('use_lora', False):
        _runtime_kwargs['lora_mask'] = lora_mask
    s_model_cls = get_subspace_model(subspace_model)
    s_model = s_model_cls(**_runtime_kwargs, **_model_params)

    return s_model, curve_mask, train_mask, _model_params


def _init_subspace_params(s_model, params, curve_mask, train_mask, init_key,
                           jitter_multiplier, lora_params, fixed_cps_train_params=False):
    """Initialize parameters on an existing subspace model.

    Calls ``init_params_from_point``, updates the train mask for LoRA,
    and maps any fixed control-point weights into the parameter vector.

    Returns:
        params (dict, updated in-place)
    """
    params = s_model.init_params_from_point(
        init_key, params, curve_mask,
        noise_fn=lambda p, key: p + jitter_multiplier * random.normal(key, p.shape, dtype=p.dtype))

    # set train mask according to config
    # Only update mask for LoRA-capable subspaces
    if lora_params and lora_params.get('use_lora', False):
        updated_train_mask = s_model.update_mask_for_lora(
            train_mask, params['params'], b_off=False, w0_off=True)  # w0 must be off for LoRA
    else:
        updated_train_mask = train_mask
    s_model.set_train_mask(updated_train_mask)

    # Map pretrained control points which are fixed into params vector
    if fixed_cps_train_params != False:
        k = s_model.k
        assert (len(fixed_cps_train_params) == (k + 1)), \
            "fixed_cps_train_params length must match k or be False"
        for i, p in enumerate(fixed_cps_train_params):
            if p != False:
                def map_fn(m, p, cp_f):
                    if m:
                        p = p.at[i].set(cp_f)
                    return p
                params['params'] = jax.tree.map(map_fn, s_model.curve_mask, params['params'], p)

    return params


def _build_subspace_model(*, model, params, k, filter_masks, optimizer_conf,
                           lora_params, model_params, subspace_model,
                           jitter_multiplier, init_key, fixed_cps_train_params):
    """Create masks, instantiate a subspace model, and initialise parameters.

    Convenience wrapper around ``_create_subspace_model`` and
    ``_init_subspace_params``.

    Returns:
        ``(s_model, params, _model_params, curve_mask, train_mask)``
    """
    s_model, curve_mask, train_mask, _model_params = _create_subspace_model(
        model=model, params_tree=params['params'], k=k, filter_masks=filter_masks,
        optimizer_conf=optimizer_conf, lora_params=lora_params,
        model_params=model_params, subspace_model=subspace_model)

    params = _init_subspace_params(
        s_model, params, curve_mask, train_mask, init_key,
        jitter_multiplier, lora_params, fixed_cps_train_params)

    return s_model, params, _model_params, curve_mask, train_mask


def _build_optimizer(*, optimizer_conf, s_model, params, fixed_cps_train_params):
    """Create the optimizer chain (gradient clipping, CP freezing, base optimizer)
    and initialise optimizer state.

    Returns:
        ``(optimizer, opt_state, n_trainable_params)``
    """
    if optimizer_conf['name'].startswith('sam'):
        raise NotImplementedError("SAM optimizer is not yet implemented for text classification")

    base_optimizer = getattr(optax, optimizer_conf['name'])(**optimizer_conf['kwargs'])

    opt_chain_list = []
    if "grad_clip_norm" in optimizer_conf:
        opt_chain_list.append(optax.clip_by_global_norm(optimizer_conf["grad_clip_norm"]))

    # Fix curve parameters during training if specified
    if fixed_cps_train_params != False:
        def get_dims(param):
            return (-1,) + (1,) * (param.ndim - 1)
        cps_fixed = [True if p != False else False for p in fixed_cps_train_params]
        fixed_cp_mask = jax.tree.map(
            lambda m, p: jnp.array(cps_fixed).reshape(get_dims(p)) if m else False,
            s_model.curve_mask, params['params'])
        opt_chain_list.append(_element_wise_mask(fixed_cp_mask))

    opt_chain_list.append(base_optimizer)
    optimizer = optax.chain(*opt_chain_list)

    only_train_params = s_model.only_trainable_params(params['params'])
    n_trainable_params = sum(p.size for p in jax.tree.flatten(only_train_params)[0])
    print(f"Number of trainable parameters: {n_trainable_params:_}")
    opt_state = optimizer.init(only_train_params)

    return optimizer, opt_state, n_trainable_params

def _make_train_batch_fn(*, s_model, optimizer, lr_schedule, t_sample_fn,
                          lora_rho, lora_rho_s, rho_scheduler, eval_every_n_batch,
                          valid_metrics_fn, empty_metric, get_best_params,
                          initial_cp_center, k,
                          train_input_ids, train_attention_mask, train_labels):
    """Return a ``train_batch(carry, batch_idx)`` closure for ``jax.lax.scan``."""

    def train_batch(carry, batch_idx):
        rng_key, params, opt_state, best_val_ll, best_params, update_idx = carry

        x_batch = jnp.take(train_input_ids, batch_idx, axis=0)
        atten_batch = jnp.take(train_attention_mask, batch_idx, axis=0)
        y_ = jnp.take(train_labels, batch_idx, axis=0)

        rng_key, subkey = random.split(rng_key)
        t = t_sample_fn(subkey)
        # t = random.uniform(subkey, (1,), minval=0., maxval=1.)
        if lora_rho > 0.0 or lora_rho_s > 0.0:
            current_rho = rho_scheduler(update_idx, lora_rho)
            current_rho_s = rho_scheduler(update_idx, lora_rho_s)
            params = s_model.set_lora_rho(current_rho, current_rho_s, params)
        loss, params, opt_state, (grad_, logs) = s_model.train_step(
            subkey, t, params, (x_batch, atten_batch), y_, opt_state, optimizer)
        # because of optax.chain [-1] gets state of optimizer and next [-1] gets state of learning rate scheduler
        current_lr = lr_schedule(update_idx)
        # validate
        rng_key, valid_key = random.split(rng_key)

        metrics = jax.lax.cond((update_idx % eval_every_n_batch) == 0, lambda x: valid_metrics_fn(
            valid_key, params)[0], lambda x: empty_metric[0], None)
        best_params = jax.lax.cond(
            metrics['val_bma_ll'] > best_val_ll, lambda x: get_best_params(params), lambda x: best_params, None)
        best_val_ll = jnp.maximum(best_val_ll, metrics['val_bma_ll'])

        if k > 0:
            cp = masked_pytree_to_matrix(params['params'], s_model.curve_mask, k)
            upper_length = _bezier_upper_bound(cp)
            lower_length = _bezier_lower_bound(cp)
            mean_center = _bezier_mass_center(cp)
            gyration_radius = _bezier_gyration(cp)
            rel_center = _bezier_rel_center(cp, initial_cp_center)

            return (rng_key, params, opt_state, best_val_ll, best_params, update_idx + 1), \
                {'loss': loss, 'current_lr': current_lr, 'upper_length': upper_length,
                 'lower_length': lower_length, 'mean_center': mean_center,
                 'gyration_radius': gyration_radius, 'rel_center': rel_center,
                 **metrics, **logs}
        else:
            return (rng_key, params, opt_state, best_val_ll, best_params, update_idx + 1), \
                {'loss': loss, 'current_lr': current_lr, **metrics, **logs}

    return train_batch


def _make_train_epoch_fn(*, s_model, optimizer, lr_schedule, t_sample_fn,
                           lora_rho, lora_rho_s, rho_scheduler, eval_every_n_batch,
                           valid_metrics_fn, empty_metric, get_best_params,
                           initial_cp_center, k,
                           train_input_ids, train_attention_mask, train_labels,
                           batch_size, smoke_test, num_epochs):
    """Return a ``train_epoch(carry, epoch)`` closure for ``jax.lax.scan``.

    Internally creates the per-batch function via ``_make_train_batch_fn``.
    """
    train_batch_fn = _make_train_batch_fn(
        s_model=s_model, optimizer=optimizer, lr_schedule=lr_schedule,
        t_sample_fn=t_sample_fn, lora_rho=lora_rho, lora_rho_s=lora_rho_s,
        rho_scheduler=rho_scheduler, eval_every_n_batch=eval_every_n_batch,
        valid_metrics_fn=valid_metrics_fn, empty_metric=empty_metric,
        get_best_params=get_best_params,
        initial_cp_center=initial_cp_center, k=k,
        train_input_ids=train_input_ids,
        train_attention_mask=train_attention_mask,
        train_labels=train_labels)
    n_train_samples = train_labels.shape[0]

    def _inf_mean(x):
        mask = jnp.isfinite(x)
        return jnp.where(mask, x, 0).sum() / mask.sum()

    @scan_tqdm(num_epochs)
    def train_epoch(carry, epoch):
        rng_key = carry[0]
        rng_key, subkey = random.split(rng_key)
        shuffel_idx = random.permutation(subkey, n_train_samples)
        shuffel_idx = es('(ib)->ib', shuffel_idx, b=batch_size)

        # Limit number of batches for smoke test
        if smoke_test:
            smoke_batch_limit = 3
            n_batches = min(smoke_batch_limit, shuffel_idx.shape[0])
            shuffel_idx = shuffel_idx[:n_batches]
            jax.debug.print(f"Smoke test: Limited to {n_batches} batches per epoch")

        # train on batches
        carry, metrics = jax.lax.scan(train_batch_fn, carry, shuffel_idx)
        val_loss_ = _inf_mean(metrics['val_loss'])
        val_acc_ = _inf_mean(metrics['val_acc'])
        jax.debug.print("Epoch {}/{}: Loss: {}, Val Loss: {}, Val Acc: {}",
                        epoch + 1, num_epochs, metrics['loss'][-1], val_loss_, val_acc_)
        return carry, metrics

    return train_epoch


def _make_expert_step_fn(*, s_model, optimizer, lr_schedule, t_sample_fn,
                           eval_every_n_batch, valid_metrics_fn, empty_metric,
                           get_best_params, num_steps, batch_size, window_size):
    """Return a ``_train_expert_step(carry, step)`` closure for ``jax.lax.scan``."""

    @scan_tqdm(num_steps)
    def _train_expert_step(carry, step):
        rng_key, params, opt_state, best_val_ll, best_params, (x_all, atten_all, y_all) = carry
        rng_key, subkey = random.split(rng_key)

        t = random.uniform(subkey, (1,), minval=0., maxval=1.)

        # get batch from data
        pos = jnp.round(t * x_all.shape[0]).squeeze().astype(int)
        rng_key, subkey = random.split(rng_key)
        idx = random.choice(subkey, window_size, shape=(batch_size,), replace=False) + pos - window_size // 2
        idx %= x_all.shape[0]  # wrap around
        x_batch = jnp.take(x_all, idx, axis=0, mode='clip')
        atten_batch = jnp.take(atten_all, idx, axis=0, mode='clip')
        y_ = jnp.take(y_all, idx, axis=0, mode='clip')

        t = t_sample_fn(subkey)
        loss, params, opt_state, (grad_, logs) = s_model.train_step(
            subkey, t, params, (x_batch, atten_batch), y_, opt_state, optimizer)
        # because of optax.chain [-1] gets state of optimizer and next [-1] gets state of learning rate scheduler
        current_lr = lr_schedule(step)

        # validate
        rng_key, valid_key = random.split(rng_key)
        metrics = jax.lax.cond((step % eval_every_n_batch) == 0, lambda x: valid_metrics_fn(
            valid_key, params)[0], lambda x: empty_metric[0], None)
        best_params = jax.lax.cond(
            metrics['val_bma_ll'] > best_val_ll, lambda x: get_best_params(params), lambda x: best_params, None)
        best_val_ll = jnp.maximum(best_val_ll, metrics['val_bma_ll'])

        return (rng_key, params, opt_state, best_val_ll, best_params, (x_all, atten_all, y_all)), \
            {'loss': loss, 'current_lr': current_lr, **metrics, **logs}

    return _train_expert_step


# ---------------------------------------------------------------------------
# Training pipeline helpers
# ---------------------------------------------------------------------------

def _setup_t_sample_fn(s_model, curve_sampling_mode):
    """Create t-sampling function based on curve sampling mode.

    Args:
        s_model: Subspace model with ``curve_mask`` and ``t_max`` attributes.
        curve_sampling_mode: ``'per_leave'`` or ``'combined'`` (and variants).

    Returns:
        Callable ``(key) -> t`` value(s).
    """
    t_max = getattr(s_model, 't_max', 1.0)
    if curve_sampling_mode == 'per_leave':
        print("Using per-leave curve sampling")
        only_curves = jax.tree.map(lambda m: True if m else None, s_model.curve_mask)
        num_curve_leaves = len(jax.tree.leaves(only_curves))
        struct = jax.tree.structure(only_curves)

        def t_sample_fn(key):
            t_flatt = random.uniform(key, (num_curve_leaves,), minval=0., maxval=t_max)
            t_tree = jax.tree.unflatten(struct, t_flatt)
            t_tree = jax.tree.map(lambda k: False if k is None else k, t_tree,
                                   is_leaf=lambda x: x is None)
            return t_tree
    else:
        def t_sample_fn(key):
            return random.uniform(key, (1,), minval=0., maxval=t_max)
    return t_sample_fn


def _setup_validation(s_model, params, val_input_ids, val_attention_mask,
                       val_labels, batch_size_eval, k,
                       lora_rho=0.0, lora_rho_s=0.0):
    """Set up validation metrics function and best-parameter tracking.

    If *val_input_ids* is ``None``, returns no-op stubs.

    Returns:
        ``(valid_metrics_fn, empty_metric, get_best_params)``
    """
    empty_metric = _empty_val_metric()
    has_val = val_input_ids is not None

    if has_val:
        print("save best parameters")
        valid_metrics_fn_inner = setup_metrics(
            s_model, batch_size_eval, val_input_ids, val_attention_mask, val_labels,
            n_samples=10 if k > 0 else 1, key_prefix='val_')

        def valid_metrics_fn(rng, params):
            if lora_rho > 0.0 or lora_rho_s > 0.0:
                params = s_model.set_lora_rho(rho_w=0.0, rho_s=0.0, params=params)
            return valid_metrics_fn_inner(rng, params)

        def get_best_params(params):
            return s_model.only_trainable_params(params['params'])
    else:
        valid_metrics_fn = lambda rng, params: empty_metric
        def get_best_params(params):
            return None

    return valid_metrics_fn, empty_metric, get_best_params


def _setup_rho_scheduler(lora_params, num_steps):
    """Create rho scheduler for LoRA noise annealing.

    Returns:
        Callable ``(step, base_rho) -> current_rho``.
    """
    if lora_params and lora_params.get('rho_scheduler_frequency', 0.0) > 0.0:
        frequency = lora_params['rho_scheduler_frequency']
        base_frequency = num_steps / (2 * frequency)
        print(f"Using cosine rho schedule with total steps {num_steps}")

        def rho_scheduler(x, base):
            return base * (1 - jnp.cos(x / base_frequency * jnp.pi))
        return rho_scheduler
    else:
        return lambda step, base: base


def _build_eval_specs(k, curve_sampling_mode, has_train_data, has_val,
                       artifact_weights_use=False):
    """Build evaluation specification list based on curve degree.

    Args:
        k: Bézier curve degree (0 = single point / pretraining).
        curve_sampling_mode: ``'combined'``, ``'per_leave'``, ``'combined_noBMA'``, etc.
        has_train_data: Whether training data is available for eval.
        has_val: Whether validation data is available.
        artifact_weights_use: Pre-computed weights artifact, if any.

    Returns:
        Ordered list of eval specification dicts for ``run_evaluation``.
    """
    eval_specs = []

    # 1) Always: test with linspace -> summary (+ plot_curve if k > 0)
    _test_metrics = list(PRETRAIN_EVAL_FNS)
    if k > 0:
        _test_metrics.append({'fn': _eval_plot_curve, 'plot_title': 'Final testset'})
    eval_specs.append({
        'name': 'test', 'sampling': 'linspace', 'key_prefix': 'final_',
        'metrics': _test_metrics,
    })

    # 2) Curve-specific evaluation (k > 0)
    if k > 0:
        if curve_sampling_mode == 'combined':
            # Posterior weights from train data
            if has_train_data:
                print("Compute posterior weights on train set")
                eval_specs.append({
                    'name': 'train', 'sampling': 'linspace',
                    'key_prefix': 'f_train_',
                    'metrics': [
                        _eval_post_pred, _eval_mutual_info, _eval_entropy_along_curve,
                        {'fn': _eval_plot_curve, 'plot_title': 'Final trainset'},
                        _eval_compute_posterior_weights,
                    ],
                })
            elif artifact_weights_use:
                eval_specs.append({
                    'name': 'test', 'sampling': 'linspace',
                    'key_prefix': '_tmp_',
                    'metrics': [_eval_load_posterior_weights],
                })

            if has_val:
                # Find best temperature on validation set
                eval_specs.append({
                    'name': 'val', 'sampling': 'linspace',
                    'key_prefix': 'final_valid_',
                    'metrics': [_eval_best_temperature],
                })
                # Apply best weight from val to test
                eval_specs.append({
                    'name': 'test', 'sampling': 'linspace',
                    'key_prefix': 'post_pred_test_ppd_',
                    'metrics': [
                        {'fn': _eval_post_pred, 'weight_key': 'best_weight'},
                        {'fn': _eval_mutual_info, 'weight_key': 'best_weight'},
                        {'fn': _eval_mean_entropy, 'weight_key': 'best_weight'},
                    ],
                })
            else:
                # Per-temperature analysis on test
                eval_specs.append({
                    'name': 'test', 'sampling': 'linspace',
                    'key_prefix': 'post_pred_test_ppd_',
                    'metrics': [
                        _eval_post_pred_per_temp,
                        _eval_mutual_info_per_temp,
                        _eval_mean_entropy_per_temp,
                    ],
                })

            # Entropy along curve on test (always for combined mode)
            eval_specs.append({
                'name': 'test', 'sampling': 'linspace',
                'key_prefix': '',
                'metrics': [_eval_entropy_along_curve],
            })

        else:
            # Non-combined mode (per_leaf, combined_noBMA, etc.)
            use_t_sample = 'combined' not in curve_sampling_mode
            eval_specs.append({
                'name': 'test',
                'sampling': 't_sample' if use_t_sample else 'linspace',
                'key_prefix': 'approx_',
                'metrics': list(CURVE_EVAL_FNS),
            })

    return eval_specs


def _run_eval_phase(*, rng_key, s_model, params, k, data,
                     curve_sampling_mode, batch_size_eval, t_sample_fn,
                     temperature, lora_rho=0.0, lora_rho_s=0.0,
                     logger, logger_prefix, smoke_test,
                     artifact_weights_use=False):
    """Disable LoRA noise and run the modular evaluation pipeline.

    Args:
        data: ``DataSplits`` with all dataset arrays.

    Returns:
        params (possibly modified with LoRA rho set to zero).
    """
    # Disable LoRA noise for evaluation
    if lora_rho > 0.0 or lora_rho_s > 0.0:
        params = s_model.set_lora_rho(rho_s=0.0, rho_w=0.0, params=params)

    eval_specs = _build_eval_specs(
        k, curve_sampling_mode,
        has_train_data=data.has_train, has_val=data.has_val,
        artifact_weights_use=artifact_weights_use)

    run_evaluation(
        s_model=s_model, params=params, rng_key=rng_key,
        data=data, eval_specs=eval_specs,
        batch_size_eval=batch_size_eval, t_sample_fn=t_sample_fn,
        temperature=temperature, logger=logger,
        logger_prefix=logger_prefix, smoke_test=smoke_test, k=k,
        artifact_weights=artifact_weights_use if artifact_weights_use else None,
    )

    return params


def _run_training_phase(rng_key, env, data, hp, logger, logger_prefix, *, scan_fn=None):
    """Execute one training phase (expert or minibatch) and log metrics.

    Reads ``params`` and ``opt_state`` from *env*; callers that need to
    override them (e.g. the per-CP pretrain loop) should mutate those fields
    on *env* before calling.

    If *scan_fn* is provided it is reused (avoids JAX retrace across
    pretraining iterations); otherwise a new ``@jit`` scan wrapper is created.

    Returns:
        ``(rng_key, params, last_trainable_params, scan_fn)``
    """
    s_model = env.s_model
    optimizer = env.optimizer
    params = env.params
    opt_state = env.opt_state
    cosine_sim = env.cosine_sim
    t_sample_fn = env.t_sample_fn
    rho_scheduler = env.rho_scheduler
    valid_metrics_fn = env.valid_metrics_fn
    empty_metric = env.empty_metric
    get_best_params = env.get_best_params

    k = s_model.k
    lr_schedule = env.lr_schedule
    dataset_sampling = hp.dataset_sampling
    lora_rho = env.lora_rho
    lora_rho_s = env.lora_rho_s
    eval_every_n_batch = hp.eval_every_n_batch
    batch_size = hp.batch_size
    num_epochs = hp.num_epochs
    num_steps = hp.num_steps
    smoke_test = hp.smoke_test

    if num_steps <= 0:
        return rng_key, params, None, scan_fn

    # Initialize best-parameter tracking
    best_val_ll = -np.inf
    best_params = s_model.only_trainable_params(params['params']) if data.has_val else None

    ds_sampling = hp.ds_sampling
    train_input_ids = data.train_input_ids
    train_attention_mask = data.train_attention_mask
    train_labels = data.train_labels

    # Initial CP center for curve metrics
    cp_w = masked_pytree_to_matrix(params['params'], s_model.curve_mask, k)
    initial_cp_center = cp_w.mean(axis=0)

    if ds_sampling == 'expert':
        print(f"Start sorted expert training for {num_steps}")
        sort_mode = dataset_sampling['expert'].get('sort_mode', False)
        if sort_mode == 'Random':
            order = random.permutation(rng_key, train_input_ids.shape[0])
            train_input_ids = train_input_ids[order]
            train_attention_mask = train_attention_mask[order]
            train_labels = train_labels[order]
        elif sort_mode:
            assert cosine_sim is not None, "cosine_sim required for non-Random expert sort"
            distance = 1 - cosine_sim
            if sort_mode == 'Hierarchical':
                condensed = squareform(distance, checks=False)
                Z = hierarchy.linkage(condensed, method='average')
                Z_opt = hierarchy.optimal_leaf_ordering(Z, condensed)
                order = hierarchy.dendrogram(Z_opt.astype(np.float64), no_plot=True)['leaves']
                order = jnp.array(order)
            elif sort_mode == 'Spectral':
                distance = distance.at[jnp.diag_indices(distance.shape[0])].set(0.0)
                D = jnp.diag(distance.sum(axis=1))
                L = D - distance
                vals, vecs = jnp.linalg.eigh(L)
                fiedler = vecs[:, -2]
                order = jnp.argsort(fiedler)
            elif sort_mode == 'First_data':
                order = jnp.argsort(distance[0])
            else:
                raise ValueError(f"Unknown sort_mode {sort_mode} for expert sampling.")
            train_input_ids = train_input_ids[order]
            train_attention_mask = train_attention_mask[order]
            train_labels = train_labels[order]
            print(f"Data sorted for expert sampling using {sort_mode} method")

        window_size = dataset_sampling['expert']['window_size']
        window_size = max(batch_size, window_size)
        train_expert_step = _make_expert_step_fn(
            s_model=s_model, optimizer=optimizer, lr_schedule=lr_schedule,
            t_sample_fn=t_sample_fn, eval_every_n_batch=eval_every_n_batch,
            valid_metrics_fn=valid_metrics_fn, empty_metric=empty_metric,
            get_best_params=get_best_params, num_steps=num_steps,
            batch_size=batch_size, window_size=window_size)

        carry = (rng_key, params, opt_state, best_val_ll, best_params,
                 (train_input_ids, train_attention_mask, train_labels))
        if scan_fn is None:
            @jit
            def scan_fn(carry, xs):
                return jax.lax.scan(train_expert_step, carry, xs)
        carry, metrics = scan_fn(carry, jnp.arange(num_steps))

    elif ds_sampling == 'minibatch':
        print(f"Start minibatch training for {num_epochs} epochs")
        train_epoch_fn = _make_train_epoch_fn(
            s_model=s_model, optimizer=optimizer, lr_schedule=lr_schedule,
            t_sample_fn=t_sample_fn, lora_rho=lora_rho, lora_rho_s=lora_rho_s,
            rho_scheduler=rho_scheduler, eval_every_n_batch=eval_every_n_batch,
            valid_metrics_fn=valid_metrics_fn, empty_metric=empty_metric,
            get_best_params=get_best_params,
            initial_cp_center=initial_cp_center, k=k,
            train_input_ids=train_input_ids,
            train_attention_mask=train_attention_mask,
            train_labels=train_labels,
            batch_size=batch_size, smoke_test=smoke_test,
            num_epochs=num_epochs)

        carry = (rng_key, params, opt_state, best_val_ll, best_params, 0)
        if scan_fn is None:
            @jit
            def scan_fn(carry, xs):
                return jax.lax.scan(train_epoch_fn, carry, xs)
        carry, metrics = scan_fn(carry, jnp.arange(num_epochs))
    else:
        raise ValueError(f"Unknown dataset_sampling method {ds_sampling}. Choose 'expert' or 'minibatch'.")

    # Clean up inf values from lazy validation
    for key, value in metrics.items():
        if key.startswith('val_'):
            metrics[key] = jnp.array(value).flatten()[::eval_every_n_batch].repeat(
                eval_every_n_batch, axis=0)[:num_steps]
        else:
            metrics[key] = jnp.array(value).flatten()

    # Log all metrics
    _ = [logger.log(dict(zip(metrics.keys(), m))) for m in zip(*metrics.values())]

    # Log metrics as artifact
    epochs = np.repeat(np.arange(num_epochs), num_steps // batch_size)
    jnp.savez(f'tmp_files/{logger.id}_{logger_prefix}metrics.npz', **metrics, epochs=epochs)
    art = wandb.Artifact(name="metrics", type="npz")
    art.add_file(f'tmp_files/{logger.id}_{logger_prefix}metrics.npz')
    logger.log_artifact(art)

    # Extract results from carry
    rng_key = carry[0]
    params = carry[1]
    last_trainable_params = None
    if data.has_val:
        last_trainable_params = s_model.only_trainable_params(params['params'])
        best_curve_params = carry[4]
        params['params'] = jax.tree.map(
            lambda x, y, m: x if m else y,
            best_curve_params, params['params'], s_model.train_mask)

    if k > 0:
        cp = masked_pytree_to_matrix(params['params'], s_model.curve_mask, k)
        length = _bezier_length(cp)
        logger.summary.update({'Curve length': length})

    return rng_key, params, last_trainable_params, scan_fn


# ---------------------------------------------------------------------------
# Pipeline stage functions – called by train() orchestrator
# ---------------------------------------------------------------------------

def _setup_lr_schedule(optimizer_conf):
    """Build optax learning-rate schedule from the optimizer config dict.

    Mutates ``optimizer_conf['kwargs']['learning_rate']`` in-place (replaces
    the raw dict/float with the compiled optax schedule object) and returns
    the schedule.
    """
    lr_conf = optimizer_conf['kwargs']['learning_rate']
    if isinstance(lr_conf, dict):
        lr_schedule = getattr(optax, lr_conf['name'])(**lr_conf['kwargs'])
    else:
        lr_schedule = optax.constant_schedule(lr_conf)
    optimizer_conf['kwargs']['learning_rate'] = lr_schedule
    return lr_schedule


def _prepare_datasets(rng_key, dataset_path, val_percentage, batch_size,
                      smoke_test, logger):
    """Load data and prepare train / val / test splits.

    Args:
        rng_key: JAX random key used for the train/val shuffle permutation.
        dataset_path: Path or wandb artifact reference for the dataset.
        val_percentage: Fraction of training data to reserve for validation.
        batch_size: Used to batch-align the train split so every epoch
            contains whole batches.  When *smoke_test* is True the dataset
            is also truncated to ``3 * batch_size`` training samples.
        smoke_test: Truncate dataset to a small size for fast iteration.
        logger: wandb run used when resolving artifact dataset paths.

    Returns:
        ``(data, unique_target_ids, rng_key)``
    """
    # Load train and test data
    input_ids, attention_mask, labels, _, unique_target_ids = \
        load_text_classification_data(dataset_path, run=logger)
    print(f"Dataset loaded: {len(input_ids)} samples, {len(unique_target_ids)} classes")

    test_input_ids, test_attention_mask, test_labels, _, unique_target_ids = \
        load_text_classification_data(dataset_path, run=logger, test=True)
    print(f"Target token IDs: {unique_target_ids}")

    # Shuffle and split into train / val
    rng_key, split_key = random.split(rng_key)
    perm_idx = random.permutation(split_key, len(input_ids))

    n_train = int(np.floor((1. - val_percentage) * len(input_ids) / batch_size) * batch_size)
    ordered_ids = input_ids[perm_idx]
    ordered_mask = attention_mask[perm_idx]
    ordered_labels = labels[perm_idx]

    train_input_ids = ordered_ids[:n_train]
    train_attention_mask = ordered_mask[:n_train]
    train_labels = ordered_labels[:n_train]
    val_input_ids = ordered_ids[n_train:] if val_percentage > 0.0 else None
    val_attention_mask = ordered_mask[n_train:] if val_percentage > 0.0 else None
    val_labels = ordered_labels[n_train:] if val_percentage > 0.0 else None

    # Smoke-test truncation: keep only a few batches so the run finishes fast
    if smoke_test:
        n_smoke = 3 * batch_size
        train_input_ids = train_input_ids[:n_smoke]
        train_attention_mask = train_attention_mask[:n_smoke]
        train_labels = train_labels[:n_smoke]
        if val_input_ids is not None:
            val_input_ids = val_input_ids[:batch_size]
            val_attention_mask = val_attention_mask[:batch_size]
            val_labels = val_labels[:batch_size]
        test_input_ids = test_input_ids[:n_smoke]
        test_attention_mask = test_attention_mask[:n_smoke]
        test_labels = test_labels[:n_smoke]
        print(f"Smoke test: dataset truncated to {n_smoke} train samples")

    has_val = val_input_ids is not None and len(val_input_ids) > 0
    if has_val:
        print("Validation data are used")
    print(f"Training data: {len(train_input_ids)} samples, "
          f"Validation data: {len(val_input_ids) if has_val else 0} samples")

    data = DataSplits(
        train_input_ids=train_input_ids,
        train_attention_mask=train_attention_mask,
        train_labels=train_labels,
        val_input_ids=val_input_ids if has_val else None,
        val_attention_mask=val_attention_mask if has_val else None,
        val_labels=val_labels if has_val else None,
        test_input_ids=test_input_ids,
        test_attention_mask=test_attention_mask,
        test_labels=test_labels,
    )

    return data, unique_target_ids, rng_key


def _compute_expert_embeddings(model, params, data, hp):
    """Compute cosine-similarity matrix for expert sorting, if needed.

    Returns ``None`` when expert sorting is not configured or not required.
    """
    if (hp.ds_sampling != 'expert') or (hp.num_steps <= 0) or not data.has_train:
        return None
    sort_mode = hp.dataset_sampling['expert'].get('sort_mode', False)
    if not sort_mode or sort_mode == 'Random':
        return None
    print("compute embeddings for expert sort")
    embedding = get_embedding(
        data.train_input_ids, data.train_attention_mask,
        model, params, hp.batch_size)
    print(f"Computed embeddings for expert sort with shape {embedding.shape}")
    device = embedding.device
    embedding = jax.device_put(embedding, jax.devices("cpu")[0])
    embedding = embedding.reshape(embedding.shape[0], -1).astype(jnp.float32)
    embedding /= jnp.linalg.norm(embedding, axis=1, keepdims=True)
    cosine_sim = jax.device_put(jnp.dot(embedding, embedding.T), device)
    print(f"Computed cosine similarity matrix with shape {cosine_sim.shape}")
    del embedding
    return cosine_sim


def _setup_training_env(*, rng_key, k, hp, config, data,
                         fixed_cps_train_params=False):
    """Create model, subspace, optimizer, and all training support objects.

    Handles both k=0 (pretraining) and k>0 (curve fitting) cases.

    *hp* carries pure training hyperparameters; object-construction config
    (model architecture, optimizer, LoRA, masking) is read from the raw
    *config* groups: ``net_kwargs``, ``model_params``, ``optimizer_conf``,
    ``curve_params``, ``lora_params``.

    Returns:
        ``(env, rng_key)`` where *env* is a ``TrainingEnv``.
    """
    net_kwargs = config['net_kwargs']
    mp = config.get('model_params', {})
    optimizer_conf = deepcopy(config['optimizer_conf'])
    model_params = dict(mp)
    filter_masks = config.get('curve_params', {}).get('filter_masks', [])
    subspace_model = mp.get('subspace_model', '')
    jitter_multiplier = mp.get('jitter_multiplier', 0.0)
    lora_params = config.get('lora_params', None)
    lr_schedule = optimizer_conf['kwargs']['learning_rate']
    lora_rho = lora_params.get('lora_rho', 0.0) if lora_params else 0.0
    lora_rho_s = lora_params.get('lora_rho_s', 0.0) if lora_params else 0.0

    model = QwenTextClassificationWrapper(**net_kwargs)
    base_params = unfreeze(model.init(None, None))

    rng_key, init_key = random.split(rng_key)

    if k == 0:
        model_params['curve_segment'] = False  # no segments at k=0
        s_model, curve_mask, train_mask, _ = _create_subspace_model(
            model=model, params_tree=base_params['params'], k=0,
            filter_masks=filter_masks,
            optimizer_conf=optimizer_conf,
            lora_params=lora_params,
            model_params=model_params,
            subspace_model=subspace_model)

        # Init once to set s_model internal state (curve_mask, LoRA structs)
        params = jax.tree.map(lambda x: x, base_params)
        params = _init_subspace_params(
            s_model, params, curve_mask, train_mask, init_key,
            jitter_multiplier, lora_params)
    else:
        s_model, params, _, curve_mask, train_mask = _build_subspace_model(
            model=model, params=base_params, k=k,
            filter_masks=filter_masks,
            optimizer_conf=optimizer_conf,
            lora_params=lora_params,
            model_params=model_params,
            subspace_model=subspace_model,
            jitter_multiplier=jitter_multiplier,
            init_key=init_key,
            fixed_cps_train_params=fixed_cps_train_params)
        base_params = None  # not needed for curve phase

    # Build optimizer
    optimizer, opt_state, n_trainable = _build_optimizer(
        optimizer_conf=optimizer_conf, s_model=s_model,
        params=params, fixed_cps_train_params=fixed_cps_train_params)

    # Expert embeddings
    cosine_sim = _compute_expert_embeddings(model, params, data, hp)

    # t sampling, rho scheduler, validation
    t_sample_fn = _setup_t_sample_fn(s_model, hp.curve_sampling_mode)
    rho_scheduler = _setup_rho_scheduler(lora_params, hp.num_steps)

    if data.has_train:
        valid_metrics_fn, empty_metric, get_best_params = \
            _setup_validation(
                s_model, params, data.val_input_ids, data.val_attention_mask,
                data.val_labels, hp.batch_size_eval, k=k,
                lora_rho=lora_rho, lora_rho_s=lora_rho_s)
    else:
        empty_metric = _empty_val_metric()
        valid_metrics_fn = lambda rng, params: empty_metric
        get_best_params = lambda params: None

    env = TrainingEnv(
        s_model=s_model, params=params, base_params=base_params,
        curve_mask=curve_mask, train_mask=train_mask,
        optimizer=optimizer, opt_state=opt_state,
        t_sample_fn=t_sample_fn, rho_scheduler=rho_scheduler,
        valid_metrics_fn=valid_metrics_fn, empty_metric=empty_metric,
        get_best_params=get_best_params,
        cosine_sim=cosine_sim, n_trainable=n_trainable,
        lr_schedule=lr_schedule, lora_rho=lora_rho, lora_rho_s=lora_rho_s)

    return env, rng_key


@contextmanager
def single_point_mask(s_model):
    """Temporarily set k=0 on *s_model* by zeroing its curve mask.

    Saves the original ``k`` and ``curve_mask`` on entry and restores both on
    exit.  The body of the ``with`` block sees the model as a single-point
    (k=0) model — useful for pretraining individual fixed control points
    without altering the outer curve model.
    """
    original_k = s_model.k
    original_curve_mask = s_model.curve_mask
    s_model.k = 0
    s_model.curve_mask = jax.tree.map(lambda x: False, original_curve_mask)
    try:
        yield s_model
    finally:
        s_model.k = original_k
        s_model.curve_mask = original_curve_mask


def _pretrain_fixed_cps(rng_key, hp, config, data, logger, artifact):
    """Train each fixed control point independently (k=0).

    Returns:
        ``(train_params, rng_key)``
    """
    cp_fix = hp.cp_fix
    lora_params = config.get('lora_params', None)

    # Validation checks
    deg_segments = config.get('model_params', {}).get('curve_segment', False)
    if deg_segments:
        all_segments_fixed_endpoints = np.all(
            [deg_segments == i for i in np.diff(
                np.argwhere(np.array(cp_fix) == 1).flatten())])
        assert all_segments_fixed_endpoints, (
            "In curve segment mode, fixed control points must be consecutive "
            "and include endpoints, e.g. [1,0,1,0,1] with deg 2.")
    if lora_params is not None and np.sum(cp_fix) < len(cp_fix):
        _lora_mode = config.get('model_params', {}).get('lora_mode', '')
        assert _lora_mode in ('A(t)B(t)', 'A(t)eB(t)'), (
            "Only A(t)B(t) or A(t)eB(t) LoRA mode makes sense for "
            "subspace curves with fixed control points")

    # Setup training environment for k=0
    env, rng_key = _setup_training_env(
        k=0, hp=hp, config=config, data=data, rng_key=rng_key)

    # Pretraining loop — single_point_mask makes the k=0 intent explicit and
    # ensures the model is restored even if an exception is raised.
    train_params = []
    scan_fn = None
    print("Pretraining fixed control points")
    with single_point_mask(env.s_model):
        for i, fixed in enumerate(cp_fix):
            if fixed:
                print(f"Training cp member {i}")
                rng_key, init_key = random.split(rng_key)
                params = jax.tree.map(lambda x: x, env.base_params)
                params = _init_subspace_params(
                    env.s_model, params, env.curve_mask, env.train_mask, init_key,
                    config.get('model_params', {}).get('jitter_multiplier', 0.0),
                    lora_params)

                # Re-init optimizer state
                only_train = env.s_model.only_trainable_params(params['params'])
                opt_state = env.optimizer.init(only_train)

                # Train
                env.params = params
                env.opt_state = opt_state
                rng_key, params, _, scan_fn = _run_training_phase(
                    rng_key, env, data, hp, logger, f'cp_fixed_{i}_',
                    scan_fn=scan_fn)

                # Evaluate
                _run_eval_phase(
                    s_model=env.s_model, params=params,
                    rng_key=rng_key, k=0,
                    t_sample_fn=env.t_sample_fn,
                    logger=logger, logger_prefix=f'cp_fixed_{i}_',
                    data=data, **hp.eval_common,
                    lora_rho=env.lora_rho, lora_rho_s=env.lora_rho_s)

                train_params.append(env.s_model.only_trainable_params(params['params']))
                rng_key, _ = random.split(rng_key)
            else:
                train_params.append(False)

    del env  # free GPU memory before curve phase

    if hp.save_params:
        print("Save pretrained trainable params ...")
        np.save(f'tmp_files/{logger.id}_pretrained_params.npy', train_params)
        time.sleep(1)
        print("Saved pretrained parameters to tmp_files")
        artifact.add_file(f'tmp_files/{logger.id}_pretrained_params.npy')

    return train_params, rng_key


def _train_full_curve(rng_key, hp, config, data, train_params, logger, artifact):
    """Fit the full Bézier curve (k > 0), evaluate, and save parameters.

    Args:
        train_params: Pretrained control-point params (list or ``False``).
    """
    cp_fix = hp.cp_fix
    k = len(cp_fix) - 1
    all_fixed = len(cp_fix) == np.sum(cp_fix)

    deg_segments = config.get('model_params', {}).get('curve_segment', False)
    indep_connected = config.get('model_params', {}).get('indepent_connected', False)
    print("Training curve model")
    if np.sum(cp_fix) == 0:
        print("No fixed control points, training full curve with all control points free.")
        if deg_segments:
            assert indep_connected is True, (
                "In curve segment mode with no fixed control points, curve "
                "optimization only makes sense if all segments are "
                "independently connected.")

    # Setup training environment for k > 0
    env, rng_key = _setup_training_env(
        k=k, hp=hp, config=config, data=data, rng_key=rng_key,
        fixed_cps_train_params=train_params)

    # Log parameter stats
    n_params = sum(
        p.size if isinstance(p, jnp.ndarray) else 0
        for p in jax.tree.flatten(env.params)[0])
    param_memory = sum(
        p.size * p.dtype.itemsize if isinstance(p, jnp.ndarray) else 0
        for p in jax.tree.flatten(env.params)[0])
    logger.summary.update({
        'n_params': n_params,
        'memory_params': param_memory,
        'n_params_trainable': env.n_trainable,
    })
    print(f"Number of parameters: {n_params:_}\n"
          f"Memory of parameters: {param_memory / (1024**2):_}MB")

    # Train (unless all CPs are fixed → eval-only)
    if not all_fixed:
        rng_key, params, last_trainable_params, _ = _run_training_phase(
            rng_key, env, data, hp, logger, '')
    else:
        params = env.params
        last_trainable_params = None

    # Evaluate
    _run_eval_phase(
        s_model=env.s_model, params=params,
        rng_key=rng_key, k=k,
        t_sample_fn=env.t_sample_fn,
        logger=logger, logger_prefix='',
        data=data, **hp.eval_common,
        lora_rho=env.lora_rho, lora_rho_s=env.lora_rho_s)

    # Save curve params
    if hp.save_params:
        print("Save final trainable params ...")
        np.save(f'tmp_files/{logger.id}_trainable_params.npy',
                env.s_model.only_trainable_params(params['params']))
        if last_trainable_params is not None:
            np.save(f'tmp_files/{logger.id}_last_trainable_params.npy',
                    last_trainable_params)
        time.sleep(1)
        print("Saved trainable parameters to tmp_files")
        artifact.add_file(f'tmp_files/{logger.id}_trainable_params.npy')
        if last_trainable_params is not None:
            artifact.add_file(f'tmp_files/{logger.id}_last_trainable_params.npy')


def main():
    logger = wandb.init()
    config = wandb.config
    train(logger, config)


def train(logger, config):
    """Main training orchestrator for Qwen curve fine-tuning."""
    rng_key = random.PRNGKey(config['rng_seed'])
    artifact = wandb.Artifact(name="params", type="pytree")

    # Resolve model path from wandb if a local path is not given
    if config['net_kwargs']['model_path'].startswith('ddold/'):
        model_artifact = logger.use_artifact(config['net_kwargs']['model_path'], type='model')
        config['net_kwargs']['model_path'] = model_artifact.download()

    data, unique_target_ids, rng_key = _prepare_datasets(
        rng_key,
        dataset_path=config['dataset_path'],
        val_percentage=config['train_hyper'].get('val_percentage', 0.0),
        batch_size=config['train_hyper']['batch_size'],
        smoke_test=config['train_hyper'].get('smoke_test', False),
        logger=logger)

    _setup_lr_schedule(config['optimizer_conf'])
    config['net_kwargs']['target_token_ids'] = unique_target_ids
    hp = TrainHyperparams.from_config(config, data)

    pretrained_params = [False] * len(hp.cp_fix)
    if any(hp.cp_fix):
        pretrained_params, rng_key = _pretrain_fixed_cps(
            rng_key, hp, config, data, logger, artifact)

    if len(hp.cp_fix) > 1:
        _train_full_curve(rng_key, hp, config, data, pretrained_params, logger, artifact)

    if hp.save_params:
        logger.log_artifact(artifact)
        time.sleep(5)
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
    
    print("Curve training")
    if args.use_sweep:
        print("Using wandb config")
        logger = wandb.init()
        config = wandb.config
        train(logger, config)
    else:
        print("Using predefined config")

        config = {
            # --- Run / experiment --------------------------------------------------
            'rng_seed': 1,
            # 'load_params_path': "ddold/qwen_PtoC/47xexal3",
            'load_params_path': False,

            # --- Data --------------------------------------------------------------
            'dataset_path': "ddold/qwen_PtoC/winogrande_m_dataset:v0",
            # 'dataset_path': "ddold/qwen_PtoC/ARC-Easy_dataset:v0",
            # 'dataset_path': "ddold/qwen_PtoC/MMLU_chem_dataset:v0",
            # 'dataset_path': "ddold/qwen_PtoC/obqa_dataset:v0",

            # --- Base model (QwenTextClassificationWrapper) ------------------------
            'net_kwargs': {'model_path': "artifacts/qwen2.5_7B_bfloat16:v0"},
            # 'net_kwargs': {'model_path': "ddold/qwen_PtoC/qwen2.5_7B_bfloat16:v0"},

            # --- Training hyperparameters → TrainHyperparams ----------------------
            # These control *how* training runs, not *what* is built.
            'train_hyper': {
                'cp_fix': args.cp_fix,        # e.g. [1,0,1]: 1=fixed, 0=trainable CP
                'batch_size': args.batch_size,
                'num_epochs': -1,             # -1 means use num_steps instead
                'num_steps': 10 if args.smoke_test else 5_000,
                'eval_every_n_batch': 20,
                'temperature': [1.0],
                'curve_sampling_mode': 'combined_noBMA',  # 'per_leave' | 'combined' | 'combined_noBMA'
                'dataset_sampling': {'minibatch': {}},
                # 'dataset_sampling': {'expert': {'window_size': 4, 'sort_mode': False}},
                'save_params': True,
                'smoke_test': args.smoke_test,
                'val_percentage': 0.1,
            },

            # --- Subspace model + LoRA architecture (→ _create_subspace_model) ----
            # All parameters that determine *what* the subspace model is.
            'model_params': {
                'subspace_model': 'lora_category',
                'jitter_multiplier': 0.,      # weight-init noise scale
                'natural_parameterization': False,
                'weight_decay': 0.,
                'curve_parameterization': 'bezier',
                'curve_segment': 3,           # int → segmented degree; False → full curve
                'indepent_connected': False,  # True → independently connected segments
                # 'gravity': 5.0,
                # 'energy_weakening': 4.,
                'r': 8,
                'lora_dtype': 'float32',
                'lora_alpha': 16.0,
                'lora_mode': 'A(t)B(t)',
                # 'lora_mode': 'Arotsd(t)eB',
                # 'lora_mode': 'Arot(t)sdeBrot(t)',
            },

            # --- Curve topology (which parameters become the Bézier curve) --------
            'curve_params': {
                'filter_masks': [
                    {'keys': ["self_attn", "q_proj", "kernel"], 'op': 'all'},
                    {'keys': ["self_attn", "v_proj", "kernel"], 'op': 'all'},
                    {'keys': ["lm_head", "kernel"], 'op': 'all'},
                ],
            },

            # --- Optimizer ---------------------------------------------------------
            'optimizer_conf': {
                'name': "adamw",
                'kwargs': {
                    'learning_rate': {
                        'name': "linear_onecycle_schedule",
                        'kwargs': {
                            'transition_steps': 5_000,
                            'peak_value': 1e-4,
                            'pct_start': 0.12,
                            'pct_final': 1.,
                            'div_factor': 300.0,
                            'final_div_factor': 300.0,
                        },
                    },
                    'weight_decay': 0.0,
                },
                'freeze_other_params': True,
                # SGD alternative:
                # 'name': "sgd",
                # 'kwargs': {'learning_rate': 0.0001, 'momentum': 0.9, 'nesterov': True},
                # 'grad_clip_norm': 100.0,
            },

            # --- LoRA noise / scheduling -------------------------------------------
            # lora_rho / lora_rho_s are training hyperparameters but sourced here
            # to keep all LoRA config in one place.
            'lora_params': {
                'use_lora': True,
                'rho_scheduler_frequency': 100.,
                'lora_rho': .25,
                'lora_rho_s': .5,
                'filter_masks': [
                    {'keys': ["self_attn", "q_proj", "kernel"], 'op': 'all', 'dims': [1, 0]},
                    {'keys': ["self_attn", "v_proj", "kernel"], 'op': 'all', 'dims': [1, 0]},
                    {'keys': ["lm_head", "kernel"], 'op': 'all', 'dims': [1, 0]},
                ],
            },
        }
        logger = wandb.init(project="qwen_PtoC", name="test_more degree", entity="ddold", config=config)
        train(logger, config)

