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
from jax import random
import optax
import wandb
from einshape import jax_einshape as es
import matplotlib.pyplot as plt
from flax.core import freeze, unfreeze

from jax_tqdm import scan_tqdm
from src.utils import calibration_error as ece_fn, post_pred_performance

# Project imports
from src.qwen_jax import QwenTextClassificationWrapper
from src.subspace_curve import (
    SubspaceBaseModel, masked_pytree_to_matrix, bezier_curve, filter_mask, bezier_coeff_fn, get_subspace_model,
    RepulsiveMixin, EntropyMixin, EntropyMixin2, JensenShannonMixin, LoraMixin,
)
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from dataclasses import dataclass, field, replace
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
    Construct via ``TrainHyperparams.from_config_dict(th, data)``; ``data``
    is only needed to resolve epoch / step counts from data size.
    """
    batch_size: int
    eval_every_n_batch: int
    num_epochs: int
    num_steps: int
    save_params: bool
    smoke_test: bool
    dataset_sampling: dict
    temperature: list

    # --- computed properties (not stored; derived from above fields) ------

    @property
    def batch_size_eval(self) -> int:
        return self.batch_size * 2

    @property
    def ds_sampling(self) -> str:
        return list(self.dataset_sampling.keys())[0]

    @classmethod
    def from_config_dict(cls, th: dict, data: 'DataSplits') -> 'TrainHyperparams':
        """Construct from ``config['train_hyper']`` dict plus loaded data.

        ``data`` is needed solely to resolve epoch / step counts from the
        training-set size.
        """
        batch_size = th['batch_size']
        dataset_sampling = th.get('dataset_sampling', {})

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
            batch_size=batch_size,
            eval_every_n_batch=th.get('eval_every_n_batch', 20),
            num_epochs=num_epochs,
            num_steps=num_steps,
            save_params=th.get('save_params', False),
            smoke_test=th.get('smoke_test', False),
            dataset_sampling=dataset_sampling,
            temperature=th.get('temperature', [1.0]),
        )


@dataclass
class LoraParams:
    """LoRA adapter configuration.

    All stored fields map 1-to-1 to keys inside ``config['model_params']['lora_params']``.
    Construct via ``LoraParams.from_config_dict(lp)``.
    """
    use_lora: bool = False
    r: int = 8
    lora_dtype: str = 'float32'
    lora_alpha: float = 16.0
    lora_mode: str = 'A(t)B(t)'
    rho_scheduler_frequency: float = 0.0
    lora_rho: float = 0.0
    lora_rho_s: float = 0.0
    filter_masks: list = field(default_factory=list)

    @classmethod
    def from_config_dict(cls, lp: dict) -> 'LoraParams':
        """Construct from the ``lora_params`` sub-dict of ``config['model_params']``."""
        return cls(
            use_lora=lp.get('use_lora', False),
            r=lp.get('r', 8),
            lora_dtype=lp.get('lora_dtype', 'float32'),
            lora_alpha=lp.get('lora_alpha', 16.0),
            lora_mode=lp.get('lora_mode', 'A(t)B(t)'),
            rho_scheduler_frequency=lp.get('rho_scheduler_frequency', 0.0),
            lora_rho=lp.get('lora_rho', 0.0),
            lora_rho_s=lp.get('lora_rho_s', 0.0),
            filter_masks=lp.get('filter_masks', []),
        )

    def model_kwargs(self) -> dict:
        """Return only the fields consumed by the subspace model constructor."""
        return dict(
            r=self.r,
            lora_dtype=self.lora_dtype,
            lora_alpha=self.lora_alpha,
            lora_mode=self.lora_mode,
            lora_rho=self.lora_rho,
            lora_rho_s=self.lora_rho_s,
        )

    def build_lora_mask(self, params_tree):
        """Build LoRA mask from ``filter_masks`` applied to *params_tree*.

        Returns an all-False pytree when ``use_lora`` is ``False``.
        """
        lora_mask = jax.tree.map(lambda x: False, params_tree)
        if self.use_lora:
            for f in self.filter_masks:
                lora_mask = filter_mask(lora_mask, *f['keys'],
                                        as_curve=np.array(f['dims']),
                                        BoolschenOperator=eval(f['op']))
        return lora_mask


@dataclass
class ModelParams:
    """Architecture and curve topology configuration.

    All stored fields map 1-to-1 to keys inside ``config['model_params']``.
    Construct via ``ModelParams.from_config_dict(mp)``.

    ``get_model_kwargs()`` returns the flat dict consumed by
    ``_create_subspace_model`` (excludes high-level routing fields such as
    ``subspace_model``, ``filter_masks``, ``cp_fix``, ``curve_sampling_mode``,
    and LoRA noise params).
    """
    cp_fix: list
    curve_sampling_mode: str
    subspace_model: str
    filter_masks: list = field(default_factory=list)
    jitter_multiplier: float = 0.0
    natural_parameterization: bool = False
    weight_decay: float = 0.0
    curve_parameterization: str = 'bezier'
    curve_segment: int | bool = False
    indepent_connected: bool = False
    lora_params: 'LoraParams | None' = None
    perm_spec: bool = False
    mutuable_param_name: bool = False
    # Mixin fields (RepulsiveMixin, EntropyMixin, JensenShannonMixin)
    gravity: float = 0.0
    energy_weakening: float = 0.1
    regularizer: str = 'force'
    entropy_weight: float = 0.0

    @classmethod
    def from_config_dict(cls, mp: dict) -> 'ModelParams':
        """Construct from ``config['model_params']`` dict."""
        cp_fix = mp['cp_fix']
        print(f"Control point fixed list: {cp_fix}")
        return cls(
            cp_fix=cp_fix,
            curve_sampling_mode=mp.get('curve_sampling_mode', 'combined'),
            subspace_model=mp.get('subspace_model', ''),
            filter_masks=mp.get('filter_masks', []),
            jitter_multiplier=mp.get('jitter_multiplier', 0.0),
            natural_parameterization=mp.get('natural_parameterization', False),
            weight_decay=mp.get('weight_decay', 0.0),
            curve_parameterization=mp.get('curve_parameterization', 'bezier'),
            curve_segment=mp.get('curve_segment', False),
            indepent_connected=mp.get('indepent_connected', False),
            lora_params=LoraParams.from_config_dict(mp['lora_params']) if mp.get('lora_params') else None,
            perm_spec=mp.get('perm_spec', False),
            mutuable_param_name=mp.get('mutuable_param_name', False),
            gravity=mp.get('gravity', 0.0),
            energy_weakening=mp.get('energy_weakening', 0.1),
            regularizer=mp.get('regularizer', 'force'),
            entropy_weight=mp.get('entropy_weight', 0.0),
        )

    @property
    def k(self) -> int:
        """Bézier curve degree: number of control points minus one."""
        return len(self.cp_fix) - 1

    def get_model_kwargs(self, *, model) -> dict:
        """Return the hard-coded kwargs dict for the subspace model constructor.

        All keys are explicitly listed so no automatic signature introspection
        is needed.  Mixin params are only included when the resolved subspace
        model class is a subclass of the corresponding mixin.

        Args:
            model: The wrapped neural-network model instance.
        """
        s_model_cls = get_subspace_model(self.subspace_model)
        d = dict(
            model=model,
            k=self.k,
            weight_decay=self.weight_decay,
            perm_spec=self.perm_spec,
            natural_parameterization=self.natural_parameterization,
            mutuable_param_name=self.mutuable_param_name,
            curve_parameterization=self.curve_parameterization,
            curve_segment=self.curve_segment,
            indepent_connected=self.indepent_connected,
        )
        # Mixin params — only included when the model class uses that mixin
        if issubclass(s_model_cls, RepulsiveMixin):
            d.update(gravity=self.gravity,
                     energy_weakening=self.energy_weakening,
                     regularizer=self.regularizer)
        if issubclass(s_model_cls, (EntropyMixin, EntropyMixin2, JensenShannonMixin)):
            d['entropy_weight'] = self.entropy_weight
        if self.lora_params and self.lora_params.use_lora:
            d.update(self.lora_params.model_kwargs())
        return d

    @property
    def lora_rho_values(self) -> tuple[float, float]:
        """Return ``(lora_rho, lora_rho_s)``, defaulting to 0.0 when no LoRA."""
        lp = self.lora_params
        return (lp.lora_rho if lp else 0.0, lp.lora_rho_s if lp else 0.0)

    def build_curve_mask(self, params_tree, k):
        """Build the curve mask from ``filter_masks`` applied to *params_tree*.

        Returns an all-False pytree when ``k == 0``.
        """
        curve_mask = jax.tree.map(lambda x: False, params_tree)
        if k > 0:
            for f in self.filter_masks:
                curve_mask = filter_mask(curve_mask, *f['keys'], as_curve=True,
                                         BoolschenOperator=eval(f['op']))
        return curve_mask


@dataclass
class OptimizerConf:
    """Optimizer configuration, including the compiled learning-rate schedule.

    All stored fields map 1-to-1 to keys inside ``config['optimizer_conf']``.
    Construct via ``OptimizerConf.from_config_dict(oc)``; the LR schedule is
    compiled from ``kwargs['learning_rate']`` during construction so callers
    never have to call ``_setup_lr_schedule`` separately.
    """
    name: str
    kwargs: dict                   # contains compiled lr_schedule under 'learning_rate'
    lr_schedule: object            # compiled optax schedule (reference into kwargs)
    freeze_other_params: bool = False
    grad_clip_norm: float | None = None

    @classmethod
    def from_config_dict(cls, oc: dict) -> 'OptimizerConf':
        """Construct from ``config['optimizer_conf']`` dict.

        Compiles ``kwargs['learning_rate']`` into an optax schedule and stores
        it both as ``lr_schedule`` and inside ``kwargs`` so downstream helpers
        receive the already-compiled object.
        """
        kwargs = dict(oc['kwargs'])
        lr_conf = kwargs['learning_rate']
        if isinstance(lr_conf, dict):
            lr_schedule = getattr(optax, lr_conf['name'])(**lr_conf['kwargs'])
        else:
            lr_schedule = optax.constant_schedule(lr_conf)
        kwargs['learning_rate'] = lr_schedule
        return cls(
            name=oc['name'],
            kwargs=kwargs,
            lr_schedule=lr_schedule,
            freeze_other_params=oc.get('freeze_other_params', False),
            grad_clip_norm=oc.get('grad_clip_norm', None),
        )


@dataclass
class Config:
    """Top-level configuration object for the training pipeline.

    Construct via ``Config.from_dict(d, data)`` after the dataset has been
    loaded so that ``train_hyper`` can resolve epoch / step counts.
    All downstream functions accept a single ``config: Config`` argument
    instead of separate ``hp`` + ``config`` dict pairs.
    """
    rng_seed: int
    load_params_path: str | bool
    data: dict
    net_kwargs: dict
    train_hyper: TrainHyperparams
    model_params: ModelParams
    optimizer_conf: OptimizerConf

    @contextmanager
    def as_pretrain(self):
        """Temporarily configure for k=0 single-point training.

        Sets ``cp_fix=[1]`` so that ``_setup_training_env`` derives ``k=0``
        naturally from ``config.model_params.k``.  Also sets
        ``curve_segment=False`` so single-point pretraining never uses
        segmented Bézier curves.  All masks, model construction, and eval
        flow correctly from config alone — no explicit k argument or
        post-hoc model patching needed.
        """
        original_cp_fix = self.model_params.cp_fix
        original_curve_segment = self.model_params.curve_segment
        self.model_params.cp_fix = [1]
        self.model_params.curve_segment = False
        try:
            yield self
        finally:
            self.model_params.cp_fix = original_cp_fix
            self.model_params.curve_segment = original_curve_segment

    @classmethod
    def from_dict(cls, d: dict, data: 'DataSplits') -> 'Config':
        """Construct a ``Config`` from a raw dict (e.g. wandb config) and loaded data.

        ``data`` is forwarded to ``TrainHyperparams.from_config_dict`` so that
        epoch / step counts can be derived from the training-set size.

        Also migrates ``weight_decay`` from ``model_params`` into the adamw
        optimizer kwargs so it is applied once (by the optimizer) rather than
        being duplicated inside the subspace model.
        """
        d = dict(d)  # shallow copy so we don't mutate the caller's dict
        model_params = ModelParams.from_config_dict(d['model_params'])
        optimizer_conf = OptimizerConf.from_config_dict(d['optimizer_conf'])
        # Migrate weight_decay: if the subspace model carries a non-zero weight
        # decay AND the optimizer is adamw, move it into the optimizer kwargs so
        # regularisation is applied exactly once.
        if model_params.weight_decay > 0.0 and optimizer_conf.name == 'adamw':
            optimizer_conf.kwargs['weight_decay'] = model_params.weight_decay
            model_params.weight_decay = 0.0

        # Validate curve-segment topology and LoRA mode at construction time
        cp_fix = model_params.cp_fix
        lora_params = model_params.lora_params
        if model_params.curve_segment:
            all_segments_fixed_endpoints = np.all(
                [model_params.curve_segment == i for i in np.diff(
                    np.argwhere(np.array(cp_fix) == 1).flatten())])
            assert all_segments_fixed_endpoints, (
                "In curve segment mode, fixed control points must be consecutive "
                "and include endpoints, e.g. [1,0,1,0,1] with deg 2.")
        if lora_params is not None and np.sum(cp_fix) < len(cp_fix):
            assert lora_params.lora_mode in ('A(t)B(t)', 'A(t)eB(t)'), (
                "Only A(t)B(t) or A(t)eB(t) LoRA mode makes sense for "
                "subspace curves with fixed control points")

        return cls(
            rng_seed=d['rng_seed'],
            load_params_path=d.get('load_params_path', False),
            data=dict(d.get('data', {})),
            net_kwargs=dict(d['net_kwargs']),
            train_hyper=TrainHyperparams.from_config_dict(d['train_hyper'], data),
            model_params=model_params,
            optimizer_conf=optimizer_conf,
        )

    def build_masks(self, params_tree):
        """Build curve / train / LoRA masks from *params_tree*.

        ``k`` is derived from ``self.model_params.k`` (i.e. ``len(cp_fix)-1``),
        so callers inside ``as_pretrain()`` automatically get the k=0 masks.

        Returns ``(curve_mask, train_mask, lora_mask)``.
        """
        mp = self.model_params
        lp = mp.lora_params

        curve_mask = mp.build_curve_mask(params_tree, mp.k)

        if self.optimizer_conf.freeze_other_params:
            train_mask = jax.tree.map(lambda x: False, params_tree)
            for f in mp.filter_masks:
                train_mask = filter_mask(train_mask, *f['keys'], as_curve=True,
                                         BoolschenOperator=eval(f['op']))
            if lp and lp.use_lora:
                for f in lp.filter_masks:
                    train_mask = filter_mask(train_mask, *f['keys'], as_curve=True,
                                             BoolschenOperator=eval(f['op']))
        else:
            train_mask = jax.tree.map(lambda x: True, params_tree)

        lora_mask = lp.build_lora_mask(params_tree) if lp else jax.tree.map(lambda x: False, params_tree)

        return curve_mask, train_mask, lora_mask


@dataclass
class TrainingEnv:
    """Reusable training environment — static attributes only (no params/opt_state)."""
    s_model: SubspaceBaseModel
    optimizer: optax.GradientTransformation
    t_sample_fn: object
    rho_scheduler: object
    valid_metrics_fn: object
    empty_metric: tuple
    get_best_params: object


# jax.config.update("jax_log_compiles", True)

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
# Evaluation pipeline
# ---------------------------------------------------------------------------

def _compute_posterior_weights(log_like, temperature):
    """Softmax posterior weights at each temperature.

    Args:
        log_like: Per-sample log-likelihood, shape ``(n_samples,)``.
        temperature: Sequence of temperature scalars.

    Returns:
        Array of shape ``(len(temperature), n_samples)``.
    """
    def _single(log_like, t):
        log_post = log_like / t
        log_post -= jax.scipy.special.logsumexp(log_post)
        return jnp.exp(log_post)
    return jax.vmap(_single, in_axes=(None, 0))(log_like, jnp.array(temperature))


def run_evaluation(*, rng_key, env: 'TrainingEnv', params,
                   data: 'DataSplits', config: 'Config',
                   logger, logger_prefix='', artifact_weights=None):
    """Evaluate model and log metrics to wandb.

    k == 0:  Compute test logits, report test ll / acc / ece.
    k > 0:
        combined mode — compute BMA weights on train, plot train curve,
            optimise temperature on val (if available).
        Compute test logits (once).
        Plot test curve (unless per_leave).
        Report test metrics (weighted in combined mode, uniform otherwise).
    """
    # Disable LoRA noise during evaluation
    lora_rho, lora_rho_s = config.model_params.lora_rho_values
    if lora_rho > 0.0 or lora_rho_s > 0.0:
        params = env.s_model.set_lora_rho(rho_s=0.0, rho_w=0.0, params=params)

    s_model = env.s_model
    t_sample_fn = env.t_sample_fn
    k = config.model_params.k
    batch_size_eval = config.train_hyper.batch_size_eval
    temperature = config.train_hyper.temperature
    smoke_test = config.train_hyper.smoke_test
    curve_sampling_mode = config.model_params.curve_sampling_mode
    t_max = getattr(s_model, 't_max', 1.0)
    n_samples = 3 if (smoke_test or k == 0) else 20

    def _logits(input_ids, attention_mask, labels,
                n_samples, use_linspace=True, key_prefix=''):
        """Compute per-sample metrics and logits for one dataset split."""
        metrics_fn = setup_metrics(
            s_model, max(batch_size_eval, 1), input_ids, attention_mask, labels,
            n_samples=n_samples, use_linspace=use_linspace, average=False,
            key_prefix=key_prefix,
            t_sample_fn=None if use_linspace else t_sample_fn)
        return metrics_fn(rng_key, params)

    # ---- k == 0: simple single-point evaluation ----
    if k == 0:
        test_ids, test_mask, test_labels = data.get('test')
        print("Evaluating test set (k=0)")
        _, logits = _logits(test_ids, test_mask, test_labels,
                            n_samples=n_samples, key_prefix='final_')
        pp = post_pred_performance(logits, test_labels)
        logger.summary.update({
            f'{logger_prefix}final_{mk}': v for mk, v in pp.items()})
        return {}

    # ---- k > 0: curve evaluation ----
    t_space = jnp.linspace(0, t_max, n_samples)
    best_weight = False          # stays False → uniform BMA
    weights_all = None

    if curve_sampling_mode == 'combined':
        # -- Compute BMA weights on train set --
        if data.has_train:
            print("Computing posterior weights on train set")
            train_ids, train_mask, train_labels = data.get('train')
            train_metrics, train_logits = _logits(
                train_ids, train_mask, train_labels,
                n_samples=n_samples, key_prefix='f_train_')

            train_pp = post_pred_performance(train_logits, train_labels)
            logger.summary.update({
                f'{logger_prefix}f_train_{mk}': v
                for mk, v in train_pp.items()})

            # Plot train curve
            fig = _plot_curve_along_t(
                t_space, train_metrics['f_train_loss'],
                train_metrics['f_train_acc'], train_metrics['f_train_ece'],
                title='Final trainset')
            logger.log({'Curve Predictive Train': wandb.Image(fig)})
            plt.close(fig)

            # Posterior weights
            log_like = -train_metrics['f_train_loss']
            weights_all = _compute_posterior_weights(log_like, temperature)

            # Save weights artifact
            art_w = wandb.Artifact(name="bma_weights", type="npz")
            jnp.savez(f'tmp_files/{logger.id}_weights.npz',
                      weights=weights_all, t_space=t_space,
                      log_like=log_like, temperature=temperature)
            art_w.add_file(f'tmp_files/{logger.id}_weights.npz')
            logger.log_artifact(art_w)
            print("Saved posterior weights artifact")

        elif artifact_weights is not None:
            for f in artifact_weights.files():
                if 'weights' in f.name:
                    wf = np.load(
                        artifact_weights.get_entry(f.name).download(),
                        allow_pickle=True)
                    weights_all = _compute_posterior_weights(
                        wf['log_like'], temperature)
                    print("Loaded posterior weights from artifact")
                    break
            else:
                raise RuntimeError("No weights file found in artifact")

        # -- Optimise temperature on val set --
        if weights_all is not None and data.has_val:
            val_ids, val_mask, val_labels = data.get('val')
            print("Optimising temperature on validation set")
            _, val_logits = _logits(
                val_ids, val_mask, val_labels,
                n_samples=n_samples, key_prefix='final_valid_')

            best_ll, best_temp = -jnp.inf, None
            for weight, temp in zip(weights_all, temperature):
                pp = post_pred_performance(val_logits, val_labels, weight)
                logger.summary.update({
                    f'{logger_prefix}final_valid_ppd_t{temp}_{mk}': v
                    for mk, v in pp.items()})
                if pp['ll'] > best_ll:
                    best_ll, best_temp, best_weight = pp['ll'], temp, weight
            logger.summary.update({
                f'{logger_prefix}best_val_log_like_temp': best_temp,
                f'{logger_prefix}best_val_log_like': best_ll,
            })
            print(f"Best temperature: {best_temp}")

    # -- Compute test logits (once) --
    use_linspace = (curve_sampling_mode != 'per_leave')
    key_prefix = 'final_' if use_linspace else 'approx_'
    print(f"Evaluating test set ({'linspace' if use_linspace else 't_sample'}, "
          f"{n_samples} samples)")
    test_ids, test_mask, test_labels = data.get('test')
    test_metrics, test_logits = _logits(
        test_ids, test_mask, test_labels,
        n_samples=n_samples, use_linspace=use_linspace,
        key_prefix=key_prefix)

    # -- Plot test performance along curve (not per_leave) --
    if use_linspace:
        fig = _plot_curve_along_t(
            t_space, test_metrics[key_prefix + 'loss'],
            test_metrics[key_prefix + 'acc'],
            test_metrics[key_prefix + 'ece'],
            title='Final testset')
        logger.log({'Curve Predictive': wandb.Image(fig)})
        plt.close(fig)

    # -- Report test performance --
    # Always log uniform-BMA test metrics
    pp = post_pred_performance(test_logits, test_labels)
    logger.summary.update({
        f'{logger_prefix}{key_prefix}{mk}': v for mk, v in pp.items()})

    # Weighted metrics (combined mode with posterior weights)
    if curve_sampling_mode == 'combined' and weights_all is not None:
        if isinstance(best_weight, jnp.ndarray):
            # Best temperature was found on val — use it
            pp_w = post_pred_performance(test_logits, test_labels, best_weight)
            logger.summary.update({
                f'{logger_prefix}post_pred_test_ppd_{mk}': v
                for mk, v in pp_w.items()})
            mi = mutual_information(
                test_logits.astype(jnp.float32), weights=best_weight)
            me = mean_entropy(
                test_logits.astype(jnp.float32), weights=best_weight)
            logger.summary.update({
                f'{logger_prefix}test_mutual_information_ppd': mi,
                f'{logger_prefix}test_mean_entropy_ppd': me,
            })
        else:
            # No val set — report per temperature
            for weight, temp in zip(weights_all, temperature):
                pp_w = post_pred_performance(
                    test_logits, test_labels, weight)
                logger.summary.update({
                    f'{logger_prefix}post_pred_test_ppd_t{temp}_{mk}': v
                    for mk, v in pp_w.items()})
                mi = mutual_information(
                    test_logits.astype(jnp.float32), weights=weight)
                me = mean_entropy(
                    test_logits.astype(jnp.float32), weights=weight)
                logger.summary.update({
                    f'{logger_prefix}test_mutual_information_ppd_t{temp}': mi,
                    f'{logger_prefix}test_mean_entropy_ppd_t{temp}': me,
                })

    # Uncertainty metrics (always for k > 0)
    mi = mutual_information(test_logits.astype(jnp.float32))
    me = mean_entropy(test_logits.astype(jnp.float32))
    entrop = entropy(test_logits.astype(jnp.float32))
    logger.summary.update({
        f'{logger_prefix}test_mutual_information': mi,
        f'{logger_prefix}test_mean_entropy': me,
        f'{logger_prefix}std_entropy_along_curve': entrop.mean(-1).std(),
        f'{logger_prefix}mean_entropy_along_curve': entrop.mean(),
    })

    return {}


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


def _init_subspace_params(config:Config, s_model, init_key, fixed_cps_train_params=False):
    """Initialize parameters on an existing subspace model.

    Calls ``init_params_from_point``, updates the train mask for LoRA,
    and maps any fixed control-point weights into the parameter vector.

    Returns:
        params (dict, updated in-place)
    """
    lora_params = config.model_params.lora_params
    params = unfreeze(s_model.model.init(None, None)) # base parameters loaded from file

    curve_mask, train_mask, lora_mask = config.build_masks(params['params'])

    params = s_model.init_params_from_point(
        init_key, params, curve_mask,
        noise_fn=lambda p, key: p + config.model_params.jitter_multiplier * random.normal(key, p.shape, dtype=p.dtype),
        lora_mask=lora_mask)

    # set train mask according to config
    # Only update mask for LoRA-capable subspaces
    if lora_params and lora_params.use_lora:
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


def _init_optimizer(config:Config, curve_mask=None, curve_params=None):
    """Create the optimizer chain (gradient clipping, CP freezing, base optimizer).

    Args:
        optimizer_conf: ``OptimizerConf`` dataclass (read directly, no dict conversion).
        s_model: Subspace model instance (provides ``curve_mask``).
        params: Current parameter pytree.
        cp_fix: ``config.model_params.cp_fix`` for the curve phase, or ``None``
            during k=0 pretraining (no CP-freeze mask needed).

    Returns:
        Configured ``optimizer``.
    """
    optimizer_conf = config.optimizer_conf
    mp = config.model_params
    if optimizer_conf.name.startswith('sam'):
        raise NotImplementedError("SAM optimizer is not yet implemented for text classification")

    base_optimizer = getattr(optax, optimizer_conf.name)(**optimizer_conf.kwargs)

    opt_chain_list = []
    if optimizer_conf.grad_clip_norm is not None:
        opt_chain_list.append(optax.clip_by_global_norm(optimizer_conf.grad_clip_norm))

    # Freeze already-trained control points during curve fitting
    if mp.k > 1 and any(mp.cp_fix) and curve_mask is not None:
        def get_dims(param):
            return (-1,) + (1,) * (param.ndim - 1)
        cps_fixed = jnp.array(mp.cp_fix, dtype=bool)
        fixed_cp_mask = jax.tree.map(
            lambda m, p: cps_fixed.reshape(get_dims(p)) if m else False,
            curve_mask, curve_params)
        opt_chain_list.append(_element_wise_mask(fixed_cp_mask))

    opt_chain_list.append(base_optimizer)
    optimizer = optax.chain(*opt_chain_list)
    return optimizer

def _make_train_batch_fn(env: 'TrainingEnv', config: 'Config', *,
                          train_input_ids, train_attention_mask, train_labels,
                          initial_cp_center):
    """Return a ``train_batch(carry, batch_idx)`` closure for ``jax.lax.scan``."""
    s_model = env.s_model
    optimizer = env.optimizer
    lr_schedule = config.optimizer_conf.lr_schedule
    t_sample_fn = env.t_sample_fn
    rho_scheduler = env.rho_scheduler
    valid_metrics_fn = env.valid_metrics_fn
    empty_metric = env.empty_metric
    get_best_params = env.get_best_params
    eval_every_n_batch = config.train_hyper.eval_every_n_batch
    k = s_model.k
    lora_rho, lora_rho_s = config.model_params.lora_rho_values

    def train_batch(carry, batch_idx):
        rng_key, params, opt_state, best_val_ll, best_params, update_idx = carry

        x_batch = jnp.take(train_input_ids, batch_idx, axis=0)
        atten_batch = jnp.take(train_attention_mask, batch_idx, axis=0)
        y_ = jnp.take(train_labels, batch_idx, axis=0)

        rng_key, subkey = random.split(rng_key)
        t = t_sample_fn(subkey)
        if lora_rho > 0.0 or lora_rho_s > 0.0:
            current_rho = rho_scheduler(update_idx, lora_rho)
            current_rho_s = rho_scheduler(update_idx, lora_rho_s)
            params = s_model.set_lora_rho(current_rho, current_rho_s, params)
        loss, params, opt_state, (grad_, logs) = s_model.train_step(
            subkey, t, params, (x_batch, atten_batch), y_, opt_state, optimizer)
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


def _make_train_epoch_fn(env: 'TrainingEnv', config: 'Config', *,
                          train_input_ids, train_attention_mask, train_labels,
                          initial_cp_center):
    """Return a ``train_epoch(carry, epoch)`` closure for ``jax.lax.scan``.

    Internally creates the per-batch function via ``_make_train_batch_fn``.
    """
    hp = config.train_hyper
    train_batch_fn = _make_train_batch_fn(
        env, config,
        train_input_ids=train_input_ids,
        train_attention_mask=train_attention_mask,
        train_labels=train_labels,
        initial_cp_center=initial_cp_center)
    n_train_samples = train_labels.shape[0]
    batch_size = hp.batch_size
    num_epochs = hp.num_epochs

    def _inf_mean(x):
        mask = jnp.isfinite(x)
        return jnp.where(mask, x, 0).sum() / mask.sum()

    @scan_tqdm(num_epochs)
    def train_epoch(carry, epoch):
        rng_key = carry[0]
        rng_key, subkey = random.split(rng_key)
        shuffel_idx = random.permutation(subkey, n_train_samples)
        shuffel_idx = es('(ib)->ib', shuffel_idx, b=batch_size)

        # train on batches
        carry, metrics = jax.lax.scan(train_batch_fn, carry, shuffel_idx)
        val_loss_ = _inf_mean(metrics['val_loss'])
        val_acc_ = _inf_mean(metrics['val_acc'])
        jax.debug.print("Epoch {}/{}: Loss: {}, Val Loss: {}, Val Acc: {}",
                        epoch + 1, num_epochs, metrics['loss'][-1], val_loss_, val_acc_)
        return carry, metrics

    return train_epoch


def _make_expert_step_fn(env: 'TrainingEnv', config: 'Config', *,
                          window_size):
    """Return a ``_train_expert_step(carry, step)`` closure for ``jax.lax.scan``."""
    s_model = env.s_model
    optimizer = env.optimizer
    lr_schedule = config.optimizer_conf.lr_schedule
    t_sample_fn = env.t_sample_fn
    valid_metrics_fn = env.valid_metrics_fn
    empty_metric = env.empty_metric
    get_best_params = env.get_best_params
    eval_every_n_batch = config.train_hyper.eval_every_n_batch
    batch_size = config.train_hyper.batch_size
    num_steps = config.train_hyper.num_steps

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
# Expert data handling (embedding computation, sorting)
# ---------------------------------------------------------------------------

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


def _compute_expert_embeddings(model:QwenTextClassificationWrapper, data, config: 'Config'):
    """Compute cosine-similarity matrix for expert sorting.

    Callers must already guard on ``ds_sampling == 'expert'`` and a
    non-Random ``sort_mode``; this function always computes embeddings.
    Hold params memory twice perhaps a problem.
    """
    params = model.init(None, None)
    hp = config.train_hyper
    print("compute embeddings for expert sort")
    train_input_ids, train_attention_mask, _ = data.get('train')
    embedding = get_embedding(
        train_input_ids, train_attention_mask,
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


def _sort_data_for_expert(rng_key, data: 'DataSplits', config: 'Config', model):
    """Sort training data according to expert sampling configuration.

    Handles ``'Random'``, ``'Hierarchical'``, ``'Spectral'``, and
    ``'First_data'`` sort modes.  Returns the (possibly reordered)
    training arrays.

    Returns:
        ``(train_input_ids, train_attention_mask, train_labels)``
    """
    train_input_ids, train_attention_mask, train_labels = data.get('train')
    sort_mode = config.train_hyper.dataset_sampling['expert'].get('sort_mode', False)

    if sort_mode == 'Random':
        order = random.permutation(rng_key, train_input_ids.shape[0])
    elif sort_mode:
        cosine_sim = _compute_expert_embeddings(model, data, config)
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
        print(f"Data sorted for expert sampling using {sort_mode} method")
    else:
        return train_input_ids, train_attention_mask, train_labels

    return train_input_ids[order], train_attention_mask[order], train_labels[order]


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


def _setup_validation(s_model, data: 'DataSplits', batch_size_eval, k):
    """Set up validation metrics function and best-parameter tracking.

    Uses ``data.val_*`` arrays; returns no-op stubs when ``data.has_val`` is False.

    Returns:
        ``(valid_metrics_fn, empty_metric, get_best_params)``
    """
    empty_metric = _empty_val_metric()

    if data.has_val:
        print("save best parameters")
        val_input_ids, val_attention_mask, val_labels = data.get('val')
        valid_metrics_fn_inner = setup_metrics(
            s_model, batch_size_eval,
            val_input_ids, val_attention_mask, val_labels,
            n_samples=10 if k > 0 else 1, key_prefix='val_')

        def valid_metrics_fn(rng, params):
            if isinstance(s_model, LoraMixin):
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
    if lora_params and lora_params.rho_scheduler_frequency > 0.0:
        frequency = lora_params.rho_scheduler_frequency
        base_frequency = num_steps / (2 * frequency)
        print(f"Using cosine rho schedule with total steps {num_steps}")

        def rho_scheduler(x, base):
            return base * (1 - jnp.cos(x / base_frequency * jnp.pi))
        return rho_scheduler
    else:
        return lambda step, base: base


def _setup_training_env(config: Config, data):    
    """Prepare the training environment for a curve or single-point run.

    This function instantiates the model wrapper and the chosen subspace
    model, configures the t-sampling function and LoRA rho scheduler,
    prepares validation helpers, and builds the optimizer. The returned
    `TrainingEnv` bundles the subspace model, optimizer and the helper
    callables required during training and evaluation.

    Args:
        config (Config): Experiment configuration containing model and
            training parameters (including `model_params` and optimizer
            settings).
        data (DataSplits): Dataset splits used to configure validation
            behavior (may be consulted when building validation functions).
        curve_mask: Optional pytree mask selecting curve control-point
            parameters (passed to the optimizer builder when applicable).
        curve_params: Optional pytree of curve parameters (used for
            dimension inference when constructing parameter masks).

    Returns:
        TrainingEnv: Container with attributes `s_model`, `optimizer`,
            `t_sample_fn`, `rho_scheduler`, `valid_metrics_fn`,
            `empty_metric`, and `get_best_params` used by the training
            and evaluation loops.
    """
    mp = config.model_params

    model = QwenTextClassificationWrapper(**config.net_kwargs)
    # Instantiate the subspace model
    logging.info("Using subspace model: %s", mp.subspace_model)
    s_model_cls = get_subspace_model(mp.subspace_model)
    s_model = s_model_cls(**mp.get_model_kwargs(model=model))
    
    # t sampling, rho scheduler, validation
    t_sample_fn = _setup_t_sample_fn(s_model, mp.curve_sampling_mode)
    rho_scheduler = _setup_rho_scheduler(mp.lora_params, config.train_hyper.num_steps)
    valid_metrics_fn, empty_metric, get_best_params = _setup_validation(
                s_model, data, config.train_hyper.batch_size_eval, k=mp.k)

    optimizer = _init_optimizer(config)
    
    env = TrainingEnv(
        s_model=s_model, optimizer=optimizer,
        t_sample_fn=t_sample_fn, rho_scheduler=rho_scheduler,
        valid_metrics_fn=valid_metrics_fn, empty_metric=empty_metric,
        get_best_params=get_best_params)
    return env


def run_training(rng_key, env, params, data, config: 'Config', logger, logger_prefix):
    """Execute one training phase (expert or minibatch) and log metrics.

    Returns:
        ``(rng_key, params, last_trainable_params)``
    """
    hp = config.train_hyper
    s_model = env.s_model

    only_train = s_model.only_trainable_params(params['params'])
    opt_state = env.optimizer.init(only_train)
    n_trainable = sum(p.size for p in jax.tree.flatten(only_train)[0])
    print(f"Number of trainable parameters: {n_trainable:_}")

    if hp.num_steps <= 0:
        return rng_key, params, None

    # Initialize best-parameter tracking
    best_val_ll = -np.inf
    best_params = s_model.only_trainable_params(params['params']) if data.has_val else None

    # Initial CP center for curve metrics
    cp_w = masked_pytree_to_matrix(params['params'], s_model.curve_mask, s_model.k)
    initial_cp_center = cp_w.mean(axis=0)

    if hp.ds_sampling == 'expert':
        print(f"Start sorted expert training for {hp.num_steps}")
        window_size = hp.dataset_sampling['expert']['window_size']
        window_size = max(hp.batch_size, window_size)
        train_step = _make_expert_step_fn(env, config, window_size=window_size)
        carry = (rng_key, params, opt_state, best_val_ll, best_params, data.get('train'))
        scan_length = hp.num_steps

    elif hp.ds_sampling == 'minibatch':
        print(f"Start minibatch training for {hp.num_epochs} epochs")
        train_input_ids, train_attention_mask, train_labels = data.get('train')
        train_step = _make_train_epoch_fn(
            env, config,
            train_input_ids=train_input_ids,
            train_attention_mask=train_attention_mask,
            train_labels=train_labels,
            initial_cp_center=initial_cp_center)

        carry = (rng_key, params, opt_state, best_val_ll, best_params, 0)
        scan_length = hp.num_epochs
    else:
        raise ValueError(f"Unknown dataset_sampling method {hp.ds_sampling}. Choose 'expert' or 'minibatch'.")

    # run training loop and collect metrics
    carry, metrics = jax.lax.scan(train_step, carry, jnp.arange(scan_length))

    # Clean up inf values from lazy validation
    for key, value in metrics.items():
        if key.startswith('val_'):
            metrics[key] = jnp.array(value).flatten()[::hp.eval_every_n_batch].repeat(
                hp.eval_every_n_batch, axis=0)[:hp.num_steps]
        else:
            metrics[key] = jnp.array(value).flatten()

    # Log all metrics
    _ = [logger.log(dict(zip(metrics.keys(), m))) for m in zip(*metrics.values())]

    # Log metrics as artifact
    batches_per_epoch = hp.num_steps // max(hp.num_epochs, 1)
    epochs = np.repeat(np.arange(hp.num_epochs), batches_per_epoch)
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

    if s_model.k > 0:
        cp = masked_pytree_to_matrix(params['params'], s_model.curve_mask, s_model.k)
        length = _bezier_length(cp)
        logger.summary.update({'Curve length': length})

    return rng_key, params, last_trainable_params


# ---------------------------------------------------------------------------
# Pipeline stage functions – called by train() orchestrator
# ---------------------------------------------------------------------------

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
    ordered_ids = input_ids[perm_idx]
    ordered_mask = attention_mask[perm_idx]
    ordered_labels = labels[perm_idx]

    n_train = int(np.floor((1. - val_percentage) * len(input_ids) / batch_size) * batch_size)
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


def _pretrain_fixed_cps(rng_key, config: 'Config', data, logger, artifact):
    """Train each fixed control point independently (k=0).

    Uses ``config.as_pretrain()`` to set ``cp_fix=[1]`` and
    ``curve_segment=False`` temporarily, so ``_setup_training_env`` derives
    k=0 and disables segmented curves naturally — no separate k argument
    or post-hoc model patching needed.

    Returns:
        ``(train_params, rng_key)``
    """
    cp_fix = config.model_params.cp_fix  # save original before context

    train_params = []
    print("Pretraining fixed control points")

    with config.as_pretrain() as pt_config:
        env = _setup_training_env(config=pt_config, data=data)

        for i, fixed in enumerate(cp_fix):
            if fixed:
                print(f"Training cp member {i}")
                rng_key, init_key = random.split(rng_key)
                params = _init_subspace_params(
                    pt_config, env.s_model, init_key)

                rng_key, params, _ = run_training(
                    rng_key, env, params, data, pt_config, logger,
                    f'cp_fixed_{i}_')

                run_evaluation(
                    rng_key=rng_key, env=env, params=params,
                    data=data, config=pt_config,
                    logger=logger, logger_prefix=f'cp_fixed_{i}_')

                train_params.append(
                    env.s_model.only_trainable_params(params['params']))
            else:
                train_params.append(False)

    if config.train_hyper.save_params:
        print("Save pretrained trainable params ...")
        np.save(f'tmp_files/{logger.id}_pretrained_params.npy', train_params)
        time.sleep(1)
        print("Saved pretrained parameters to tmp_files")
        artifact.add_file(f'tmp_files/{logger.id}_pretrained_params.npy')

    return train_params, rng_key


def _train_full_curve(rng_key, config: 'Config', data, train_params, logger, artifact):
    """Fit the full Bézier curve (k > 0), evaluate, and save parameters.

    Initialises the subspace model for the full curve phase (``k > 0``),
    brings in any pretrained fixed control-point weights, rebuilds the
    optimizer so that the CP-freeze mask covers the now-known parameter
    structure, runs the training loop (or skips it when all CPs are fixed),
    evaluates, and serialises the final parameters as wandb artifacts.

    Args:
        rng_key: JAX random key.
        config (Config): Full experiment configuration.  ``k`` and ``cp_fix``
            are read from ``config.model_params``.
        data (DataSplits): Train / val / test dataset splits.
        train_params: List of pretrained parameter pytrees for each control
            point (``False`` for free / untrained CPs), as returned by
            ``_pretrain_fixed_cps``.
        logger: wandb run object used for metric and artifact logging.
        artifact: wandb ``Artifact`` to which final parameter files are
            attached.
    """
    cp_fix = config.model_params.cp_fix
    all_fixed = len(cp_fix) == np.sum(cp_fix)

    print("Training curve model")
    if np.sum(cp_fix) == 0:
        print("No fixed control points, training full curve with all control points free.")
        if config.model_params.curve_segment:
            assert config.model_params.indepent_connected is True, (
                "In curve segment mode with no fixed control points, curve "
                "optimization only makes sense if all segments are "
                "independently connected.")

    # Setup training environment — k is derived from config.model_params.k
    env = _setup_training_env(config=config, data=data)

    # Initialize subspace parameters from pretrained control-point params
    rng_key, init_key = random.split(rng_key)
    params = _init_subspace_params(config, env.s_model, init_key, fixed_cps_train_params=train_params)

    # Rebuild optimizer with curve mask/params so CP-freeze mask is applied
    env.optimizer = _init_optimizer(config, curve_mask=env.s_model.curve_mask, curve_params=params['params'])

    # Log parameter stats
    flat_params = jax.tree.flatten(params)[0]
    n_params = sum(p.size if isinstance(p, jnp.ndarray) else 0 for p in flat_params)
    param_memory = sum(
        p.size * p.dtype.itemsize if isinstance(p, jnp.ndarray) else 0
        for p in flat_params)
    n_trainable = sum(p.size for p in jax.tree.flatten(
        env.s_model.only_trainable_params(params['params']))[0])
    logger.summary.update({
        'n_params': n_params,
        'memory_params': param_memory,
        'n_params_trainable': n_trainable,
    })
    print(f"Number of parameters: {n_params:_}\n"
          f"Memory of parameters: {param_memory / (1024**2):_}MB")

    # Train (unless all CPs are fixed → eval-only)
    if not all_fixed:
        rng_key, params, last_trainable_params = run_training(
            rng_key, env, params, data, config, logger, '')
    else:
        last_trainable_params = None

    # Evaluate
    run_evaluation(
        rng_key=rng_key, env=env, params=params,
        data=data, config=config,
        logger=logger)

    # Save curve params
    if config.train_hyper.save_params:
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
    config_dict = dict(wandb.config)
    train(logger, config_dict)


def train(logger, config_dict):
    """Main training orchestrator for Qwen curve fine-tuning."""
    config_dict = dict(config_dict)  # shallow copy so we can mutate
    rng_key = random.PRNGKey(config_dict['rng_seed'])
    artifact = wandb.Artifact(name="params", type="pytree")

    # Resolve model path from wandb if a local path is not given
    net_kwargs = dict(config_dict['net_kwargs'])
    if net_kwargs['model_path'].startswith('ddold/'):
        model_artifact = logger.use_artifact(net_kwargs['model_path'], type='model')
        net_kwargs['model_path'] = model_artifact.download()
    config_dict['net_kwargs'] = net_kwargs

    data, unique_target_ids, rng_key = _prepare_datasets(
        rng_key,
        dataset_path=config_dict['data']['dataset_path'],
        val_percentage=config_dict['data'].get('val_percentage', 0.0),
        batch_size=config_dict['train_hyper']['batch_size'],
        smoke_test=config_dict['train_hyper'].get('smoke_test', False),
        logger=logger)

    config_dict['net_kwargs']['target_token_ids'] = unique_target_ids
    config = Config.from_dict(config_dict, data)

    # sort data for expert sampling, if needed
    if config.train_hyper.ds_sampling == 'expert':
        base_model = QwenTextClassificationWrapper(**config.net_kwargs)
        train_input_ids, train_attention_mask, train_labels = \
            _sort_data_for_expert(rng_key, data, config, base_model)
        data = replace(data,
                       train_input_ids=train_input_ids,
                       train_attention_mask=train_attention_mask,
                       train_labels=train_labels)

    pretrained_params = [False] * len(config.model_params.cp_fix)
    if any(config.model_params.cp_fix):
        pretrained_params, rng_key = _pretrain_fixed_cps(
            rng_key, config, data, logger, artifact)

    if len(config.model_params.cp_fix) > 1:
        _train_full_curve(rng_key, config, data, pretrained_params, logger, artifact)

    if config.train_hyper.save_params:
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
            'data': {
                'dataset_path': "ddold/qwen_PtoC/winogrande_m_dataset:v0",
                # 'dataset_path': "ddold/qwen_PtoC/ARC-Easy_dataset:v0",
                # 'dataset_path': "ddold/qwen_PtoC/MMLU_chem_dataset:v0",
                # 'dataset_path': "ddold/qwen_PtoC/obqa_dataset:v0",
                'val_percentage': 0.1,
            },

            # --- Base model (QwenTextClassificationWrapper) ------------------------
            # 'target_token_ids' is injected automatically after dataset load
            'net_kwargs': {'model_path': "artifacts/qwen2.5_7B_bfloat16:v0"},
            # 'net_kwargs': {'model_path': "ddold/qwen_PtoC/qwen2.5_7B_bfloat16:v0"},

            # --- Training hyperparameters → TrainHyperparams ----------------------
            # These control *how* training runs, not *what* is built.
            'train_hyper': {
                'batch_size': args.batch_size,
                'num_epochs': -1,             # -1 means use num_steps instead
                'num_steps': 10 if args.smoke_test else 5_000,
                'eval_every_n_batch': 20,
                'temperature': [1.0],
                'dataset_sampling': {'minibatch': {}},
                # 'dataset_sampling': {'expert': {'window_size': 4, 'sort_mode': False}},
                'save_params': True,
                'smoke_test': args.smoke_test,
            },

            # --- Subspace model + LoRA architecture → ModelParams -----------------
            # All parameters that determine *what* the subspace model is.
            'model_params': {
                'cp_fix': args.cp_fix,        # e.g. [1,0,1]: 1=fixed, 0=trainable CP
                'curve_sampling_mode': 'combined_noBMA',  # 'per_leave' | 'combined' | 'combined_noBMA'
                'subspace_model': 'lora_category',
                'jitter_multiplier': 0.,      # weight-init noise scale
                'natural_parameterization': False,
                'weight_decay': 0.,
                'curve_parameterization': 'bezier',
                'curve_segment': 3,           # int → segmented degree; False → full curve
                'indepent_connected': False,  # True → independently connected segments
                # 'gravity': 5.0,
                # 'energy_weakening': 4.,

                # curve topology: which parameters become the Bézier curve
                'filter_masks': [
                    {'keys': ["self_attn", "q_proj", "kernel"], 'op': 'all'},
                    {'keys': ["self_attn", "v_proj", "kernel"], 'op': 'all'},
                    {'keys': ["lm_head", "kernel"], 'op': 'all'},
                ],

                # LoRA sub-config (architecture + noise scheduling)
                'lora_params': {
                    'use_lora': True,
                    'r': 8,
                    'lora_dtype': 'float32',
                    'lora_alpha': 16.0,
                    'lora_mode': 'A(t)B(t)',
                    # 'lora_mode': 'Arotsd(t)eB',
                    # 'lora_mode': 'Arot(t)sdeBrot(t)',
                    'rho_scheduler_frequency': 100.,
                    'lora_rho': .25,
                    'lora_rho_s': .5,
                    'filter_masks': [
                        {'keys': ["self_attn", "q_proj", "kernel"], 'op': 'all', 'dims': [1, 0]},
                        {'keys': ["self_attn", "v_proj", "kernel"], 'op': 'all', 'dims': [1, 0]},
                        {'keys': ["lm_head", "kernel"], 'op': 'all', 'dims': [1, 0]},
                    ],
                },
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
        }
        logger = wandb.init(project="qwen_PtoC", name="test_more degree", entity="ddold", config=config)
        train(logger, config)

