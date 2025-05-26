from numpyro import distributions as dist
import numpyro
import time
import wandb
from numpyro import handlers
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from jax import random, jit, grad
import jax
from jax import random
import jax.numpy as jnp
import optax
import arviz as az
from numpyro.diagnostics import hpdi
from copy import deepcopy
from src.subspace_sampling import setup_inference_chain
from src.subspace_curve import pytree_to_matrix, vec_to_pytree, bezier_coeff_fn, bezier_curve
from src.subspace_curve import OrthoSpan, CategorySubspaceBatchNorm, RepulsiveCategorySubspaceBatchNorm
from src.models import ResNet, ResNetBlock, init_t_lambda_to_phi
import blackjax
from numpyro.infer.util import initialize_model
from jax_tqdm import scan_tqdm
import os
import torchvision
from torch.utils.data import random_split, TensorDataset
import torch
from flax import linen as nn
from src.weight_matching import resnet20_permutation_spec2
import argparse
import src.models


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=10"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# jax.config.update('jax_platform_name', 'cpu')
# Assuming equivalent JAX implementations


def init_model_tube(model, params, k, t_lambda_to_phi, t_phi_to_weight, prior_correct: bool, prior_scale=0.5, d_bezier=None, log_norm_d_bezier=jnp.array(0.)):
    def model_tube(x, y=None, temperature=1.):
        # prior definition
        t = numpyro.sample(
            "t", dist.Uniform(-0.1, 1.1).expand((1,)).to_event(1))
        tube = numpyro.sample("tube", dist.Normal(
            0., prior_scale).expand((k-1,)).to_event(1))

        lambda_ = jnp.concat([t, tube], axis=-1)
        space_point = t_lambda_to_phi(lambda_)

        if prior_correct:
            # jacobian = jax.jacrev(transform)(jax.lax.stop_gradient(lambda_)).squeeze()
            jacobian = jax.jacrev(t_lambda_to_phi)(lambda_)
            sign, logabsdet = jnp.linalg.slogdet(jacobian)
            numpyro.factor("logabsdet", logabsdet)
        else:
            t_det = jnp.linalg.norm(d_bezier(t), axis=-1)
            numpyro.factor("t_scale", jnp.log(t_det) - log_norm_d_bezier)

        varphi = numpyro.deterministic('varphi', space_point)
        weight_vec = t_phi_to_weight(varphi)

        def apply_model(weight_vec, x):
            weight_pytree = vec_to_pytree(weight_vec, params)
            out = model.apply({'params': weight_pytree}, x)
            return out
        logit = apply_model(weight_vec, x)
        with numpyro.plate("data", x.shape[0], dim=-1):
            with handlers.scale(scale=1/temperature):
                numpyro.sample("obs", dist.CategoricalLogits(logit), obs=y)
    return model_tube


def load_cifar10(seed=0):
    PATH_DATASETS = "./datasets/cifar/cifar10_ds"
    os.makedirs(PATH_DATASETS, exist_ok=True)
    cifar10_normalization = torchvision.transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

    def cifar10_undo_normalization(image):
        mean = jnp.array([x / 255.0 for x in [125.3, 123.0, 113.9]])
        std = jnp.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
        return image * std + mean

    def split_dataset(dataset, val_split=0.2):
        """Splits the dataset into train and validation set."""
        len_dataset = len(dataset)
        splits = get_splits(len_dataset, val_split)
        dataset_train, dataset_val = random_split(
            dataset, splits, generator=torch.Generator().manual_seed(seed))

        return dataset_train, dataset_val

    def get_splits(len_dataset, val_split):
        """Computes split lengths for train and validation set."""
        if isinstance(val_split, int):
            train_len = len_dataset - val_split
            splits = [train_len, val_split]
        elif isinstance(val_split, float):
            val_len = int(val_split * len_dataset)
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(val_split)}")

        return splits

    # train_transforms = torchvision.transforms.Compose(
    #     [
    #         # torchvision.transforms.RandomCrop(32, padding=4),
    #         # torchvision.transforms.RandomHorizontalFlip(),
    #         torchvision.transforms.ToTensor(),
    #         cifar10_normalization,
    #         torchvision.transforms.Lambda(lambda x: x.permute(1, 2, 0)),
    #     ]
    # )
    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            cifar10_normalization,
            torchvision.transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ]
    )

    # dataset_train = torchvision.datasets.CIFAR10(PATH_DATASETS, train=True,
    #                                 download=True, transform=train_transforms)
    dataset = torchvision.datasets.CIFAR10(PATH_DATASETS, train=True,
                                download=True, transform=test_transforms)
    dataset_train, dataset_val = split_dataset(dataset, val_split=0.02)
    dataset_test = torchvision.datasets.CIFAR10(PATH_DATASETS, train=False,
                                    download=True, transform=test_transforms)

    def dataset_to_tensordataset(dataset):
        with torch.no_grad():
            test1 = dataset[0]
            x_stack = torch.zeros(
                (len(dataset), *(test1[0].shape)), dtype=test1[0].dtype)
            if type(test1[1]) is int:
                y_stack = torch.zeros((len(dataset)), dtype=torch.int64)
            else:
                y_stack = torch.zeros(
                    (len(dataset), *(test1[1].shape)), dtype=test1[1].dtype)
            for i in range(len(dataset)):
                x_stack[i], y_stack[i] = dataset[i]
            return TensorDataset(x_stack, y_stack)


    dataset_train = dataset_to_tensordataset(dataset_train)
    dataset_val = dataset_to_tensordataset(dataset_val)
    dataset_test = dataset_to_tensordataset(dataset_test)

    x = jnp.array(dataset_train[:][0].numpy())
    y = jnp.array(dataset_train[:][1].numpy())
    x_val = jnp.array(dataset_val[:][0].numpy())
    y_val = jnp.array(dataset_val[:][1].numpy())
    x_test = jnp.array(dataset_test[:][0].numpy())
    y_test = jnp.array(dataset_test[:][1].numpy())
    
    return x, y, x_val, y_val, x_test, y_test, cifar10_undo_normalization


def augment(key, image):
    """Performs data augmentation."""
    # random crop
    image = jnp.pad(image, ((4, 4), (4, 4), (0, 0)), mode='edge')
    key, subkey = random.split(key)
    crop_x = random.randint(subkey, (), 0, 8)
    key, subkey = random.split(key)
    crop_y = random.randint(subkey, (), 0, 8)
    image = jax.lax.dynamic_slice_in_dim(image, crop_x, 32, 0)
    image = jax.lax.dynamic_slice_in_dim(image, crop_y, 32, 1)
    # image = image[crop_x:crop_x+32, crop_y:crop_y+32, :]
    # horizontal flip
    key, subkey = random.split(key)
    flip = random.bernoulli(subkey, 0.5)
    # image = jax.lax.cond(flip, image, lambda x: x, jnp.flip, image)
    image = jnp.where(flip, image, jnp.flip(image, axis=1))
    return key, image


def setup_metrics_curve(eval_model, batch_size, x, target, n_samples, use_linspace=False, average=True):
    if use_linspace:
        get_t = lambda key: jnp.linspace(0, 1., n_samples)
    else:
        get_t = lambda key: random.uniform(key, (n_samples,), minval=0., maxval=1.)

    def acc_fn(rng_key, params):
        x_ = x.reshape(-1, batch_size, 32, 32, 3)
        y_ = target.reshape(-1, batch_size)

        def pred(key, data):
            x_batch, y_batch = data
            
            def eval(t):
                logits, _ = eval_model.eval(
                    params['params'], params['batch_stats'], t, x_batch)  # params, batch_stats, t, x
                loss = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(
                    logits=logits, labels=y_batch))
                acc = jnp.mean(jnp.argmax(logits, axis=-1) == y_batch)
                return loss, acc  # mean over batch
            
            key, subkey = random.split(key)
            # draw 10 samples to approximate curve performance
            t = get_t(subkey)
            loss, acc = jax.vmap(eval)(t)
            if average:
                return key, (jnp.mean(loss), jnp.mean(acc))  # mean over samples
            else:
                return key, (loss, acc)  # no mean over samples
        # shape (n_data, draws, output_dim)
        rng_key, (loss, acc) = jax.lax.scan(pred, rng_key, (x_, y_))
        loss = jnp.mean(loss, axis=0)
        acc = jnp.mean(acc, axis=0)
        return loss, acc  # mean over epochs
    return acc_fn



def single_model_train(logger, config):
    '''' example config:
    config = {
        'rng_seed': 0,
        'num_epochs': 500,
        'batch_size': 100,
        'optimizer_conf': {'name': "optax.sgd", 'kwargs': {'learning_rate': 0.05, 'momentum': 0.9, 'nesterov': True}},
        'net_kwargs':{'num_blocks': (3, 3, 3),
                      'c_hidden': (16, 32, 64),
                      'num_classes': 10,
                      'act_fn': relu,
                      'block_class': ResNetBlock}}
    '''
    config['net_kwargs']['act_fn'] = getattr(nn, config['net_kwargs']['act_fn'])
    config['net_kwargs']['block_class'] = getattr(src.models, config['net_kwargs']['block_class'])
    rng_key = random.PRNGKey(config['rng_seed'])

    # Load data
    x, y, x_val, y_val, x_test, y_test, data_revert_norm_fn = load_cifar10(
        config['rng_seed'])  # get fold from seed
    

    net = ResNet(**config['net_kwargs'])
    rng_key, init_key = random.split(rng_key)
    params = net.init(init_key, x[0:2])
    
    def _predict(params, bn_params, inputs, train):
        # Predict logits from inputs and parameters
        all_params = {"params": params, "batch_stats": bn_params}

        def train_fn(inputs):
            logits, net_state = net.apply(
                all_params, inputs, train=True, mutable=["batch_stats"]
            )
            return logits, net_state

        def eval_fn(inputs):
            logits = net.apply(all_params, inputs, train=False, mutable=False)
            net_state = {"batch_stats": bn_params}
            return logits, net_state

        logits, net_state = jax.lax.cond(
            train, lambda x: train_fn(x), lambda x: eval_fn(x), inputs
        )
        return logits, net_state['batch_stats']


    def loss_fn(params, bn_params, inputs, targets, train:bool):
        def l2_loss(x, alpha):
            return alpha * (x ** 2).sum()
        
        logits, stats_params = _predict(params, bn_params, inputs, train)
        #   print(logits.shape, targets.shape)
        loss = jnp.mean(
            optax.losses.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=targets)
        )
        loss = jax.lax.cond(train, lambda x: x+sum(l2_loss(w, alpha=1e-4)
                    for w in jax.tree.leaves(params)), lambda x: x, loss)
        return loss, stats_params
    

    # Train
    optimizer_conf = config['optimizer_conf']
    if isinstance(optimizer_conf['kwargs']['learning_rate'], dict):
        lr_schedule = eval(optimizer_conf['kwargs']['learning_rate']['name'])(
            **optimizer_conf['kwargs']['learning_rate']['kwargs'])
    else:
        iter_per_epoch_train = len(x) // config['batch_size']
        lr_schedule = optax.linear_onecycle_schedule(
            config['num_epochs'] * iter_per_epoch_train, config['optimizer_conf']['kwargs']['learning_rate'])
    optimizer_conf['kwargs']['learning_rate'] = lr_schedule
    optimizer = eval(optimizer_conf['name'])(**optimizer_conf['kwargs'])
    opt_state = optimizer.init(params['params'])

    # Train
    best_loss = np.inf
    best_params = params
    def setup_metrics(batch_size, x, target):
        def acc_fn(params):
            x_ = x.reshape(-1, batch_size, 32, 32, 3)
            y_ = target.reshape(-1, batch_size)

            def pred(carry, x):
                logits, _ = _predict(
                    params['params'], params['batch_stats'], x, train=False)
                return None, logits
            _, logits = jax.lax.scan(pred, None, x_)
            loss = jnp.mean(
                optax.losses.softmax_cross_entropy_with_integer_labels(
                    logits=logits, labels=y_)
            )
            acc = jnp.mean(jnp.argmax(logits, axis=-1) == y_)
            return {'Acc':acc, 'Loss':loss}
        return acc_fn

    valid_metrics_fn = setup_metrics(250, x_val, y_val)

    def train_batch(carry, batch):
        x_, y_ = batch
        rng_key, params, opt_state, best_loss, best_params = carry
        rng_key, subkey = random.split(rng_key)

        net_params, stats_params = params['params'], params['batch_stats']

        (loss, stats_params), grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(net_params, stats_params, x_, y_, train=True)
        updates, opt_state = optimizer.update(grads, opt_state, net_params)
        net_params = optax.apply_updates(net_params, updates)

        return (rng_key, {'params': net_params, 'batch_stats': stats_params}, opt_state, best_loss, best_params), loss

    @scan_tqdm(config['num_epochs'])
    def train(carry, _):
        rng_key = carry[0]
        rng_key, subkey = random.split(rng_key)
        shuffel = jax.random.permutation(subkey, jnp.arange(y.shape[0]))
        # data augmentation
        rng_key, x_ = jax.lax.scan(augment, rng_key, x)
        # x_ = x
        x_ = x_[shuffel].reshape(-1, config['batch_size'], 32, 32, 3)
        y_ = y[shuffel].reshape(-1, config['batch_size'])
        carry, loss_batch = jax.lax.scan(
            train_batch, carry, (x_, y_))
        
        # validate
        loss_epoch = jnp.mean(loss_batch)
        params, best_loss, best_params = carry[1], carry[3], carry[4]

        metrics = valid_metrics_fn(params)
        val_loss = metrics['Loss']
        best_params = jax.lax.cond(
            val_loss < best_loss, lambda x: params, lambda x: best_params, None)
        best_loss = jnp.minimum(best_loss, val_loss)

        return (rng_key, params, carry[2], best_loss, best_params), (loss_batch, val_loss, metrics['Acc'])


    carry = (rng_key, params, opt_state, best_loss, best_params)
    # start training
    carry, (losses, valid_losses, valid_acc) = jax.lax.scan(train, carry, jnp.arange(config['num_epochs']))
    rng_key, params, opt_state, best_loss, best_params = carry

    # log losses for wandb
    n_steps_e = losses.shape[1]
    epochs = np.repeat(np.arange(config['num_epochs']), n_steps_e)
    _ = [logger.log({"train_loss": l, "valid_loss": vl, 'valid ACC': vacc, 'epoch': e})
         for e, l, vl, vacc in zip(epochs, losses.flatten(), np.repeat(valid_losses, n_steps_e), np.repeat(valid_acc, n_steps_e))]

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.array(losses).flatten(), alpha=0.2, label="train")
    ax.plot(np.arange(n_steps_e, n_steps_e*config['num_epochs']+1, n_steps_e), np.array(
        valid_losses).flatten(), alpha=0.2, label="valid")
    ax.set_ylabel("nll")
    ax.set_xlabel("steps")
    ax.legend()
    logger.log(
            {'Nll vs epoch': wandb.Image(fig)})
    
    # log test performance
    test_metrics_fn = setup_metrics(250, x_test, y_test)
    test_metrics = test_metrics_fn(params)
    logger.log({'test_loss': test_metrics['Loss'], 'test_acc': test_metrics['Acc']})
    test_metrics = test_metrics_fn(best_params)
    logger.log({'test_loss_best': test_metrics['Loss'], 'test_acc_best': test_metrics['Acc']})
    
    print("Save artifacts")
    jnp.save(f'tmp_files/{logger.id}_params.npy', best_params)
    jnp.save(f'tmp_files/{logger.id}_last_params.npy', params)
    print("add files to artifact")
    # create artifact for storing parameters
    artifact = wandb.Artifact(name="params", type="pytree")
    artifact.add_file(f'tmp_files/{logger.id}_params.npy')
    artifact.add_file(f'tmp_files/{logger.id}_last_params.npy')
    print("Log artifact")
    logger.log_artifact(artifact)

    logger.finish()


def main():
    logger = wandb.init()
    config = wandb.config
    single_run(logger, config)


def single_run(logger, config, n_samples_nuts=None):
    rng_key = random.PRNGKey(config['rng_seed'])
    EVAL_EVERY_N_BATCH = config['eval_every_n_batch']

    # create artifact for storing parameters
    artifact = wandb.Artifact(name="params", type="pytree")

    def train_curve(rng_key,
                    k: int,
                    num_epochs: int,
                    batch_size: int,
                    optimizer_conf: dict,
                    natural_parameterization: bool,
                    gravity: float, 
                    weight_decay: float = 0.0, 
                    remove_initial_permutation: bool = False,
                    net_kwargs: dict = None,
                    jitter: float = 0.0,):
        rng_key, init_key = random.split(rng_key)

        
        net_kwargs['act_fn'] = getattr(nn, net_kwargs['act_fn'])
        net_kwargs['block_class'] = getattr(src.models, net_kwargs['block_class'])
        model = ResNet(**net_kwargs)

        perm_spec = False
        if remove_initial_permutation:
            perm_spec = resnet20_permutation_spec2()

        s_model = RepulsiveCategorySubspaceBatchNorm(model, k=k, weight_decay=weight_decay, 
                                                     perm_spec=perm_spec,
                                                     natural_parameterization=natural_parameterization,
                                                     gravity=gravity)
        
        # load params from wandb artifact if path is given in config
        if 'params_path' in config:
            art = logger.use_artifact(config['params_path'], type='pytree')
            artifact_dir = art.download()
            filename = os.path.join(artifact_dir, f"{art.logged_by().id}_params.npy")
            init_params = jnp.load(filename, allow_pickle=True).item()
            curve_params = [deepcopy(init_params['params']) for _ in range(k+1)]
            
            # add noise to the parameters
            param, struct = jax.tree.flatten(curve_params)
            keys = random.split(rng_key, len(param))
            def apply_noise(p, key, jitter):
                return p+jitter*jax.random.normal(key, p.shape)
            noise_params = [apply_noise(p, key, jitter=jitter) for p, key in zip(param, keys)]
            curve_params = jax.tree.unflatten(struct, noise_params)
            
            if remove_initial_permutation:
                curve_params = s_model.remove_permutation(init_key, curve_params)
            
            curve_params = jax.tree.map(lambda *x: jnp.stack(x), *curve_params)
            params = {'params': curve_params, 'batch_stats': init_params['batch_stats'], 'dist_params': {}}

            # load data
            x, y, x_val, y_val, x_test, y_test, data_revert_norm_fn = load_cifar10(art.logged_by().config['rng_seed'])  # get fold from seed

            def setup_init_metrics(batch_size, x, target, prefix='valid'):
                def acc_fn(params):
                    x_ = x.reshape(-1, batch_size, 32, 32, 3)
                    y_ = target.reshape(-1, batch_size)

                    def pred(carry, x):
                        logits, _ = model.apply(params, x, train=False, mutable=["batch_stats"])
                        return None, logits
                    _, logits = jax.lax.scan(pred, None, x_)
                    loss = jnp.mean(
                        optax.losses.softmax_cross_entropy_with_integer_labels(
                            logits=logits, labels=y_)
                    )
                    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y_)
                    return {f'init_{prefix}_acc':acc, f'init_{prefix}_loss':loss}
                return acc_fn
            init_metrics_fn = setup_init_metrics(250, x_val, y_val)
            init_metrics = init_metrics_fn(init_params)
            logger.summary.update(init_metrics)
            init_metrics_fn = setup_init_metrics(250, x_test, y_test, prefix='test')
            init_metrics = init_metrics_fn(init_params)
            logger.summary.update(init_metrics)
        else:
            params = s_model.init_params(init_key, x[:2])
            # load data
            x, y, x_val, y_val, x_test, y_test, data_revert_norm_fn = load_cifar10(config['rng_seed'])  # get fold from seed

        jnp.save(f'tmp_files/{logger.id}_init_params.npy', params)
        artifact.add_file(f'tmp_files/{logger.id}_init_params.npy')

        # Train
        if isinstance(optimizer_conf['kwargs']['learning_rate'], dict):
            lr_schedule = eval(optimizer_conf['kwargs']['learning_rate']['name'])(
                **optimizer_conf['kwargs']['learning_rate']['kwargs'])
            optimizer_conf['kwargs']['learning_rate'] = lr_schedule
        optimizer = eval(optimizer_conf['name'])(**optimizer_conf['kwargs'])

        # iter_per_epoch_train = len(x) // batch_size
        # # lr_schedule = optax.linear_onecycle_schedule(
        # #     num_epochs * iter_per_epoch_train, lr, pct_start=0.4, div_factor=10.0, final_div_factor=20.0
        # # )
        # lr_schedule = optax.piecewise_interpolate_schedule(
        #     "cosine", 0.01, {iter_per_epoch_train*50: 10., iter_per_epoch_train*200: 0.5, (iter_per_epoch_train*num_epochs*4)//5: 0.5})

        # optimizer = optax.sgd(lr_schedule, momentum=0.9, nesterov=True)
        opt_state = optimizer.init(
            {'params': params['params']})  # only dnn parameters
        best_loss = np.inf
        best_params = params

        def setup_metrics(batch_size, x, target, n_samples, use_linspace=False, average=True):
            if use_linspace:
                get_t = lambda key: jnp.linspace(0, 1., n_samples)
            else:
                get_t = lambda key: random.uniform(key, (n_samples,), minval=0., maxval=1.)

            def acc_fn(rng_key, params):
                x_ = x.reshape(-1, batch_size, 32, 32, 3)
                y_ = target.reshape(-1, batch_size)

                def pred(key, data):
                    x_batch, y_batch = data

                    def eval(t):
                        logits, _ = s_model.eval(
                            params['params'], params['batch_stats'], t, x_batch)  # params, batch_stats, t, x
                        loss = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(
                            logits=logits, labels=y_batch))
                        acc = jnp.mean(jnp.argmax(logits, axis=-1) == y_batch)
                        return loss, acc  # mean over batch
                    key, subkey = random.split(key)

                    # draw 10 samples to approximate curve performance
                    t = get_t(subkey)
                    loss, acc = jax.vmap(eval)(t)

                    if average:
                        # mean over samples
                        return key, (jnp.mean(loss), jnp.mean(acc))
                    else:
                        return key, (loss, acc)  # no mean over samples

                # shape (n_data, draws, output_dim)
                rng_key, (loss, acc) = jax.lax.scan(pred, rng_key, (x_, y_))
                loss = jnp.mean(loss, axis=0)
                acc = jnp.mean(acc, axis=0)
                return loss, acc  # mean over epochs
            return acc_fn
        valid_metrics_fn = setup_metrics(250, x_val, y_val, n_samples=10)

        # Bezier curve length
        # define function to compute bezier curve length
        def get_d_bezier(t):
            bezier_coeff_inv = bezier_coeff_fn(k)
            coeff = jax.vmap(bezier_coeff_inv)(t)

            def clousure(cp):
                n = cp.shape[0] - 1
                cp_diff = cp[1:] - cp[:-1]
                return jnp.einsum('tj,j...->t...', coeff, cp_diff) * n
            return clousure

        @jit
        def bezier_length(cp):
            t = jnp.linspace(0, 1, 1000)
            d_bezier = get_d_bezier(t)
            d_t = jnp.linalg.norm(d_bezier(cp), axis=-1)
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
        cp_w = pytree_to_matrix(params['params'], k)
        rel_center_fn = init_rel_center(cp_w)

        def train_batch(carry, batch):
            x_, y_, batch_idx = batch
            rng_key, params, opt_state, best_loss, best_params = carry
            rng_key, subkey = random.split(rng_key)
            loss, params, opt_state, grad_ = s_model.train_step(
                subkey, params, x_, y_, opt_state, optimizer)
            # validate
            rng_key, valid_key = random.split(rng_key)
            # val_loss = s_model.compute_loss(
            #     subkey, params, x_val, y_val, n_samples=10)
            # best_params = jax.lax.cond(
            #     val_loss < best_loss, lambda x: params, lambda x: best_params, None)
            # best_loss = jnp.minimum(best_loss, val_loss)

            # acc, val_loss = valid_metrics_fn(valid_key, params)
            val_loss, acc = jax.lax.cond((batch_idx % EVAL_EVERY_N_BATCH) == 0, lambda x: valid_metrics_fn(
                valid_key, params), lambda x: (jnp.inf, -jnp.inf), None)
            best_params = jax.lax.cond(
                val_loss < best_loss, lambda x: params, lambda x: best_params, None)
            best_loss = jnp.minimum(best_loss, val_loss)

            cp_w = pytree_to_matrix(params['params'], k)
            length = bezier_length(cp_w)
            l_bound = lower_bound(cp_w)
            u_bound = upper_bound(cp_w)
            gyra = gyration(cp_w)
            center = mass_center(cp_w)
            # mean_curv = mean_curvature2(cp_w)
            rel_center = rel_center_fn(cp_w)
            grad_norm = jnp.linalg.norm(pytree_to_matrix(grad_['params'], k), axis=1).mean()

            return (rng_key, params, opt_state, best_loss, best_params), (loss, val_loss, acc, length, l_bound, u_bound, gyra, center, rel_center, grad_norm)

        @scan_tqdm(num_epochs)
        def train_epoch(carry, epoch):
            rng_key = carry[0]
            rng_key, subkey = random.split(rng_key)
            shuffel = jax.random.permutation(subkey, jnp.arange(y.shape[0]))
            # data augmentation
            rng_key, x_ = jax.lax.scan(augment, rng_key, x)
            # x_ = x
            x_ = x_[shuffel].reshape(-1, batch_size, 32, 32, 3)
            y_ = y[shuffel].reshape(-1, batch_size)
            carry, (loss, val_loss, val_acc, length, l_bound, u_bound, gyra, center, rel_center, grad_norm) = jax.lax.scan(
                train_batch, carry, (x_, y_, jnp.arange(x_.shape[0])))
            return carry, (loss, val_loss, val_acc, length, l_bound, u_bound, gyra, center, rel_center, grad_norm)

        carry = (rng_key, params, opt_state, best_loss, best_params)
        carry, (losses, valid_losses, valid_acc, length, l_bound, u_bound, gyra, center, rel_center, grad_norm) = jax.lax.scan(
            train_epoch, carry, jnp.arange(num_epochs))

        # log losses for wandb
        n_batches = np.array(losses).shape[-1]
        epochs = np.repeat(np.arange(num_epochs), n_batches)
        losses = jnp.array(losses).flatten()

        # clean up inf values because of lazy validation
        valid_losses = valid_losses[:, ::EVAL_EVERY_N_BATCH].repeat(
            EVAL_EVERY_N_BATCH, axis=1)[:,:valid_losses.shape[1]]
        valid_acc = valid_acc[:, ::EVAL_EVERY_N_BATCH].repeat(
            EVAL_EVERY_N_BATCH, axis=1)[:, :valid_acc.shape[1]]

        valid_losses = jnp.array(valid_losses).flatten()
        valid_acc = jnp.array(valid_acc).flatten()
        length = jnp.array(length).flatten()
        l_bound = jnp.array(l_bound).flatten()
        u_bound = jnp.array(u_bound).flatten()
        gyra = jnp.array(gyra).flatten()
        center = jnp.array(center).flatten()
        rel_center = jnp.array(rel_center).flatten()
        grad_norm = jnp.array(grad_norm).flatten()
        _ = [
            logger.log({
            "train_loss": l,
            "Curve valid_loss": vl,
            "Curve valid ACC": vacc,
            "epoch": e,
            "curve_length": le,
            "curve_lower_bound": lb,
            "curve_upper_bound": ub,
            "curve_gyration": gy,
            "curve_center": ce,
            "curve_rel_center": rc,
            "curve_grad_norm": gn,
            })
            for e, l, vl, vacc, le, lb, ub, gy, ce, rc, gn in zip(
            epochs, losses, valid_losses, valid_acc, length, l_bound, u_bound, gyra, center, rel_center, grad_norm
            )
        ]
        params = carry[4]  # select best parameters
        last_params = carry[1]
        rng_key = carry[0]

        # Bezier curve length        
        cp = pytree_to_matrix(params['params'], k)
        length_best = bezier_length(cp)
        cp = pytree_to_matrix(last_params['params'], k)
        length_last = bezier_length(cp)
        logger.log({'Curve length best': length_best, 'Curve length last': length_last})

        # plot likelihood and acc along curve
        fig, ax = plt.subplots(1, 1)
        t_space = jnp.linspace(0, 1, 50)
        valid_metrics_fn_final = setup_metrics(100, x_val, y_val, n_samples=50, use_linspace=True, average=False)
        train_metrics_fn_final = setup_metrics(100, x, y, n_samples=50, use_linspace=True, average=False)
        test_metrics_fn_final = setup_metrics(100, x_test, y_test, n_samples=50, use_linspace=True, average=False)
    

        # valid
        nll, acc = valid_metrics_fn_final(rng_key, params)
        logger.log({'acc_val_curve': acc.mean(), 'nll_val_curve': nll.mean()})
        ax.plot(t_space, -nll, label="valid ll", c=plt.get_cmap('tab10')(1))
        ax2 = ax.twinx()
        ax2.plot(t_space, acc, label="valid acc",
                 c=plt.get_cmap('tab10')(1), linestyle="--")
        # train
        nll, acc = train_metrics_fn_final(rng_key, params)
        ax.plot(t_space, -nll, label="train", c=plt.get_cmap('tab10')(0))
        ax2.plot(t_space, acc, label="train acc",
                 c=plt.get_cmap('tab10')(0), linestyle="--")
        # test
        nll, acc = test_metrics_fn_final(rng_key, params)
        logger.log({'acc_test_curve': acc.mean(), 'nll_test_curve': nll.mean()})
        ax.plot(t_space, -nll, label="test", c=plt.get_cmap('tab10')(2))
        ax2.plot(t_space, acc, label="test acc",
                 c=plt.get_cmap('tab10')(2), linestyle="--")
        ax.set_xlabel("t")
        ax.set_ylabel("mean log likelihood")
        ax2.set_ylabel("acc")
        ax.legend()
        ax2.legend(loc='lower center')
        logger.log(
            {'Nll alonge curve': wandb.Image(fig)})
        
        # valid
        fig, ax = plt.subplots(1, 1)
        nll, acc = valid_metrics_fn_final(rng_key, last_params)
        logger.log({'acc_val_curve_last': acc.mean(), 'nll_val_curve_last': nll.mean()})
        ax.plot(t_space, -nll, label="valid ll", c=plt.get_cmap('tab10')(1))
        ax2 = ax.twinx()
        ax2.plot(t_space, acc, label="valid acc",
                 c=plt.get_cmap('tab10')(1), linestyle="--")
        # train
        nll, acc = train_metrics_fn_final(rng_key, last_params)
        ax.plot(t_space, -nll, label="train", c=plt.get_cmap('tab10')(0))
        ax2.plot(t_space, acc, label="train acc",
                 c=plt.get_cmap('tab10')(0), linestyle="--")
        # test
        nll, acc = test_metrics_fn_final(rng_key, last_params)
        logger.log({'acc_test_curve_last': acc.mean(), 'nll_test_curve_last': nll.mean()})
        ax.plot(t_space, -nll, label="test", c=plt.get_cmap('tab10')(2))
        ax2.plot(t_space, acc, label="test acc",
                 c=plt.get_cmap('tab10')(2), linestyle="--")
        ax.set_xlabel("t")
        ax.set_ylabel("mean log likelihood")
        ax2.set_ylabel("acc")
        ax.legend()
        ax2.legend(loc='lower center')
        logger.log(
            {'Nll alonge curve last': wandb.Image(fig)})

        return rng_key, params, model, last_params

    rng_key, params, model, last_params = train_curve(
        rng_key, **config['curve_params'], net_kwargs=config['net_kwargs'])

    print("Save artifacts")
    jnp.save(f'tmp_files/{logger.id}_params.npy', params)
    jnp.save(f'tmp_files/{logger.id}_last_params.npy', last_params)
    print("add files to artifact")
    artifact.add_file(f'tmp_files/{logger.id}_params.npy')
    artifact.add_file(f'tmp_files/{logger.id}_last_params.npy')
    print("Log artifact")
    logger.log_artifact(artifact)

    logger.finish()


if __name__ == '__main__':
    # Add requirement for wandb core
    wandb.require("core")
    os.makedirs('tmp_files', exist_ok=True)

    # config = {
    #     'curve_params': {'k': 10,
    #                      'num_epochs': 500,
    #                      'batch_size': 100,
    #                      'uniform_t_optimize': False,
    #                      'weight_decay': 0.0001,
    #                      'with_multiplier': 1,
    #                      'remove_initial_permutation': True,
    #                     #  'optimizer_conf': {'name': "optax.adamw", 'kwargs': {'learning_rate': 0.05, 'weight_decay': 0.0001}},
    #                     #  'optimizer_conf': {'name': "optax.adamw", 'learning_rate': 0.05, 'momentum': 0.9, 'nesterov': True},
    #                      'optimizer_conf': {'name': "optax.sgd", 'kwargs': {'learning_rate': {'name': "optax.linear_onecycle_schedule", 
    #                                                                                           'kwargs': {'transition_steps': 490_000, 'peak_value': 0.2, 'pct_start': 0.2, 'pct_final': 0.6, 'div_factor': 5.0, 'final_div_factor': 10.0}},
    #                                                                         'momentum': 0.9, 
    #                                                                         'nesterov': True},},
    #                     },
    #     'sampling': {'space_config': {'space': 'lambda',
    #                                   'prior_scale': .5,
    #                                   'tube_scale': .1,
    #                                   'epsilon': 25.,
    #                                   'prior_correct': False},
    #                  'num_chains': 2,
    #                  'temperature': 10.,
    #                  'num_warmup': 100,
    #                  'num_draws': 100,
    #                  'sampler': 'mclmc'
    #                  },
    #     'rng_seed': 0,
    #     # 'nuts_sweep_id': '1',
    #     'nuts_sweep_id': 'nuts',
    #     'eval_every_n_batch': 50
    # }
    # logger = wandb.init(project="subspace_test", name="cifar", entity="ddold", config=config)
    # single_run(logger, config, n_samples_nuts=200)
    
    parser = argparse.ArgumentParser(description="Train a single model or a curve model.")
    parser.add_argument("--train", choices=["single", "curve"], required=True, help="Specify the training mode: 'single' for single model training, 'curve' for curve model training.")
    parser.add_argument("--use_sweep", action="store_true", help="Use wandb sweep configuration if set.")

    args = parser.parse_args()

    if args.train == "single":
        print("Single model training")
        config = {
            'rng_seed': 0,
            'num_epochs': 500,
            'batch_size': 100,
            'optimizer_conf': {'name': "optax.sgd", 'kwargs': {'learning_rate': 0.05, 'momentum': 0.9, 'nesterov': True}},
            'net_kwargs':{'num_blocks': (3, 3, 3),
                          'c_hidden': (16, 32, 64),
                          'num_classes': 10,
                          'act_fn': 'relu',
                          'block_class': 'ResNetBlock'}}
        logger = wandb.init(project="cifar_PtoC", name="cifar_point", entity="ddold", config=config)
        single_model_train(logger, config)
    elif args.train == "curve":
        print("Curve training")
        if args.use_sweep:
            print("Using wandb config")
            logger = wandb.init()
            config = wandb.config
            single_run(logger, config)
        else:
            print("Using perdefined config")
            config = {
                'rng_seed': 0,
                'net_kwargs':{'num_blocks': (3, 3, 3),
                            'c_hidden': (16, 32, 64),
                            'num_classes': 10,
                            'act_fn': 'relu',
                            'block_class': 'ResNetBlock'},
                'params_path': 'ddold/cifar_PtoC/params:v1',
                'eval_every_n_batch': 20,
                'curve_params': {'k': 10,
                                'num_epochs': 50,
                                'batch_size': 100,
                                'natural_parameterization': False,
                                'gravity': 0.,
                                'weight_decay': 0.,
                                'remove_initial_permutation': False,
                                'jitter': 1e-6,
                                #  'optimizer_conf': {'name': "optax.adamw", 'kwargs': {'learning_rate': 0.05, 'weight_decay': 0.0001}},
                                #  'optimizer_conf': {'name': "optax.adamw", 'learning_rate': 0.05, 'momentum': 0.9, 'nesterov': True},
                                'optimizer_conf': {'name': "optax.sgd", 'kwargs': {'learning_rate': {'name': "optax.linear_onecycle_schedule", 
                                                                                                'kwargs': {'transition_steps': 490_000, 'peak_value': 0.1, 'pct_start': 0.2, 'pct_final': 0.6, 'div_factor': 5.0, 'final_div_factor': 10.0}},
                                                                                'momentum': 0.9, 
                                                                                'nesterov': True},},
                            }}
            logger = wandb.init(project="cifar_PtoC", name="cifar_curve", entity="ddold", config=config)
            single_run(logger, config)

        # main()

    # main()
