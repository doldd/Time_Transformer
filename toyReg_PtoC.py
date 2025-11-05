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
from src.subspace_curve import masked_pytree_to_matrix, vec_to_pytree, bezier_coeff_fn, bezier_curve
from src.subspace_curve import OrthoSpan, RepulsiveDistRegressionSubspace, DistRegressionSubspace
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
from src.models import MLPModel
import optax


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=10"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# jax.config.update('jax_platform_name', 'cpu')
# Assuming equivalent JAX implementations


# Generate data by using the model itself
def gen_data(rng_seed, net_kwargs, **kwargs):
    rng_key = random.PRNGKey(rng_seed + 1133)
    x = jnp.linspace(-2, 2, 100)
    x = x[(jnp.abs(x) > 0.6)].reshape(-1, 1)

    # define ground trouth model
    rng_key, rng_init = random.split(rng_key)
    model_gen = MLPModel(**net_kwargs)
    params = model_gen.init(rng_init, x)

    # generate train data
    y_gt = model_gen.apply(params, x).squeeze()
    rng_key, rng = random.split(rng_key)
    y = y_gt + jax.random.normal(rng, y_gt.shape) * 0.05

    # generate valid data
    x_val = jnp.linspace(-2, 2, 25)
    x_val = x_val[(jnp.abs(x_val) > 0.6)].reshape(-1, 1)
    y_gt_val = model_gen.apply(params, x_val).squeeze()
    rng_key, rng = random.split(rng_key)
    y_val = y_gt_val + jax.random.normal(rng, y_gt_val.shape) * 0.05

    xt = jnp.linspace(-2.5, 2.5, 1000).reshape(-1, 1)
    rng_key, rng = random.split(rng_key)
    y_gtt = model_gen.apply(params, xt).squeeze()
    yt = y_gtt + jax.random.normal(rng, y_gtt.shape) * 0.05

    return x, y, x_val, y_val, xt, yt, y_gtt


def setup_metrics(model, batch_size, x, target, n_samples, use_linspace=False, average=True):
    if use_linspace:
        get_t = lambda key: jnp.linspace(0, 1., n_samples)
    else:
        get_t = lambda key: random.uniform(key, (n_samples,), minval=0., maxval=1.)

    def acc_fn(rng_key, params):
        x_ = x.reshape(-1, batch_size, 1)
        y_ = target.reshape(-1, batch_size)

        def pred(key, data):
            x_batch, y_batch = data

            def eval(t):
                loss, _, out = model.nll(params['params'], {}, t, x_batch, y_batch, train=False)  # params, batch_stats, t, x
                mse = jnp.mean((out.squeeze(axis=-1) - y_batch)**2)
                return jnp.mean(loss), mse  # mean over batch
            key, subkey = random.split(key)

            # draw samples to approximate curve performance
            t = get_t(subkey)
            loss, mse = jax.vmap(eval)(t)

            if average:
                # mean over samples
                return key, (jnp.mean(loss), jnp.mean(mse))
            else:
                return key, (loss, mse)  # no mean over samples

        # shape (n_data, draws, output_dim)
        rng_key, (loss, mse) = jax.lax.scan(pred, rng_key, (x_, y_))
        loss = jnp.mean(loss, axis=0)
        mse = jnp.mean(mse, axis=0)
        return loss, mse  # mean over epochs
    return acc_fn


def main():
    logger = wandb.init()
    config = wandb.config
    train_curve(logger, config)


def train_curve(logger, config):
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
                    energy_weakening: float,
                    regularizer: bool|str,
                    weight_decay: float = 0.0, 
                    remove_initial_permutation: bool = False,
                    net_kwargs: dict = None,
                    jitter: float = 0.0,):
        rng_key, init_key = random.split(rng_key)

        model = MLPModel(**net_kwargs)

        perm_spec = False
        if remove_initial_permutation:
            raise NotImplementedError("Removing initial permutation is not implemented for this model.")
            perm_spec = resnet20_permutation_spec2()

        if regularizer:
            s_model = RepulsiveDistRegressionSubspace(model=model, k=k, weight_decay=weight_decay, 
                                                    perm_spec=perm_spec,
                                                    natural_parameterization=natural_parameterization,
                                                    mutuable_param_name = False,
                                                    gravity=gravity, energy_weakening=energy_weakening, regularizer=regularizer)
        else:
            s_model = DistRegressionSubspace(model=model, k=k, weight_decay=weight_decay, 
                                             perm_spec=perm_spec,
                                             natural_parameterization=natural_parameterization,
                                             mutuable_param_name = False)
        


        # load params from wandb artifact if path is given in config
        if 'params_path' in config:
            assert k > 0, "Subspace dimension k must be greater than 0 when training a curve from point."
            art = logger.use_artifact(config['params_path'], type='pytree')
            artifact_dir = art.download()
            filename = os.path.join(artifact_dir, f"{art.logged_by().id}_params.npy")
            init_params = jnp.load(filename, allow_pickle=True).item()

            # load data
            x, y, x_val, y_val, x_test, y_test, _ = gen_data(art.logged_by().config['rng_seed'], net_kwargs)  # get fold from seed

            # initialize curve parameter with pretrained point parameter
            mask = s_model.init_mask(x_val[:10], all_curves=True)
            params = s_model.init_params_from_point(init_key, init_params, mask, noise_fn=lambda p, key: p+jitter*jax.random.normal(key, p.shape, dtype=p.dtype))
            
            init_metrics_fn = setup_metrics(s_model, 18, x_val, y_val, 1, use_linspace=True, average=True)
            loss, mse = init_metrics_fn(init_key, params)
            logger.summary.update({'init_val_loss': loss, 'init_val_mse': mse})
            init_metrics_fn = setup_metrics(s_model, 250, x_test, y_test, 1, use_linspace=True, average=True)
            loss, mse  = init_metrics_fn(init_key, params)
            logger.summary.update({'init_test_loss': loss, 'init_test_mse': mse})
        else:
            # load data
            x, y, x_val, y_val, x_test, y_test, _ = gen_data(config['rng_seed'], net_kwargs)  # get fold from seed
            
            if k > 0:
                # initialize curve parameter with random parameters
                mask = s_model.init_mask(x_val[:2], all_curves=True)
                params = s_model.init_params(init_key, x[:2], mask=mask)
            else:
                # initialize point model (vanilla training)
                mask = s_model.init_mask(x_val[:2], all_curves=False)
                params = s_model.init_params(init_key, x[:2], mask=mask)

        jnp.save(f'tmp_files/{logger.id}_init_params.npy', params)
        artifact.add_file(f'tmp_files/{logger.id}_init_params.npy')

        # Train
        if isinstance(optimizer_conf['kwargs']['learning_rate'], dict):
            lr_schedule = getattr(optax, optimizer_conf['kwargs']['learning_rate']['name'])(
                **optimizer_conf['kwargs']['learning_rate']['kwargs'])
            optimizer_conf['kwargs']['learning_rate'] = lr_schedule
        optimizer = getattr(optax, optimizer_conf['name'])(**optimizer_conf['kwargs'])
        opt_state = optimizer.init(params['params'])  # only dnn parameters

        best_loss = np.inf
        best_params = params

        valid_metrics_fn = setup_metrics(s_model, 18, x_val, y_val, n_samples=20)

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
        cp_w = masked_pytree_to_matrix(params['params'], s_model.curve_mask, k)
        rel_center_fn = init_rel_center(cp_w)

        if k > 0:
            def train_batch(carry, batch):
                x_, y_, batch_idx = batch
                rng_key, params, opt_state, best_loss, best_params = carry
                rng_key, subkey = random.split(rng_key)
                t = random.uniform(subkey, (1,), minval=0., maxval=1.)
                loss, params, opt_state, grad_ = s_model.train_step(
                    subkey, t, params, x_, y_, opt_state, optimizer)
                # validate
                rng_key, valid_key = random.split(rng_key)

                # mse, val_loss = valid_metrics_fn(valid_key, params)
                val_loss, mse = jax.lax.cond((batch_idx % EVAL_EVERY_N_BATCH) == 0, lambda x: valid_metrics_fn(
                    valid_key, params), lambda x: (jnp.inf, -jnp.inf), None)
                best_params = jax.lax.cond(
                    val_loss < best_loss, lambda x: params, lambda x: best_params, None)
                best_loss = jnp.minimum(best_loss, val_loss)

                cp_w = masked_pytree_to_matrix(params['params'], s_model.curve_mask, k)
                length = bezier_length(cp_w)
                l_bound = lower_bound(cp_w)
                u_bound = upper_bound(cp_w)
                gyra = gyration(cp_w)
                center = mass_center(cp_w)
                # mean_curv = mean_curvature2(cp_w)
                rel_center = rel_center_fn(cp_w)
                grad_norm = jnp.linalg.norm(masked_pytree_to_matrix(grad_, mask, k), axis=1).mean()
                cos_sim = s_model.summed_cosine_similarity(params['params'])
                cos_sim /= ((k+1)*k/2)
                return (rng_key, params, opt_state, best_loss, best_params), (loss, val_loss, mse, length, l_bound, u_bound, gyra, center, rel_center, grad_norm, cos_sim)
        else:
            def train_batch(carry, batch):
                x_, y_, batch_idx = batch
                rng_key, params, opt_state, best_loss, best_params = carry
                rng_key, subkey = random.split(rng_key)
                t = random.uniform(subkey, (1,), minval=0., maxval=1.)
                loss, params, opt_state, grad_ = s_model.train_step(
                    subkey, t, params, x_, y_, opt_state, optimizer)
                # validate
                rng_key, valid_key = random.split(rng_key)

                # mse, val_loss = valid_metrics_fn(valid_key, params)
                val_loss, mse = jax.lax.cond((batch_idx % EVAL_EVERY_N_BATCH) == 0, lambda x: valid_metrics_fn(
                    valid_key, params), lambda x: (jnp.inf, -jnp.inf), None)
                best_params = jax.lax.cond(
                    val_loss < best_loss, lambda x: params, lambda x: best_params, None)
                best_loss = jnp.minimum(best_loss, val_loss)

                length = 0.
                l_bound = 0.
                u_bound = 0.
                gyra = 0.
                center = 0.
                rel_center = 0.
                grad_norm = 0.
                cos_sim = 0.
                return (rng_key, params, opt_state, best_loss, best_params), (loss, val_loss, mse, length, l_bound, u_bound, gyra, center, rel_center, grad_norm, cos_sim)


        @scan_tqdm(num_epochs)
        def train_epoch(carry, epoch):
            rng_key = carry[0]
            rng_key, subkey = random.split(rng_key)
            shuffel = jax.random.permutation(subkey, jnp.arange(y.shape[0]))
            x_ = x
            x_ = x_[shuffel].reshape(-1, batch_size, 1)
            y_ = y[shuffel].reshape(-1, batch_size)
            carry, (loss, val_loss, val_mse, length, l_bound, u_bound, gyra, center, rel_center, grad_norm, cos_sim) = jax.lax.scan(
                train_batch, carry, (x_, y_, jnp.arange(x_.shape[0])))
            return carry, (loss, val_loss, val_mse, length, l_bound, u_bound, gyra, center, rel_center, grad_norm, cos_sim)

        carry = (rng_key, params, opt_state, best_loss, best_params)
        carry, (losses, val_losses, val_mse, length, l_bound, u_bound, gyra, center, rel_center, grad_norm, cos_sim) = jax.lax.scan(
            train_epoch, carry, jnp.arange(num_epochs))

        # log losses for wandb
        n_batches = np.array(losses).shape[-1]
        epochs = np.repeat(np.arange(num_epochs), n_batches)
        losses = jnp.array(losses).flatten()

        # clean up inf values because of lazy validation
        val_losses = val_losses[:, ::EVAL_EVERY_N_BATCH].repeat(
            EVAL_EVERY_N_BATCH, axis=1)[:,:val_losses.shape[1]]
        val_mse = val_mse[:, ::EVAL_EVERY_N_BATCH].repeat(
            EVAL_EVERY_N_BATCH, axis=1)[:, :val_mse.shape[1]]

        val_losses = jnp.array(val_losses).flatten()
        val_mse = jnp.array(val_mse).flatten()
        length = jnp.array(length).flatten()
        l_bound = jnp.array(l_bound).flatten()
        u_bound = jnp.array(u_bound).flatten()
        gyra = jnp.array(gyra).flatten()
        center = jnp.array(center).flatten()
        rel_center = jnp.array(rel_center).flatten()
        grad_norm = jnp.array(grad_norm).flatten()
        cos_sim = jnp.array(cos_sim).flatten()
        _ = [
            logger.log({
            "train_loss": l,
            "Curve valid_loss": vl,
            "Curve valid MSE": vacc,
            "epoch": e,
            "curve_length": le,
            "curve_lower_bound": lb,
            "curve_upper_bound": ub,
            "curve_gyration": gy,
            "curve_center": ce,
            "curve_rel_center": rc,
            "curve_grad_norm": gn,
            "cosine_similarity": cs,
            })
            for e, l, vl, vacc, le, lb, ub, gy, ce, rc, gn, cs in zip(
            epochs, losses, val_losses, val_mse, length, l_bound, u_bound, gyra, center, rel_center, grad_norm, cos_sim
            )
        ]
        # log metrics as artifact
        jnp.savez(f'tmp_files/{logger.id}_metrics.npz', 
                  loss=losses,
                  epochs=epochs,
                  val_loss=val_losses,
                  val_mse=val_mse,
                  curve_length=length,
                  curve_lower_bound=l_bound,
                  curve_upper_bound=u_bound,
                  curve_gyration=gyra,
                  curve_center=center,
                  curve_rel_center=rel_center,
                  curve_grad_norm=grad_norm,
                  cosine_similarity=cos_sim)
        art = wandb.Artifact(name="metrics", type="npz")
        art.add_file(f'tmp_files/{logger.id}_metrics.npz')
        logger.log_artifact(art)

        params = carry[4]  # select best parameters
        last_params = carry[1]
        rng_key = carry[0]

        # Bezier curve length        
        cp = masked_pytree_to_matrix(params['params'], s_model.curve_mask, k)
        length_best = bezier_length(cp)
        cp = masked_pytree_to_matrix(last_params['params'], s_model.curve_mask, k)
        length_last = bezier_length(cp)
        logger.log({'Curve length best': length_best, 'Curve length last': length_last})

        # plot likelihood and acc along curve
        fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
        t_space = jnp.linspace(0, 1, 50)
        valid_metrics_fn_final = setup_metrics(s_model, 18, x_val, y_val, n_samples=50, use_linspace=True, average=False)
        train_metrics_fn_final = setup_metrics(s_model, 70, x, y, n_samples=50, use_linspace=True, average=False)
        test_metrics_fn_final = setup_metrics(s_model, 250, x_test, y_test, n_samples=50, use_linspace=True, average=False)
    
        # valid
        nll, mse = valid_metrics_fn_final(rng_key, params)
        logger.log({'mse_val_curve': mse.mean(), 'nll_val_curve': nll.mean(), 'mse_val_curve_min': mse.min(), 'nll_val_curve_min': nll.min()})
        ax[0].plot(t_space, -nll, label="valid ll", c=plt.get_cmap('tab10')(1))
        ax2 = ax[0].twinx()
        ax2.plot(t_space, mse, label="valid mse",
                 c=plt.get_cmap('tab10')(1), linestyle="--")
        # train
        nll, mse = train_metrics_fn_final(rng_key, params)
        ax[0].plot(t_space, -nll, label="train", c=plt.get_cmap('tab10')(0))
        ax2.plot(t_space, mse, label="train mse",
                 c=plt.get_cmap('tab10')(0), linestyle="--")
        # test
        nll, mse = test_metrics_fn_final(rng_key, params)
        logger.log({'mse_test_curve': mse.mean(), 'nll_test_curve': nll.mean(), 'mse_test_curve_min': mse.min(), 'nll_test_curve_min': nll.min()})
        ax[0].plot(t_space, -nll, label="test", c=plt.get_cmap('tab10')(2))
        ax2.plot(t_space, mse, label="test mse",
                 c=plt.get_cmap('tab10')(2), linestyle="--")
        ax[0].set_xlabel("t")
        ax[0].set_ylabel("mean log likelihood")
        ax[0].set_title("Best params")
        
        # valid
        nll, mse = valid_metrics_fn_final(rng_key, last_params)
        logger.log({'mse_val_curve_last': mse.mean(), 'nll_val_curve_last': nll.mean(), 'mse_val_curve_last_min': mse.min(), 'nll_val_curve_last_min': nll.min()})
        ax[1].plot(t_space, -nll, label="valid ll", c=plt.get_cmap('tab10')(1))
        ax2 = ax[1].twinx()
        ax2.plot(t_space, mse, label="valid mse",
                 c=plt.get_cmap('tab10')(1), linestyle="--")
        # train
        nll, mse = train_metrics_fn_final(rng_key, last_params)
        ax[1].plot(t_space, -nll, label="train", c=plt.get_cmap('tab10')(0))
        ax2.plot(t_space, mse, label="train mse",
                 c=plt.get_cmap('tab10')(0), linestyle="--")
        # test
        nll, mse = test_metrics_fn_final(rng_key, last_params)
        logger.log({'mse_test_curve_last': mse.mean(), 'nll_test_curve_last': nll.mean(), 'mse_test_curve_last_min': mse.min(), 'nll_test_curve_last_min': nll.min()})
        ax[1].plot(t_space, -nll, label="test", c=plt.get_cmap('tab10')(2))
        ax2.plot(t_space, mse, label="test mse",
                 c=plt.get_cmap('tab10')(2), linestyle="--")
        ax[1].set_xlabel("t")
        ax2.set_ylabel("mse")
        ax[0].legend(loc='center left', bbox_to_anchor=(-0.1, 1.25), ncol=3)
        ax[1].set_title("Last params")
        ax2.legend(loc='center left', bbox_to_anchor=(-0.1, 1.25), ncol=3)
        plt.tight_layout()
        logger.log(
            {'Nll alonge curve': wandb.Image(fig)})
        
        # plot predictive performance
        fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
        t_space = jnp.linspace(0, 1, 100)
        x_lin = jnp.linspace(-3, 3, 100)[:, None]
        # with best params
        out, _ = jax.vmap(s_model, in_axes=(None,None,0,None,None))(params['params'], {}, t_space, x_lin, False)
        out = out.squeeze(axis=-1)  # shape (n_samples, n_data)
        ax[0].plot(x, y, 'o', label='train')
        # plt.fill_between(x.squeeze(), out.mean(axis=0) - out.std(axis=0), out.mean(axis=0) + out.std(axis=0), alpha=0.5)
        # plot lines using viridis color map
        colors = plt.cm.viridis(t_space)
        for o, c in zip(out, colors):
            ax[0].plot(x_lin, o, color=c, alpha=0.3)
        ax[0].plot(x_lin, out.mean(axis=0), label='mean',
                c='red', linewidth=2, alpha=0.8)
        ax[0].set_ylabel("y")
        ax[0].set_xlabel("x")
        ax[0].set_title("Best params")
        # compute functional diversity
        functional_diff = {"Sample Std best":out.std(axis=0).mean()}  # std over samples
        
        # with last params
        out, _ = jax.vmap(s_model, in_axes=(None,None,0,None,None))(last_params['params'], {}, t_space, x_lin, False)
        out = out.squeeze(axis=-1)  # shape (n_samples, n_data)
        ax[1].plot(x, y, 'o', label='train')
        # plt.fill_between(x.squeeze(), out.mean(axis=0) - out.std(axis=0), out.mean(axis=0) + out.std(axis=0), alpha=0.5)
        # plot lines using viridis color map
        colors = plt.cm.viridis(t_space)
        for o, c in zip(out, colors):
            ax[1].plot(x_lin, o, color=c, alpha=0.3)
        ax[1].plot(x_lin, out.mean(axis=0), label='mean',
                c='red', linewidth=2, alpha=0.8)
        ax[1].set_ylabel("y")
        ax[1].set_xlabel("x")
        ax[1].set_title("Last params")
        functional_diff["Sample Std last"] = out.std(axis=0).mean()  # std over samples
        logger.log(functional_diff)

        plt.tight_layout()
        logger.log(
            {'Curve Predictive': wandb.Image(fig)})


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
    
    parser = argparse.ArgumentParser(description="Train a single model or a curve model.")
    parser.add_argument("--train", choices=["single", "curve"], required=True, help="Specify the training mode: 'single' for single model training, 'curve' for curve model training.")
    parser.add_argument("--use_sweep", action="store_true", help="Use wandb sweep configuration if set.")

    args = parser.parse_args()

    if args.train == "single":
        print("Single model training")
        config = {
            'rng_seed': 5,
            'net_kwargs':{'depth': 10, 
                            'width': 10, 
                            'activation': 'elu'},
            'eval_every_n_batch': 1,
            'curve_params': {'k': 0,
                            'num_epochs': 100,
                            'batch_size': 14,
                            'natural_parameterization': False,
                            'gravity': 0.0,
                            'energy_weakening': 1., 
                            'regularizer': False,
                            'weight_decay': 0.,
                            'remove_initial_permutation': False,
                            'jitter': 1e-8,
                            #  'optimizer_conf': {'name': "adamw", 'kwargs': {'learning_rate': 0.05, 'weight_decay': 0.0001}},
                            #  'optimizer_conf': {'name': "adamw", 'learning_rate': 0.05, 'momentum': 0.9, 'nesterov': True},
                                'optimizer_conf': {'name': "sgd", 'kwargs': {'learning_rate': 0.002, 'momentum': 0.9, 'nesterov': True}},
                            # 'optimizer_conf': {'name': "sgd", 'kwargs': {'learning_rate': {'name': "linear_onecycle_schedule", 
                                                                                            # 'kwargs': {'transition_steps': 490_000, 'peak_value': 0.1, 'pct_start': 0.2, 'pct_final': 0.6, 'div_factor': 5.0, 'final_div_factor': 10.0}},
                                                                            # 'momentum': 0.9, 
                                                                            # 'nesterov': True},},
                        }}
        logger = wandb.init(project="toyReg_PtoC", name="toyReg_single", entity="ddold", config=config)
        train_curve(logger, config)
    elif args.train == "curve":
        print("Curve training")
        if args.use_sweep:
            print("Using wandb config")
            logger = wandb.init()
            config = wandb.config
            train_curve(logger, config)
        else:
            print("Using perdefined config")
            config = {
                'rng_seed': 3,
                'net_kwargs':{'depth': 10, 
                              'width': 10, 
                              'activation': 'elu'},
                'eval_every_n_batch': 1,
                'params_path': "ddold/toyReg_PtoC/params:v16",
                'curve_params': {'k': 2,
                                'num_epochs': 1000,
                                'batch_size': 14,
                                'natural_parameterization': False,
                                'gravity': 1.,
                                'energy_weakening': 1.,
                                'weight_decay': 0.,
                                'remove_initial_permutation': False,
                                'regularizer': 'cosine',
                                'jitter': 1e-8,
                                #  'optimizer_conf': {'name': "adamw", 'kwargs': {'learning_rate': 0.05, 'weight_decay': 0.0001}},
                                #  'optimizer_conf': {'name': "adamw", 'learning_rate': 0.05, 'momentum': 0.9, 'nesterov': True},
                                 'optimizer_conf': {'name': "sgd", 'kwargs': {'learning_rate': 0.001, 'momentum': 0.9, 'nesterov': True}},
                                # 'optimizer_conf': {'name': "sgd", 'kwargs': {'learning_rate': {'name': "linear_onecycle_schedule", 
                                                                                                # 'kwargs': {'transition_steps': 490_000, 'peak_value': 0.1, 'pct_start': 0.2, 'pct_final': 0.6, 'div_factor': 5.0, 'final_div_factor': 10.0}},
                                                                                # 'momentum': 0.9, 
                                                                                # 'nesterov': True},},
                            }}
            logger = wandb.init(project="toyReg_PtoC", name="toyReg_curve", entity="ddold", config=config)
            train_curve(logger, config)

        # main()

    # main()
