import os
os.environ["JAX_COMPILATION_CACHE_DIR"] = "./jax_cache"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=10"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"

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
from src.subspace_curve import OrthoSpan, QuantileSubspace, RepulsiveQuantileSubspace, SubspaceBaseModel, LoRAQuantileSubspace, LoRARepulsiveQuantileSubspace
from src.models import ResNet, ResNetBlock, init_t_lambda_to_phi
import blackjax
from numpyro.infer.util import initialize_model
from jax_tqdm import scan_tqdm
from flax import linen as nn
from src.weight_matching import resnet20_permutation_spec2
import argparse
from src.models import MLPModel
import optax
import timesfm
import pandas as pd
from einshape import jax_einshape as es
from src.subspace_curve import filter_mask
from scipy.interpolate import PchipInterpolator
import pprint



# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# jax.config.update('jax_platform_name', 'cpu')
# Assuming equivalent JAX implementations


# Generate data by using the model itself
def gen_data(rng_seed, context_len, horizon_len, max_time_diff, **kwargs):
    path = "datasets/SDWPF/SDWPF_dataset/sdwpf_full/sdwpf_2001_2112_full.parquet"
    df = pd.read_parquet(path)
    df.dropna(inplace=True)
    df_sub = df[(df['TurbID'] < 120)]
    df_ood = df[(df["TurbID"] >= 120)]

    df_sub.set_index(["TurbID", 'Tmstamp'], inplace=True)
    df_ood.set_index(["TurbID", 'Tmstamp'], inplace=True)
    # standardize the data
    data_mean = df_sub.mean()
    data_std = df_sub.std()
    df_sub = (df_sub - data_mean) / data_std
    df_ood = (df_ood - data_mean) / data_std
    df_sub = df_sub.reset_index()
    df_ood = df_ood.reset_index()
    df_sub.sort_values(by=["Tmstamp",], inplace=True)
    df_ood.sort_values(by=["Tmstamp",], inplace=True)

    # df_sub = df

    splits = df_sub['Tmstamp'].quantile([0.75, 0.85, 1.]).values
    # split train test data
    df_train = df_sub[df_sub["Tmstamp"] < splits[0]].copy(deep=True)
    df_val = df_sub[(df_sub["Tmstamp"] >= splits[0]) & (df_sub["Tmstamp"] < splits[1])].copy(deep=True)
    df_test = df_sub[df_sub["Tmstamp"] >= splits[1]].copy(deep=True)

    def get_batched_data_fn(df):
        inputs = []
        outputs = []
        num_examples = 0
        num_skipped = 0

        dfg = df.groupby('TurbID')
        time_diff = dfg['Tmstamp'].diff().dt.total_seconds() / 60  # Convert to minutes
        time_diff = time_diff.fillna(0)  # Fill NaN values with 0
        df.loc[:, 'Tmstamp_diff'] = time_diff 

        # loop over each TurbID
        for t_id, df_ in df.groupby('TurbID'):
            # split df into frames where continous time difference is < 30 (each frame has a time difference smaller 30 minutes)
            dfg = df_.groupby((df_['Tmstamp_diff'] > max_time_diff).cumsum())
            # loop over continous time group
            for g_id, group in dfg:
                if len(group) < (context_len + horizon_len):
                    num_skipped += len(group)
                    continue
                # interploate in each group the get time difference = 10
                interp = PchipInterpolator(group['Tmstamp'], group['Patv'], extrapolate=True)
                # create new time stamps with 10 minutes difference
                new_time = pd.date_range(start=group['Tmstamp'].min(), end=group['Tmstamp'].max(), freq='10min')
                # interpolate the Patv values for the new time stamps
                new_patv = interp(new_time)

                for start in range(0, len(new_patv) - (context_len + horizon_len), horizon_len):
                    num_examples += 1
                    inputs.append(new_patv[start:(context_end := start + context_len)])
                    outputs.append(new_patv[context_end:(context_end + horizon_len)])

        return jnp.array(inputs), jnp.array(outputs), num_examples, num_skipped

    num_skipped_all = 0
    x_train, y_train, _, num_skipped = get_batched_data_fn(df_train)
    num_skipped_all += num_skipped
    x_val, y_val, _, num_skipped = get_batched_data_fn(df_val)
    num_skipped_all += num_skipped
    x_test, y_test, _, num_skipped = get_batched_data_fn(df_test)
    num_skipped_all += num_skipped
    x_ood, y_ood, _, num_skipped = get_batched_data_fn(df_ood)
    num_skipped_all += num_skipped
    print(f"Number of skipped examples: {num_skipped_all}")
    return x_train, y_train, x_val, y_val, x_test, y_test, x_ood, y_ood, (data_mean['Patv'], data_std['Patv'])

def load_or_generate_data(rng_key, context_len, horizon_len, max_time_diff=30, **kwargs):
    if os.path.exists(f"datasets/SDWPF/SDWPF_dataset/sdwpf_full/prepared_data_{context_len}.npz"):
        print(f"Data already exists for context length {context_len}. Loading from file.")
        with np.load(f"datasets/SDWPF/SDWPF_dataset/sdwpf_full/prepared_data_{context_len}.npz") as data:
            x_train = jnp.array(data['x_train'])
            y_train = jnp.array(data['y_train'])
            x_val = jnp.array(data['x_val'])
            y_val = jnp.array(data['y_val'])
            x_test = jnp.array(data['x_test'])
            y_test = jnp.array(data['y_test'])
            x_ood = jnp.array(data['x_ood'])
            y_ood = jnp.array(data['y_ood'])
            data_mean = jnp.array(data['data_mean'])
            data_std = jnp.array(data['data_std'])
    else:
        print(f"Generating data for context length {context_len}.")
        x_train, y_train, x_val, y_val, x_test, y_test, x_ood, y_ood, (data_mean, data_std) = gen_data(rng_key, context_len, horizon_len=horizon_len, max_time_diff=max_time_diff)
        print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        np.savez(f"datasets/SDWPF/SDWPF_dataset/sdwpf_full/prepared_data_{context_len}.npz", 
                x_train=x_train,
                y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test,
                x_ood=x_ood, y_ood=y_ood, data_mean=data_mean, data_std=data_std)
    return x_train, y_train, x_val, y_val, x_test, y_test, x_ood, y_ood, (data_mean, data_std)


def setup_metrics(model, batch_size, x, target, n_samples, use_linspace=False, average=True, rescale_fn=lambda x: x):
    if use_linspace:
        get_t = lambda key: jnp.linspace(0, 1., n_samples)
    else:
        get_t = lambda key: random.uniform(key, (n_samples,), minval=0., maxval=1.)

    def acc_fn(rng_key, params):
        x_ = x[:(matching_limit:=(x.shape[0] // batch_size) * batch_size)]  # drop non matching batch size
        x_last = x[matching_limit:]
        y_ = target[:matching_limit]  # drop non matching batch size
        y_last = target[matching_limit:]
        x_ = es('(ib)...->ib...', x_, b=batch_size)
        y_ = es('(ib)...->ib...', y_, b=batch_size)

        def pred(key_param, data):
            x_batch, y_batch = data
            key, param = key_param

            def eval(t):
                loss, _, out = model.nll(param, {}, t, x_batch, y_batch, train=False)  # params, batch_stats, t, x ; out shape (batch_size, output_dim)
                mse = jnp.sum(jnp.mean(jnp.square(rescale_fn(out[0]) - rescale_fn(y_batch)), axis=-1))
                return loss*batch_size, mse  # mean over horizon length but not over batch size
            
            key, subkey = random.split(key)

            # draw samples to approximate curve performance
            t = get_t(subkey)
            loss, mse = jax.vmap(eval)(t)

            if average:
                # mean over samples
                return (key, param), (jnp.sum(loss), jnp.sum(mse))
            else:
                return (key, param), (loss, mse)  # no mean over samples

        # shape (n_data, draws, output_dim)
        rng_key, (loss, mse) = jax.lax.scan(pred, (rng_key, params['params']), (x_, y_))
        # compute loss of last irregular batch
        if len(x_last) > 0:
            rng_key, (loss_last, mse_last) = pred(rng_key, (x_last, y_last))
            loss = jnp.concatenate([loss, loss_last[jnp.newaxis, ...]], axis=0)
            mse = jnp.concatenate([mse, mse_last[jnp.newaxis, ...]], axis=0)
        if average:
            loss = loss.sum() / (n_samples * x.shape[0])
            mse = mse.sum() / (n_samples * x.shape[0])
        else:
            loss = jnp.sum(loss, axis=0) / x.shape[0]
            mse = jnp.sum(mse, axis=0) / x.shape[0]
        return loss, mse  # mean over epochs
    return acc_fn


# def main():
#     logger = wandb.init()
#     config = wandb.config
#     train_curve(logger, config)


class timesfmWrapper:
    def __init__(self, model:timesfm.TimesFm, frequency=0, autoregressive=False):
        self.model = model
        self.freq = frequency  # Frequency of the model, e.g., 0 for daily data
        self.autoregressive = autoregressive  # Set to True if you want to use autoregressive decoding

    def _preprocess(self, inputs, freq) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Formats and pads raw inputs to feed into the model.

        This function both pads each time series to match the context length, and
        pads the inputs to meet the SPMD shape requirement.

        Args:
        inputs: A list of 1d JTensors. Each JTensor is the context time series of
            a single forecast task.
        freq: list of frequencies

        Returns:
        A tuple of:
        - the padded input time series to meet the model required context.
        - the padding indicator.
        - the frequency of each input time series.
        - the number of padded examples for SPMD so that each core has the same
            number (a multiple of `batch_size`) of examples.
        """

        input_ts, input_padding, inp_freq = [], [], []

        pmap_pad = ((len(inputs) - 1) // self.model.global_batch_size +
                    1) * self.model.global_batch_size - len(inputs)

        for i, ts in enumerate(inputs):
            input_len = ts.shape[0]
            padding = jnp.zeros(shape=(input_len + self.model.horizon_len,), dtype=float)
            if input_len < self.model.context_len:
                num_front_pad = self.model.context_len - input_len
                ts = jnp.concatenate([jnp.zeros(shape=(num_front_pad,), dtype=float), ts],
                                    axis=0)
                padding = jnp.concatenate(
                    [jnp.ones(shape=(num_front_pad,), dtype=float), padding], axis=0)
            elif input_len > self.model.context_len:
                ts = ts[-self.model.context_len:]
                padding = padding[-(self.model.context_len + self.model.horizon_len):]

            input_ts.append(ts)
            input_padding.append(padding)
            inp_freq.append(freq[i])

        # # Padding the remainder batch.
        # for _ in range(pmap_pad):
        #     input_ts.append(input_ts[-1])
        #     input_padding.append(input_padding[-1])
        #     inp_freq.append(inp_freq[-1])

        return (
            jnp.stack(input_ts, axis=0),
            jnp.stack(input_padding, axis=0),
            jnp.array(inp_freq).astype(jnp.int32).reshape(-1, 1),
            pmap_pad,
        )


    def _prepare_data(self, x):
        # Prepare the data for the model, e.g., add frequency information
        freq = jnp.full((len(x),1), self.freq, dtype=jnp.int32)
        input_ts = {}
        input_ts['input_ts'], input_ts['input_padding'], input_ts['freq'], _ = self._preprocess(x, freq)
        input_ts['input_padding'] = input_ts['input_padding'][:, :input_ts['input_ts'].shape[1]]
        return input_ts

    def apply(self, params, x, train=False, mutable=False):
        param_model = self.model._model.bind(params)
        if self.autoregressive:
            # use decode of patched_decoder model to run autoregressive decoding for larger horzion lengths than output_patch_len=128
            freq = jnp.full((len(x),1), self.freq, dtype=jnp.int32)
            input_ts = {}
            input_ts['input_ts'], input_ts['input_padding'], input_ts['freq'], _ = self._preprocess(x, freq)
            return param_model.decode(inputs=input_ts,
                                      horizon_len=self.model.horizon_len)
        else:
            # use __call__ of patched_decoder model to run single forward pass with horizon length equal to output_patch_len=128
            input_ts = self._prepare_data(x)
            fprop_outputs = param_model(input_ts)['output_ts'] # shape (BatchShape, Tokens, output_patch_len, output_dim) output_dim is 10 for quantile regression tasks
            o_batch_len = self.model.output_patch_len
            return (fprop_outputs[:, -1, :o_batch_len, 0], fprop_outputs[:, -1, :o_batch_len, :])

    def init(self, key, x):
        return self.model._model.init(key, self._prepare_data(x))


def train(logger, config):
    rng_key = random.PRNGKey(config['rng_seed'])
    EVAL_EVERY_N_BATCH = config['eval_every_n_batch']
    BATCH_SIZE_EVAL = config['curve_params']['batch_size']

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
                    pretrained: bool,
                    weight_decay: float = 0.0, 
                    remove_initial_permutation: bool = False,
                    net_kwargs: dict = None,
                    filter_masks: list = None,
                    jitter: float = 0.0,
                    lora_params: dict = None):
        rng_key, init_key = random.split(rng_key)

        print(f"Initialize TimesFM model {batch_size}, {net_kwargs}")

        class noJitDecodeTimes(timesfm.TimesFm):
            def jit_decode(self):
                pass
        model_a = noJitDecodeTimes(
            hparams=timesfm.TimesFmHparams(
                backend='gpu',
                per_core_batch_size=batch_size,
                num_layers=50,
                **net_kwargs
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-jax"),
        )
        model = timesfmWrapper(model_a, frequency=0, autoregressive=net_kwargs['horizon_len'] > 128) # learnd output_patchlen was 128. So larger horizon lengths require autoregressive decoding
        print("Model is initialized")

        print("Generate data")
        times = time.time()
        x, y, x_val, y_val, x_test, y_test, x_ood, y_ood, (data_mean, data_std) = load_or_generate_data(config['rng_seed'], **net_kwargs, max_time_diff=30)  # get fold from seed
        del x_ood, y_ood  # remove ood data for now
        rescale_fn = lambda x: (x * data_std) + data_mean  # rescale function for predictions
        # drop non matching batch size
        x = x[:(x.shape[0]// batch_size) * batch_size]
        y = y[:(x.shape[0]// batch_size) * batch_size]
        print(f"Data is generated in {(time.time() - times):.2f} seconds")
        print("Shapes: ", (x.shape, y.shape), (x_val.shape, y_val.shape), (x_test.shape, y_test.shape))

        curve_mask = jax.tree.map(lambda x: False, model_a._train_state.mdl_vars['params'])
        # only if k > 0, otherwise no curve parameters are used
        if k > 0:
            for f in filter_masks:
                curve_mask = filter_mask(curve_mask, *f['keys'], as_curve=True, BoolschenOperator=eval(f['op']))

        if "freeze_other_params" in optimizer_conf and optimizer_conf["freeze_other_params"]:
            train_mask = jax.tree.map(lambda x: False, model_a._train_state.mdl_vars['params'])
            # all curve and lora params are trainable
            for f in filter_masks:
                train_mask = filter_mask(train_mask, *f['keys'], as_curve=True, BoolschenOperator=eval(f['op']))
            if lora_params['use_lora']:
                for f in lora_params['filter_masks']:
                    train_mask = filter_mask(train_mask, *f['keys'], as_curve=True, BoolschenOperator=eval(f['op']))
        else:
            train_mask = jax.tree.map(lambda x: True, model_a._train_state.mdl_vars['params'])
        # set LoRA mask for all parameters
        lora_mask = jax.tree.map(lambda x: False, model_a._train_state.mdl_vars['params'])  # LoRA mask for all parameters
        if lora_params['use_lora']:
            for f in lora_params['filter_masks']:
                lora_mask = filter_mask(lora_mask, *f['keys'], as_curve=np.array(f['dims']), BoolschenOperator=eval(f['op']))
        # pprint.pprint(f"LoRA mask: {lora_mask}")
        # pprint.pprint(f"Curve mask: {curve_mask}")
        # pprint.pprint(f"Train mask: {train_mask}")

        perm_spec = False
        if remove_initial_permutation:
            raise NotImplementedError("Removing initial permutation is not implemented for this model.")
            perm_spec = resnet20_permutation_spec2()


        # add weight decay to optimizer if adamw is specified
        if (weight_decay > 0.0) and (optimizer_conf['name'] == 'adamw'):
            optimizer_conf['kwargs']['weight_decay'] = weight_decay
            weight_decay = 0.0  # set to 0.0 to avoid double weight decay in model
        if regularizer:
            print("Using LoRARepulsiveQuantileSubspace with regularizer")
            s_model = LoRARepulsiveQuantileSubspace(lora_mask=lora_mask, r=lora_params['r'],
                                                    model=model, k=k, weight_decay=weight_decay,
                                                    quantiles=model_a._model.quantiles, 
                                                    perm_spec=perm_spec,
                                                    natural_parameterization=natural_parameterization,
                                                    mutuable_param_name=False,
                                                    gravity=gravity, energy_weakening=energy_weakening, regularizer=regularizer)
        else:
            print("Using QuantileSubspace without repulsive force")
            s_model = LoRAQuantileSubspace(lora_mask=lora_mask, r=lora_params['r'],
                                           model=model, k=k, weight_decay=weight_decay, 
                                           quantiles=model_a._model.quantiles,
                                           perm_spec=perm_spec,
                                           natural_parameterization=natural_parameterization,
                                           mutuable_param_name=False)

        # load params from wandb artifact if path is given in config
        if pretrained:
            print("Initialize model with pretrained parameters")
            assert k > 0, "Subspace dimension k must be greater than 0 when training a curve from point."

            # initialize curve parameter with pretrained point parameter
            params = s_model.init_params_from_point(init_key, model_a._train_state.mdl_vars, curve_mask, noise_fn=lambda p, key: p+jitter*jax.random.normal(key, p.shape, dtype=p.dtype))
            
            del model_a._train_state.mdl_vars['params']  # free memory
            
            init_metrics_fn = setup_metrics(s_model, BATCH_SIZE_EVAL, x_val, y_val, 1, use_linspace=True, average=True, rescale_fn=rescale_fn)
            loss, mse = init_metrics_fn(init_key, params)
            logger.summary.update({'init_val_loss': loss, 'init_val_mse': mse})
            init_metrics_fn = setup_metrics(s_model, BATCH_SIZE_EVAL, x_test, y_test, 1, use_linspace=True, average=True, rescale_fn=rescale_fn)
            loss, mse  = init_metrics_fn(init_key, params)
            logger.summary.update({'init_test_loss': loss, 'init_test_mse': mse})
        else:
            del model_a._train_state.mdl_vars['params']  # free memory
            assert lora_params['use_lora'] == False, "LoRA only possible for finetuning."
            
            # initialize curve parameter with random parameters
            params = s_model.init_params(init_key, x[:2], mask=curve_mask)
        
        train_mask = s_model.update_mask_for_lora(train_mask, params['params'], b_off=False, w0_off=True) # w0 must be off for LoRA 
        s_model.set_train_mask(train_mask)

        # compute number of parameters
        n_params = sum(p.size for p in jax.tree.flatten(params)[0])
        logger.summary.update({'n_params': n_params})
        print(f"Number of parameters: {n_params:_}")

        jnp.save(f'tmp_files/{logger.id}_init_params.npy', params)
        artifact.add_file(f'tmp_files/{logger.id}_init_params.npy')

        # Define optimizer
        if isinstance(optimizer_conf['kwargs']['learning_rate'], dict):
            lr_schedule = getattr(optax, optimizer_conf['kwargs']['learning_rate']['name'])(
                **optimizer_conf['kwargs']['learning_rate']['kwargs'])
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
        # loss, params, opt_state, grad_ = s_model.train_step(
        #             random.PRNGKey(0), params, x[:32], y[:32], opt_state, optimizer)
        # print(f"Initial loss: {loss}, grad norm: {jnp.linalg.norm(masked_pytree_to_matrix(grad_, mask, k))}")

        best_loss = np.inf
        best_params = params

        valid_metrics_fn = setup_metrics(s_model, BATCH_SIZE_EVAL, x_val, y_val, n_samples=10, rescale_fn=rescale_fn)

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
            def train_batch(carry, batch):
                x_, y_, batch_idx = batch
                rng_key, params, opt_state, best_loss, best_params = carry
                rng_key, subkey = random.split(rng_key)
                loss, params, opt_state, grad_ = s_model.train_step(
                    subkey, params, x_, y_, opt_state, optimizer)
                # validate
                rng_key, valid_key = random.split(rng_key)

                # mse, val_loss = valid_metrics_fn(valid_key, params)
                val_loss, mse = jax.lax.cond((batch_idx % EVAL_EVERY_N_BATCH) == 0, lambda x: valid_metrics_fn(
                    valid_key, params), lambda x: (jnp.inf, -jnp.inf), None)
                best_params = jax.lax.cond(
                    val_loss < best_loss, lambda x: params, lambda x: best_params, None)
                best_loss = jnp.minimum(best_loss, val_loss)

                # cp_w = masked_pytree_to_matrix(params['params'], s_model.curve_mask, k)
                # length = bezier_length(cp_w)
                # l_bound = lower_bound(cp_w)
                # u_bound = upper_bound(cp_w)
                # gyra = gyration(cp_w)
                # center = mass_center(cp_w)
                length = 0.
                l_bound = 0.
                u_bound = 0.
                gyra = 0.
                center = 0.
                
                # mean_curv = mean_curvature2(cp_w)
                # rel_center = rel_center_fn(cp_w)
                rel_center = 0.
                grad_norm = jnp.linalg.norm(masked_pytree_to_matrix(grad_, s_model.curve_mask, k), axis=1).mean()
                # cos_sim = s_model.summed_cosine_similarity(params['params'])
                # cos_sim /= ((k+1)*k/2)
                # grad_norm = 0.
                cos_sim = 0.
                return (rng_key, params, opt_state, best_loss, best_params), (loss, val_loss, mse, length, l_bound, u_bound, gyra, center, rel_center, grad_norm, cos_sim)
        else:
            def train_batch(carry, batch):
                x_, y_, batch_idx = batch
                rng_key, params, opt_state, best_loss, best_params = carry
                rng_key, subkey = random.split(rng_key)
                loss, params, opt_state, grad_ = s_model.train_step(
                    subkey, params, x_, y_, opt_state, optimizer)
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
            x_ = es('(ib)...->ib...', x[shuffel], b=batch_size)
            y_ = es('(ib)...->ib...', y[shuffel], b=batch_size)
            carry, (loss, val_loss, val_mse, length, l_bound, u_bound, gyra, center, rel_center, grad_norm, cos_sim) = jax.lax.scan(
                train_batch, carry, (x_, y_, jnp.arange(x_.shape[0])))
            return carry, (loss, val_loss, val_mse, length, l_bound, u_bound, gyra, center, rel_center, grad_norm, cos_sim)

        # # print memory stats
        # memory_stats = jax.devices()[0].memory_stats()
        # for stats_name, v in memory_stats.items():
        #     print(f"{stats_name}: {v / (1024 * 1024):.2f} MB" if "bytes" in stats_name else f"{stats_name}: {v}")
        

        # # jax.profiler.save_device_memory_profile(f'memory_profile/memory_profile_0.json')
        # for e in tqdm(range(num_epochs)):
        #     print(f"Epoch {e+1}/{num_epochs}")
        #     rng_key, subkey = random.split(rng_key)
        #     shuffel = jax.random.permutation(subkey, jnp.arange(y.shape[0]))
        #     x_ = es('(ib)...->ib...', x[shuffel], b=batch_size)
        #     y_ = es('(ib)...->ib...', y[shuffel], b=batch_size)

        #     # train on batches
        #     batch_loop = tqdm(enumerate(zip(x_, y_)), leave=False)
        #     for b, (xb_, yb_) in batch_loop:
        #         rng_key, subkey = random.split(rng_key)
        #         # jax.profiler.save_device_memory_profile(f'mem_profile/memory_profile_{e}_{b}.json')
        #         loss, params, opt_state, grad_ = s_model.train_step(
        #             subkey, params, xb_, yb_, opt_state, optimizer)
        #         grad_norm = jnp.linalg.norm(masked_pytree_to_matrix(grad_, mask, k), axis=1).mean()

        #         batch_loop.set_postfix({"Loss": loss})
        #         logs = {
        #             'epoch': e+1, 'batch': b+1, 'stepp': e*x_.shape[0]+b+1,
        #             'train_loss': loss,
        #             'grad_norm': grad_norm
        #         }
        #         if jnp.isnan(loss):
        #             print(f"Loss is NaN at epoch {e+1}, batch {b+1}. Stopping training.")
        #             break
                
        #         # validate
        #         rng_key, valid_key = random.split(rng_key)
        #         if b % EVAL_EVERY_N_BATCH == 0:
        #             val_loss, mse = valid_metrics_fn(valid_key, params)
        #             logs = {**logs, 'valid_loss': val_loss, 'valid_mse': mse}
        #             if val_loss < best_loss:
        #                 best_params = params
        #                 best_loss = val_loss
        #         logger.log(logs)
                
        #     batch_loop.close()
        #     if jnp.isnan(loss):
        #         break
        # last_params = params  

        # jax.profiler.save_device_memory_profile(f'memory_profile/memory_profile_1.json')

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

        # print memory stats
        memory_stats = jax.devices()[0].memory_stats()
        for stats_name, v in memory_stats.items():
            print(f"{stats_name}: {v / (1024 * 1024):.2f} MB" if "bytes" in stats_name else f"{stats_name}: {v}")

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

        if k > 0:
            # Bezier curve length        
            cp = masked_pytree_to_matrix(best_params['params'], s_model.curve_mask, k)
            length_best = bezier_length(cp)
            cp = masked_pytree_to_matrix(last_params['params'], s_model.curve_mask, k)
            length_last = bezier_length(cp)
            logger.log({'Curve length best': length_best, 'Curve length last': length_last})

        # plot likelihood and acc along curve
        fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
        t_space = jnp.linspace(0, 1, 25)
        valid_metrics_fn_final = setup_metrics(s_model, BATCH_SIZE_EVAL, x_val, y_val, n_samples=25, use_linspace=True, average=False, rescale_fn=rescale_fn)
        train_metrics_fn_final = setup_metrics(s_model, BATCH_SIZE_EVAL, x, y, n_samples=25, use_linspace=True, average=False, rescale_fn=rescale_fn)
        test_metrics_fn_final = setup_metrics(s_model, BATCH_SIZE_EVAL, x_test, y_test, n_samples=25, use_linspace=True, average=False, rescale_fn=rescale_fn)
        # ood_metrics_fn_final = setup_metrics(s_model, BATCH_SIZE_EVAL, x_ood, y_ood, n_samples=25, use_linspace=True, average=False, rescale_fn=rescale_fn)
    
        # # plot best params
        # # valid
        # nll, mse = valid_metrics_fn_final(rng_key, best_params)
        # logger.log({'mse_val_curve': mse.mean(), 'nll_val_curve': nll.mean(), 'mse_val_curve_min': mse.min(), 'nll_val_curve_min': nll.min()})
        # ax[0].plot(t_space, -nll, label="valid ll", c=plt.get_cmap('tab10')(1))
        # ax2 = ax[0].twinx()
        # ax2.plot(t_space, mse, label="valid mse",
        #          c=plt.get_cmap('tab10')(1), linestyle="--")
        # # train
        # nll, mse = train_metrics_fn_final(rng_key, best_params)
        # ax[0].plot(t_space, -nll, label="train", c=plt.get_cmap('tab10')(0))
        # ax2.plot(t_space, mse, label="train mse",
        #          c=plt.get_cmap('tab10')(0), linestyle="--")
        # # test
        # nll, mse = test_metrics_fn_final(rng_key, best_params)
        # logger.log({'mse_test_curve': mse.mean(), 'nll_test_curve': nll.mean(), 'mse_test_curve_min': mse.min(), 'nll_test_curve_min': nll.min()})
        # ax[0].plot(t_space, -nll, label="test", c=plt.get_cmap('tab10')(2))
        # ax2.plot(t_space, mse, label="test mse",
        #          c=plt.get_cmap('tab10')(2), linestyle="--")
        # ax[0].set_xlabel("t")
        # ax[0].set_ylabel("mean log likelihood")
        # ax[0].set_title("Best params")
        
        # plot last params
        # valid
        nll, mse = valid_metrics_fn_final(rng_key, last_params)
        logger.log({'mse_val_curve_last': mse.mean(), 'nll_val_curve_last': nll.mean(), 'mse_val_curve_last_min': mse.min(), 'nll_val_curve_last_min': nll.min()})
        ax[1].plot(t_space, -nll, label="valid ll", c=plt.get_cmap('tab10')(1))
        ax2 = ax[1].twinx()
        ax2.plot(t_space, mse, label="valid mse",
                 c=plt.get_cmap('tab10')(1), linestyle="--")
        # # train
        # nll, mse = train_metrics_fn_final(rng_key, last_params)
        # ax[1].plot(t_space, -nll, label="train", c=plt.get_cmap('tab10')(0))
        # ax2.plot(t_space, mse, label="train mse",
        #          c=plt.get_cmap('tab10')(0), linestyle="--")
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
        # out, _ = jax.vmap(s_model, in_axes=(None,None,0,None,None))(last_params['params'], {}, t_space, x_val[:2], False)
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

        plt.tight_layout()
        logger.log(
            {'Curve Predictive for x_val[0]': wandb.Image(fig)})


        return rng_key, params, model, last_params

    rng_key, params, model, last_params = train_curve(
        rng_key, **config['curve_params'], net_kwargs=config['net_kwargs'], lora_params=config.get('lora_params', None))

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
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained set.")
    parser.add_argument("--use-sweep", action="store_true", help="Use sweep configuration.")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for training. Default is 128.")

    args = parser.parse_args()

    if args.train == "single":
        print("Single model training")
        config = {
            'rng_seed': 5,
            'net_kwargs':{'horizon_len': 128,
                          'use_positional_embedding': False,
                          'context_len': 8*60*24//10
                          },
            'eval_every_n_batch': 10,
            'curve_params': {'k': 0,
                            'num_epochs': 100,
                            'batch_size': args.batch_size,
                            'natural_parameterization': False,
                            'gravity': 0.0,
                            'energy_weakening': 1., 
                            'regularizer': False,
                            'weight_decay': 0.,
                            'remove_initial_permutation': False,
                            'jitter': 1e-2,
                            'pretrained': args.pretrained,
                            'filter_masks': [{'keys': ("value", "x_layers_0"), 'op': 'all'}],
                            #  'optimizer_conf': {'name': "adamw", 'kwargs': {'learning_rate': 0.05, 'weight_decay': 0.0001}},
                            #  'optimizer_conf': {'name': "adamw", 'learning_rate': 0.05, 'momentum': 0.9, 'nesterov': True},
                                'optimizer_conf': {'name': "sgd", 
                                                   'kwargs': {'learning_rate': 0.0005, 'momentum': 0.9, 'nesterov': True},
                                                   'grad_clip_norm': 15.0,
                                                   'freeze_other_params': True},
                            # 'optimizer_conf': {'name': "sgd", 'kwargs': {'learning_rate': {'name': "linear_onecycle_schedule", 
                                                                                            # 'kwargs': {'transition_steps': 490_000, 'peak_value': 0.1, 'pct_start': 0.2, 'pct_final': 0.6, 'div_factor': 5.0, 'final_div_factor': 10.0}},
                                                                            # 'momentum': 0.9, 
                                                                            # 'nesterov': True},},
                        }}
        logger = wandb.init(project="timeSeries_PtoC", name="timeSeries_single", entity="ddold", config=config)
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
            config = {
                'rng_seed': 5,
                'net_kwargs':{'horizon_len': 128,
                              'use_positional_embedding': False,
                              'context_len': 8*60*24//10
                              },
                'eval_every_n_batch': 50,
                'lora_params': {'use_lora': True, 'r': 8, 
                                'filter_masks': 
                                    [{'keys': ("query", "x_layers_0", "w"), 'op': 'all', 'dims': [0, 2]}]},  # use lora for query projection
                'curve_params': {'k': 2,
                                'num_epochs': 2,
                                'batch_size': args.batch_size,
                                'natural_parameterization': False,
                                'gravity': 0.0,
                                'energy_weakening': 1., 
                                'regularizer': False,
                                'weight_decay': 0.,
                                'remove_initial_permutation': False,
                                'jitter': 1e-2,  # np.sqrt(2./fan_in, dtype=np.float32) fan_in = 1280 * 16 or 1280 = 1e-2
                                'pretrained': args.pretrained,
                                'filter_masks': 
                                    [{'keys': ("value", "x_layers_0"), 'op': 'all'},
                                     {'keys': ("query", "x_layers_0"), 'op': 'all'}],  # filter query projection
                                #  'optimizer_conf': {'name': "adamw", 'kwargs': {'learning_rate': 0.05, 'weight_decay': 0.0001}},
                                #  'optimizer_conf': {'name': "adamw", 'learning_rate': 0.05, 'momentum': 0.9, 'nesterov': True},
                                    'optimizer_conf': {'name': "sgd", 
                                                       'kwargs': {'learning_rate': 0.0001, 'momentum': 0.9, 'nesterov': True},
                                                       'grad_clip_norm': 15.0,
                                                       'freeze_other_params': True},
                                # 'optimizer_conf': {'name': "sgd", 'kwargs': {'learning_rate': {'name': "linear_onecycle_schedule", 
                                                                                                # 'kwargs': {'transition_steps': 490_000, 'peak_value': 0.1, 'pct_start': 0.2, 'pct_final': 0.6, 'div_factor': 5.0, 'final_div_factor': 10.0}},
                                                                                # 'momentum': 0.9, 
                                                                                # 'nesterov': True},},
                            }}
            logger = wandb.init(project="timeSeries_PtoC", name="timeSeries_curve", entity="ddold", config=config)
            train(logger, config)

        # main()

    # main()
