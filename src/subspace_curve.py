import jax.numpy as jnp
from jax import jit, random
import jax
import optax
import numpy as np
from functools import partial
from jax import custom_jvp
from typing import Callable, Tuple
from numpyro import distributions as dist
from jaxopt import Bisection
from jax.scipy.special import gammaln
from src.weight_matching import PermutationSpec, apply_permutation, weight_matching
from flax import traverse_util
from copy import deepcopy
# from flax.core import freeze, unfreeze

__all__ = ['OrthoSpan', 'SubspaceModel', 'pytree_to_matrix', 'pytree_to_vec',
           'vec_to_pytree', 'bezier_curve', 'UniformTSubspace', 'UniformTCategory']

@custom_jvp
def pos_pow(x, pow):
    # we expect only positive powers => multiple grad is 0 istead of nan
    return jnp.where(pow < 0, 0, jnp.pow(x, pow))

def pos_pow_jvp(primals, tangents):
    x, pow = primals
    grad = pow * pos_pow(x, pow-1)
    grad *= tangents[0]
    return pos_pow(x, pow), grad
pos_pow.defjvp(pos_pow_jvp)


def comb(N, k):
  return jnp.exp(gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1)).round()

def bezier_coeff_fn(num_bends) -> Callable:
    range = jnp.arange(0, num_bends)
    rev_range = jnp.arange((num_bends - 1), -1, -1)
    binom_ = comb(num_bends - 1, jnp.arange(num_bends))

    def wrapper(t):
        return binom_ * pos_pow(t, range) * pos_pow((1.0 - t), rev_range)
    return wrapper

def bezier_curve(num_bends, cp):
    """
    Compute a Bezier curve and its derivative.

    Args:
        num_bends (int): The number of bends in the curve.
        cp (ndarray): Control points of the curve.

    Returns:
        tuple: A tuple containing two functions:
            - f (function): The Bezier curve function that takes a parameter `t`(only single t value) and returns the corresponding point on the curve.
            - derivative_bezier (function): The derivative of the Bezier curve function that takes a parameter `t` and returns the derivative at that point.
    """
    bezier_coeff_inv = bezier_coeff_fn(num_bends - 1)
    bezier_coeff = bezier_coeff_fn(num_bends)

    def derivative_bezier(t):
        n = cp.shape[0] - 1
        coeff = bezier_coeff_inv(t)
        cp_diff = cp[1:] - cp[:-1]
        return jnp.einsum('j,j...->...', coeff, cp_diff) * n

    # forward function
    @custom_jvp
    def f(t):
        p_t = jnp.einsum('k,k...->...', bezier_coeff(t), cp)
        return p_t

    def f_jvp(primals, tangents):
        t = primals[0]
        tangent_out = derivative_bezier(t)
        primal_out = f(t)
        return primal_out, tangent_out*tangents[0]

    f.defjvp(f_jvp)
    return f, derivative_bezier


class OrthoSpan:
    """
    A class that defines the transformation between the subspace spanned by the control points and the weight space.

    Args:
        cp (torch.Tensor): A tensor representing the set of vectors that span the subspace. shape (k+1, dim)

    Attributes:
        mu (torch.Tensor): The mean vector of the input vectors. shape (dim,)
        pi (torch.Tensor): The projection matrix from the subspace to the weight space. shape (k, dim)
        pi_inv (torch.Tensor): The inverse projection matrix. shape (dim, k)

    Methods:
        __call__(self, p): Applies the projection matrix to the input tensor.
        inv(self, p): Applies the inverse projection matrix to the input tensor.

    """

    def __init__(self, cp):
        super().__init__()
        subspace_dim = cp.shape[0] - 1
        self.mu = cp.mean(0)
        cp -= self.mu
        u, self.s, vh = jnp.linalg.svd(cp, full_matrices=False) # u: (k+1, k+1), s: (k+1,), vh: (k+1, dim)
        self.pi = self.s[:subspace_dim, None] * vh[:subspace_dim]
        self.pi_inv = vh[:subspace_dim].T @ jnp.diag(1/self.s[:subspace_dim])

    def __call__(self, p) -> jnp.ndarray:
        """
        Applies the projection matrix to the input tensor.

        Args:
            p (torch.Tensor): The input tensor. shape (n, k)

        Returns:
            torch.Tensor: The tensor after applying the projection matrix.

        """
        return p @ self.pi + self.mu

    def inv(self, p) -> jnp.ndarray:
        """
        Applies the inverse projection matrix to the input tensor.

        Args:
            p (torch.Tensor): The input tensor. shape (n, dim)

        Returns:
            torch.Tensor: The tensor after applying the inverse projection matrix.

        """
        return (p - self.mu) @ self.pi_inv
    

# class OrthoSpanQR(OrthoSpan):
#     def __init__(self, cp):
#         subspace_dim = cp.shape[0] - 1
#         self.mu = cp.mean(0)
#         cp -= self.mu
#         Q, R = jnp.linalg.qr(cp)
#         self.pi = self.s[:subspace_dim, None] * vh[:subspace_dim]
#         self.pi_inv = vh[:subspace_dim].T @ jnp.diag(1/self.s[:subspace_dim])


class BezierTspaceUnifrom(dist.Uniform):
    def __init__(self, d_bezier, validate_args=None):
        self.d_bezier = d_bezier
        tt = jnp.linspace(0., 1., 1_000)
        bezier_grad = jax.vmap(d_bezier)(tt)
        length = jnp.trapezoid(jnp.linalg.norm(bezier_grad, axis=-1), tt)
        self.total_length = length
        print("Total length: ", length)
        super(BezierTspaceUnifrom, self).__init__(validate_args=validate_args)

    @staticmethod
    def arc_length(t, d_bezier):
        tt = jnp.linspace(0., t, 1_000)
        grads = jax.vmap(d_bezier)(tt)
        return jnp.trapezoid(jnp.linalg.norm(grads, axis=-1), tt)

    def t_from_length(self, length):
        def loss(t):
            return self.arc_length(t, self.d_bezier) - length
        root_solver = Bisection(loss, lower=0., upper=1.,
                                check_bracket=False, tol=1e-12, maxiter=30)
        return root_solver.run()

    def sample(self, key, sample_shape=()):
        s = random.uniform(key, sample_shape) * self.total_length
        shape = s.shape
        t = jax.vmap(self.t_from_length)(s.flatten())
        return t.params.reshape(shape)
        # return t

    def log_prob(self, value):
        absdet = jnp.linalg.norm(jax.vmap(self.d_bezier)(value), axis=-1)
        return jnp.log(absdet) - jnp.log(self.total_length)

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, q):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def variance(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError


class SubspaceBaseModel:
    """
    A class representing a subspace model which can be used to train the curve model.

    Parameters:
    - model: The underlying model used in the subspace model.
    - k: The number of parameter sets to sample.
    - mutuable_param_name (str or bool, optional): Name of the mutable parameter or False if not used. Default is False.

    Attributes:
    - model: The underlying model used in the subspace model.
    - k: The number of parameter sets to sample.
    - bezier: The BezierCoeff object for computing Bezier coefficients.

    Methods:
    - init_params(x): Initializes the parameters of the model.
    - __call__(params, t, x): Computes the output of the model for given parameters, time, and input.
    - nll(params, t, x, y): Computes the negative log-likelihood loss of the model.
    - compute_loss(key, params, x, y, n_samples=1): Computes the loss of the model.
    - train_step(key, params, x, y, opt_state, optimizer): Performs a single training step.

    """

    def __init__(self, model, k, weight_decay=0., perm_spec: PermutationSpec = False, natural_parameterization=False, mutuable_param_name: str | bool = False):
        self.model = model
        self.k = k
        self.bezier = bezier_coeff_fn(k + 1)
        self.weight_decay = weight_decay
        self.perm_spec = perm_spec
        self.mutable_name = mutuable_param_name
        self.compute_loss = self.compute_loss_natural_t if natural_parameterization else self.compute_loss_t
        self.curve_mask = None

    def remove_permutation(self, key, params_list):
        """
        Removes the permutation from the parameters.

        Parameters:
        - params: The parameters of the model.

        Returns:
        - The parameters without permutation.

        """
        def flatten_params(params):
            return {"/".join(k): v for k, v in traverse_util.flatten_dict(params).items()}

        def unflatten_params(flat_params):
            return traverse_util.unflatten_dict({tuple(k.split("/")): v
                                                for k, v in flat_params.items()})

        for i, p in enumerate(params_list[1:]):
            print(f'Remove permutation from k {i+1}')
            key, perm_key = random.split(key)
            final_permutation = weight_matching(perm_key, self.perm_spec,
                                                flatten_params(params_list[0]), flatten_params(p))
            params_list[i] = unflatten_params(apply_permutation(
                self.perm_spec, final_permutation, flatten_params(p)))
        return params_list

    def init_mask(self, x, all_curves=False):
        """
        Initializes the mask for the model parameters.

        Parameters:
        - x: The input data.

        Returns:
        - A pytree of masks with the same structure as the model parameters, where each leaf is a boolean indicating whether the parameter is trainable.

        """
        params = self.model.init(random.PRNGKey(0), x)
        mask = jax.tree.map(lambda x: all_curves, params['params'])
        return mask
    
    def init_params_from_point(self, key, point, mask, jitter):
        """
        Initializes curve parameters from a given parameter point, optionally adding noise and handling permutation invariance.
        Args:
            key: A JAX PRNGKey used for random number generation.
            point (dict): A dictionary containing at least a 'params' key with the base parameters to initialize from.
            mask: A tree-like structure of booleans indicating which parameters should be handled by the curve.
            jitter (float): The standard deviation of the Gaussian noise to add to the parameters.
        Returns:
            dict: A copy of the input `point` dictionary, with the 'params' key replaced by the initialized curve parameters.
        Notes:
            - If `self.perm_spec` is set, permutation invariance is handled by removing permutations from the curve parameters.
            - The method adds Gaussian noise to each parameter in the curve, controlled by `jitter`.
            - Non-curve parameters are reverted to their original values from `point['params']`.
            - The mask used for curve parameters is stored in `self.curve_mask`.
        """
        curve_params = [deepcopy(point['params']) for _ in range(self.k+1)]
        
        # add noise to the parameters
        param, struct = jax.tree.flatten(curve_params)
        keys = random.split(key, len(param))
        def apply_noise(p, key, jitter):
            return p+jitter*jax.random.normal(key, p.shape)
        noise_params = [apply_noise(p, key, jitter=jitter) for p, key in zip(param, keys)]
        curve_params = jax.tree.unflatten(struct, noise_params)
        
        if self.perm_spec:
            curve_params = self.remove_permutation(key, curve_params)
        
        curve_params = jax.tree.map(lambda *x: jnp.stack(x), *curve_params)

        # revert all non-curve parameters to the original point
        curve_params = jax.tree.map(
            lambda p, m, o: p if m else o, curve_params, mask, point['params'])
        all_params = deepcopy(point)
        all_params['params'] = curve_params

        self.curve_mask = mask
        return all_params
    
    def init_params(self, key, x, mask):
        """
        Initializes the parameters of the model.

        Parameters:
        - key: The random key for parameter initialization.
        - x: The input data.

        Returns:
        - The initialized parameters {params, batch_stats}.

        """
        keys = random.split(key, self.k+1)

        # list of params and variables (such as batch statistics)
        all_params = self.model.init(keys[0], x)
        stacked_params = [all_params['params']]
        stacked_params += [self.model.init(k, x)['params'] for k in keys[1:]]
        # if permutation is specified then remove the permutation by using the first point as anchor
        if self.perm_spec:
            stacked_params = self.remove_permutation(key, stacked_params)
        # now pytree of stacked params (each leaf has leading dim of k+1)
        all_params['params'] = jax.tree.map(
            lambda m, *x: jnp.stack(x) if m else x[0], mask, *stacked_params)

        self.curve_mask = mask
        return all_params
    

    @partial(jit, static_argnums=(0,))
    def _predict(self, params, state, inputs, train) -> Tuple[jnp.ndarray, dict]:
        # Predict out from inputs and parameters
        if self.mutable_name:
            all_params = {'params': params, self.mutable_name: state}
            def train_fn(inputs):
                out, net_state = self.model.apply(
                    all_params, inputs, train=True, mutable=self.mutable_name)
                return out, net_state[self.mutable_name]
        else:
            all_params = {'params': params}
            def train_fn(inputs):
                out = self.model.apply(all_params, inputs, train=True, mutable=False)
                return out, {}

        def eval_fn(inputs):
            out = self.model.apply(all_params, inputs,
                                   train=False, mutable=False)
            return out, state

        out, net_state = jax.lax.cond(
            train, train_fn, eval_fn, inputs
        )
        return out, net_state

    @partial(jit, static_argnums=(0,))
    def __call__(self, params, state, t, x, train=True) -> Tuple[jnp.ndarray, dict]:
        """
        Computes the output of the model for given parameters, time, and input.

        Parameters:
        - params: The parameters of the model.
        - t: The time parameter. Only single value supported.
        - x: The input data.

        Returns:
        - The output of the model.

        """
        # sample Bezier coefficient
        bezier_coeff = self.bezier(t)
        # Compute one parameter set per sample
        params = jax.tree.map(lambda p, m: jnp.einsum(
            'k,k...->...', bezier_coeff, p) if m else p, params, self.curve_mask)

        # forward pass per sample
        out, state = self._predict(params, state, x, train=train)
        return out, state

    @partial(jit, static_argnums=(0,))
    def nll(self, params, state, t, x, y, train:bool) -> Tuple[jnp.ndarray, dict, jnp.ndarray]:
        """
        Abstract method for computing the negative log-likelihood loss of the model.

        Parameters:
        - params: The parameters of the model.
        - t: The time parameter.
        - x: The input data.
        - y: The target output.

        Returns:
        - The negative log-likelihood loss.

        Raises:
        - NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("nll must be implemented by subclasses of SubspaceBaseModel.")

    @staticmethod
    def l2_loss(x, alpha):
        return alpha * (x ** 2).sum()

    @partial(jit, static_argnums=(0,))
    def compute_loss_t(self, key, params, state, x, y):
        """
        Computes the loss of the model.

        Parameters:
        - key: The random key for generating random numbers.
        - params: The parameters of the model.
        - x: The input data.
        - y: The target output.
        - n_samples: The number of samples to use for computing the loss.

        Returns:
        - The computed loss.

        """
        # Sample t param of Bezier curve
        t = random.uniform(key, (1,), minval=0., maxval=1.)
        loss, state, out = self.nll(params, state, t, x, y, train=True)
        loss = loss.mean() + sum(self.l2_loss(w, alpha=self.weight_decay)
                                 for w in jax.tree.leaves(params))
        return loss, state

    @partial(jit, static_argnums=(0,))
    def compute_loss_natural_t(self, key, params, state, x, y):
        """
        Computes the loss of the model.

        Parameters:
        - key: The random key for generating random numbers.
        - params: The parameters of the model.
        - x: The input data.
        - y: The target output.
        - n_samples: The number of samples to use for computing the loss.

        Returns:
        - The computed loss.

        """
        # Sample t param of Bezier curve

        t = random.uniform(key, (1,), minval=0., maxval=1.)
        loss, state, out = self.nll(params, state, t, x, y, train=True)

        raise ValueError("Natural parameterization currently not implemented for SubspaceBaseModel (pytree_to_matrix needs masking)")
        cp_w = pytree_to_matrix(params['params'], self.k)
        # cp_phi = 
        curve, d_bezier = bezier_curve(self.k+1, cp_w)
        tt = jnp.linspace(0., 1., 1_000)
        bezier_grad = jax.vmap(d_bezier)(tt)
        normalized_grad = jnp.trapezoid(
            jnp.linalg.norm(bezier_grad, axis=-1), tt)
        grads = jax.vmap(d_bezier)(t)
        weights = jnp.linalg.norm(grads, axis=-1) / normalized_grad

        # t_dist = BezierTspaceUnifrom(d_bezier)
        # t = t_dist.sample(key, (n_samples,))

        # weighted_loss = jnp.einsum('n,n->', weights, loss)
        weighted_loss = weights.squeeze() * loss
        return weighted_loss + sum(self.l2_loss(w, alpha=self.weight_decay)
                                 for w in jax.tree.leaves(params['params'])), state

    @partial(jit, static_argnums=(0, 6))
    def train_step(self, key, params, x, y, opt_state, optimizer):
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

        (loss, stats_params), grads = jax.value_and_grad(
            self.compute_loss, argnums=1, has_aux=True)(key, params['params'], params[self.mutable_name] if self.mutable_name else {}, x, y)
        # update params
        updates, opt_state = optimizer.update(
            grads, opt_state, params['params'])
        params['params'] = optax.apply_updates(params['params'], updates)
        if self.mutable_name:
            params[self.mutable_name] = stats_params
        return loss, params, opt_state, grads


class CategorySubspace(SubspaceBaseModel):

    @partial(jit, static_argnums=(0,))
    def nll(self, params, state, t, x, y, train:bool=True) -> Tuple[jnp.ndarray, dict, jnp.ndarray]:
        """
        Computes the negative log-likelihood loss of the model.

        Parameters:
        - params: The parameters of the model.
        - t: The time parameter.
        - x: The input data.
        - y: The target output.

        Returns:
        - The negative log-likelihood loss.

        """
        out, state = self(params, state, t, x, train=train)  # shape (n_samples, n_data, output_dim)

        nll = jnp.mean(
            optax.losses.softmax_cross_entropy_with_integer_labels(
                logits=out, labels=y)
        )
        return nll, state, out
    
class RegressionSubspace(SubspaceBaseModel):
    def __init__(self, out_dist_log_scale=jnp.log(1.0), **kwargs):
        """
        Initializes the RegressionSubspace class.

        Parameters:
        - model: The model to be used.
        - k: The number of control points.
        - weight_decay: The weight decay factor for regularization.
        - perm_spec: The permutation specification for the model.
        - natural_parameterization: Whether to use natural parameterization.
        """
        super().__init__(**kwargs)
        self.log_scale = out_dist_log_scale

    @partial(jit, static_argnums=(0,))
    def nll(self, params, state, t, x, y, train:bool=True) -> Tuple[jnp.ndarray, dict, jnp.ndarray]:
        """
        Computes the negative log-likelihood loss of the model.

        Parameters:
        - params: The parameters of the model.
        - t: The time parameter.
        - x: The input data.
        - y: The target output.

        Returns:
        - The negative log-likelihood loss.

        """
        out, state = self(params, state, t, x, train=train)  # shape (n_data, output_dim)
        nll = -jax.scipy.stats.norm.logpdf(y, loc=out.squeeze(axis=-1), scale=jnp.exp(self.log_scale)+1e-8)
        return nll, state, out


class DistRegressionSubspace(SubspaceBaseModel):
    def __init__(self, init_outDist_log_scale=jnp.log(1.), **kwargs):
        """
        Initializes the RegressionSubspace class.

        Parameters:
        - model: The model to be used.
        - k: The number of control points.
        - weight_decay: The weight decay factor for regularization.
        - perm_spec: The permutation specification for the model.
        - natural_parameterization: Whether to use natural parameterization.
        """
        super().__init__(**kwargs)
        self.init_log_scale = jnp.array(init_outDist_log_scale, dtype=jnp.float32)

    def init_params_from_point(self, key, point, mask, jitter):
        mask['log_scale'] = False  # mask log_scale parameter as non curve parameter
        return super().init_params_from_point(key, point, mask, jitter)

    def init_params(self, key, x, mask):
        """
        Initializes the parameters of the model.

        Parameters:
        - key: The random key for parameter initialization.
        - x: The input data.
        - mask: The mask for the parameters.

        Returns:
        - The initialized parameters.

        """
        params = super().init_params(key, x, mask)
        # initialize log scale
        params['params']['log_scale'] = self.init_log_scale
        print(self.curve_mask)
        self.curve_mask['log_scale'] = False  # mask log_scale parameter as non curve parameter
        return params

    @partial(jit, static_argnums=(0,))
    def nll(self, params, state, t, x, y, train:bool=True) -> Tuple[jnp.ndarray, dict, jnp.ndarray]:
        """
        Computes the negative log-likelihood loss of the model.

        Parameters:
        - params: The parameters of the model.
        - t: The time parameter.
        - x: The input data.
        - y: The target output.

        Returns:
        - The negative log-likelihood loss.

        """
        out, state = self(params, state, t, x, train=train)  # shape (n_data, output_dim)
        nll = -jax.scipy.stats.norm.logpdf(y, loc=out.squeeze(axis=-1), scale=jnp.exp(params['log_scale'])+1e-8)
        return nll, state, out
    
class QuantileSubspace(SubspaceBaseModel):
    def __init__(self, model, k, quantiles, weight_decay=0, perm_spec = False, natural_parameterization=False, mutuable_param_name = False):
        super().__init__(model, k, weight_decay, perm_spec, natural_parameterization, mutuable_param_name)
        self.quantiles = jnp.array(quantiles, dtype=jnp.float32)

    @staticmethod
    @jit
    def _quantile_loss(pred, actual, quantile: float):
        """Calculates quantile loss.

        Args:
        pred: B x T
        actual: B x T
        quantile: quantile at which loss is computed.

        Returns:
        per coordinate loss.
        """
        dev = actual - pred
        loss_first = dev * quantile
        loss_second = -dev * (1.0 - quantile)
        return 2 * jnp.where(loss_first >= 0, loss_first, loss_second)

    def nll(self, params, state, t, x, y, train:bool=True) -> Tuple[jnp.ndarray, dict, jnp.ndarray]:
        """
        Computes the negative log-likelihood loss of the model.

        Parameters:
        - params: The parameters of the model.
        - t: The time parameter.
        - x: The input data.
        - y: The target output.

        Returns:
        - The negative log-likelihood loss.

        """
        out, state = self(params, state, t, x, train=train)  # shape (n_samples, n_data, output_dim)

        out_quantile = out[1] # shape (n_samples, batch, tokens, dims, out_dim)

        loss = jnp.square(out_quantile[:,:,0] - y)
        def body_fun(loss, pred_and_quantile):
            loss += self._quantile_loss(pred_and_quantile['preds'], y, quantile=pred_and_quantile['quantile'])
            return loss, _
        loss, _ = jax.lax.scan(body_fun, loss, {'quantile': self.quantiles, 'preds': out_quantile.transpose(2, 0, 1)[1:]})
        return loss.mean(), state, out


class RepulsiveMixin:
    def __init__(self, gravity=0.0, energy_weakening=0.1, regularizer="force", **kwargs):
        super().__init__(**kwargs)
        self.gravity = gravity
        self.energy_weakening = energy_weakening
        self.regularizer = self._get_regularizer(regularizer)

    def _get_regularizer(self, regularizer):
        reg = regularizer.lower()
        if reg == "force":
            return self.repulsive_force
        elif reg == "energy":
            return self.repulsive_energy
        elif reg in ("cosine_similarity", "cosine", "similarity"):
            return self.summed_cosine_similarity_square
        else:
            raise ValueError(f"Unknown regularizer type: {regularizer}")

    def repulsive_force(self, params):
        cp_w = masked_pytree_to_matrix(params, self.curve_mask, self.k)
        n = cp_w.shape[0]
        def body_fun(i, acc):
            diffs = cp_w - cp_w[i]
            dists = jnp.sum(diffs**2, axis=-1)
            dists = 1. / (dists + self.energy_weakening + 1e-8)
            dists = jnp.where(jnp.arange(0, n) > i, dists, 0.)
            return acc + jnp.sum(dists)
        force = jax.lax.fori_loop(0, n - 1, body_fun, 0.0)
        return force

    def repulsive_energy(self, params):
        cp_w = masked_pytree_to_matrix(params, self.curve_mask, self.k)
        n = cp_w.shape[0]
        def body_fun(i, acc):
            diffs = jnp.abs(cp_w - cp_w[i])
            dists = jnp.sum(diffs, axis=-1)
            energy = 1. / (dists + self.energy_weakening + 1e-8)
            log_energy = jnp.log(energy)
            log_energy = jnp.where(jnp.arange(0, n) > i, log_energy, 0.)
            return acc + jnp.sum(log_energy)
        log_energy = jax.lax.fori_loop(0, n - 1, body_fun, 0.0)
        return log_energy

    def summed_cosine_similarity(self, params):
        cp_w = masked_pytree_to_matrix(params, self.curve_mask, self.k)
        n = cp_w.shape[0]
        cp_w = cp_w / jnp.linalg.norm(cp_w, axis=-1, keepdims=True)
        dot_res = jnp.dot(cp_w, cp_w.T)
        upper_triangular_indices = jnp.triu_indices(n, k=1)
        cosine_sim = jnp.sum(dot_res[upper_triangular_indices])
        return cosine_sim

    def summed_cosine_similarity_square(self, params):
        return self.summed_cosine_similarity(params) ** 2
    
    @partial(jit, static_argnums=(0,))
    def compute_loss_t(self, key, params, state, x, y):
        # Sample t param of Bezier curve
        nll, state = super().compute_loss_t(key, params, state, x, y)
        rep_force = self.regularizer(params)
        return nll + self.gravity * rep_force, state

# Define classes:
class RepulsiveRegressionSubspace(RepulsiveMixin, RegressionSubspace):
    pass

class RepulsiveDistRegressionSubspace(RepulsiveMixin, DistRegressionSubspace):
    pass

class RepulsiveCategorySubspace(RepulsiveMixin, CategorySubspace):
    pass

class RepulsiveQuantileSubspace(RepulsiveMixin, QuantileSubspace):
    pass


#@partial(jit, static_argnums=(1))
def pytree_to_matrix(pytree, k):
    """
    Converts a pytree of the subspace model into a matrix.

    Args:
        pytree: The pytree to be converted.

    Returns:
        matrix: The matrix representation of the pytree with shape (k+1, n_params).
    """
    # first argument of reduce fn is the accumulator and second is the current value
    return jax.tree.reduce(lambda x, y: jnp.concatenate(
        [x, y.reshape(k+1, -1)], axis=-1), pytree)

#@partial(jit, static_argnums=(1))
def masked_pytree_to_matrix(pytree, mask, k):
    """
    Converts a pytree of the subspace model into a matrix.

    Args:
        pytree: The pytree to be converted.

    Returns:
        matrix: The matrix representation of the pytree with shape (k+1, n_params).
    """
    # first argument of reduce fn is the accumulator and second is the current value
    only_curve_params = jax.tree.map(lambda x, m: x if m else jnp.reshape(jnp.array([]), (k+1, 0)), pytree, mask)
    return jax.tree.reduce(lambda x, y: jnp.concatenate(
        [x, y.reshape(k+1, -1)], axis=-1), only_curve_params)

# @partial(jit, static_argnums=(1))
def pytree_to_vec(pytree):
    """
    Converts a pytree of the subspace model into a matrix.

    Args:
        pytree: The pytree to be converted.

    Returns:
        matrix: The matrix representation of the pytree with shape (k+1, n_params).
    """
    # first argument of reduce fn is the accumulator and second is the current value
    return jax.tree.reduce(lambda x, y: jnp.concatenate(
        [x, y.reshape(-1)], axis=-1), pytree)

@jit
def vec_to_single_pytree(vec, subspace_params):
    """
    Converts a vector representation of parameters to a pytree structure with no loading dimension.
    opposite of pytree_to_vec function.

    Args:
        vec: A 2-D numpy array or JAX array representing the flattened parameters; shape (k, D).
        subspace_params: A pytree of the subspace parameters. Used to inherit the structure and leaf shapes.

    Returns:
        A pytree structure with the same structure as `subspace_params` and with
        the parameters replaced by the values from `vec`. (Converts the whole parameterset)

    Example:
        vec = np.array([[1, 2, 3, 4]])
        subspace_params = {'a': np.array([[0, 0],]), 'b': np.array([[0, 0, 0],])}
        result = matrix_to_pytree(vec, subspace_params)
        # Output: {'a': np.array([[1, 2]]), 'b': np.array([[3, 4, 0]])}
    """
    leafs_params, structure = jax.tree.flatten(subspace_params)
    # get leaf shapes without stacking dimension
    leaf_vec_shapes = [p.shape for p in leafs_params]

    flatten_leafs = []
    index = 0
    for s in leaf_vec_shapes:
        upper_index = index + np.prod(s)
        flatten_leafs.append(jnp.reshape(vec[index: upper_index], s))
        index = upper_index
    return jax.tree.unflatten(structure, flatten_leafs)


@jit
def vec_to_pytree(vec, subspace_params):
    """
    Converts a vector representation of parameters to a pytree structure.
    With leading dimension.

    Args:
        vec: A 1-D numpy array or JAX array representing the flattened parameters.
        subspace_params: A pytree of the subspace parameters. Used to inherit the structure and leaf shapes.

    Returns:
        A pytree structure with the same structure as `subspace_params` (Without leading dimension), but with
        the parameters replaced by the values from `vec`.

    Example:
        vec = np.array([1, 2, 3, 4])
        subspace_params = {'a': np.array([[0, 0],]), 'b': np.array([[0, 0, 0],])}
        result = vec_to_pytree(vec, subspace_params)
        # Output: {'a': np.array([1, 2]), 'b': np.array([3, 4, 0])}
    """
    leafs_params, structure = jax.tree.flatten(subspace_params)
    # get leaf shapes without stacking dimension
    leaf_vec_shapes = [p.shape[1:] for p in leafs_params]

    flatten_leafs = []
    index = 0
    for s in leaf_vec_shapes:
        upper_index = index + np.prod(s)
        flatten_leafs.append(jnp.reshape(vec[index: upper_index], s))
        index = upper_index
    return jax.tree.unflatten(structure, flatten_leafs)

@jit
def matrix_to_pytree(vec, subspace_params):
    """
    Converts a vector representation of parameters to a pytree structure.

    Args:
        vec: A 2-D numpy array or JAX array representing the flattened parameters; shape (k, D).
        subspace_params: A pytree of the subspace parameters. Used to inherit the structure and leaf shapes.

    Returns:
        A pytree structure with the same structure as `subspace_params` and with
        the parameters replaced by the values from `vec`. (Converts the whole parameterset)

    Example:
        vec = np.array([[1, 2, 3, 4]])
        subspace_params = {'a': np.array([[0, 0],]), 'b': np.array([[0, 0, 0],])}
        result = matrix_to_pytree(vec, subspace_params)
        # Output: {'a': np.array([[1, 2]]), 'b': np.array([[3, 4, 0]])}
    """
    leafs_params, structure = jax.tree.flatten(subspace_params)
    # get leaf shapes without stacking dimension
    leaf_vec_shapes = [p.shape for p in leafs_params]

    flatten_leafs = []
    index = 0
    for s in leaf_vec_shapes:
        upper_index = index + np.prod(s[1:])
        flatten_leafs.append(jnp.reshape(vec[:, index: upper_index], s))
        index = upper_index
    return jax.tree.unflatten(structure, flatten_leafs)
