import jax.numpy as jnp
from jax import jit, random
import jax
import optax
import numpy as np
from functools import partial
from jax import custom_jvp
from typing import Callable
from numpyro import distributions as dist
from jaxopt import Bisection
from jax.scipy.special import gammaln
from src.weight_matching import PermutationSpec, apply_permutation, weight_matching
from flax import traverse_util
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


class SubspaceModel:
    """
    A class representing a subspace model which can be used to train the curve model.

    Parameters:
    - model: The underlying model used in the subspace model.
    - k: The number of parameter sets to sample.

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

    def __init__(self, model, k, n_samples=1, out_scale=0.05, optimize_distparams=False):
        self.model = model
        self.k = k
        self.bezier = jax.vmap(bezier_coeff_fn(k + 1))
        self.n_samples = n_samples
        self.out_scale = out_scale
        self.optimize_distparams = optimize_distparams


    def init_params(self, key, x):
            """
            Initializes the parameters of the model.

            Parameters:
            - key: The random key for parameter initialization.
            - x: The input data.

            Returns:
            - The initialized parameters.

            """
            keys = random.split(key, self.k+1)
            # list of params and variables (such as batch statistics)
            params_and_var = [self.model.init(k, x) for k in keys]
            # now pytree of stacked params (each leaf has leading dim of k+1)
            stacked_params_var = jax.tree.map(
                lambda *x: jnp.stack(x), *params_and_var)
            
            stacked_params_var['dist_params'] = {'log_scale': jnp.log(
                jnp.array(self.out_scale))}
            # only trainable params from pytree
            return stacked_params_var

    @partial(jit, static_argnums=(0,))
    def __call__(self, params, t, x):
        """
        Computes the output of the model for given parameters, time, and input.

        Parameters:
        - params: The parameters of the model.
        - t: The time parameter.
        - x: The input data.

        Returns:
        - The output of the model.

        """
        # sample Bezier coefficient
        bezier_coeff = self.bezier(t)
        # Compute one parameter set per sample
        sample_param = jax.tree.map(lambda p: jnp.einsum(
            'sk,k...->s...', bezier_coeff, p), params)
        sample_param = dict(params=sample_param)
        # forward pass per sample
        out = jax.vmap(self.model.apply, in_axes=(0, None))(sample_param, x)
        return out

    @partial(jit, static_argnums=(0,))
    def nll(self, params, t, x, y):
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
        out = self(params['params'], t, x)  # shape (n_samples, n_data, output_dim)
        nll = -jax.scipy.stats.norm.logpdf(y,
                                           loc=out.squeeze(axis=-1), scale=jnp.exp(params['dist_params']['log_scale'])+1e-8)
        return nll

    @partial(jit, static_argnums=(0, 5))
    def compute_loss(self, key, params, x, y, n_samples=1):
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
        t = random.uniform(key, (n_samples,), minval=0., maxval=1.)
        loss = self.nll(params, t, x, y)
        return loss.mean()
    
    @partial(jit, static_argnums=(0, 5))
    def compute_loss_natural_t(self, key, params, x, y, n_samples=1):
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

        t = random.uniform(key, (n_samples,), minval=0., maxval=1.)
        loss = self.nll(params, t, x, y).mean(-1)
        # jax.debug.print(f'loss shape: {loss.shape}')

        cp_w = pytree_to_matrix(params['params'], self.k)
        curve, d_bezier = bezier_curve(self.k+1, cp_w)
        tt = jnp.linspace(0., 1., 10_000)
        bezier_grad = jax.vmap(d_bezier)(tt)
        normalized_grad = jnp.trapezoid(
            jnp.linalg.norm(bezier_grad, axis=-1), tt)
        grads = jax.vmap(d_bezier)(t)
        weights = jnp.linalg.norm(grads, axis=-1) / normalized_grad

        # t_dist = BezierTspaceUnifrom(d_bezier)
        # t = t_dist.sample(key, (n_samples,))

        weighted_loss = jnp.einsum('n,n->', weights, loss)/n_samples
        return weighted_loss

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
        loss, grads = jax.value_and_grad(
            self.compute_loss, argnums=(1,), )(key, params, x, y, self.n_samples)
        if self.optimize_distparams:
            updates, opt_state = optimizer.update(grads[0], opt_state, params)
            params = optax.apply_updates(params, updates)
        else:
            updates, opt_state = optimizer.update(grads[0]['params'], opt_state, params['params'])
            params['params'] = optax.apply_updates(params['params'], updates)
        return loss, params, opt_state, grads[0]
    

class UniformTSubspace(SubspaceModel):
    # slow implementation and not working because of gradient computation
    # @partial(jit, static_argnums=(0, 5))
    # def compute_loss(self, key, params, x, y, n_samples=1):
    #     """
    #     Computes the loss of the model.

    #     Parameters:
    #     - key: The random key for generating random numbers.
    #     - params: The parameters of the model.
    #     - x: The input data.
    #     - y: The target output.
    #     - n_samples: The number of samples to use for computing the loss.

    #     Returns:
    #     - The computed loss.

    #     """
    #     # Sample t param of Bezier curve
    #     cp_w = pytree_to_matrix(params['params'], self.k)
    #     curve, d_bezier = bezier_curve(self.k+1, cp_w)
    #     t_dist = BezierTspaceUnifrom(d_bezier)
    #     t = t_dist.sample(key, (n_samples,))
    #     loss = self.nll(params, t, x, y)
    #     return loss.mean()
    
    @partial(jit, static_argnums=(0, 5))
    def compute_loss(self, key, params, x, y, n_samples=1):
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

        t = random.uniform(key, (n_samples,), minval=0., maxval=1.)
        loss = self.nll(params, t, x, y).mean(-1)
        # jax.debug.print(f'loss shape: {loss.shape}')

        cp_w = pytree_to_matrix(params['params'], self.k)
        curve, d_bezier = bezier_curve(self.k+1, cp_w)
        tt = jnp.linspace(0., 1., 10_000)
        bezier_grad = jax.vmap(d_bezier)(tt)
        curve_length = jnp.trapezoid(
            jnp.linalg.norm(bezier_grad, axis=-1), tt)
        grads = jax.vmap(d_bezier)(t)
        weights = jnp.linalg.norm(grads, axis=-1) / curve_length

        # t_dist = BezierTspaceUnifrom(d_bezier)
        # t = t_dist.sample(key, (n_samples,))

        weighted_loss = jnp.einsum('n,n->', weights, loss)/n_samples
        return weighted_loss

class CategorySubspace(SubspaceModel):
    def __init__(self, model, k, n_samples):
        super().__init__(model, k, n_samples, np.inf, False)

    @partial(jit, static_argnums=(0,))
    def nll(self, params, t, x, y):
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
        logits = self(params['params'], t,
                      x)  # shape (n_samples, n_data, output_dim)

        nll = jax.vmap(
            optax.softmax_cross_entropy_with_integer_labels, in_axes=(0, None))(logits, y)
        return nll


class UniformTCategory(CategorySubspace, UniformTSubspace):
    pass


class SubspaceModelBatchNorm:
    """
    A class representing a subspace model which can be used to train the curve model.

    Parameters:
    - model: The underlying model used in the subspace model.
    - k: The number of parameter sets to sample.

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

    def __init__(self, model, k, out_scale=0.05, optimize_distparams=False, weight_decay=0., perm_spec: PermutationSpec = False, natural_parameterization=False):
        self.model = model
        self.k = k
        self.bezier = bezier_coeff_fn(k + 1)
        self.out_scale = out_scale
        self.optimize_distparams = optimize_distparams
        self.weight_decay = weight_decay
        self.perm_spec = perm_spec
        self.compute_loss = self.compute_loss_natural_t if natural_parameterization else self.compute_loss_t

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
    
    def init_params(self, key, x):
        """
        Initializes the parameters of the model.

        Parameters:
        - key: The random key for parameter initialization.
        - x: The input data.

        Returns:
        - The initialized parameters.

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
            lambda *x: jnp.stack(x), *stacked_params)

        all_params['dist_params'] = {'log_scale': jnp.log(
            jnp.array(self.out_scale))}
        # only trainable params from pytree
        return all_params

    @partial(jit, static_argnums=(0,))
    def _predict(self, params, bn_params, inputs, train):
        # Predict out from inputs and parameters
        all_params = {"params": params, "batch_stats": bn_params}

        def train_fn(inputs):
            out, net_state = self.model.apply(
                all_params, inputs, train=True, mutable=["batch_stats"]
            )
            return out, net_state

        def eval_fn(inputs):
            out = self.model.apply(all_params, inputs,
                                   train=False, mutable=False)
            net_state = {"batch_stats": bn_params}
            return out, net_state

        out, net_state = jax.lax.cond(
            train, lambda x: train_fn(x), lambda x: eval_fn(x), inputs
        )
        return out, net_state['batch_stats']

    @partial(jit, static_argnums=(0,))
    def __call__(self, params, batch_stats, t, x):
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
        sample_param = jax.tree.map(lambda p: jnp.einsum(
            'k,k...->...', bezier_coeff, p), params)

        # forward pass per sample
        out, batch_stats = self._predict(
            sample_param, batch_stats, x, train=True)
        return out, batch_stats

    def eval(self, params, batch_stats, t, x):
        # sample Bezier coefficient
        bezier_coeff = self.bezier(t)
        # Compute one parameter set per sample
        sample_param = jax.tree.map(lambda p: jnp.einsum(
            'k,k...->...', bezier_coeff, p), params)

        # forward pass per sample
        out, batch_stats = self._predict(
            sample_param, batch_stats, x, train=False)
        return out, batch_stats

    @partial(jit, static_argnums=(0,))
    def nll(self, params, param_stats, t, x, y):
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
        out, batch_stats = self(
            params['params'], param_stats['batch_stats'], t, x)  # shape (n_data, output_dim)
        if self.optimize_distparams:
            nll = -jax.scipy.stats.norm.logpdf(y,
                                               loc=out.squeeze(axis=-1), scale=jnp.exp(params['dist_params']['log_scale'])+1e-8)
        else:
            nll = -jax.scipy.stats.norm.logpdf(y,
                                               loc=out.squeeze(axis=-1), scale=jnp.exp(param_stats['dist_params']['log_scale'])+1e-8)
        return nll, batch_stats

    @staticmethod
    def l2_loss(x, alpha):
        return alpha * (x ** 2).sum()

    @partial(jit, static_argnums=(0,))
    def compute_loss_t(self, key, params, param_stats, x, y):
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
        loss, batch_stats = self.nll(params, param_stats, t, x, y)
        loss = loss.mean() + sum(self.l2_loss(w, alpha=self.weight_decay)
                                 for w in jax.tree.leaves(params))
        return loss, batch_stats

    @partial(jit, static_argnums=(0,))
    def compute_loss_natural_t(self, key, params, param_stats, x, y):
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
        loss, batch_stats = self.nll(params, param_stats, t, x, y)

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
                                 for w in jax.tree.leaves(params)), batch_stats

    @partial(jit, static_argnums=(0, 6))
    def train_step(self, key, all_params, x, y, opt_state, optimizer):
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
        trainable_params = {'params': all_params['params']}
        param_state = {'batch_stats': all_params['batch_stats']}
        if self.optimize_distparams:
            trainable_params['dist_params'] = all_params['dist_params']
        else:
            param_state['dist_params'] = all_params['dist_params']

        (loss, stats_params), grads = jax.value_and_grad(
            self.compute_loss, argnums=1, has_aux=True)(key, trainable_params, param_state, x, y)
        # update params
        updates, opt_state = optimizer.update(
            grads, opt_state, trainable_params)
        trainable_params = optax.apply_updates(trainable_params, updates)

        if self.optimize_distparams:
            params = {'params': trainable_params['params'], 'batch_stats': stats_params,
                      'dist_params': trainable_params['dist_params']}
        else:
            params = {'params': trainable_params['params'],
                      'batch_stats': stats_params, 'dist_params': param_state['dist_params']}
        return loss, params, opt_state, grads


class CategorySubspaceBatchNorm(SubspaceModelBatchNorm):
    def __init__(self, model, k, weight_decay, perm_spec: PermutationSpec=False, natural_parameterization=False):
        super().__init__(model, k, np.inf, False, weight_decay, perm_spec, natural_parameterization)

    @partial(jit, static_argnums=(0,))
    def nll(self, params, param_stats, t, x, y):
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
        out, batch_stats = self(
            params['params'], param_stats['batch_stats'], t, x)  # shape (n_samples, n_data, output_dim)

        nll = jnp.mean(
            optax.losses.softmax_cross_entropy_with_integer_labels(
                logits=out, labels=y)
        )
        return nll, batch_stats


class RepulsiveCategorySubspaceBatchNorm(CategorySubspaceBatchNorm):
    def __init__(self, model, k, weight_decay, perm_spec: PermutationSpec=False, natural_parameterization=False, gravity=0.0, energy_weakening=0.1, regularizer="force"):
        """
        Initializes the RepulsiveCategorySubspaceBatchNorm class.

        Parameters:
        - model: The model to be used.
        - k: The number of control points.
        - out_scale: The output scale.
        - optimize_distparams: Whether to optimize the distribution parameters.
        - gravity: The gravity parameter.
        - energy_weakening: The energy weakening parameter.
        - regularizer: The regularizer type ("force", "energy", "cosine_similarity").
        """
        super().__init__(model, k, weight_decay, perm_spec, natural_parameterization)
        self.gravity = gravity
        self.energy_weaker = energy_weakening
        self.regularizer = self._get_regularizer(regularizer)

    def _get_regularizer(self, regularizer):
        reg = regularizer.lower()
        if reg == "force":
            return self.repulsive_force
        elif reg == "energy":
            return self.repulsive_energy
        elif reg in ("cosine_similarity", "cosine", "similarity"):
            return self.summed_cosine_similarity
        else:
            raise ValueError(f"Unknown regularizer type: {regularizer}")

    def repulsive_force(self, params):
        """
        Computes the repulsive force between the parameters.

        Parameters:
        - params: The parameters of the model.

        Returns:
        - The repulsive force.
        """
        cp_w = pytree_to_matrix(params['params'], self.k)
        def sum_pairwise_distances(A):
            n = A.shape[0]
            def body_fun(i, acc):
                diffs = A - A[i]
                dists = jnp.sum(diffs**2, axis=-1)      # shape: (n)    
                # dists = jnp.linalg.norm(diffs, axis=-1)      # shape: (n)
                dists = self.gravity/(dists+self.energy_weaker+1e-8)
                dists = jnp.where(jnp.arange(0,n)>i, dists, 0.) # masking: use only i+1: elements
                return acc + jnp.sum(dists)
            return jax.lax.fori_loop(0, n - 1, body_fun, 0.0)
        force = sum_pairwise_distances(cp_w)
        return force

    def repulsive_energy(self, params):
        """
        Computes the repulsive force between the parameters.

        Parameters:
        - params: The parameters of the model.

        Returns:
        - The repulsive force.
        """
        cp_w = pytree_to_matrix(params['params'], self.k)
        def sum_pairwise_distances(A):
            n = A.shape[0]
            def body_fun(i, acc):
                diffs = jnp.abs(A - A[i])
                dists = jnp.sum(diffs, axis=-1)      # shape: (n)    
                # dists = jnp.linalg.norm(diffs, axis=-1)      # shape: (n)
                energy = self.gravity/(dists+self.energy_weaker+1e-8)
                log_energy = jnp.log(energy)
                log_energy = jnp.where(jnp.arange(0,n)>i, log_energy, 0.) # masking: use only i+1: elements
                return acc + jnp.sum(log_energy)
            return jax.lax.fori_loop(0, n - 1, body_fun, 0.0)
        log_energy = sum_pairwise_distances(cp_w)
        return log_energy

    def summed_cosine_similarity(self, params):
        """
        Computes the sum of pairwise cosine similarities to minimize.

        Equation:
            β · Σ_{j ≠ k} cos²(ω_j, ω_k) = β · Σ_{j ≠ k} (⟨ω_j, ω_k⟩ / (||ω_j|| ||ω_k||))²
            $\beta \cdot \sum_{j \neq k} \cos^2(\omega_j, \omega_k) = \beta \cdot \sum_{j \neq k} \left( \frac{\langle \omega_j, \omega_k \rangle}{\|\omega_j\| \|\omega_k\|} \right)^2$

        Parameters:
        - params: The parameters of the model.

        Returns:
        - The summed pairwise cosine similarity.
        """
        cp_w = pytree_to_matrix(params['params'], self.k)
        n = cp_w.shape[0]
        cp_w = cp_w / jnp.linalg.norm(cp_w, axis=-1, keepdims=True)
        dot_res = jnp.dot(cp_w, cp_w.T) # k x k matrix of cosine similarities
        # upper triangular part without diagonal
        upper_triangular_indices = jnp.triu_indices(n, k=1)
        cosine_sim = self.gravity * jnp.sum(dot_res[upper_triangular_indices])  # summed cosine similarity
        return cosine_sim

    
    def nll(self, params, param_stats, t, x, y):
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
        nll, batch_stats = super().nll(params, param_stats, t, x, y)
        rep_force = self.regularizer(params)
        # jax.debug.print("nll: {}",nll)
        # jax.debug.print("forece: {}",rep_force)
        return nll + rep_force, batch_stats

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
