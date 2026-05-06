import jax.numpy as jnp
from jax import jit, random
import jax
import optax
import numpy as np
from functools import partial
from jax import custom_jvp
from typing import Callable, Tuple, Any
from numpyro import distributions as dist
from jaxopt import Bisection
from jax.scipy.special import gammaln
from src.weight_matching import PermutationSpec, apply_permutation, weight_matching
from flax import traverse_util
from flax.core import freeze, unfreeze
from jax.tree_util import register_pytree_node_class
import math
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subspace model registry – maps string names to SubspaceBaseModel subclasses
# ---------------------------------------------------------------------------
SUBSPACE_REGISTRY: dict[str, type] = {}


def register_subspace_model(name: str):
    """Class decorator that registers a SubspaceBaseModel subclass."""

    def decorator(cls):
        if name in SUBSPACE_REGISTRY:
            raise ValueError(f"Duplicate subspace model name: '{name}'")
        SUBSPACE_REGISTRY[name] = cls
        return cls

    return decorator


def get_subspace_model(name: str) -> type:
    """Look up a registered SubspaceBaseModel subclass by *name*.

    Raises ``ValueError`` with a list of known names when *name* is unknown.
    """
    if name not in SUBSPACE_REGISTRY:
        raise ValueError(
            f"Unknown subspace model: '{name}'. "
            f"Available: {sorted(SUBSPACE_REGISTRY.keys())}"
        )
    return SUBSPACE_REGISTRY[name]


def tree_broadcast(
    prefix_tree: Any, full_tree: Any, is_leaf: Callable[[Any], bool] | None = None
) -> Any:
    """Alias of :func:`jax.tree.broadcast`."""

    # broadcast_prefix is not exported
    def broadcast_prefix(
        prefix_tree: Any, full_tree: Any, is_leaf: Callable[[Any], bool] | None = None
    ) -> list[Any]:
        """Broadcasts tree prefix leaves into the full set of leaves for a given full tree.

        Args:
        prefix_tree: a pytree that is a tree prefix of full_tree.
        full_tree: a pytree with the structure to broadcast the prefix leaves into.
        is_leaf: an optionally specified function that will be called at each
            flattening step. It should return a boolean, with true stopping the
            traversal and the whole subtree being treated as a leaf, and false
            indicating the flattening should traverse the current object.

        Returns:
        A list of leaves matching the expected count for the full tree,
        with the leaf of each prefix tree being duplicated to match the count of
        its corresponding subtree.
        """
        result = []

        def num_leaves(t):
            return jax.tree.structure(t).num_leaves

        def add_leaves(x, subtree):
            return result.extend([x] * num_leaves(subtree))

        jax.tree.map(add_leaves, prefix_tree, full_tree, is_leaf=is_leaf)
        return result

    broadcast_leaves = broadcast_prefix(prefix_tree, full_tree, is_leaf=is_leaf)
    return jax.tree.structure(full_tree).unflatten(broadcast_leaves)


@custom_jvp
def pos_pow(x, pow):
    # we expect only positive powers => multiple grad is 0 istead of nan
    return jnp.where(pow < 0, 0, jnp.pow(x, pow))


def pos_pow_jvp(primals, tangents):
    x, pow = primals
    grad = pow * pos_pow(x, pow - 1)
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
        return jnp.einsum("j,j...->...", coeff, cp_diff) * n

    # forward function
    @custom_jvp
    def f(t):
        p_t = jnp.einsum("k,k...->...", bezier_coeff(t), cp)
        return p_t

    def f_jvp(primals, tangents):
        t = primals[0]
        tangent_out = derivative_bezier(t)
        primal_out = f(t)
        return primal_out, tangent_out * tangents[0]

    f.defjvp(f_jvp)
    return f, derivative_bezier


def segmented_bezier_coeff_fn(
    num_bends, segment_degree, indepent_connected: bool = False
) -> Callable:
    """Piecewise Bézier coefficient function with non-overlapping segments.

    Connects *num_bends* control points through consecutive Bézier segments
    of degree *segment_degree*.  Each segment uses (segment_degree + 1)
    control points, and consecutive segments share only their boundary CP
    (stride = segment_degree).

    **Open curve** (odd-style): when ``(num_bends - 1) % segment_degree == 0``,
    e.g. 5 CPs / degree 2 → segments [0,1,2], [2,3,4] → 2 segments.

    **Closed loop** (even-style): when ``num_bends % segment_degree == 0``,
    e.g. 6 CPs / degree 2 → segments [0,1,2], [2,3,4], [4,5,0] → 3 segments.
    The last segment wraps back to the first CP.

    The global parameter *t* lives in ``[0, num_segments]``.

    Args:
        num_bends:      Total number of control points (k + 1).
        segment_degree: Polynomial degree of each Bézier segment
                        (e.g. 2 → quadratic).

    Returns:
        A callable  ``t  →  coefficients``  that returns an array of shape
        ``(num_bends,)`` whose entries sum to 1.
    """
    is_open = (num_bends - 1) % segment_degree == 0
    if is_open:
        num_segments = (num_bends - 1) // segment_degree
    else:
        assert num_bends % segment_degree == 0, (
            f"num_bends={num_bends} is neither compatible with open curve "
            f"((n-1) % d == 0) nor closed loop (n % d == 0) for degree {segment_degree}"
        )
        num_segments = num_bends // segment_degree

    assert num_segments >= 1, (
        f"Need at least {segment_degree + 1} control points for degree "
        f"{segment_degree}, got {num_bends}"
    )

    # Bézier basis for a single segment of the given degree
    seg_coeff_fn = bezier_coeff_fn(segment_degree + 1)

    # Precompute CP indices for each segment: shape (num_segments, segment_degree+1)
    seg_indices = jnp.array(
        [
            [(i * segment_degree + j) % num_bends for j in range(segment_degree + 1)]
            for i in range(num_segments)
        ]
    )  # (num_segments, segment_degree+1)

    def wrapper(t):
        # Determine which segment we are in
        segment_idx = jnp.floor(t).astype(jnp.int32)
        segment_idx = jnp.clip(segment_idx, 0, num_segments - 1)
        segment_idx = jnp.squeeze(segment_idx)  # ensure scalar for jax.lax.cond
        local_t = t - segment_idx.astype(t.dtype)  # ∈ [0, 1]
        local_t = jnp.clip(local_t, 0.0, 1.0)

        # Local Bézier coefficients (length: segment_degree + 1)
        local_coeffs = seg_coeff_fn(local_t)
        if indepent_connected:
            if is_open:
                # For open curve, don't zero out the last CP of the last segment
                local_coeffs = jax.lax.cond(
                    segment_idx == num_segments - 1,
                    lambda: local_coeffs,
                    lambda: local_coeffs.at[-1].set(0.0),
                )
            else:
                local_coeffs = local_coeffs.at[-1].set(0.0)  #

        # Get CP indices for this segment via dynamic indexing
        cp_indices = seg_indices[segment_idx]  # (segment_degree+1,)

        # Scatter local coefficients into global coefficient vector
        full_coeffs = jnp.zeros(num_bends, dtype=local_coeffs.dtype)
        full_coeffs = full_coeffs.at[cp_indices].add(local_coeffs)
        return full_coeffs

    wrapper.num_segments = num_segments
    wrapper.is_open = is_open
    return wrapper


# B-Spline basis function using Cox-de Boor recursion formula
def bspline_basis(i, p, knots, t):
    if p == 0:
        return jnp.where((knots[i] <= t) & (t < knots[i + 1]), 1.0, 0.0)
    else:
        denom1 = knots[i + p] - knots[i]
        denom2 = knots[i + p + 1] - knots[i + 1]

        term1 = jnp.where(
            denom1 > 0, (t - knots[i]) / denom1 * bspline_basis(i, p - 1, knots, t), 0.0
        )
        term2 = jnp.where(
            denom2 > 0,
            (knots[i + p + 1] - t) / denom2 * bspline_basis(i + 1, p - 1, knots, t),
            0.0,
        )
        return term1 + term2


def bspline_coeff_fn(num_ctrl, degree, knots):
    # @jax.jit
    def coeff(t):
        t = t.squeeze()  # ensure t is scalar
        # special case: t == last knot
        t = jnp.where(
            t == knots[-1],
            t - 1e-7,  # move slightly inside
            t,
        )
        return jnp.array([bspline_basis(i, degree, knots, t) for i in range(num_ctrl)])

    return coeff


def open_uniform_knots(num_ctrl, degree):
    n = num_ctrl - 1
    return jnp.concatenate(
        [jnp.zeros(degree), jnp.linspace(0, 1, n - degree + 2), jnp.ones(degree)]
    )


def bspline_curve(cp, degree, knots=None):
    num_ctrl = cp.shape[0]
    if knots is None:
        knots = open_uniform_knots(num_ctrl, degree)
    coeff_fn = bspline_coeff_fn(num_ctrl, degree, knots)

    def derivative_bspline(t):
        # analytische Ableitung der Basisfunktionen
        def dN(i):
            denom1 = knots[i + degree] - knots[i]
            denom2 = knots[i + degree + 1] - knots[i + 1]

            term1 = jnp.where(
                denom1 > 0,
                degree / denom1 * bspline_basis(i, degree - 1, knots, t),
                0.0,
            )
            term2 = jnp.where(
                denom2 > 0,
                degree / denom2 * bspline_basis(i + 1, degree - 1, knots, t),
                0.0,
            )
            return term1 - term2

        dcoeff = jnp.array([dN(i) for i in range(num_ctrl)])
        return jnp.einsum("i,i...->...", dcoeff, cp)

    @custom_jvp
    def f(t):
        coeff = coeff_fn(t)
        return jnp.einsum("i,i...->...", coeff, cp)

    def f_jvp(primals, tangents):
        (t,) = primals
        (t_dot,) = tangents
        primal_out = f(t)
        tangent_out = derivative_bspline(t)
        return primal_out, tangent_out * t_dot

    f.defjvp(f_jvp)
    return f, derivative_bspline


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
        u, self.s, vh = jnp.linalg.svd(
            cp, full_matrices=False
        )  # u: (k+1, k+1), s: (k+1,), vh: (k+1, dim)
        self.pi = self.s[:subspace_dim, None] * vh[:subspace_dim]
        self.pi_inv = vh[:subspace_dim].T @ jnp.diag(1 / self.s[:subspace_dim])

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


class BezierTspaceUnifrom(dist.Uniform):
    def __init__(self, d_bezier, validate_args=None):
        self.d_bezier = d_bezier
        tt = jnp.linspace(0.0, 1.0, 1_000)
        bezier_grad = jax.vmap(d_bezier)(tt)
        length = jnp.trapezoid(jnp.linalg.norm(bezier_grad, axis=-1), tt)
        self.total_length = length
        logger.debug("Total length: %s", length)
        super(BezierTspaceUnifrom, self).__init__(validate_args=validate_args)

    @staticmethod
    def arc_length(t, d_bezier):
        tt = jnp.linspace(0.0, t, 1_000)
        grads = jax.vmap(d_bezier)(tt)
        return jnp.trapezoid(jnp.linalg.norm(grads, axis=-1), tt)

    def t_from_length(self, length):
        def loss(t):
            return self.arc_length(t, self.d_bezier) - length

        root_solver = Bisection(
            loss, lower=0.0, upper=1.0, check_bracket=False, tol=1e-12, maxiter=30
        )
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
    - train_step(key, t, params, x, y, opt_state, optimizer): Performs a single training step.

    """

    def __init__(
        self,
        model,
        k,
        weight_decay: float = 0.0,
        perm_spec: PermutationSpec = False,
        natural_parameterization=False,
        mutuable_param_name: str | bool = False,
        curve_parameterization="bezier",
        SegDeg: int | bool = False,
        indepent_connected: bool = False,
    ):
        self.model = model
        self.k = k
        if SegDeg and SegDeg < k:
            segment_degree = int(SegDeg)
            self.bezier = segmented_bezier_coeff_fn(
                k + 1, segment_degree, indepent_connected=indepent_connected
            )
            num_segments = self.bezier.num_segments
            is_open = self.bezier.is_open
            logger.info(
                f"Using segmented Bézier ({'open' if is_open else 'closed'}): "
                f"{k + 1} CPs, degree {segment_degree}, "
                f"{num_segments} segments, t ∈ [0, {num_segments}]"
            )
            self.t_max = float(num_segments)
        elif "b-spline" in curve_parameterization:
            logger.info("Using B-Spline curve parameterization")
            degree = 2
            knots = open_uniform_knots(k + 1, degree)
            self.bezier = bspline_coeff_fn(k + 1, degree, knots)
            self.t_max = 1.0
        else:
            logger.info("Using Bezier curve parameterization")
            self.bezier = bezier_coeff_fn(k + 1)
            self.t_max = 1.0
        self.weight_decay = weight_decay
        self.perm_spec = perm_spec
        self.mutable_name = mutuable_param_name
        self.compute_loss = (
            self.compute_loss_natural_t
            if natural_parameterization
            else self.compute_loss_t
        )
        self.curve_mask = None
        self.train_mask = None

    def only_trainable_params(self, params):
        return jax.tree.map(
            lambda x, m: x if m else jnp.array([]), params, self.train_mask
        )

    def remove_permutation(self, key, params_list):
        """
        Removes the permutation from the parameters.

        Parameters:
        - params: The parameters of the model.

        Returns:
        - The parameters without permutation.

        """

        def flatten_params(params):
            return {
                "/".join(k): v for k, v in traverse_util.flatten_dict(params).items()
            }

        def unflatten_params(flat_params):
            return traverse_util.unflatten_dict(
                {tuple(k.split("/")): v for k, v in flat_params.items()}
            )

        for i, p in enumerate(params_list[1:]):
            logger.info("Remove permutation from k %d", i + 1)
            key, perm_key = random.split(key)
            final_permutation = weight_matching(
                perm_key,
                self.perm_spec,
                flatten_params(params_list[0]),
                flatten_params(p),
            )
            params_list[i] = unflatten_params(
                apply_permutation(self.perm_spec, final_permutation, flatten_params(p))
            )
        return params_list

    def set_train_mask(self, train_mask):
        """
        Sets the train mask and curve mask for the model.

        Parameters:
        - train_mask: The mask indicating which parameters are trainable.
        - curve_mask: The mask indicating which parameters are part of the curve.

        """
        self.train_mask = train_mask

    def init_mask(self, x, all_curves=False):
        """
        Initializes the mask for the model parameters.

        Parameters:
        - x: The input data.

        Returns:
        - A pytree of masks with the same structure as the model parameters, where each leaf is a boolean indicating whether the parameter is trainable.

        """
        params = self.model.init(random.PRNGKey(0), x)
        mask = jax.tree.map(lambda x: all_curves, params["params"])
        self.curve_mask = mask
        return mask

    def init_params_from_point(self, key, params, mask, noise_fn):
        """
        Initializes curve parameters from a given parameter point, optionally adding noise and handling permutation invariance.
        Args:
            key: A JAX PRNGKey used for random number generation.
            point (dict): A dictionary containing at least a 'params' key with the base parameters to initialize from.
            mask: A tree-like structure of booleans indicating which parameters should be handled by the curve.
            jitter (float): The standard deviation multiplier of the Gaussian noise to add to the parameters.
        Returns:
            dict: A copy of the input `point` dictionary, with the 'params' key replaced by the initialized curve parameters.
        Notes:
            - If `self.perm_spec` is set, permutation invariance is handled by removing permutations from the curve parameters.
            - The method adds Gaussian noise to each parameter in the curve, controlled by `jitter`.
            - Non-curve parameters are reverted to their original values from `point['params']`.
            - The mask used for curve parameters is stored in `self.curve_mask`.
        """
        assert ("params" in params) and len(
            params.keys()
        ) == 1, "Point must contain 'params' key and no other keys."

        if self.perm_spec:
            raise ValueError(
                "Permutation specification is not supported in init_params_from_point method."
            )

        # add a dimension if mask is true and initialize with noise function or repeate LoRA A matrix initialization
        # use the same weight matrix initialization as Hugging Face PEFT library does for the LoRA A matrix
        def make_curve(m, p, key):
            if m:
                if (type(p) is LoRAParams) or (type(p) is LoRAepsilonAParams):
                    gain = math.sqrt(
                        2 / (1 + 5)
                    )  # Gain for initialization sqrt(2/(1+math.sqrt(5)^2)) for ReLU
                    shape_ = (self.k + 1,) + p.A.shape
                    init_a = jax.nn.initializers.variance_scaling(
                        in_axis=-1,
                        out_axis=range(len(shape_))[:-1],
                        distribution="uniform",
                        mode="fan_in",
                        scale=gain**2,
                    )  # fan_in = all shapes except m[1]
                    # keys = random.split(key, self.k + 1)
                    p.A = init_a(key, shape_, dtype=p.A.dtype)
                    return p
                elif type(p) is LoRAScaleParams:
                    shape_ = (self.k + 1,) + p.s.shape
                    init_s = jax.nn.initializers.orthogonal()
                    p.s = init_s(key, shape_, dtype=p.s.dtype)
                    return p
                elif (type(p) is LoRAScaleDiagParams) or (
                    type(p) is LoRAepsilonScaleDiagParams
                ):
                    shape_ = (self.k + 1,) + p.s.shape
                    init_s = jax.nn.initializers.normal(stddev=1.0)
                    p.s = init_s(key, shape_, dtype=p.s.dtype) + 1.0
                    return p
                    # raise NotImplementedError("LoRAScaleParams initialization not implemented in init_params_from_point.")
                elif (type(p) is LoRAABParam) or (type(p) is LoRAepsilonABParams):
                    gain = math.sqrt(
                        2 / (1 + 5)
                    )  # Gain for initialization sqrt(2/(1+math.sqrt(5)^2)) for ReLU
                    shape_ = (self.k + 1,) + p.A.shape
                    init_a = jax.nn.initializers.variance_scaling(
                        in_axis=-1,
                        out_axis=range(len(shape_))[:-1],
                        distribution="uniform",
                        mode="fan_in",
                        scale=gain**2,
                    )  # fan_in = all shapes except m[1]
                    # keys = random.split(key, self.k + 1)
                    p.A = init_a(key, shape_, dtype=p.A.dtype)
                    p.B = jnp.zeros((self.k + 1,) + p.B.shape, dtype=p.B.dtype)
                    return p
                elif type(p) is LoRARotaEpsilonDiagscaleRotbParams:
                    key, subkey = random.split(key)
                    # p.A = jax.nn.initializers.orthogonal()(subkey, (self.k+1,) + p.A.shape, dtype=p.A.dtype)
                    p.A = jax.nn.initializers.normal(stddev=0.01)(
                        subkey, (self.k + 1,) + p.A.shape, dtype=p.A.dtype
                    )
                    key, subkey = random.split(key)
                    # p.B = jax.nn.initializers.orthogonal()(subkey, (self.k+1,) + p.B.shape, dtype=p.B.dtype)
                    p.B = jax.nn.initializers.normal(stddev=0.01)(
                        subkey, (self.k + 1,) + p.B.shape, dtype=p.B.dtype
                    )
                    return p
                elif type(p) is LoRAAEpsilonRotdiagscaleBParams:
                    shape_ = (self.k + 1,) + p.s.shape
                    init_s = jax.nn.initializers.constant(value=-10.0)
                    p.s = init_s(key, shape_, dtype=p.s.dtype)
                    return p
                else:
                    curve_param = jnp.stack([p] * (self.k + 1))
                    curve_param = noise_fn(curve_param, key)
                    return curve_param
            else:
                return p

        mf, struct = jax.tree.flatten(mask)
        rng_keys = random.split(key, len(mf))
        rng_keys = jax.tree.unflatten(struct, rng_keys)
        params["params"] = jax.tree.map(
            make_curve,
            mask,
            params["params"],
            rng_keys,
            is_leaf=lambda x: isinstance(x, LoraAbstractParams),
        )

        self.curve_mask = mask
        return params

    def init_params(self, key, x, mask):
        """
        Initializes the parameters of the model.

        Parameters:
        - key: The random key for parameter initialization.
        - x: The input data.

        Returns:
        - The initialized parameters {params, batch_stats}.

        """
        keys = random.split(key, self.k + 1)

        # list of params and variables (such as batch statistics)
        all_params = self.model.init(keys[0], x)
        stacked_params = [all_params["params"]]
        stacked_params += [self.model.init(k, x)["params"] for k in keys[1:]]
        # if permutation is specified then remove the permutation by using the first point as anchor
        if self.perm_spec:
            stacked_params = self.remove_permutation(key, stacked_params)
        # now pytree of stacked params (each leaf has leading dim of k+1)
        all_params["params"] = jax.tree.map(
            lambda m, *x: jnp.stack(x) if m else x[0], mask, *stacked_params
        )
        del stacked_params

        self.curve_mask = mask
        return all_params

    # @partial(jit, static_argnums=(0,), donate_argnums=(2,))
    # @partial(jit, static_argnums=(0,))
    def _predict(self, params, state, inputs, train) -> Tuple[jnp.ndarray, dict]:
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        # Predict out from inputs and parameters
        if self.mutable_name:
            all_params = {"params": params, self.mutable_name: state}

            def train_fn(inputs):
                out, net_state = self.model.apply(
                    all_params, *inputs, train=True, mutable=self.mutable_name
                )
                return out, net_state[self.mutable_name]
        else:
            all_params = {"params": params}

            def train_fn(inputs):
                out = self.model.apply(all_params, *inputs, train=True, mutable=False)
                return out, {}

        def eval_fn(inputs):
            out = self.model.apply(all_params, *inputs, train=False, mutable=False)
            return out, state

        out, net_state = jax.lax.cond(train, train_fn, eval_fn, inputs)
        return out, net_state

    # @partial(jit, static_argnums=(0,), donate_argnums=(2,))
    # @partial(jit, static_argnums=(0,))
    def __call__(
        self, params, state, t, x, train=True, key=None
    ) -> Tuple[jnp.ndarray, dict]:
        """
        Computes the output of the model for given parameters, time, and input.

        Parameters:
        - params: The parameters of the model.
        - t: The time parameter. Only single value supported.
        - x: The input data.

        Returns:
        - The output of the model.

        """
        if isinstance(t, dict):

            def curve_to_point(t, p, m):
                if m:
                    bezier_coeff = self.bezier(t)
                    return jnp.einsum("k,k...->...", bezier_coeff, p).astype(p.dtype)
                else:
                    return p

            params = jax.tree.map(curve_to_point, t, params, self.curve_mask)
        else:
            # sample Bezier coefficient
            bezier_coeff = self.bezier(t)
            # Compute one parameter set per sample
            params = jax.tree.map(
                lambda p, m: (
                    jnp.einsum("k,k...->...", bezier_coeff, p).astype(p.dtype)
                    if m
                    else p
                ),
                params,
                self.curve_mask,
            )

        # forward pass per sample
        out, state = self._predict(params, state, x, train=train)
        return out, state

    # @partial(jit, static_argnums=(0,))
    def nll(
        self, params, state, t, x, y, train: bool, key
    ) -> Tuple[jnp.ndarray, dict, jnp.ndarray]:
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
        raise NotImplementedError(
            "nll must be implemented by subclasses of SubspaceBaseModel."
        )

    @staticmethod
    def l2_loss(x, alpha):
        return alpha * (x**2).sum()

    # @partial(jit, static_argnums=(0,))
    def compute_loss_t(self, key, t, params, state, freezed_params, x, y):
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
        # t = random.uniform(key, (1,), minval=0., maxval=1.)
        joind_param = jax.tree.map(
            lambda x, y, m: x if m else y,
            params,
            unfreeze(freezed_params),
            self.train_mask,
        )
        nll, state, out = self.nll(joind_param, state, t, x, y, train=True, key=key)
        nll = nll.mean()
        if self.weight_decay == 0.0:
            return nll, (state, {"nll": nll})
        else:
            w_norm = sum(
                self.l2_loss(w, alpha=self.weight_decay)
                for w in jax.tree.leaves(params)
            )
            loss = nll + w_norm
            return loss, (state, {"nll": nll, "weight_decay": w_norm})

    # @partial(jit, static_argnums=(0,))
    def compute_loss_natural_t(self, key, t, params, state, x, y):
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

        t = random.uniform(key, (1,), minval=0.0, maxval=1.0)
        loss, state, out = self.nll(params, state, t, x, y, train=True, key=key)

        raise ValueError(
            "Natural parameterization currently not implemented for SubspaceBaseModel (pytree_to_matrix needs masking)"
        )
        cp_w = pytree_to_matrix(params["params"], self.k)
        # cp_phi =
        curve, d_bezier = bezier_curve(self.k + 1, cp_w)
        tt = jnp.linspace(0.0, 1.0, 1_000)
        bezier_grad = jax.vmap(d_bezier)(tt)
        normalized_grad = jnp.trapezoid(jnp.linalg.norm(bezier_grad, axis=-1), tt)
        grads = jax.vmap(d_bezier)(t)
        weights = jnp.linalg.norm(grads, axis=-1) / normalized_grad

        # t_dist = BezierTspaceUnifrom(d_bezier)
        # t = t_dist.sample(key, (n_samples,))

        # weighted_loss = jnp.einsum('n,n->', weights, loss)
        weighted_loss = weights.squeeze() * loss
        return weighted_loss + sum(
            self.l2_loss(w, alpha=self.weight_decay)
            for w in jax.tree.leaves(params["params"])
        ), state

    # @partial(jit, static_argnums=(0, 7))
    # @profile
    @partial(jit, static_argnums=(0, 7), donate_argnums=(3, 6))
    def train_step(self, key, t, params, x, y, opt_state, optimizer):
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
        trainable_params = self.only_trainable_params(params["params"])
        freezed_params = freeze(
            jax.tree.map(
                lambda x, m: x if not m else jnp.array([]),
                params["params"],
                self.train_mask,
            )
        )
        (loss, (stats_params, logs)), grads = jax.value_and_grad(
            self.compute_loss, argnums=2, has_aux=True
        )(
            key,
            t,
            trainable_params,
            params[self.mutable_name] if self.mutable_name else {},
            freezed_params,
            x,
            y,
        )
        # update params
        updates, opt_state = optimizer.update(grads, opt_state, trainable_params)
        trainable_params = optax.apply_updates(trainable_params, updates)
        params = jax.tree.map(
            lambda x, y, m: x if m else y,
            trainable_params,
            unfreeze(freezed_params),
            self.train_mask,
        )
        params = {"params": params}
        if self.mutable_name:
            params[self.mutable_name] = stats_params
        return loss, params, opt_state, (grads, logs)


@register_subspace_model("category")
class CategorySubspace(SubspaceBaseModel):
    # @partial(jit, static_argnums=(0,), donate_argnums=(2,))
    # @partial(jit, static_argnums=(0,))
    def nll(
        self, params, state, t, x, y, train: bool = True, key=None
    ) -> Tuple[jnp.ndarray, dict, jnp.ndarray]:
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
        out, state = self(
            params, state, t, x, train=train, key=key
        )  # shape (n_samples, n_data, output_dim)

        nll = optax.losses.softmax_cross_entropy_with_integer_labels(
            logits=out, labels=y
        )
        return jnp.mean(nll), state, out


@register_subspace_model("regression")
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

    # @partial(jit, static_argnums=(0,), donate_argnums=(2,))
    # @partial(jit, static_argnums=(0,))
    def nll(
        self, params, state, t, x, y, train: bool = True, key=None
    ) -> Tuple[jnp.ndarray, dict, jnp.ndarray]:
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
        out, state = self(
            params, state, t, x, train=train, key=key
        )  # shape (n_data, output_dim)
        assert out.shape[-1] == 1, "RegressionSubspace expects output_dim=1"
        assert y.ndim == 1, "RegressionSubspace expects y to be 1D"
        nll = -jax.scipy.stats.norm.logpdf(
            y, loc=out.squeeze(axis=-1), scale=jnp.exp(self.log_scale) + 1e-8
        )
        return nll, state, out


@register_subspace_model("dist_regression")
class DistRegressionSubspace(SubspaceBaseModel):
    def __init__(self, init_outDist_log_scale=jnp.log(1.0), **kwargs):
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

    def init_params_from_point(self, key, point, mask, noise_fn):
        raise ValueError("need some adjustements for lora, train and curve mask.")
        mask["log_scale"] = False  # mask log_scale parameter as non curve parameter
        return super().init_params_from_point(key, point, mask, noise_fn)

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
        raise ValueError("need some adjustements for lora, train and curve mask.")
        params = super().init_params(key, x, mask)
        # initialize log scale
        params["params"]["log_scale"] = self.init_log_scale
        logger.debug("curve_mask: %s", self.curve_mask)
        self.curve_mask[
            "log_scale"
        ] = False  # mask log_scale parameter as non curve parameter
        return params

    # @partial(jit, static_argnums=(0,), donate_argnums=(2,))
    # @partial(jit, static_argnums=(0,))
    def nll(
        self, params, state, t, x, y, train: bool = True, key=None
    ) -> Tuple[jnp.ndarray, dict, jnp.ndarray]:
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
        out, state = self(
            params, state, t, x, train=train, key=key
        )  # shape (n_data, output_dim)
        nll = -jax.scipy.stats.norm.logpdf(
            y, loc=out.squeeze(axis=-1), scale=jnp.exp(params["log_scale"]) + 1e-8
        )
        return nll, state, out


@register_subspace_model("quantile")
class QuantileSubspace(SubspaceBaseModel):
    def __init__(
        self,
        model,
        k,
        quantiles,
        weight_decay=0,
        perm_spec=False,
        natural_parameterization=False,
        mutuable_param_name=False,
    ):
        super().__init__(
            model,
            k,
            weight_decay,
            perm_spec,
            natural_parameterization,
            mutuable_param_name,
        )
        self.quantiles = jnp.array(quantiles, dtype=jnp.float32)

    @staticmethod
    # @jit
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

    # @partial(jit, static_argnums=(0,), donate_argnums=(2,))
    # @partial(jit, static_argnums=(0,))
    def nll(
        self, params, state, t, x, y, train: bool = True, key=None
    ) -> Tuple[jnp.ndarray, dict, jnp.ndarray]:
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
        out, state = self(
            params, state, t, x, train=train, key=key
        )  # shape (n_samples, n_data, output_dim)
        out_quantile = out[1]

        loss = jnp.square(out_quantile[:, :, 0] - y)

        def body_fun(loss, pred_and_quantile):
            loss += self._quantile_loss(
                pred_and_quantile["preds"], y, quantile=pred_and_quantile["quantile"]
            )
            return loss, None

        loss, _ = jax.lax.scan(
            body_fun,
            loss,
            {"quantile": self.quantiles, "preds": out_quantile.transpose(2, 0, 1)[1:]},
        )
        return loss.mean(), state, out


class RepulsiveMixin:
    def __init__(
        self, gravity=0.0, energy_weakening=0.1, regularizer="force", **kwargs
    ):
        logger.info("Using Repulsive Mixin with regularizer: %s", regularizer)
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

    # @partial(jit, static_argnums=(0,))
    def repulsive_force(self, params):
        cp_w = masked_pytree_to_matrix(params, self.curve_mask, self.k)
        n = cp_w.shape[0]

        def body_fun(i, acc):
            diffs = cp_w - cp_w[i]
            dists = jnp.sum(diffs**2, axis=-1)
            dists = 1.0 / (dists + self.energy_weakening + 1e-8)
            dists = jnp.where(jnp.arange(0, n) > i, dists, 0.0)
            return acc + jnp.sum(dists)

        force = jax.lax.fori_loop(0, n - 1, body_fun, 0.0)
        return force

    # @partial(jit, static_argnums=(0,))
    def repulsive_energy(self, params):
        cp_w = masked_pytree_to_matrix(params, self.curve_mask, self.k)
        n = cp_w.shape[0]

        def body_fun(i, acc):
            diffs = jnp.abs(cp_w - cp_w[i])
            dists = jnp.sum(diffs, axis=-1)
            energy = 1.0 / (dists + self.energy_weakening + 1e-8)
            log_energy = jnp.log(energy)
            log_energy = jnp.where(jnp.arange(0, n) > i, log_energy, 0.0)
            return acc + jnp.sum(log_energy)

        log_energy = jax.lax.fori_loop(0, n - 1, body_fun, 0.0)
        return log_energy

    # @partial(jit, static_argnums=(0,))
    def summed_cosine_similarity(self, params):
        cp_w = masked_pytree_to_matrix(params, self.curve_mask, self.k)
        n = cp_w.shape[0]
        cp_w = cp_w / jnp.linalg.norm(cp_w, axis=-1, keepdims=True)
        dot_res = jnp.dot(cp_w, cp_w.T)
        upper_triangular_indices = jnp.triu_indices(n, k=1)
        cosine_sim = jnp.sum(dot_res[upper_triangular_indices])
        return cosine_sim

    # @partial(jit, static_argnums=(0,))
    def summed_cosine_similarity_square(self, params):
        return self.summed_cosine_similarity(params) ** 2

    # @partial(jit, static_argnums=(0,), donate_argnums=(3,))
    # @partial(jit, static_argnums=(0,))
    def compute_loss_t(self, key, t, params, state, freezed_params, x, y):
        # Sample t param of Bezier curve
        nll, (state, logs) = super().compute_loss_t(
            key, t, params, state, freezed_params, x, y
        )
        rep_force = self.regularizer(params)
        return nll + self.gravity * rep_force, (
            state,
            {"repulsive_force": rep_force, **logs},
        )


class EntropyMixin:
    def __init__(self, entropy_weight=0.0, **kwargs):
        logger.info("Using Entropy Mixin with weight: %s", entropy_weight)
        super().__init__(**kwargs)
        self.entropy_weight = entropy_weight

    # @partial(jit, static_argnums=(0,), donate_argnums=(3,))
    # @partial(jit, static_argnums=(0,))
    def compute_loss_t(self, key, t, params, state, freezed_params, x, y):
        joind_param = jax.tree.map(
            lambda x, y, m: x if m else y,
            params,
            unfreeze(freezed_params),
            self.train_mask,
        )
        nll, state, out = self.nll(joind_param, state, t, x, y, train=True, key=key)
        probs = jax.nn.softmax(out, axis=-1)
        n_entropy = jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1).mean()
        loss = nll.mean() + self.entropy_weight * n_entropy
        if self.weight_decay != 0.0:
            loss = loss + sum(
                self.l2_loss(w, alpha=self.weight_decay)
                for w in jax.tree.leaves(params)
            )
        return loss, (state, {"nll": nll.mean(), "entropy": n_entropy.mean()})


class EntropyMixin2:
    def __init__(self, entropy_weight=0.0, **kwargs):
        logger.info("Using Entropy Mixin 2 with weight: %s", entropy_weight)
        super().__init__(**kwargs)
        self.entropy_weight = entropy_weight

    # @partial(jit, static_argnums=(0,), donate_argnums=(3,))
    # @partial(jit, static_argnums=(0,))
    def compute_loss_t(self, key, t, params, state, freezed_params, x, y):
        joind_param = jax.tree.map(
            lambda x, y, m: x if m else y,
            params,
            unfreeze(freezed_params),
            self.train_mask,
        )
        nll, state, out = self.nll(joind_param, state, t, x, y, train=True, key=key)
        probs = jax.nn.softmax(out, axis=-1)

        # apply entropy loss only to misclassified samples
        n_mask = jnp.argmax(probs, axis=-1) != y  # shape (n_data,)
        n_mask = n_mask.astype(probs.dtype)
        n_entropy = jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
        loss = nll.mean() + (self.entropy_weight * n_entropy * n_mask).mean()
        if self.weight_decay != 0.0:
            loss = loss + sum(
                self.l2_loss(w, alpha=self.weight_decay)
                for w in jax.tree.leaves(params)
            )
        return loss, (state, {"nll": nll.mean(), "entropy": n_entropy.mean()})


class JensenShannonMixin:
    def __init__(self, entropy_weight=0.0, **kwargs):
        logger.info("Using Jensen-Shannon Mixin with weight: %s", entropy_weight)
        super().__init__(**kwargs)
        self.entropy_weight = entropy_weight

    # @partial(jit, static_argnums=(0,), donate_argnums=(3,))
    # @partial(jit, static_argnums=(0,))
    def compute_loss_t(self, key, t, params, state, freezed_params, x, y):
        joind_param = jax.tree.map(
            lambda x, y, m: x if m else y,
            params,
            unfreeze(freezed_params),
            self.train_mask,
        )
        nll, state, out = self.nll(joind_param, state, t, x, y, train=True, key=key)
        probs_1 = jax.nn.softmax(out, axis=-1)

        key, t2 = random.split(key)
        t2 = random.uniform(key, (1,), minval=0.0, maxval=1.0)
        nll2, state, out2 = self.nll(joind_param, state, t2, x, y, train=True, key=key)
        probs_2 = jax.nn.softmax(out2, axis=-1)

        # jsd divergence between probs_1 and probs_2
        m = 0.5 * (probs_1 + probs_2)
        kl_1 = jnp.sum(probs_1 * (jnp.log(probs_1 + 1e-8) - jnp.log(m + 1e-8)), axis=-1)
        kl_2 = jnp.sum(probs_2 * (jnp.log(probs_2 + 1e-8) - jnp.log(m + 1e-8)), axis=-1)
        jsd = (0.5 * (kl_1 + kl_2)).mean()

        # loss = nll.mean() + nll2.mean() - self.entropy_weight * jsd  * jnp.maximum(jnp.abs(t - t2), 0.1)
        loss = nll.mean() - self.entropy_weight * jsd * jnp.maximum(
            jnp.abs(t - t2), 0.1
        )
        if self.weight_decay != 0.0:
            loss = loss + sum(
                self.l2_loss(w, alpha=self.weight_decay)
                for w in jax.tree.leaves(params)
            )
        return loss.squeeze(), (
            state,
            {"nll1": nll.mean(), "nll2": nll2.mean(), "jsd": jsd},
        )


class JensenShannonMixin_v2:
    def __init__(self, entropy_weight=0.0, target_jsd=0.0, **kwargs):
        logger.info(
            "Using Jensen-Shannon Mixin v2 with weight: %s, target_jsd: %s",
            entropy_weight,
            target_jsd,
        )
        super().__init__(**kwargs)
        self.entropy_weight = entropy_weight
        self.target_jsd = target_jsd

    def compute_loss_t(self, key, t, params, state, freezed_params, x, y):
        joind_param = jax.tree.map(
            lambda x, y, m: x if m else y,
            params,
            unfreeze(freezed_params),
            self.train_mask,
        )

        # Symmetric NLL on clean data
        nll1, state, out1 = self.nll(joind_param, state, t, x, y, train=True, key=key)
        probs_1 = jax.nn.softmax(out1, axis=-1)

        # Fixed offset t2 = (t + t_max/2) % t_max
        t2 = (t + self.t_max / 2) % self.t_max
        nll2, state, out2 = self.nll(joind_param, state, t2, x, y, train=True, key=key)
        probs_2 = jax.nn.softmax(out2, axis=-1)

        # JSD divergence between probs_1 and probs_2
        m = 0.5 * (probs_1 + probs_2)
        kl_1 = jnp.sum(probs_1 * (jnp.log(probs_1 + 1e-8) - jnp.log(m + 1e-8)), axis=-1)
        kl_2 = jnp.sum(probs_2 * (jnp.log(probs_2 + 1e-8) - jnp.log(m + 1e-8)), axis=-1)
        jsd = (0.5 * (kl_1 + kl_2)).mean()

        # Margin-based JSD (target_jsd)
        jsd_loss = jnp.maximum(self.target_jsd - jsd, 0.0)

        loss = 0.5 * (nll1.mean() + nll2.mean()) + self.entropy_weight * jsd_loss

        if self.weight_decay != 0.0:
            loss = loss + sum(
                self.l2_loss(w, alpha=self.weight_decay)
                for w in jax.tree.leaves(params)
            )
        return loss.squeeze(), (
            state,
            {"nll1": nll1.mean(), "nll2": nll2.mean(), "jsd": jsd},
        )


class JensenShannonNoiseMixin:
    def __init__(
        self,
        entropy_weight=0.0,
        noise_rate=0.1,
        vocab_size=151936,
        **kwargs,
    ):
        logger.info(
            "Using Jensen-Shannon Noise Mixin with weight: %s, noise_rate: %s, vocab_size: %s",
            entropy_weight,
            noise_rate,
            vocab_size,
        )
        super().__init__(**kwargs)
        self.entropy_weight = entropy_weight
        self.noise_rate = noise_rate
        self.vocab_size = vocab_size

    def compute_loss_t(self, key, t, params, state, freezed_params, x, y):
        # x is (input_ids, attention_mask)
        input_ids, attention_mask = x
        joind_param = jax.tree.map(
            lambda x, y, m: x if m else y,
            params,
            unfreeze(freezed_params),
            self.train_mask,
        )

        # Symmetric NLL on clean data
        nll1, state, out1 = self.nll(joind_param, state, t, x, y, train=True, key=key)

        t2 = (t + self.t_max / 2) % self.t_max
        nll2, state, out2 = self.nll(joind_param, state, t2, x, y, train=True, key=key)

        # Create OOD data (uniform batch noise)
        key, token_key = jax.random.split(key)
        noise_mask = (
            jax.random.uniform(token_key, input_ids.shape) < self.noise_rate
        ) & (attention_mask > 0)

        random_tokens = jax.random.randint(
            token_key, input_ids.shape, 0, self.vocab_size
        )
        x_ood = (
            jnp.where(noise_mask, random_tokens, input_ids),
            attention_mask,
        )

        # JSD on OOD data
        out_ood1, state = self(joind_param, state, t, x_ood, train=True, key=key)
        probs_ood1 = jax.nn.softmax(out_ood1, axis=-1)

        out_ood2, state = self(joind_param, state, t2, x_ood, train=True, key=key)
        probs_ood2 = jax.nn.softmax(out_ood2, axis=-1)

        m_ood = 0.5 * (probs_ood1 + probs_ood2)
        kl_1 = jnp.sum(
            probs_ood1 * (jnp.log(probs_ood1 + 1e-8) - jnp.log(m_ood + 1e-8)), axis=-1
        )
        kl_2 = jnp.sum(
            probs_ood2 * (jnp.log(probs_ood2 + 1e-8) - jnp.log(m_ood + 1e-8)), axis=-1
        )
        jsd_ood = (0.5 * (kl_1 + kl_2)).mean()

        loss = 0.5 * (nll1.mean() + nll2.mean()) - self.entropy_weight * jsd_ood

        if self.weight_decay != 0.0:
            loss = loss + sum(
                self.l2_loss(w, alpha=self.weight_decay)
                for w in jax.tree.leaves(params)
            )
        return loss.squeeze(), (
            state,
            {"nll1": nll1.mean(), "nll2": nll2.mean(), "jsd_ood": jsd_ood},
        )


class JensenShannonNoiseSamplingMixin:
    def __init__(
        self,
        entropy_weight=0.0,
        noise_rate=0.1,
        target_jsd=0.0,
        vocab_size=151936,
        **kwargs,
    ):
        logger.info(
            "Using Jensen-Shannon Noise Sampling Mixin with weight: %s, noise_rate: %s, target_jsd: %s, vocab_size: %s",
            entropy_weight,
            noise_rate,
            target_jsd,
            vocab_size,
        )
        super().__init__(**kwargs)
        self.entropy_weight = entropy_weight
        self.noise_rate = noise_rate
        self.target_jsd = target_jsd
        self.vocab_size = vocab_size

    def compute_loss_t(self, key, t, params, state, freezed_params, x, y):
        # x is (input_ids, attention_mask)
        input_ids, attention_mask = x
        joind_param = jax.tree.map(
            lambda x, y, m: x if m else y,
            params,
            unfreeze(freezed_params),
            self.train_mask,
        )

        # Symmetric NLL on clean data
        nll1, state, out1 = self.nll(joind_param, state, t, x, y, train=True, key=key)

        t2 = (t + self.t_max / 2) % self.t_max
        nll2, state, out2 = self.nll(joind_param, state, t2, x, y, train=True, key=key)

        # Create OOD data (dynamic per-sample noise rate)
        key, rate_key, token_key = jax.random.split(key, 3)
        batch_size = input_ids.shape[0]

        # Sample a different noise rate for EACH sequence in the batch [0, noise_rate]
        sample_noise_rates = jax.random.uniform(
            rate_key, shape=(batch_size, 1), minval=0.0, maxval=self.noise_rate
        )

        noise_mask = (
            jax.random.uniform(token_key, input_ids.shape) < sample_noise_rates
        ) & (attention_mask > 0)

        random_tokens = jax.random.randint(
            token_key, input_ids.shape, 0, self.vocab_size
        )
        x_ood = (
            jnp.where(noise_mask, random_tokens, input_ids),
            attention_mask,
        )

        # JSD on OOD data
        out_ood1, state = self(joind_param, state, t, x_ood, train=True, key=key)
        probs_ood1 = jax.nn.softmax(out_ood1, axis=-1)

        out_ood2, state = self(joind_param, state, t2, x_ood, train=True, key=key)
        probs_ood2 = jax.nn.softmax(out_ood2, axis=-1)

        m_ood = 0.5 * (probs_ood1 + probs_ood2)
        kl_1 = jnp.sum(
            probs_ood1 * (jnp.log(probs_ood1 + 1e-8) - jnp.log(m_ood + 1e-8)), axis=-1
        )
        kl_2 = jnp.sum(
            probs_ood2 * (jnp.log(probs_ood2 + 1e-8) - jnp.log(m_ood + 1e-8)), axis=-1
        )
        jsd_ood = (0.5 * (kl_1 + kl_2)).mean()

        # Margin-based JSD (target_jsd) - only penalize if below target
        jsd_loss = jnp.maximum(self.target_jsd - jsd_ood, 0.0)

        loss = 0.5 * (nll1.mean() + nll2.mean()) + self.entropy_weight * jsd_loss

        if self.weight_decay != 0.0:
            loss = loss + sum(
                self.l2_loss(w, alpha=self.weight_decay)
                for w in jax.tree.leaves(params)
            )
        return loss.squeeze(), (
            state,
            {"nll1": nll1.mean(), "nll2": nll2.mean(), "jsd_ood": jsd_ood},
        )


class JensenShannonNoiseSamplingDropoutMixin:
    def __init__(
        self,
        entropy_weight=0.0,
        noise_rate=1.0,
        target_jsd=0.0,
        vocab_size=151936,
        **kwargs,
    ):
        logger.info(
            "Using Jensen-Shannon Noise Sampling Dropout Mixin with weight: %s, noise_rate (K): %s, target_jsd: %s, vocab_size: %s",
            entropy_weight,
            noise_rate,
            target_jsd,
            vocab_size,
        )
        super().__init__(**kwargs)
        self.entropy_weight = entropy_weight
        self.noise_rate = noise_rate
        self.target_jsd = target_jsd
        self.vocab_size = vocab_size

    def compute_loss_t(self, key, t, params, state, freezed_params, x, y):
        # x is (input_ids, attention_mask)
        input_ids, attention_mask = x
        joind_param = jax.tree.map(
            lambda x, y, m: x if m else y,
            params,
            unfreeze(freezed_params),
            self.train_mask,
        )

        # Symmetric NLL on clean data
        nll1, state, out1 = self.nll(joind_param, state, t, x, y, train=True, key=key)

        t2 = (t + self.t_max / 2) % self.t_max
        nll2, state, out2 = self.nll(joind_param, state, t2, x, y, train=True, key=key)

        # Create OOD data (dynamic attention dropout based on percentage)
        key, rate_key, drop_key = jax.random.split(key, 3)
        batch_size, seq_len = input_ids.shape

        # Determine valid sequence lengths per batch element
        seq_lengths = jnp.sum(attention_mask > 0, axis=-1)

        # Sample a different noise rate (percentage) for EACH sequence in the batch
        # Values are uniformly distributed between [0.0, self.noise_rate]
        sample_noise_rates = jax.random.uniform(
            rate_key, shape=(batch_size,), minval=0.0, maxval=self.noise_rate
        )

        # Calculate K dynamically: round(length * percentage)
        sample_k = jnp.round(seq_lengths * sample_noise_rates).astype(jnp.int32)

        # Create a random noise matrix for the top-k selection
        noise = jax.random.uniform(drop_key, attention_mask.shape)
        # Only consider valid tokens (mask > 0), set others to -1.0
        noise = jnp.where(attention_mask > 0, noise, -1.0)

        # Sort the noise. Invalid tokens (-1.0) are naturally pushed to the front.
        sorted_noise = jnp.sort(noise, axis=-1)  # (batch_size, seq_len)

        # The threshold value to drop the top 'k' tokens is at index (seq_len - k)
        threshold_idx = jnp.clip(seq_len - sample_k, 0, seq_len - 1)
        thresholds = jnp.take_along_axis(sorted_noise, threshold_idx[:, None], axis=-1)

        # Tokens with noise >= threshold AND that are valid are dropped.
        # We also ensure sample_k > 0 so we don't accidentally drop a token when K=0.
        to_drop = (noise >= thresholds) & (attention_mask > 0) & (sample_k[:, None] > 0)
        new_attention_mask = jnp.where(to_drop, 0, attention_mask)

        # Construct the OOD input (clean tokens, dropped attention)
        x_ood = (input_ids, new_attention_mask)

        # JSD on OOD data
        out_ood1, state = self(joind_param, state, t, x_ood, train=True, key=key)
        probs_ood1 = jax.nn.softmax(out_ood1, axis=-1)

        out_ood2, state = self(joind_param, state, t2, x_ood, train=True, key=key)
        probs_ood2 = jax.nn.softmax(out_ood2, axis=-1)

        m_ood = 0.5 * (probs_ood1 + probs_ood2)
        kl_1 = jnp.sum(
            probs_ood1 * (jnp.log(probs_ood1 + 1e-8) - jnp.log(m_ood + 1e-8)), axis=-1
        )
        kl_2 = jnp.sum(
            probs_ood2 * (jnp.log(probs_ood2 + 1e-8) - jnp.log(m_ood + 1e-8)), axis=-1
        )
        jsd_ood = (0.5 * (kl_1 + kl_2)).mean()

        # Margin-based JSD (target_jsd) - only penalize if below target
        jsd_loss = jnp.maximum(self.target_jsd - jsd_ood, 0.0)

        loss = 0.5 * (nll1.mean() + nll2.mean()) + self.entropy_weight * jsd_loss

        if self.weight_decay != 0.0:
            loss = loss + sum(
                self.l2_loss(w, alpha=self.weight_decay)
                for w in jax.tree.leaves(params)
            )
        return loss.squeeze(), (
            state,
            {"nll1": nll1.mean(), "nll2": nll2.mean(), "jsd_ood": jsd_ood},
        )


@register_subspace_model("repulsive_regression")
class RepulsiveRegressionSubspace(RepulsiveMixin, RegressionSubspace):
    pass


@register_subspace_model("repulsive_dist_regression")
class RepulsiveDistRegressionSubspace(RepulsiveMixin, DistRegressionSubspace):
    pass


@register_subspace_model("repulsive_category")
class RepulsiveCategorySubspace(RepulsiveMixin, CategorySubspace):
    pass


@register_subspace_model("repulsive_quantile")
class RepulsiveQuantileSubspace(RepulsiveMixin, QuantileSubspace):
    pass


class LoraAbstractParams:
    def __init__(self, A, B, w0, lora_rho_w: float, dims: tuple, lora_alpha: float):
        self.A = A
        self.B = B
        self.w0 = w0
        self.dims = tuple(
            int(d) for d in dims
        )  # tuple of indices where the LoRA parameters are applied e.g. (0, 1) for the first two axes (First axis is for B second for A)
        self.lora_alpha = (
            lora_alpha  # Scaling factor for LoRA, can be adjusted as needed
        )
        self.lora_rho_w = lora_rho_w

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

    def add_w_noise(self, p, k):
        scaling = self.lora_rho_w / jnp.sqrt(p.shape[0])
        w_noise = (
            scaling
            * jnp.linalg.norm(p, axis=0, keepdims=True)
            * jax.random.normal(k, p.shape, dtype=p.dtype)
        )
        return p + w_noise

    def _apply(self, key):
        raise NotImplementedError(
            "_apply must be implemented by subclasses of LoraAbstractParams."
        )

    def apply(self, key):
        p = self._apply(key)
        # if self.lora_rho_w > 0.:
        key, subkey = random.split(key)
        return self.add_w_noise(p, subkey)
        # return p

    def set_rho(self, rho_s, rho_w):
        # if isinstance(p, LoraAbstractParams):
        self.lora_rho_w = rho_w
        if hasattr(self, "lora_rho_s"):
            self.lora_rho_s = rho_s
        return self
        # return p


@register_pytree_node_class
class LoRAParams(LoraAbstractParams):
    @property
    def shape(self):
        return {"A": self.A.shape, "B": self.B.shape, "w0": self.w0.shape}

    def __repr__(self):
        # return f"LoRA (A={self.A.shape}, B={self.B.shape}, w0={self.w0.shape}) with dims={self.dims} and lora_alpha={self.lora_alpha})"
        return f"LoRA (A={self.A}, B={self.B}, w0={self.w0}) with dims={self.dims} and lora_alpha={self.lora_alpha}, lora_rho_w={self.lora_rho_w})"

    def tree_flatten(self):
        children = (self.A, self.B, self.w0, self.lora_rho_w)
        aux_data = (self.dims, self.lora_alpha)
        return children, aux_data

    def _apply(self, key):
        # print(f"Applying LoRA to {p.w0.shape} with A: {p.A.shape}, B: {p.B.shape}")
        w = jnp.matmul(self.B, self.A)  # Apply LoRA transformation
        w = jnp.moveaxis(
            w, [-2, -1], self.dims
        )  # Move the last two axes to the positions specified by m
        scale = self.lora_alpha / self.B.shape[-1]  # Rank of the LoRA transformation
        return (w * scale).astype(
            self.w0.dtype
        ) + self.w0  # Add the original parameter w0


@register_pytree_node_class
class LoRAABParam(LoRAParams):
    pass


@register_pytree_node_class
class LoRAScaleParams(LoraAbstractParams):
    def __init__(self, A, s, B, w0, lora_rho_w: float, dims: tuple, lora_alpha: float):
        super().__init__(A, B, w0, lora_rho_w, dims, lora_alpha)
        self.s = s

    @property
    def shape(self):
        return {
            "A": self.A.shape,
            "s": self.s.shape,
            "B": self.B.shape,
            "w0": self.w0.shape,
        }

    def __repr__(self):
        # return f"LoRA(A={self.A.shape}, s={self.s.shape}, B={self.B.shape}, w0={self.w0.shape}) with dims={self.dims} and lora_alpha={self.lora_alpha})"
        return f"LoRA(A={self.A}, s={self.s}, B={self.B}, w0={self.w0}) with dims={self.dims} and lora_alpha={self.lora_alpha}, lora_rho_w={self.lora_rho_w})"

    def tree_flatten(self):
        children = (self.A, self.s, self.B, self.w0, self.lora_rho_w)
        aux_data = (self.dims, self.lora_alpha)
        return children, aux_data

    def _apply(self, key):
        # print(f"Applying LoRA to {p.w0.shape} with A: {p.A.shape}, B: {p.B.shape}")
        # A_ = jnp.matmul(self.s, self.A)  # Scale A by s
        # w = jnp.matmul(self.B, A_)  # Apply LoRA transformation
        w = jnp.einsum("...j,ji,ik->...k", self.B, self.s, self.A)
        w = jnp.moveaxis(
            w, [-2, -1], self.dims
        )  # Move the last two axes to the positions specified by m
        scale = self.lora_alpha / self.B.shape[-1]  # Rank of the LoRA transformation
        return (w * scale).astype(
            self.w0.dtype
        ) + self.w0  # Add the original parameter w0


@register_pytree_node_class
class LoRAScaleDiagParams(LoRAScaleParams):
    def _apply(self, key):
        # print(f"Applying LoRA to {p.w0.shape} with A: {p.A.shape}, B: {p.B.shape}")
        # A_ = jnp.matmul(jnp.diag(self.s), self.A)  # Scale A by s
        # w = jnp.matmul(self.B, A_)  # Apply LoRA transformation
        w = jnp.einsum("...i,i,ij->...j", self.B, self.s, self.A)
        w = jnp.moveaxis(
            w, [-2, -1], self.dims
        )  # Move the last two axes to the positions specified by m
        scale = self.lora_alpha / self.B.shape[-1]  # Rank of the LoRA transformation
        return (w * scale).astype(
            self.w0.dtype
        ) + self.w0  # Add the original parameter w0


@register_pytree_node_class
class LoRAepsilonAParams(LoRAABParam):
    def __init__(
        self,
        A,
        B,
        w0,
        lora_rho_w: float,
        lora_rho_s: float,
        dims: tuple,
        lora_alpha: float,
    ):
        super().__init__(A, B, w0, lora_rho_w, dims, lora_alpha)
        self.lora_rho_s = lora_rho_s

    def __repr__(self):
        # return f"LoRA(A={self.A.shape}, s={self.s.shape}, B={self.B.shape}, w0={self.w0.shape}) with dims={self.dims} and lora_alpha={self.lora_alpha})"
        return f"LoRA(A={self.A}, B={self.B}, w0={self.w0}) with dims={self.dims} and lora_alpha={self.lora_alpha}, rho_w={self.lora_rho_w}, rho_s={self.lora_rho_s})"

    def tree_flatten(self):
        children = (self.A, self.B, self.w0, self.lora_rho_w, self.lora_rho_s)
        aux_data = (self.dims, self.lora_alpha)
        return children, aux_data

    def _apply(self, key):
        # print(f"Applying LoRA to {p.w0.shape} with A: {p.A.shape}, B: {p.B.shape}")
        epsilon_s = (
            random.normal(key, self.A.shape[0]) * self.lora_rho_s + 1.0
        )  # Add noise around 1
        w = jnp.einsum("...i,i,ij->...j", self.B, epsilon_s, self.A)
        # jax.debug.print(f"LoRAepsilonABParams: Applied noise with shape {w.shape}")
        # A_ = jnp.matmul(jnp.diag(epsilon_s), self.A)  # Scale A by s
        # w = jnp.matmul(self.B, A_)  # Apply LoRA transformation
        w = jnp.moveaxis(
            w, [-2, -1], self.dims
        )  # Move the last two axes to the positions specified by m
        scale = self.lora_alpha / self.B.shape[-1]  # Rank of the LoRA transformation
        return (w * scale).astype(
            self.w0.dtype
        ) + self.w0  # Add the original parameter w0


@register_pytree_node_class
class LoRAepsilonABParams(LoRAepsilonAParams):
    pass


@register_pytree_node_class
class LoRAepsilonScaleDiagParams(LoRAScaleDiagParams):
    def __init__(
        self,
        A,
        s,
        B,
        w0,
        lora_rho_w: float,
        lora_rho_s: float,
        dims: tuple,
        lora_alpha: float,
    ):
        super().__init__(A, s, B, w0, lora_rho_w, dims, lora_alpha)
        self.lora_rho_s = lora_rho_s

    def __repr__(self):
        # return f"LoRA(A={self.A.shape}, s={self.s.shape}, B={self.B.shape}, w0={self.w0.shape}) with dims={self.dims} and lora_alpha={self.lora_alpha})"
        return f"LoRA(A={self.A}, s={self.s}, B={self.B}, w0={self.w0}) with dims={self.dims} and lora_alpha={self.lora_alpha}, rho_w={self.lora_rho_w}, rho_s={self.lora_rho_s})"

    def tree_flatten(self):
        children = (self.A, self.s, self.B, self.w0, self.lora_rho_w, self.lora_rho_s)
        aux_data = (self.dims, self.lora_alpha)
        return children, aux_data

    def _apply(self, key):
        # print(f"Applying LoRA to {p.w0.shape} with A: {p.A.shape}, B: {p.B.shape}")
        epsilon_s = (
            random.normal(key, self.s.shape) * self.lora_rho_s + 1.0
        )  # Add noise around 1
        w = jnp.einsum("...r,r,r,rj->...j", self.B, self.s, epsilon_s, self.A)
        w = jnp.moveaxis(
            w, [-2, -1], self.dims
        )  # Move the last two axes to the positions specified by m
        scale = self.lora_alpha / self.B.shape[-1]  # Rank of the LoRA transformation
        return (w * scale).astype(
            self.w0.dtype
        ) + self.w0  # Add the original parameter w0


# get stiefel manifold projection for tall matrices
def constrain_to_stiefel_mainfold(V):
    """
    Erzeugt eine m x r Matrix mit orthonormalen Spalten.
    V: (m, r) - Die Matrix der r Reflektor-Vektoren.
    """
    m, r = V.shape
    # Wir starten mit den ersten r Spalten der Einheitsmatrix
    # Das ist eine m x r Matrix
    A_init = jnp.eye(m, r)
    V /= (
        jnp.linalg.norm(V, axis=0, keepdims=True) + 1e-8
    )  # Normierung der Reflektor-Vektoren

    # Wir wenden die r Spiegelungen nacheinander an
    def housholder(A, v):
        # v must be a unit vector
        # H @ A = (I - 2vv^T) @ A = A - 2 * v @ (v^T @ A)
        # v^T @ A ist ein Vektor der Größe (r,)
        A = A - 2.0 * jnp.outer(v, v.T @ A)
        # A_next = apply_householder_to_tall(A, v)
        return A, None

    A_final, _ = jax.lax.scan(housholder, A_init, V.T, unroll=False)
    return A_final


@register_pytree_node_class
class LoRARotaEpsilonDiagscaleRotbParams(LoRAepsilonScaleDiagParams):
    def _apply(self, key):
        # print(f"Applying LoRA to {p.w0.shape} with A: {p.A.shape}, B: {p.B.shape}")
        # A_ = jnp.matmul(self.s, self.A)  # Scale A by s
        # w = jnp.matmul(self.B, A_)  # Apply LoRA transformation
        A = constrain_to_stiefel_mainfold(
            self.A.T
        ).T  # Constrain A to the Stiefel manifold
        B = constrain_to_stiefel_mainfold(self.B)  # Constrain B to the Stiefel manifold
        epsilon_s = (
            random.normal(key, self.s.shape[0]) * self.lora_rho_s + 1.0
        )  # Add noise around 1
        epsilon_s = jax.nn.softplus(epsilon_s) + 1e-8  # Ensure epsilon_s is positive
        s = jax.nn.softplus(self.s) + 1e-8  # Ensure s is positive
        w = jnp.einsum("...r,r,r,ri->...i", B, s, epsilon_s, A)
        w = jnp.moveaxis(
            w, [-2, -1], self.dims
        )  # Move the last two axes to the positions specified by m
        scale = self.lora_alpha / self.B.shape[-1]  # Rank of the LoRA transformation
        return (w * scale).astype(
            self.w0.dtype
        ) + self.w0  # Add the original parameter w0


@register_pytree_node_class
class LoRAAEpsilonRotdiagscaleBParams(LoRAepsilonScaleDiagParams):
    pass


class LoRAMixin:
    """
    Mixin class for LoRA (Low-Rank Adaptation) functionality.
    This class provides methods to convert a pytree of parameters into a matrix and vice versa.
    Parameters:
    - r (int): The rank for the LoRA adaptation.
    - lora_mask (dict): A mask indicating which parameters should be adapted using LoRA.
    - lora_dtype (str): The data type for LoRA parameters. Options are "original", "float16", "bfloat16", "float32".
    - lora_alpha (float): Scaling factor for LoRA.
    - lora_mode (str): The mode of LoRA adaptation. Options are "A(t)B", "As(t)B", "A(t)B(t)".
    - lora_rho (float): if > 0, use Flat-LoRA with adding weight noise of factor rho.
    - kwargs: Additional keyword arguments for the parent class.
    """

    def __init__(
        self,
        r,
        lora_dtype: str,
        lora_alpha: float,
        lora_mode: str = "A(t)B",
        lora_rho: float = 0.0,
        lora_rho_s: float = 0.0,
        **kwargs,
    ):
        logger.info(
            "Using LoRA Mixin with rank=%s, dtype=%s, alpha=%s, mode=%s, rho=%s, rho_s=%s",
            r,
            lora_dtype,
            lora_alpha,
            lora_mode,
            lora_rho,
            lora_rho_s,
        )
        super().__init__(**kwargs)
        self.r = r
        self.w0_params = None  # Placeholder for non-LoRA parameters
        self.lora_dtype = lora_dtype
        self.lora_alpha = lora_alpha
        self.lora_mode = lora_mode
        self.lora_rho_w = lora_rho
        self.lora_rho_s = lora_rho_s
        if "e" in lora_mode:
            if lora_rho_s == 0.0:
                logger.warning(
                    "Using LoRA mode %s with lora_rho_s=0 makes no sense and is equivalent to %s.",
                    lora_mode,
                    lora_mode.replace("e", ""),
                )

    def get_dtype(self, param):
        if self.lora_dtype == "original":
            return param.dtype
        else:
            if self.lora_dtype == "float16":
                return jnp.float16
            elif self.lora_dtype == "bfloat16":
                return jnp.bfloat16
            elif self.lora_dtype == "float32":
                return jnp.float32
            else:
                raise ValueError(f"Unknown lora_dtype: {self.lora_dtype}")

    def set_lora_rho(self, rho_w, rho_s, params):
        # inplace update of rho for all lora params in the pytree
        params["params"] = jax.tree.map(
            lambda p: (
                p.set_rho(rho_s=rho_s, rho_w=rho_w)
                if isinstance(p, LoraAbstractParams)
                else p
            ),
            params["params"],
            is_leaf=lambda p: isinstance(p, LoraAbstractParams),
        )
        # self.curve_mask = jax.tree.map(lambda p: p.set_rho(rho_s=rho_s, rho_w=rho_w) if isinstance(p, LoraAbstractParams) else p, self.curve_mask, is_leaf=lambda p: isinstance(p, LoraAbstractParams))
        # self.train_mask = jax.tree.map(lambda p: p.set_rho(rho_s=rho_s, rho_w=rho_w) if isinstance(p, LoraAbstractParams) else p, self.train_mask, is_leaf=lambda p: isinstance(p, LoraAbstractParams))
        return params

    @staticmethod
    def update_mask_for_lora(
        mask, lora_params, b_off=True, w0_off=True, a_off=False, s_on=None
    ):
        mask = tree_broadcast(
            mask, lora_params
        )  # broadcast mask to the shape of lora_params

        def adapt_mask(mask):
            """Set B and w0 to False and A corresponds to the original curve mask."""
            if isinstance(mask, LoraAbstractParams):
                # Do not use LoRA for w0 and B
                if b_off:
                    mask.B = False
                if w0_off:
                    mask.w0 = False
                if a_off:
                    mask.A = False
                if isinstance(s_on, bool):
                    mask.s = s_on
                if hasattr(mask, "lora_rho_s"):
                    mask.lora_rho_s = False
                mask.lora_rho_w = False
                return mask
            else:
                # For other parameters, keep the mask as is
                return mask

        mask = jax.tree.map(
            adapt_mask, mask, is_leaf=lambda x: isinstance(x, LoraAbstractParams)
        )
        return mask

    def init_params_from_point(self, key, params, mask, noise_fn, lora_mask=None):
        """Initialize parameters from a point in the latent space.

        Args:
            key (jax.random.PRNGKey): Random key for initialization.
            params (dict): params in the latent space. {params: ...}
            mask (dict): Mask indicating which parameters to be handled by the curve. {...: bool}
            lora_mask: LoRA mask pytree indicating which parameters to adapt with LoRA.
            jitter (float): Jitter to add to the initialization for the curve params. As A matrix of LoRA is initialized with N(0,1) this could probably be set to 1.

        Returns:
            _type_: _description_
        """
        assert ("params" in params) and len(
            params.keys()
        ) == 1, "Point must contain 'params' key and no other keys."

        def initialize_lora_params(rng_key, params, lora_mask, rank):
            """
            Initialize LoRA parameters based on the base model and the provided LoRA mask.
            lora_mask of the form:
                'Dense_0': {
                    'kernel': [1, 0],  # Apply Low Rank for kernel on dimension 1 and 0
                        lora_mask must be always a list of two integers indicating the dimensions to apply LoRA
                        ==> first mask dimension is for A matrix (determines low rank dimension), second for B matrix
                        e.g kernel of shape (inC, outC) with mask [0, 1] will have A of shape (rank, outC) and B of shape (inC, rank)
                        e.g kernel of shape (inC, c, outC) with mask [2, 0] will have A of shape (c, rank, inC) and B of shape (c, outC, rank)
                    'bias': False,  # Do not use LoRA for bias
                    },
            returns:
                w0_params: Parameters of the model used as w0 by LoRA.
                lora_params: Parameters of the model. LoRA marked parameters are tuples of (A, B) where:
                    A: The low-rank matrix of shape (rank, param_shape[m[1]])
                    B: The low-rank matrix of shape (param_shape[m[0]], rank)
                    Non LoRA parameters are returned as is.
            """
            # Initialize the base model parameters
            mask_, structure = jax.tree_util.tree_flatten_with_path(lora_mask)

            def shape_builder(shape, r_dim):
                shape = list(shape)
                shape[r_dim] = rank
                return tuple(shape)

            lora_params = []
            for (key, m), p in zip(mask_, jax.tree.leaves(params)):
                param_name = "/".join([k.key for k in key])
                # change to a lora param
                if not isinstance(m, bool):
                    assert (
                        p.ndim > 1
                    ), f"Param {param_name} cannot be used for LoRA. Too few dimensions for parameter with shape {p.shape}"
                    assert (
                        p.ndim > np.max(m)
                    ), f"Mask {m} incompatible with param {param_name} and shape {p.shape}"
                    dtype = self.get_dtype(p)
                    B = jnp.zeros(shape_builder(p.shape, m[1]), dtype=dtype)
                    B = jnp.moveaxis(
                        B, [m[0], m[1]], [-2, -1]
                    )  # Move the first axis to the last position

                    # Initialize A matrix with He normal initialization
                    rng_key, subkey = random.split(rng_key)
                    # init_a = jax.nn.initializers.he_normal(out_axis=m[1].item()) # fan_in are all other dimensions except the m[1] (last dim is output of nn.Dense)
                    # from PEFT
                    #     nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
                    gain = math.sqrt(
                        2 / (1 + 5)
                    )  # Gain for initialization sqrt(2/(1+math.sqrt(5)^2)) for ReLU
                    init_a = jax.nn.initializers.variance_scaling(
                        out_axis=m[0].item(),
                        distribution="uniform",
                        mode="fan_in",
                        scale=gain**2,
                    )  # fan_in = all shapes except m[1]
                    shape_ = shape_builder(p.shape, m[0])
                    A = init_a(subkey, shape_, dtype=dtype)
                    # shape_[m[1]] = 0
                    # gain = jnp.sqrt(6 / np.prod(shape_))  # Gain for initialization
                    # A = jax.random.uniform(subkey, shape_builder(p.shape, m[0]), dtype=p.dtype)
                    A = jnp.moveaxis(
                        A, [m[0], m[1]], [-2, -1]
                    )  # Move the first axis to the last position
                    if self.lora_mode == "A(t)B":
                        lora_params.append(
                            LoRAParams(
                                A=A,
                                B=B,
                                w0=p,
                                dims=tuple(m),
                                lora_alpha=self.lora_alpha,
                                lora_rho_w=self.lora_rho_w,
                            )
                        )  # Store A and B as a tuple
                    elif self.lora_mode == "As(t)B":
                        rng_key, subkey = random.split(rng_key)
                        s = jax.nn.initializers.orthogonal()(
                            subkey, (rank, rank), dtype=dtype
                        )
                        # s = jnp.ones((rank, 1), dtype=dtype)  # Scaling vector for A
                        lora_params.append(
                            LoRAScaleParams(
                                A=A,
                                s=s,
                                B=B,
                                w0=p,
                                dims=tuple(m),
                                lora_alpha=self.lora_alpha,
                                lora_rho_w=self.lora_rho_w,
                            )
                        )  # Store A, s and B
                    elif self.lora_mode == "A(t)B(t)":
                        lora_params.append(
                            LoRAABParam(
                                A=A,
                                B=B,
                                w0=p,
                                dims=tuple(m),
                                lora_alpha=self.lora_alpha,
                                lora_rho_w=self.lora_rho_w,
                            )
                        )  # used to indicate both A and B are time dependent
                    elif (self.lora_mode == "Asd(t)B") or (
                        self.lora_mode == "Asd(t)eB"
                    ):
                        rng_key, subkey = random.split(rng_key)
                        s = (
                            jax.nn.initializers.normal(stddev=1.0)(
                                subkey, (rank,), dtype=dtype
                            )
                            + 1.0
                        )
                        if self.lora_mode == "Asd(t)B":
                            # lora_params.append(LoRAScaleDiagParams(A=A, s=s, B=B, w0=p, dims=tuple(m), lora_alpha=self.lora_alpha, lora_rho_w=self.lora_rho_w))  # Store A, s and B
                            lora_params.append(
                                LoRAScaleDiagParams(
                                    A=A,
                                    s=s,
                                    B=B,
                                    w0=p,
                                    dims=tuple(m),
                                    lora_alpha=self.lora_alpha,
                                    lora_rho_w=self.lora_rho_w,
                                )
                            )  # Store A, s and B
                        else:  # Asd(t)eB
                            lora_params.append(
                                LoRAepsilonScaleDiagParams(
                                    A=A,
                                    s=s,
                                    B=B,
                                    w0=p,
                                    dims=tuple(m),
                                    lora_alpha=self.lora_alpha,
                                    lora_rho_w=self.lora_rho_w,
                                    lora_rho_s=self.lora_rho_s,
                                )
                            )  # Store A, s and B
                    elif self.lora_mode == "A(t)eB(t)":
                        lora_params.append(
                            LoRAepsilonABParams(
                                A=A,
                                B=B,
                                w0=p,
                                dims=tuple(m),
                                lora_alpha=self.lora_alpha,
                                lora_rho_w=self.lora_rho_w,
                                lora_rho_s=self.lora_rho_s,
                            )
                        )  # used to indicate both A and B are time dependent
                    elif self.lora_mode == "A(t)eB":
                        lora_params.append(
                            LoRAepsilonAParams(
                                A=A,
                                B=B,
                                w0=p,
                                dims=tuple(m),
                                lora_alpha=self.lora_alpha,
                                lora_rho_w=self.lora_rho_w,
                                lora_rho_s=self.lora_rho_s,
                            )
                        )  # used to indicate both A and B are time dependent
                    elif (self.lora_mode == "Arot(t)sdeBrot(t)") or (
                        self.lora_mode == "Arotsd(t)eB"
                    ):
                        # because B is identiy => Bs = 0 => defaut Lora behavior
                        rng_key, subkey = random.split(rng_key)
                        s = jax.nn.initializers.constant(-10.0)(
                            subkey, (rank,), dtype=dtype
                        )
                        rng_key, subkey = random.split(rng_key)
                        # A = jax.nn.initializers.orthogonal()(subkey, A.shape, dtype=dtype)
                        A = jax.nn.initializers.normal(
                            stddev=0.01
                        )(
                            subkey, A.shape, dtype=dtype
                        )  # add small noise to make sure we are not exactly on the stiefel manifold at the beginning
                        rng_key, subkey = random.split(rng_key)
                        # B = jax.nn.initializers.orthogonal()(subkey, B.shape, dtype=dtype)
                        B = jax.nn.initializers.normal(
                            stddev=0.01
                        )(
                            subkey, B.shape, dtype=dtype
                        )  # add small noise to make sure we are not exactly on the stiefel manifold at the beginning
                        if self.lora_mode == "Arot(t)sdeBrot(t)":
                            lora_params.append(
                                LoRARotaEpsilonDiagscaleRotbParams(
                                    A=A,
                                    s=s,
                                    B=B,
                                    w0=p,
                                    dims=tuple(m),
                                    lora_alpha=self.lora_alpha,
                                    lora_rho_w=self.lora_rho_w,
                                    lora_rho_s=self.lora_rho_s,
                                )
                            )  # Store A, s and B
                        else:  # Arotsd(t)eB
                            lora_params.append(
                                LoRAAEpsilonRotdiagscaleBParams(
                                    A=A,
                                    s=s,
                                    B=B,
                                    w0=p,
                                    dims=tuple(m),
                                    lora_alpha=self.lora_alpha,
                                    lora_rho_w=self.lora_rho_w,
                                    lora_rho_s=self.lora_rho_s,
                                )
                            )  # Store A, s and B
                    else:
                        raise ValueError(
                            f"Unknown lora_mode: {self.lora_mode} || Supported modes are 'A(t)B', 'As(t)B', 'A(t)B(t)'"
                        )
                    # lora_params.append(dict(A=A, B=B))  # Store A and B
                else:
                    lora_params.append(p)

            lora_params = jax.tree_util.tree_unflatten(structure, lora_params)
            return lora_params

        params["params"] = initialize_lora_params(
            key, params["params"], lora_mask, rank=self.r
        )
        params = super().init_params_from_point(key, params, mask, noise_fn)

        if (
            (self.lora_mode == "A(t)B(t)")
            or (self.lora_mode == "A(t)eB(t)")
            or (self.lora_mode == "Arot(t)sdeBrot(t)")
        ):
            # s for mode "Arot(t)sdeBrot(t)" is automatically set to zero
            if self.lora_mode == "Arot(t)sdeBrot(t)":
                mask_update = self.update_mask_for_lora(
                    mask, params["params"], b_off=False, w0_off=True, s_on=False
                )
            else:
                mask_update = self.update_mask_for_lora(
                    mask, params["params"], b_off=False, w0_off=True
                )
        elif (
            (self.lora_mode == "As(t)B")
            or (self.lora_mode == "Asd(t)B")
            or (self.lora_mode == "Asd(t)eB")
            or (self.lora_mode == "Arotsd(t)eB")
        ):
            mask_update = self.update_mask_for_lora(
                mask, params["params"], b_off=True, w0_off=True, a_off=True
            )
        else:
            # update the original curve mask to include LoRA parameter B into curve approach if the parameter was originally selected by the mask
            mask_update = self.update_mask_for_lora(
                mask, params["params"], b_off=True, w0_off=True
            )
        self.curve_mask = mask_update

        # initialize parameters for Flat-LoRA weight noise
        only_lora_params = jax.tree.map(
            lambda m: True if isinstance(m, LoraAbstractParams) else None,
            self.curve_mask,
            is_leaf=lambda x: isinstance(x, LoraAbstractParams),
        )
        self.num_train_leaves = len(jax.tree.leaves(only_lora_params))
        self.struct_lora = jax.tree.structure(only_lora_params)

        return params

        # # update the original curve mask to include LoRA parameter A into curve approach if the parameter was originally selected by the mask
        # params = super().init_params_from_point(key, {'params': lora_params}, mask_update, noise_fn)
        # # initialize curve parameters according mask
        # return params

    def init_params(self, key, x, mask):
        raise NotImplementedError("init_params not possible for LoRA.")

    # @partial(jit, static_argnums=(0,), donate_argnums=(2,))
    # @partial(jit, static_argnums=(0,))
    def __call__(
        self, params, state, t, x, train=True, key=None
    ) -> Tuple[jnp.ndarray, dict]:
        """
        Computes the output of the model for given parameters, time, and input.

        Parameters:
        - params: The parameters of the model.
        - t: The time parameter. Only single value supported.
        - x: The input data.

        Returns:
        - The output of the model.

        """
        if isinstance(t, dict):

            def curve_to_point(t, p, m):
                if m:
                    bezier_coeff = self.bezier(t)
                    return jnp.einsum("k,k...->...", bezier_coeff, p).astype(p.dtype)
                return p

            params = jax.tree.map(curve_to_point, t, params, self.curve_mask)
        else:
            # sample Bezier coefficient
            bezier_coeff = self.bezier(t)
            # Compute one parameter set per sample
            params = jax.tree.map(
                lambda p, m: (
                    jnp.einsum("k,k...->...", bezier_coeff, p).astype(p.dtype)
                    if m
                    else p
                ),
                params,
                self.curve_mask,
            )

        # # sample Bezier coefficient
        # bezier_coeff = self.bezier(t)
        # # Compute one parameter set per sample
        # params = jax.tree.map(lambda p, m: jnp.einsum(
        #     'k,k...->...', bezier_coeff, p).astype(p.dtype) if m else p, params, self.curve_mask)

        # Apply LoRA to the parameters
        def get_t_tree(key):
            keys_flatt = jax.random.split(key, self.num_train_leaves)
            keys_tree = jax.tree.unflatten(self.struct_lora, keys_flatt)
            keys_tree = jax.tree.map(
                lambda k: False if k is None else k,
                keys_tree,
                is_leaf=lambda x: x is None,
            )  # clean None leaves to False for propper tree.map
            return keys_tree

        key_tree = get_t_tree(key)
        params = jax.tree.map(
            lambda lp, key: lp.apply(key) if isinstance(lp, LoraAbstractParams) else lp,
            params,
            key_tree,
            is_leaf=lambda x: isinstance(x, LoraAbstractParams),
        )

        # already applied in LoRAParams.apply method
        # if (self.lora_rho > 0.) and (key != None):
        #     # Apply Flat-LoRA weight noise
        #     def apply_noise(p, k):
        #         scaling = self.lora_rho / jnp.sqrt(p.shape[0])
        #         w_noise = scaling * jnp.linalg.norm(p, axis=0, keepdims=True) * jax.random.normal(k, p.shape, dtype=p.dtype)
        #         return p + w_noise
        #     params = jax.tree.map(lambda p, k: p if k is False else apply_noise(p, k), params, key_tree)

        # forward pass per sample
        out, state = self._predict(params, state, x, train=train)
        return out, state


@register_subspace_model("lora_regression")
class LoRARegressionSubspace(LoRAMixin, RegressionSubspace):
    pass


@register_subspace_model("lora_category")
class LoRACategorySubspace(LoRAMixin, CategorySubspace):
    pass


@register_subspace_model("lora_repulsive_category")
class LoRARepulsiveCategorySubspace(LoRAMixin, RepulsiveMixin, CategorySubspace):
    pass


@register_subspace_model("lora_repulsive_jsd_category")
class LoRARepulsiveEntropyCategorySubspace(
    LoRAMixin, RepulsiveMixin, JensenShannonMixin, CategorySubspace
):
    pass


@register_subspace_model("lora_quantile")
class LoRAQuantileSubspace(LoRAMixin, QuantileSubspace):
    pass


@register_subspace_model("lora_repulsive_quantile")
class LoRARepulsiveQuantileSubspace(LoRAMixin, RepulsiveMixin, QuantileSubspace):
    pass


@register_subspace_model("lora_entropy_category")
class LoRAEntropyCategorySubspace(LoRAMixin, EntropyMixin2, CategorySubspace):
    pass


@register_subspace_model("lora_jsd_category")
class LoRAJSDCategorySubspace(LoRAMixin, JensenShannonMixin, CategorySubspace):
    pass


@register_subspace_model("jsd_v2_category")
class LoRAJSDV2CategorySubspace(LoRAMixin, JensenShannonMixin_v2, CategorySubspace):
    pass


@register_subspace_model("jsd_noise_category")
class LoRAJSDNoiseCategorySubspace(
    LoRAMixin, JensenShannonNoiseMixin, CategorySubspace
):
    pass


@register_subspace_model("jsd_noise_sampling_category")
class LoRAJSDNoiseSamplingCategorySubspace(
    LoRAMixin, JensenShannonNoiseSamplingMixin, CategorySubspace
):
    pass


@register_subspace_model("jsd_noise_sampling_dropout_category")
class JSDNoiseSamplingDropoutCategorySubspace(
    LoRAMixin, JensenShannonNoiseSamplingDropoutMixin, CategorySubspace
):
    pass


# @partial(jit, static_argnums=(1))
def pytree_to_matrix(pytree, k):
    """
    Converts a pytree of the subspace model into a matrix.

    Args:
        pytree: The pytree to be converted.

    Returns:
        matrix: The matrix representation of the pytree with shape (k+1, n_params).
    """
    # first argument of reduce fn is the accumulator and second is the current value
    return jax.tree.reduce(
        lambda x, y: jnp.concatenate([x, y.reshape(k + 1, -1)], axis=-1), pytree
    )


# @partial(jit, static_argnums=(1))
def masked_pytree_to_matrix(pytree, mask, k):
    """
    Converts a pytree of the subspace model into a matrix.

    Args:
        pytree: The pytree to be converted.

    Returns:
        matrix: The matrix representation of the pytree with shape (k+1, n_params).
    """
    # first argument of reduce fn is the accumulator and second is the current value
    only_curve_params = jax.tree.map(
        lambda x, m: x if m else jnp.reshape(jnp.array([]), (k + 1, 0)), pytree, mask
    )
    return jax.tree.reduce(
        lambda x, y: jnp.concatenate(
            [x.reshape(k + 1, -1), y.reshape(k + 1, -1)], axis=-1
        ),
        only_curve_params,
    )


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
    return jax.tree.reduce(
        lambda x, y: jnp.concatenate([x, y.reshape(-1)], axis=-1), pytree
    )


# @jit
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
        flatten_leafs.append(jnp.reshape(vec[index:upper_index], s))
        index = upper_index
    return jax.tree.unflatten(structure, flatten_leafs)


# @jit
def vec_to_pytree(vec, subspace_params):
    """
    Converts a vector representation of parameters to a pytree structure.
    Without leading dimension.

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
        flatten_leafs.append(jnp.reshape(vec[index:upper_index], s))
        index = upper_index
    return jax.tree.unflatten(structure, flatten_leafs)


# @jit
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
        flatten_leafs.append(jnp.reshape(vec[:, index:upper_index], s))
        index = upper_index
    return jax.tree.unflatten(structure, flatten_leafs)


def filter_mask(
    mask, *key_names, as_curve: bool | np.ndarray = False, BoolschenOperator=all
):
    """Convert a mask to a no-curve mask according key_name. Perfomrs boolschen AND for all key_names or OR with BoolschenOperator=any.
    Note: Function opperates in-place.
    BoolschenOperator can be used to change the behavior of the filter, e.g. to use `any` instead of `all`.
    as_curve can be bool or np.array (but not pytree or jnp.array as mask)
    """
    if len(key_names) == 0:
        return mask

    # Check if path contains all key names
    def key_in_path(path):
        return BoolschenOperator(k in path for k in set(key_names))

    path_values, structure = jax.tree_util.tree_flatten_with_path(mask)
    new_mask = jax.tree.leaves(mask)
    for i, (k, v) in enumerate(path_values):
        path = "/".join((dk.key for dk in k))
        if key_in_path(path):
            new_mask[i] = as_curve
    return jax.tree.unflatten(structure, new_mask)
