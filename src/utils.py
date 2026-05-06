import numpy as np
from jax import numpy as jnp
import jax


def bezier_length(cp):
    """Compute arc length of a Bézier curve defined by control points *cp*."""
    from src.subspace_curve import bezier_coeff_fn

    k = cp.shape[0] - 1
    t = jnp.linspace(0, 1, 1000)
    bezier_coeff_inv = bezier_coeff_fn(k)
    coeff = jax.vmap(bezier_coeff_inv)(t)
    cp_diff = cp[1:] - cp[:-1]

    def _d_length_t(cp_diff, c):
        d_t = jnp.linalg.norm(jnp.einsum("j,j...->...", c, cp_diff) * k)
        return cp_diff, d_t

    _, d_t = jax.lax.scan(_d_length_t, cp_diff, coeff)
    return jax.scipy.integrate.trapezoid(d_t, t)


def lower_bound(cp):
    """Chord length between first and last control point."""
    return jnp.linalg.norm(cp[-1] - cp[0])


def upper_bound(cp):
    """Sum of segment lengths of the control polygon."""
    return jnp.linalg.norm(jnp.diff(cp, axis=0), axis=1).sum()


def bezier_mass_center(cp):
    """Norm of the mass centre of control points."""
    return jnp.linalg.norm(cp.mean(axis=0))


def bezier_gyration(cp):
    """Mean distance of control points from their centre of mass."""
    mc = cp.mean(0)
    return jnp.linalg.norm(cp - mc, axis=1).mean()


def bezier_mean_curvature(cp):
    """Integrated mean curvature of the Bézier curve defined by *cp*."""
    from src.subspace_curve import bezier_curve

    tt = jnp.linspace(0, 1, 10)
    _, d_bezier = bezier_curve(cp.shape[0], cp)

    def _single(t):
        grad_ = d_bezier(t)
        second_dev = jax.jacrev(d_bezier)(t).squeeze()
        norm_r_prime_sq = jnp.dot(grad_, grad_)
        norm_r_double_prime_sq = jnp.dot(second_dev, second_dev)
        dot_r_prime_double_prime = jnp.dot(grad_, second_dev)
        numerator = jnp.sqrt(
            norm_r_prime_sq * norm_r_double_prime_sq - dot_r_prime_double_prime**2
        )
        denominator = norm_r_prime_sq ** (3 / 2)
        return numerator / denominator

    curvature = jax.vmap(_single)(tt)
    return jax.scipy.integrate.trapezoid(curvature, tt)


def bezier_rel_center(cp, initial_center):
    """Distance of the current mass centre from *initial_center*."""
    return jnp.linalg.norm(cp.mean(axis=0) - initial_center)


def calibration_error(logits, labels, num_bins):
    probs = jax.nn.softmax(logits, axis=-1)
    confidences = jnp.max(probs, axis=-1)
    predictions = jnp.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(jnp.float32)

    bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    def body_fun(ece, bins):
        bin_lower, bin_upper = bins
        # get mask for samples with confidence in (bin_lower, bin_upper]
        in_bin = jnp.logical_and(confidences > bin_lower, confidences <= bin_upper)

        prop_in_bin = jnp.mean(in_bin.astype(jnp.float32))
        # same as jnp.mean(accuracies[in_bin]) but jit friendly
        accuracy_in_bin = jnp.where(
            prop_in_bin > 0,
            jnp.sum(jnp.where(in_bin, accuracies, 0.0)) / jnp.sum(in_bin),
            0.0,
        )
        # same as jnp.mean(confidences[in_bin]) but jit friendly
        avg_confidence_in_bin = jnp.where(
            prop_in_bin > 0,
            jnp.sum(jnp.where(in_bin, confidences, 0.0)) / jnp.sum(in_bin),
            0.0,
        )
        ece += jnp.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece, None

    ece, _ = jax.lax.scan(body_fun, 0.0, (bin_lowers, bin_uppers))
    return ece


def post_pred_performance(
    logits,
    labels,
    weights: bool | jnp.ndarray = False,
    key_prefix: str = "",
    num_bins: int = 15,
):
    """Compute Bayesian model averaging predictive metrics from sampled logits.

    Args:
        logits: array shape (n_samples, n_data, n_classes)
        labels: array shape (n_data,)
        weights: False or array of shape (n_samples,) for weighted logsumexp
        key_prefix: if provided, prefix returned metric keys with this string
        num_bins: bins for ECE computation

    Returns:
        dict with keys either ('ll','acc','ece') or with prefix '{key_prefix}bma_ll', etc.
    """
    logits = jax.nn.log_softmax(
        logits, axis=-1
    )  # ensure logits are normalized for logsumexp
    if isinstance(weights, bool):
        # treating True as uniform average
        post_predict_logits = jax.nn.logsumexp(logits, axis=0) - jnp.log(
            logits.shape[0]
        )
    else:
        post_predict_logits = jax.nn.logsumexp(logits, b=weights[:, None, None], axis=0)

    post_acc = jnp.mean(jnp.argmax(post_predict_logits, axis=-1) == labels)
    post_ece = calibration_error(post_predict_logits, labels, num_bins)
    ll = (
        jnp.take_along_axis(post_predict_logits, labels[:, None], axis=-1)
        .squeeze()
        .mean()
    )
    # alternative ll computation using Categorical distribution (should be the same)
    # ll = dist.Categorical(logits=post_predict_logits).log_prob(labels).mean()

    # Brier score
    post_predict_probs = jnp.exp(post_predict_logits)
    true_probs = jax.nn.one_hot(labels, post_predict_logits.shape[-1])
    brier = jnp.mean(jnp.sum((post_predict_probs - true_probs) ** 2, axis=-1))

    return {
        f"{key_prefix}ll": ll,
        f"{key_prefix}acc": post_acc,
        f"{key_prefix}ece": post_ece,
        f"{key_prefix}brier": brier,
    }


def load_fn(m, p, pl):
    """Merge a loaded parameter into the current pytree leaf based on the train mask."""
    if m:
        assert (
            p.shape == pl.shape
        ), f"Shape mismatch: current {p.shape} vs loaded {pl.shape}"
        return pl
    return p


def load_checkpoint_new(run, params, s_model):
    """Load trainable parameters from a wandb run artifact into the model parameters.

    Args:
        run: wandb run object containing the artifacts.
        params: current model parameters (flax format).
        s_model: subspace model (provides train_mask).

    Returns:
        params, bma_weights, trainable_params, init_trainable_params,
        pretrained_params, last_trainable_params
    """
    from src.subspace_curve import LoraAbstractParams

    def load_art(load_run):
        artifact_use, artifact_weights_use = False, False
        print(f"Found run: {load_run.name} ({load_run.id})")
        for art_l in load_run.logged_artifacts():
            if art_l.type == "pytree" and art_l.name.startswith("params"):
                print(f"Downloading params from {art_l.name}")
                artifact_use = art_l
            if art_l.type == "npz" and art_l.name.startswith("bma_weights"):
                print(f"Downloading bma_weights from {art_l.name}")
                artifact_weights_use = art_l
        return artifact_use, artifact_weights_use

    artifact_use, artifact_weights_use = load_art(run)

    trainable_params = False
    init_trainable_params = False
    pretrained_params = False
    last_trainable_params = False

    for f in artifact_use.files():
        if "trainable_params" in f.name:
            if "init" in f.name:
                init_trainable_params = np.load(
                    artifact_use.get_entry(f.name).download(), allow_pickle=True
                ).item()
                print(f"Loaded init_trainable_params from {f.name}")
            elif "last" in f.name:
                last_trainable_params = np.load(
                    artifact_use.get_entry(f.name).download(), allow_pickle=True
                ).item()
                print(f"Loaded last_trainable_params from {f.name}")
            else:
                trainable_params = np.load(
                    artifact_use.get_entry(f.name).download(), allow_pickle=True
                ).item()
                print(f"Loaded trainable_params from {f.name}")
                assert (
                    len(set(trainable_params.keys()) - set(params["params"].keys()))
                    == 0
                ), "Loaded trainable_params do not match model params"
        elif "pretrained_params" in f.name:
            pretrained_params = np.load(
                artifact_use.get_entry(f.name).download(), allow_pickle=True
            )
            print(f"Loaded pretrained_params from {f.name}")

    assert (
        trainable_params
    ), f"trainable_params not found in artifact {artifact_use.name}"

    try:
        params["params"] = jax.tree.map(
            load_fn, s_model.train_mask, params["params"], trainable_params
        )
    except Exception as e:
        if "lora_rho_w" in str(e):
            lora_rho_default = run.config.get("lora_params", {}).get("lora_rho", 0.0)

            def set_rho_zero(p):
                if isinstance(p, LoraAbstractParams):
                    p.lora_rho_w = lora_rho_default
                return p

            trainable_params = jax.tree.map(
                set_rho_zero,
                trainable_params,
                is_leaf=lambda p: isinstance(p, LoraAbstractParams),
            )
            params["params"] = jax.tree.map(
                load_fn, s_model.train_mask, params["params"], trainable_params
            )
        else:
            print(f"\033[91mError loading trainable parameters: {e}\033[0m")

    bma_weights = False
    if artifact_weights_use:
        for f in artifact_weights_use.files():
            if "weights" in f.name:
                bma_weights = np.load(
                    artifact_weights_use.get_entry(f.name).download(), allow_pickle=True
                )
                print(f"Loaded bma_weights from {f.name}")

    return (
        params,
        bma_weights,
        trainable_params,
        init_trainable_params,
        pretrained_params,
        last_trainable_params,
    )
