from matplotlib import bezier, colors
import numpy as np
from scipy.integrate import quad
from scipy.special import binom
import matplotlib.pyplot as plt
from jax import numpy as jnp
import jax
from numpyro import distributions as dist
from copy import deepcopy
from flax.core import unfreeze
# from src.models import MLPModel, MLPModelUCI, init_model_tube, init_t_lambda_to_phi, init_model_phi, init_model_line
# from src.subspace_curve import OrthoSpan, pytree_to_matrix
# import arviz as az

__all__ = ['bezier_length', 'lower_bound', 'upper_bound',
           'plot_subspace', 'get_cp_w', 'get_data',
           'init_model', 'load_checkpoint_new', 'load_fn']


# https://jwalton.info/Embed-Publication-Matplotlib-Latex/
tex_fonts = {
    # Use LaTeX to write all text
    # for the align enivironment
    #"text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsfonts}\usepackage{amsmath}\usepackage{amssymb}",
    #"text.usetex": True,  # use inline math for ticks
    #"font.family": "serif",
    "text.usetex": False,  # use inline math for ticks
    "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    "savefig.transparent": True,
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 9,
    "font.size": 9,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
}
plt.rcParams.update(tex_fonts)
textwidth = 430
# columnwidth = 252

# https://jwalton.info/Embed-Publication-Matplotlib-Latex/


def set_size(width=textwidth, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def bezier_length(control_points):
    # Number of dimensions
    dims = control_points.shape[1]
    n = len(control_points) - 1  # Degree of the Bezier curve
    # Calculate factorial terms for the Bernstein basis (n choose i)
    # def factorial(n):
    #     if n == 0:
    #         return 1
    #     else:
    #         return n * factorial(n - 1)

    # def binom(n, k):
    #     return factorial(n) / (factorial(k) * factorial(n - k))

    # Calculate the Bernstein basis polynomials
    def bernstein_basis(i, n, t):
        return binom(n, i) * (t**i) * ((1 - t)**(n - i))
    # Calculate the derivative of the Bezier curve at a given t

    def bezier_derivative(t):
        derivative = np.zeros(dims)
        for i in range(n):
            for k in range(dims):
                derivative[k] += (n * (control_points[i+1][k] - control_points[i][k]) *
                                  bernstein_basis(i, n - 1, t))
        return np.linalg.norm(derivative)

    # Compute the length of the Bezier curve as an integral
    length, error = quad(bezier_derivative, 0, 1)
    return length, error


def lower_bound(control_points):
    return np.linalg.norm(control_points[..., 0, :] - control_points[..., -1, :], axis=-1)


def upper_bound(control_points):
    n_points = control_points.shape[-2]
    pair_wise_length = [np.linalg.norm(
        control_points[..., i, :] - control_points[..., i+1, :], axis=-1) for i in range(n_points-1)]
    return np.sum(pair_wise_length, axis=0)


def plot_subspace(X, Y, Z, t0=None, t_bend=None, t2=None, aspect_equal=True, linear_color=False,
                  interpolate=False, vmax=None, vmin=None, ax=None, label=None, **kwargs):
    """
    :param df: pandas dataframe (must contain xx and yy columns for axis)
    :param df_name: column name of data
    :param t0: two dimensional point P1 of Bézier curve
    :param t_bend: two dimensional point Bend point of Bézier curve
    :param t2: two dimensional point P2 of Bézier curve
    :param aspect_equal: Draw x and y axis in same scale
    :param linear_color: if True use linear coloring else linear color steps according their quantiles
    :param df_samples: pandas dataframe of samples (must contain xx and yy columns for axis) or tuple with (dataframe, xname, yname)
    :param interpolate: if True use counterf to plot else pcolormesh
    :param vmax: max value of coloring
    :param vmin: min value of coloring
    """
    plt.rcParams['axes.grid'] = False
    if ax is None:
        fig = plt.figure(figsize=set_size(), dpi=150)
        ax = fig.gca()
    else:
        fig = ax.get_figure()
    if 'cmap' in kwargs:
        cmap = plt.colormaps[kwargs['cmap']]
        kwargs.pop('cmap')
    else:
        cmap = plt.colormaps['viridis']
    if vmax is None:
        vmax = Z.max()
    if vmin is None:
        vmin = Z.min()
    if linear_color:
        norm = colors.Normalize(clip=True, vmax=vmax, vmin=vmin)
    else:
        # compute boundaries according data quantile (step_size=1/255)
        levels = np.quantile(Z[(Z <= vmax) & (Z >= vmin)],
                             np.arange(0, 1, 1/cmap.N)).tolist()
        levels.append(vmax)
        # df_min = df[df_name].min()
        # shrink = (vmax - vmin) / (df[df_name].max() - df_min)
        # levels = shrink * (np.array(levels) - df_min) + vmin
        norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    # shape = np.sqrt(len(df['xx'].to_numpy())).astype(np.int32)
    # shape = (shape, shape)
    # xx = df['xx'].to_numpy().reshape(shape)
    # yy = df['yy'].to_numpy().reshape(shape)
    # zz = df[df_name].to_numpy().reshape(shape)
    if interpolate:
        if linear_color:
            pcm = ax.contourf(X, Y, Z, levels=cmap.N,
                              cmap=cmap, norm=norm, **kwargs)
        else:
            pcm = ax.contourf(X, Y, Z, levels=levels,
                              cmap=cmap, norm=norm, **kwargs)
    else:
        pcm = ax.pcolormesh(X, Y, Z, cmap=cmap,
                            norm=norm, shading='nearest', **kwargs)
    if aspect_equal:
        ax.set_aspect('equal')
    if linear_color:
        fig.colorbar(pcm, ax=ax, label=label, shrink=0.5,
                     aspect=20*0.5, format="%1.2f")
    else:
        cb = fig.colorbar(pcm, ax=ax, label=label, shrink=0.5,
                          aspect=20*0.5, format="%1.2f", ticks=levels[::50])
        cb.ax.minorticks_off()
    if t0 is not None and t2 is not None:
        ax.scatter(*t0, c='red')
        ax.scatter(*t2, c='red')
        if t_bend is not None:
            # ax.scatter(*t_bend, c='red')
            bz = bezier.BezierSegment(np.vstack([t0, t_bend, t2]))
            curve_t = np.array([bz.point_at_t(i)
                               for i in np.linspace(0, 1, 100)]).T
            ax.plot(*curve_t, c='red', label="Bezier curve", linewidth=1.)
    fig.legend()
    fig.tight_layout()
    return fig


def get_cp_w(run, from_path_optim_sweep=True, last=False):
    # load samples, data and curve params
    if from_path_optim_sweep:
        rrun_art = run.logged_artifacts()
    else:
        rrun_art = run.used_artifacts()
    for art in rrun_art:
        if art.type == "pytree":
            if from_path_optim_sweep:
                filename = f"{run.id}_params"
            else:
                filename = f"{art.logged_by().id}_params"
            if last:
                filename += "_last"
            filename += ".npy" 

            file = art.get_entry(filename).download(
                root="./art_tmp")
            params = jnp.load(file, allow_pickle=True).item()

            # define model
            # get design matrix from curve parameters stroed as pytree
            return params


def get_data(run, from_path_optim_sweep=True, get_file_name=lambda c: f"{c['rng_seed']}_{c['dataset']}_data.npz"):
    if from_path_optim_sweep:
        rrun_art = run.logged_artifacts()
    else:
        rrun_art = run.used_artifacts()
    # load samples, data and curve params
    for art in rrun_art:
        if art.type == "numpy":
            file = art.get_entry(get_file_name(run.config)).download(
                root="./art_tmp")
            with jnp.load(file, allow_pickle=True) as f:
                x = f['x']
                y = f['y']
                x_val = f['x_val']
                y_val = f['y_val']
                x_test = f['x_test']
                y_test = f['y_test']
            return x, y, x_val, y_val, x_test, y_test


def get_samples_from_run(run):
    import arviz as az
    for art in run.logged_artifacts():
        if art.type == "xarray":
            file = art.get_entry(f"{run.id}_samples.nc").download(
                root="./art_tmp")
            samples = az.from_netcdf(file)
            return samples


def get_model_from_run_path(run, from_path_optim_sweep=False):
    from src.models import MLPModel, MLPModelUCI, init_model_tube, init_t_lambda_to_phi, init_model_phi, init_model_line
    from src.subspace_curve import OrthoSpan, pytree_to_matrix

    config = run.config
    k = config['curve_params']['k']
    if (config['dataset'] == 'generate') or (config['dataset'] == 'default'):
        model = MLPModel(**config['curve_params']['model_kwargs'])
    else:
        model = MLPModelUCI(**config['curve_params']['model_kwargs'])

    params = get_cp_w(run, from_path_optim_sweep=from_path_optim_sweep)
    cp_w = pytree_to_matrix(params['params'], k)
    # initialize transformation functions
    t_phi_to_weight = OrthoSpan(cp_w)
    # control points in the varphi space
    cp_phi = t_phi_to_weight.inv(cp_w)
    t_lambda_to_phi, curve, d_bezier = init_t_lambda_to_phi(cp_phi, k,
                                                            epsilon=25,
                                                            tube_scale=0.1)

    tt = jnp.linspace(0., 1., 10_000)
    bezier_grad = jax.vmap(d_bezier)(tt)
    log_normalized_bezier_grad = jnp.log(jnp.trapezoid(
        jnp.linalg.norm(bezier_grad, axis=-1), tt))

    if run.config["sampling"]["space_config"]["space"] == 'lambda':
        sampling_model = init_model_tube(model,
                                         params['params'],
                                         k,
                                         t_lambda_to_phi,
                                         t_phi_to_weight,
                                         prior_scale=config['sampling']['space_config']['prior_scale'],
                                         prior_correct=config['sampling']['space_config']['prior_correct'],
                                         d_bezier=d_bezier,
                                         log_norm_d_bezier=log_normalized_bezier_grad,
                                         dist_scale="homo" if config['curve_params']['optimize_distparams'] else np.exp(
                                             params['dist_params']['log_scale'].item())
                                         )
    elif run.config["sampling"]["space_config"]["space"] == 'varphi':
        sampling_model = init_model_phi(model,
                                        params['params'],
                                        k,
                                        t_phi_to_weight,
                                        prior_scale=config['sampling']['space_config']['prior_scale'],
                                        dist_scale="homo" if config['curve_params']['optimize_distparams'] else np.exp(
                                            params['dist_params']['log_scale'].item())
                                        )
    elif run.config["sampling"]["space_config"]["space"] == 'line':
        sampling_model = init_model_line(model,
                                         params['params'],
                                         k,
                                         t_lambda_to_phi,
                                         t_phi_to_weight,
                                         d_bezier=d_bezier,
                                         log_norm_d_bezier=log_normalized_bezier_grad,
                                         dist_scale="homo" if config['curve_params']['optimize_distparams'] else np.exp(
                                             params['dist_params']['log_scale'].item())
                                         )

    return sampling_model, config, params


def get_params_from_run(run, file_name_without_id="params.npy"):
    for art in run.logged_artifacts():
        if art.type == "pytree":
            file = art.get_entry(f"{run.id}_{file_name_without_id}").download(
                root="./art_tmp")
            params_loaded = jnp.load(file, allow_pickle=True).item()
            return params_loaded


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
        accuracy_in_bin = jnp.where(prop_in_bin > 0, jnp.sum(jnp.where(in_bin == True, accuracies, 0.0))/jnp.sum(in_bin), 0.0)
        # same as jnp.mean(confidences[in_bin]) but jit friendly
        avg_confidence_in_bin = jnp.where(prop_in_bin > 0, jnp.sum(jnp.where(in_bin == True, confidences, 0.0))/jnp.sum(in_bin), 0.0)
        ece += jnp.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece, None

    ece, _ = jax.lax.scan(body_fun, 0.0, (bin_lowers, bin_uppers))
    return ece

def post_pred_performance(logits, labels, weights: bool|jnp.ndarray=False, key_prefix: str="", num_bins: int=15):
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
    if isinstance(weights, bool):
        # treating True as uniform average
        post_predict_logits = jax.nn.logsumexp(logits, axis=0) - jnp.log(logits.shape[0])
    else:
        post_predict_logits = jax.nn.logsumexp(logits, b=weights[:, None, None], axis=0)

    post_acc = jnp.mean(jnp.argmax(post_predict_logits, axis=-1) == labels)
    post_ece = calibration_error(post_predict_logits, labels, num_bins)
    ll = dist.Categorical(logits=post_predict_logits).log_prob(labels).mean()

    return {f'{key_prefix}ll': ll, f'{key_prefix}acc': post_acc, f'{key_prefix}ece': post_ece}


# ---------------------------------------------------------------------------
# Model init / checkpoint helpers (for Qwen LoRA subspace experiments)
# ---------------------------------------------------------------------------

def init_model(config, unique_target_ids):
    """Initialize a LoRAEntropyCategorySubspace model from a wandb run config.

    Args:
        config: dict – wandb run config (may be mutated in-place for model_path).
        unique_target_ids: array of unique target token ids for the classifier head.

    Returns:
        s_model: initialised subspace model.
        params: initialised parameter pytree.
    """
    # Import here to avoid circular / heavy imports at module load time
    from qwen_fineTuning_PtoC import QwenTextClassificationWrapper
    from src.subspace_curve import filter_mask, LoRAEntropyCategorySubspace

    if config['net_kwargs']['model_path'].startswith('ddold/'):
        import wandb
        api = wandb.Api()
        model_artifact = api.artifact(config['net_kwargs']['model_path'], type='model')
        config['net_kwargs']['model_path'] = model_artifact.download()

    model = QwenTextClassificationWrapper(**config['net_kwargs'], target_token_ids=unique_target_ids)
    params = unfreeze(model.init(None, None))

    curve_mask = jax.tree.map(lambda x: False, params['params'])
    if config['curve_params']['k'] > 0:
        for f in config['curve_params']['filter_masks']:
            curve_mask = filter_mask(curve_mask, *f['keys'], as_curve=True, BoolschenOperator=eval(f['op']))

    optimizer_conf = config['curve_params']['optimizer_conf']
    lora_params = config.get('lora_params', None)
    filter_masks = config['curve_params']['filter_masks']
    if "freeze_other_params" in optimizer_conf and optimizer_conf["freeze_other_params"]:
        train_mask = jax.tree.map(lambda x: False, params['params'])
        for f in filter_masks:
            train_mask = filter_mask(train_mask, *f['keys'], as_curve=True, BoolschenOperator=eval(f['op']))
        if lora_params and lora_params['use_lora']:
            for f in lora_params['filter_masks']:
                train_mask = filter_mask(train_mask, *f['keys'], as_curve=True, BoolschenOperator=eval(f['op']))
    else:
        train_mask = jax.tree.map(lambda x: True, params['params'])

    lora_mask = jax.tree.map(lambda x: False, params['params'])
    if lora_params and lora_params['use_lora']:
        for f in lora_params['filter_masks']:
            lora_mask = filter_mask(lora_mask, *f['keys'], as_curve=np.array(f['dims']), BoolschenOperator=eval(f['op']))

    curve_pps = deepcopy(config['curve_params'])
    curve_pps.pop('cp_fix')
    s_model = LoRAEntropyCategorySubspace(
        lora_mask=lora_mask,
        r=lora_params['r'],
        lora_dtype=lora_params['lora_dtype'],
        lora_alpha=lora_params['lora_alpha'],
        lora_mode=lora_params.get('lora_mode', 'A(t)B'),
        lora_rho=lora_params.get('lora_rho', 0.0),
        lora_rho_s=lora_params.get('lora_rho_s', 0.0),
        model=model,
        **curve_pps,
    )

    init_key = jax.random.PRNGKey(config.get('rng_seed', 1))
    params = s_model.init_params_from_point(
        init_key, params, curve_mask,
        noise_fn=lambda p, key: p + config['curve_params']['jitter_multiplier'] * jax.random.normal(key, p.shape, dtype=p.dtype),
    )

    if lora_params and lora_params['use_lora']:
        train_mask = s_model.update_mask_for_lora(train_mask, params['params'], b_off=False, w0_off=True)
    s_model.set_train_mask(train_mask)
    return s_model, params


def load_fn(m, p, pl):
    """Merge a loaded parameter into the current pytree leaf based on the train mask."""
    if m:
        assert p.shape == pl.shape, f"Shape mismatch: current {p.shape} vs loaded {pl.shape}"
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
        if 'trainable_params' in f.name:
            if 'init' in f.name:
                init_trainable_params = np.load(artifact_use.get_entry(f.name).download(), allow_pickle=True).item()
                print(f"Loaded init_trainable_params from {f.name}")
            elif 'last' in f.name:
                last_trainable_params = np.load(artifact_use.get_entry(f.name).download(), allow_pickle=True).item()
                print(f"Loaded last_trainable_params from {f.name}")
            else:
                trainable_params = np.load(artifact_use.get_entry(f.name).download(), allow_pickle=True).item()
                print(f"Loaded trainable_params from {f.name}")
                assert len(set(trainable_params.keys()) - set(params['params'].keys())) == 0, \
                    "Loaded trainable_params do not match model params"
        elif 'pretrained_params' in f.name:
            pretrained_params = np.load(artifact_use.get_entry(f.name).download(), allow_pickle=True)
            print(f"Loaded pretrained_params from {f.name}")

    assert trainable_params, f"trainable_params not found in artifact {artifact_use.name}"

    try:
        params['params'] = jax.tree.map(load_fn, s_model.train_mask, params['params'], trainable_params)
    except Exception as e:
        if "lora_rho_w" in str(e):
            lora_rho_default = run.config.get('lora_params', {}).get('lora_rho', 0.0)

            def set_rho_zero(p):
                if isinstance(p, LoraAbstractParams):
                    p.lora_rho_w = lora_rho_default
                return p

            trainable_params = jax.tree.map(set_rho_zero, trainable_params, is_leaf=lambda p: isinstance(p, LoraAbstractParams))
            params['params'] = jax.tree.map(load_fn, s_model.train_mask, params['params'], trainable_params)
        else:
            print(f"\033[91mError loading trainable parameters: {e}\033[0m")

    bma_weights = False
    if artifact_weights_use:
        for f in artifact_weights_use.files():
            if 'weights' in f.name:
                bma_weights = np.load(artifact_weights_use.get_entry(f.name).download(), allow_pickle=True)
                print(f"Loaded bma_weights from {f.name}")

    return params, bma_weights, trainable_params, init_trainable_params, pretrained_params, last_trainable_params
