import os
import logging
import wandb
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_figsize(columnwidth=234.8, wf=1.0, hf=(np.sqrt(5) - 1.0) / 2.0):
    """
    Returns [fig_width, fig_height] in inches for matplotlib.
    Default columnwidth is 234.8pt (standard for many IEEE/ACM/ACL journals).
    """
    fig_width_pt = columnwidth * wf
    inches_per_pt = 1.0 / 72.27
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width * hf
    return [fig_width, fig_height]


# Configure LaTeX fonts and figure size
figsize = get_figsize(wf=1.0)
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "Nimbus Sans",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": figsize,
}
plt.rcParams.update(tex_fonts)


def load_and_compute_metrics(
    project_path="ddold/qwen_PtoC_seg", target_dataset="winogrande_s"
):
    """
    Fetches runs from W&B, downloads logit artifacts for the target dataset,
    and computes metrics along the continuous curve parameter t.
    """
    api = wandb.Api()
    logger.info("Fetching runs from %s for dataset %s...", project_path, target_dataset)

    # The dataset name appears to be in config['data']['dataset_path'] or similar
    # So we'll iterate through all runs and filter locally
    runs = api.runs(project_path)

    filtered_runs = []
    for r in runs:
        data_cfg = r.config.get("data", {})
        if isinstance(data_cfg, dict):
            ds_path = data_cfg.get("dataset_path", "")
            if target_dataset in ds_path:
                # filter out runs that don't have the expected artifacts (e.g., logits)
                # Exclude runs that have a "hide" tag
                if hasattr(r, "tags") and r.tags:
                    if any("hide" in str(tag).lower() for tag in r.tags):
                        logger.warning(
                            "\033[91mSkipping r %s (tags contain 'hide': %s)\033[0m",
                            r.name,
                            r.tags,
                        )
                        continue
                # exclude runs which has no sweep
                if hasattr(r, "sweep") and r.sweep == None:
                    logger.warning(
                        "\033[91mSkipping r %s (has sweep: %s)\033[0m", r.name, r.sweep
                    )
                    continue

                filtered_runs.append(r)

    logger.info(
        "Found %d runs matching dataset '%s'.", len(filtered_runs), target_dataset
    )

    data_dict = {}
    download_dir = Path("tmp_artifacts")
    download_dir.mkdir(exist_ok=True)

    for run in filtered_runs:

        # Create a group label based on the curve configuration
        cfg = run.config.get("model_params", {})

        cp_fix = cfg.get("cp_fix", [])
        seg_deg = cfg.get("curve_segment", 1)  # Default to 1
        if seg_deg == False:
            seg_deg = len(cp_fix) - 1

        if isinstance(cp_fix, list) and len(cp_fix) > 1:
            n_seg = (len(cp_fix) - 1) // seg_deg
        else:
            n_seg = cfg.get("n_seg", "None")

        cp_fix_str = str(cp_fix)

        # Skip if not a curve run
        if cp_fix_str == "None" and n_seg == "None":
            continue

        # --- REFINED GROUPING LOGIC ---
        group_label = f"Fix={cp_fix_str}, Segs={n_seg}, Deg={seg_deg}"
        
        category_label = "Other"
        if cp_fix is not None:
            if all(val == 1 for val in cp_fix):
                category_label = "Linear"
            elif all(val == 0 for val in cp_fix):
                category_label = "Open"
            else:
                category_label = "Pretrained"

        if group_label not in data_dict:
            data_dict[group_label] = {"train": [], "val": [], "test": [], "category": category_label}

        logger.info("Processing run: %s (%s, cat: %s)", run.name, group_label, category_label)

        # Find the logits artifact
        artifacts = list(run.logged_artifacts())
        logits_artifact = next(
            (a for a in artifacts if "logits" in a.name and a.type == "npz"), None
        )

        if not logits_artifact:
            logger.warning("No logits artifact found for run %s", run.name)
            continue

        # Download files
        art_dir = download_dir / run.id
        logits_artifact.download(root=str(art_dir))

        for split in ["train", "val", "test"]:
            file_path = list(art_dir.glob(f"*{split}_logits.npz"))
            if not file_path:
                continue

            npz = np.load(file_path[0])
            logits = npz["logits"]  # shape: (n_t, n_data, n_classes)
            labels = npz["labels"]  # shape: (n_data,)

            n_t = logits.shape[0]
            max_t = float(n_seg) if n_seg != "None" and n_seg is not None else 1.0
            t_space = np.linspace(0, max_t, n_t)

            # Compute log probabilities: log(softmax(logits))
            log_probs = logits - logsumexp(logits, axis=-1, keepdims=True)

            # Vectorized metrics along t
            # true_log_probs: shape (n_t, n_data) -> mean over n_data -> shape (n_t,)
            true_log_probs = log_probs[:, np.arange(len(labels)), labels]
            ll_list = np.mean(true_log_probs, axis=1)

            # Accuracy: shape (n_t, n_data) -> mean over n_data -> shape (n_t,)
            preds = np.argmax(logits, axis=-1)
            acc_list = np.mean(preds == labels[np.newaxis, :], axis=1)

            data_dict[group_label][split].append(
                {"t": t_space, "acc": acc_list, "ll": ll_list, "category": category_label}
            )

    return data_dict


def plot_curve_metrics(data_dict, save_path="curve_metrics.png"):
    """
    Plots Train (left) and Test/Val (right) metrics across t.
    Creates a new figure row for each group label.
    Plots individual runs thin with alpha=0.25, and averages thick.
    Uses distinct colors for Accuracy and Log-Likelihood, and for different splits.
    """
    n_groups = len(data_dict)
    if n_groups == 0:
        logger.warning("No data to plot.")
        return

    # Create a subplot grid with n_groups rows and 2 columns
    fig, axes = plt.subplots(n_groups, 2, figsize=(14, 5 * n_groups), squeeze=False)

    for i, (group_label, splits) in enumerate(data_dict.items()):
        ax_train = axes[i, 0]
        ax_testval = axes[i, 1]

        # Twin axes for Log-Likelihood
        ax_train_ll = ax_train.twinx()
        ax_testval_ll = ax_testval.twinx()

        def plot_split(
            split_name, ax, ax_ll, color_acc, color_ll, line_style, label_suffix
        ):
            if not splits[split_name]:
                return
            t_space = splits[split_name][0]["t"]

            # 1. Plot individual runs
            for run_data in splits[split_name]:
                ax.plot(
                    t_space,
                    run_data["acc"],
                    color=color_acc,
                    linestyle=line_style,
                    alpha=0.25,
                    linewidth=1,
                )
                ax_ll.plot(
                    t_space,
                    run_data["ll"],
                    color=color_ll,
                    linestyle=line_style,
                    alpha=0.25,
                    linewidth=1,
                )

            # 2. Plot the average with thick lines
            avg_acc = np.mean([s["acc"] for s in splits[split_name]], axis=0)
            avg_ll = np.mean([s["ll"] for s in splits[split_name]], axis=0)

            ax.plot(
                t_space,
                avg_acc,
                color=color_acc,
                linestyle=line_style,
                linewidth=3.0,
                label=f"Acc {label_suffix}",
            )
            ax_ll.plot(
                t_space,
                avg_ll,
                color=color_ll,
                linestyle=line_style,
                linewidth=3.0,
                alpha=0.8,
                label=f"LL {label_suffix}",
            )

        cmap = plt.cm.Paired
        # --- TRAIN PLOT ---
        # Acc: Light Blue (0), LL: Dark Blue (1)
        plot_split("train", ax_train, ax_train_ll, cmap(0), cmap(1), "-", "(Train)")

        # --- TEST / VAL PLOT ---
        # Test - Acc: Light Green (2), LL: Dark Green (3)
        plot_split("test", ax_testval, ax_testval_ll, cmap(2), cmap(3), "-", "(Test)")
        # Val - Acc: Light Orange (6), LL: Dark Orange (7)
        plot_split("val", ax_testval, ax_testval_ll, cmap(6), cmap(7), "--", "(Val)")

        # --- FORMATTING ---
        ax_train.set_title(f"Train - {group_label}")
        ax_train.set_xlabel("t (Curve Position)")
        ax_train.set_ylabel("Accuracy")
        ax_train_ll.set_ylabel("Log-Likelihood")

        # Combine legends for Train
        lines_1, labels_1 = ax_train.get_legend_handles_labels()
        lines_2, labels_2 = ax_train_ll.get_legend_handles_labels()
        if lines_1 or lines_2:
            ax_train.legend(
                lines_1 + lines_2,
                labels_1 + labels_2,
                loc="lower center",
                fontsize="small",
            )

        ax_testval.set_title(f"Test/Val - {group_label}")
        ax_testval.set_xlabel("t (Curve Position)")
        ax_testval.set_ylabel("Accuracy")
        ax_testval_ll.set_ylabel("Log-Likelihood")

        # Combine legends for Test/Val
        lines_3, labels_3 = ax_testval.get_legend_handles_labels()
        lines_4, labels_4 = ax_testval_ll.get_legend_handles_labels()
        if lines_3 or lines_4:
            ax_testval.legend(
                lines_3 + lines_4,
                labels_3 + labels_4,
                loc="lower center",
                fontsize="small",
            )

    plt.tight_layout()
    # plt.savefig(save_path + ".png", dpi=300)
    plt.savefig(save_path + ".jpeg",
        format='jpeg',            # Use JPEG instead of PNG
        dpi=60,                   # Lower resolution (80-100 is usually plenty)
        bbox_inches='tight',      # Remove empty whitespace
        pil_kwargs={'quality': 50} # Tell PIL to compress the JPEG (0-100)
    )
    plt.savefig(save_path + ".pdf", dpi=300)
    logger.info("Plot saved to %s", save_path)
    plt.show()


def plot_mean_std_shaded(data_dict, save_path="curve_metrics_mean_std.png"):
    """
    Plots Train (left) and Test (right) metrics across t.
    Re-aggregates groups into Linear, Open, and Pretrained for this plot.
    Plots mean lines thick and shaded std areas.
    """
    # 1. Re-aggregate data by category
    category_dict = {}
    for group_label, splits in data_dict.items():
        cat = splits.get("category", "Other")
        if cat not in category_dict:
            category_dict[cat] = {"train": [], "test": []}
        
        category_dict[cat]["train"].extend(splits["train"])
        category_dict[cat]["test"].extend(splits["test"])

    n_cats = len(category_dict)
    if n_cats == 0:
        logger.warning("No data to plot.")
        return

    # Create a subplot grid with 1 row and 2 columns
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(16, 6))

    # Twin axes for Log-Likelihood
    ax_train_ll = ax_train.twinx()
    ax_test_ll = ax_test.twinx()

    # Use a categorical color palette for the categories
    colors = plt.cm.tab10(np.linspace(0, 1, n_cats))

    for i, (cat_label, splits) in enumerate(category_dict.items()):
        color = colors[i]

        def plot_split_aggregated(split_name, ax, ax_ll, label_prefix):
            if not splits[split_name]:
                return
            
            # 1. Collect all unique t values from all runs in this category
            all_t = np.unique(np.concatenate([run_data["t"] for run_data in splits[split_name]]))
            
            # 2. Interpolate each run onto this global t_space and track where it exists
            acc_interp = []
            ll_interp = []
            mask = [] # To track which runs contribute to which t-point
            
            for run_data in splits[split_name]:
                t_run = run_data["t"]
                acc_run = run_data["acc"]
                ll_run = run_data["ll"]
                
                # Interpolate run data to all_t
                # Only keep values within the original range of this run
                acc_val = np.interp(all_t, t_run, acc_run, left=np.nan, right=np.nan)
                ll_val = np.interp(all_t, t_run, ll_run, left=np.nan, right=np.nan)
                
                acc_interp.append(acc_val)
                ll_interp.append(ll_val)
                mask.append(np.isfinite(acc_val))
            
            # 3. Calculate mean and std ignoring NaNs (where runs don't reach)
            acc_matrix = np.array(acc_interp)
            ll_matrix = np.array(ll_interp)
            
            avg_acc = np.nanmean(acc_matrix, axis=0)
            std_acc = np.nanstd(acc_matrix, axis=0)
            avg_ll = np.nanmean(ll_matrix, axis=0)
            std_ll = np.nanstd(ll_matrix, axis=0)

            # Accuracy (Solid line)
            ax.plot(
                all_t,
                avg_acc,
                color=color,
                linestyle="-",
                linewidth=2.5,
                label=f"{label_prefix} (Acc)",
            )
            ax.fill_between(
                all_t,
                avg_acc - std_acc,
                avg_acc + std_acc,
                color=color,
                alpha=0.15,
            )

            # Log-Likelihood (Dashed line)
            ax_ll.plot(
                all_t,
                avg_ll,
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label=f"{label_prefix} (LL)",
            )
            ax_ll.fill_between(
                all_t,
                avg_ll - std_ll,
                avg_ll + std_ll,
                color=color,
                alpha=0.1,
            )

        # Plot Train and Test only
        plot_split_aggregated("train", ax_train, ax_train_ll, cat_label)
        plot_split_aggregated("test", ax_test, ax_test_ll, cat_label)

    # --- FORMATTING ---
    for ax, ax_ll, title in zip(
        [ax_train, ax_test],
        [ax_train_ll, ax_test_ll],
        ["Train Metrics", "Test Metrics"],
    ):
        ax.set_title(title)
        ax.set_xlabel("t (Curve Position)")
        ax.set_ylabel("Accuracy (Solid)")
        ax_ll.set_ylabel("Log-Likelihood (Dashed)")
        ax.grid(True, linestyle=":", alpha=0.5)

    # Unitary Legend for both subplots
    lines, labels = ax_train.get_legend_handles_labels()
    lines_ll, labels_ll = ax_train_ll.get_legend_handles_labels()
    fig.legend(
        lines + lines_ll,
        labels + labels_ll,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=3,
        fontsize="x-small",
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    # Save as JPEG for compression as in the original function
    plt.savefig(
        save_path + ".jpeg",
        format="jpeg",
        dpi=100,
        bbox_inches="tight",
        pil_kwargs={"quality": 60},
    )
    plt.savefig(save_path + ".pdf", dpi=300)
    logger.info("Mean/Std plot saved to %s", save_path)
    plt.show()


def plot_only_means(data_dict, save_path="curve_metrics_means_only.png"):
    """
    Plots a single row with Train (left) and Test (right) containing only the 
    mean Log-Likelihood lines.
    Hue: Category (Linear, Open, Pretrained)
    Style: Group (Specific Fix Pattern)
    """
    n_groups = len(data_dict)
    if n_groups == 0:
        return

    # Using the LaTeX-compatible figsize from earlier
    # Each subplot width: columnwidth*wf*0.5 (roughly)
    # Total width is figsize[0]*2 as we want a single row of two subplots
    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    # Define categories and their colors
    unique_cats = sorted(list(set(splits.get("category", "Other") for splits in data_dict.values())))
    cat_colors = dict(zip(unique_cats, plt.cm.Set1(np.linspace(0, 1, len(unique_cats)))))
    
    # Define group styles (line styles)
    line_styles = ["-", "--", "-.", ":"]
    group_labels = sorted(data_dict.keys())
    group_to_style = {label: line_styles[i % len(line_styles)] for i, label in enumerate(group_labels)}

    for i, (group_label, splits) in enumerate(data_dict.items()):
        cat_label = splits.get("category", "Other")
        color = cat_colors[cat_label]
        ls = group_to_style[group_label]

        def plot_mean_line(split_name, ax):
            if not splits[split_name]:
                return
            
            all_t = np.unique(np.concatenate([run_data["t"] for run_data in splits[split_name]]))
            ll_interp = []
            
            for run_data in splits[split_name]:
                t_run = run_data["t"]
                ll_val = np.interp(all_t, t_run, run_data["ll"], left=np.nan, right=np.nan)
                ll_interp.append(ll_val)
            
            avg_ll = np.nanmean(np.array(ll_interp), axis=0)

            ax.plot(
                all_t, 
                avg_ll, 
                color=color, 
                linestyle=ls, 
                linewidth=2, 
                label=f"[{cat_label}] {group_label}"
            )

        plot_mean_line("train", ax_train)
        plot_mean_line("test", ax_test)

    for ax, title in zip([ax_train, ax_test], ["Train Log-Likelihood", "Test Log-Likelihood"]):
        ax.set_title(title)
        ax.set_xlabel("t")
        ax.set_ylabel("Log-Likelihood")
        ax.grid(True, linestyle=":", alpha=0.5)
        
    # Single Legend for both subplots
    handles, labels = ax_train.get_legend_handles_labels()
    fig.legend(
        handles, 
        labels, 
        loc="upper center", 
        bbox_to_anchor=(0.5, 0.05), 
        ncol=2, 
        fontsize="xx-small"
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path + ".jpeg", format="jpeg", dpi=100, bbox_inches="tight", pil_kwargs={"quality": 70})
    plt.savefig(save_path + ".pdf", dpi=300)
    logger.info("Means-only (LL) plot saved to %s", save_path)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot curve metrics from W&B.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="winogrande_s",
        help="Target dataset name (e.g., winogrande_s)",
    )
    parser.add_argument(
        "--project", type=str, default="ddold/qwen_PtoC_seg", help="W&B project path"
    )
    args = parser.parse_args()

    # 1. Fetch and compute
    data = load_and_compute_metrics(args.project, args.dataset)

    # 2. Plot
    if data:
        plot_curve_metrics(data, save_path=f"result/{args.dataset}_along_curve")
        plot_mean_std_shaded(data, save_path=f"result/{args.dataset}_mean_std")
        plot_only_means(data, save_path=f"result/{args.dataset}_means_only")
    else:
        logger.warning("No valid curve data found to plot.")
