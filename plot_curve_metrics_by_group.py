import os
import logging
import wandb
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KNOWN_DATASETS = [
    "winogrande_s",
    "winogrande_m",
    "ARC-Challenge",
    "ARC-Easy",
    "obqa",
    "boolq",
]


def get_dataset_name(ds_path):
    for kd in KNOWN_DATASETS:
        if kd in ds_path:
            return kd
    return Path(ds_path).stem


def load_and_compute_metrics_by_group(project_path="ddold/qwen_PtoC_seg"):
    """
    Fetches runs from W&B, groups them by their curve configuration and then by dataset.
    Downloads logit artifacts and computes metrics.
    """
    api = wandb.Api()
    logger.info("Fetching runs from %s...", project_path)

    runs = api.runs(project_path)

    # data_dict[group_label][dataset_name] = {"train": [], "val": [], "test": []}
    data_dict = {}
    download_dir = Path("tmp_artifacts")
    download_dir.mkdir(exist_ok=True)

    valid_runs = []
    for r in runs:
        # Exclude runs that have a "hide" tag
        if hasattr(r, "tags") and r.tags:
            if any("hide" in str(tag).lower() for tag in r.tags):
                continue
        # Exclude runs which have no sweep
        if hasattr(r, "sweep") and r.sweep is None:
            continue

        valid_runs.append(r)

    logger.info("Found %d valid runs in project.", len(valid_runs))

    for run in valid_runs:
        data_cfg = run.config.get("data", {})
        if not isinstance(data_cfg, dict):
            continue

        ds_path = data_cfg.get("dataset_path", "")
        if not ds_path:
            continue

        dataset_name = get_dataset_name(ds_path)

        # Create a group label based on the curve configuration
        cfg = run.config.get("model_params", {})

        cp_fix = cfg.get("cp_fix", [])
        seg_deg = cfg.get("curve_segment", False)  # Default to 1
        if seg_deg == False:
            seg_deg = len(cp_fix) - 1

        if isinstance(cp_fix, list) and len(cp_fix) > 1:
            n_seg = (len(cp_fix) - 1) // seg_deg
        else:
            raise ValueError(f"Unexpected cp_fix format in run {run.name}: {cp_fix}")

        cp_fix_str = str(cp_fix)

        # Skip if not a curve run
        if cp_fix_str == "None" and n_seg == "None":
            continue

        group_label = f"Fix={cp_fix_str}, Segs={n_seg}, Deg={seg_deg}"

        if group_label not in data_dict:
            data_dict[group_label] = {}
        if dataset_name not in data_dict[group_label]:
            data_dict[group_label][dataset_name] = {"train": [], "val": [], "test": []}

        logger.info(
            "Processing run: %s | Group: %s | Dataset: %s",
            run.name,
            group_label,
            dataset_name,
        )

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
            true_log_probs = log_probs[:, np.arange(len(labels)), labels]
            ll_list = np.mean(true_log_probs, axis=1)

            # Accuracy
            preds = np.argmax(logits, axis=-1)
            acc_list = np.mean(preds == labels[np.newaxis, :], axis=1)

            data_dict[group_label][dataset_name][split].append(
                {"t": t_space, "acc": acc_list, "ll": ll_list}
            )

    return data_dict


def sanitize_filename(name):
    """Removes invalid characters for filenames"""
    clean_name = re.sub(r"[^\w\-]", "_", name)
    # Remove multiple underscores
    clean_name = re.sub(r"_+", "_", clean_name).strip("_")
    return clean_name


def plot_group_metrics(data_dict, output_dir="result/groups"):
    """
    Creates one figure per group_label.
    Inside the figure, each dataset gets its own row.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not data_dict:
        logger.warning("No data to plot.")
        return

    for group_label, datasets_data in data_dict.items():
        datasets = list(datasets_data.keys())
        n_rows = len(datasets)

        if n_rows == 0:
            continue

        logger.info(
            "Plotting figure for group: %s with %d datasets", group_label, n_rows
        )

        # Create a subplot grid with n_rows rows and 2 columns
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows), squeeze=False)
        fig.suptitle(f"Group: {group_label}", fontsize=16, y=1.02)

        for i, dataset_name in enumerate(datasets):
            splits = datasets_data[dataset_name]
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
            plot_split(
                "test", ax_testval, ax_testval_ll, cmap(2), cmap(3), "-", "(Test)"
            )
            # Val - Acc: Light Orange (6), LL: Dark Orange (7)
            plot_split(
                "val", ax_testval, ax_testval_ll, cmap(6), cmap(7), "--", "(Val)"
            )

            # --- FORMATTING ---
            ax_train.set_title(f"{dataset_name} | Train")
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

            ax_testval.set_title(f"{dataset_name} | Test/Val")
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

        # Save figure
        clean_group_name = sanitize_filename(group_label)
        save_base = out_path / clean_group_name

        plt.savefig(
            f"{save_base}.jpeg",
            format="jpeg",
            dpi=60,
            bbox_inches="tight",
            pil_kwargs={"quality": 50},
        )
        plt.savefig(f"{save_base}.pdf", dpi=300)
        logger.info("Saved plots for group %s to %s.jpeg/.pdf", group_label, save_base)
        plt.close(fig)  # Close the figure to free memory


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot curve metrics by cp_fix groups.")
    parser.add_argument(
        "--project", type=str, default="ddold/qwen_PtoC_seg", help="W&B project path"
    )
    args = parser.parse_args()

    # 1. Fetch and compute
    data = load_and_compute_metrics_by_group(args.project)

    # 2. Plot
    plot_group_metrics(data, output_dir="result/groups")
