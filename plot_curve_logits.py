#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns


def model_sort_key(model_name):
    """Sort key for model names: ALC(5,0) → FLC(5,0) → ALC(5,1) → FLC(5,1) → ..."""
    match = re.search(r"(ALC|FLC)\((\d+),(\d+)\)", model_name)
    if match:
        prefix = match.group(1)  # "ALC" or "FLC"
        n = int(match.group(2))  # N
        m = int(match.group(3))  # m
        # Sort key: (N, m, is_FLC) where is_FLC=0 for ALC, 1 for FLC
        is_flc = 1 if prefix == "FLC" else 0
        return (n, m, is_flc)
    return (0, 0, 1)  # Default for unknown formats


def mean_entropy(probs, eps=1e-10):
    # probs: (T, K)
    ent = -np.sum(probs * np.log(probs + eps), axis=-1)
    return np.mean(ent)


def mutual_information(probs, eps=1e-10):
    # probs: (T, K)
    avg_probs = np.mean(probs, axis=0)
    bma_ent = -np.sum(avg_probs * np.log(avg_probs + eps), axis=-1)
    avg_ent = mean_entropy(probs, eps)
    return bma_ent - avg_ent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--plot-type", choices=["heatmap", "lineplot"], default="heatmap"
    )
    parser.add_argument("--input-dir", default="funct_smooth/logits_backup")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--files", type=str, nargs="+", default=None)
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    parser.add_argument("--indices_ood", type=int, nargs="+", default=None)
    args = parser.parse_args()

    import wandb

    api = wandb.Api()

    if not os.path.exists(args.input_dir):
        print(f"Directory {args.input_dir} does not exist.")
        return

    # FIX 1: Ignore OOD files during the initial dataset compilation
    if args.files:
        files_to_scan = [f for f in args.files if f.endswith(".npz") and not f.endswith("_ood.npz")]
    else:
        files_to_scan = [f for f in os.listdir(args.input_dir) if f.endswith(".npz") and not f.endswith("_ood.npz")]

    if not files_to_scan:
        print("No primary .npz files found (excluding OOD files).")
        return

    datasets = {}
    for f in files_to_scan:
        if "seed_" in f:
            ds_name = f.split("_seed_")[0]
        elif "_QWEN.npz" in f:
            ds_name = f.replace("_QWEN.npz", "")
        else:
            ds_name = f.split(".")[0]  # Fallback

        if args.dataset and ds_name != args.dataset:
            continue
        if ds_name not in datasets:
            datasets[ds_name] = []
        datasets[ds_name].append(f)

    for ds_name, ds_files in datasets.items():
        all_data = []
        for f in ds_files:
            path = os.path.join(args.input_dir, f)
            if not os.path.exists(path):
                print(f"File {path} not found, skipping.")
                continue
            d = dict(np.load(path, allow_pickle=True))

            # Metadata extraction from filename or content
            if "_QWEN.npz" in f:
                run_id = "BASE"
                model_name = "QWEN"
            else:
                parts = f.replace(".npz", "").split("_")
                run_id = parts[-1]
                model_name = parts[-2]

            identifier = f"{model_name}_{run_id}"
            d["run_id"] = run_id
            d["identifier"] = identifier
            d["filename"] = f

            # Load subspace_model and N from W&B if possible
            if "wandb_path" in d:
                try:
                    wp = d["wandb_path"]
                    if isinstance(wp, np.ndarray) and wp.size == 3:
                        run_path = "/".join(wp.tolist())
                    elif isinstance(wp, np.ndarray) and wp.size == 1:
                        run_path = str(wp.flatten()[0])
                    else:
                        run_path = str(wp)

                    print(f"Loading W&B config for {run_path}...")
                    run = api.run(run_path)
                    sub_model = run.config.get("model_params", {}).get(
                        "subspace_model", "unknown"
                    )
                    num_curve_segment = run.config.get("model_params", {}).get(
                        "num_curve_segment", 0
                    )
                    d["N"] = num_curve_segment + 1
                    d["model_name"] = f"{model_name} ({run_id})\n({sub_model})"
                except Exception as e:
                    print(f"Error loading W&B config for {f}: {e}")
                    d["model_name"] = f"{model_name} ({run_id})"
                    d["N"] = 1
            else:
                d["model_name"] = f"{model_name} ({run_id})"
                d["N"] = 1

            if "t_max" not in d:
                d["t_max"] = 1.0  # Default for QWEN or DE

            all_data.append(d)

        # Sort all_data by filename order if args.files provided, else model_sort_key
        if args.files:
            # Match the order in args.files
            file_to_data = {d["filename"]: d for d in all_data}
            all_data = [file_to_data[f] for f in args.files if f in file_to_data]
        else:
            all_data.sort(key=lambda d: model_sort_key(d["model_name"]))

        if not all_data:
            continue

        if args.indices:
            top_k_indices = args.indices
        else:
            # FIX: Find the minimum number of samples across all models
            min_samples = min([d["logits"].shape[1] for d in all_data])

            sample_entropies = []
            for d in all_data:
                logits = d["logits"][:, :min_samples, :]  # (T, samples, classes)
                print(f"Processing {d['filename']} with logits shape {logits.shape}...")
                probs = softmax(logits, axis=-1)
                avg_probs = np.mean(probs, axis=0)  # (samples, classes)
                ent = -np.sum(avg_probs * np.log(avg_probs + 1e-10), axis=-1)
                sample_entropies.append(ent)

            # Max entropy across all models for each sample
            max_ent_per_sample = np.max(np.stack(sample_entropies), axis=0)
            top_k_indices = np.argsort(max_ent_per_sample)[::-1][: args.k]

        # FIX 2: Correct scoping and simplified OOD matching logic
        ood_rows = []
        all_files_in_dir = os.listdir(args.input_dir)
        
        for d in all_data:
            expected_ood = d["filename"].replace(".npz", "_ood.npz")
            if expected_ood in all_files_in_dir:
                ood_data = dict(
                    np.load(
                        os.path.join(args.input_dir, expected_ood), allow_pickle=True
                    )
                )
                ood_rows.append(ood_data)
            else:
                ood_rows.append(None)

        # Check if we have OOD examples (all OOD examples must be shown as rows)
        num_ood_samples = 0
        for row in ood_rows:
            if row is not None:
                num_ood_samples = max(num_ood_samples, row["logits"].shape[1])

        # Determine which OOD sample indices to display
        if args.indices_ood is not None:
            # Use explicitly provided OOD indices
            ood_sample_indices = args.indices_ood
            num_ood_samples = len(ood_sample_indices)
        else:
            # Show all OOD samples (indices 0, 1, 2, ...)
            ood_sample_indices = list(range(num_ood_samples))

        total_rows = len(top_k_indices) + num_ood_samples
        num_models = len(all_data)

        # Adjust width ratios: make QWEN small (10/100) and DE members small (e.g., 0.1 per member)
        width_ratios = []
        for d in all_data:
            if d.get("run_id") == "BASE":
                width_ratios.append(0.1)
            elif "DE(" in d.get("filename", ""):
                # DE files have shape (N_checkpoints, n_valid, n_classes)
                n_checkpoints = d["logits"].shape[0]
                # Assign 0.1 width per member
                width_ratios.append(0.1 * n_checkpoints)
            else:
                width_ratios.append(d["t_max"])

        sum_widths = sum(width_ratios)
        # Use a smaller multiplier (1.2) for the width to keep it compact
        fig = plt.figure(figsize=(1.2 * sum_widths, 2.5 * total_rows))
        gs = fig.add_gridspec(total_rows, num_models, width_ratios=width_ratios)
        axes = np.array(
            [
                [fig.add_subplot(gs[i, j]) for j in range(num_models)]
                for i in range(total_rows)
            ]
        )

        for i in range(total_rows):
            is_ood = i >= len(top_k_indices)
            if is_ood:
                sample_idx = ood_sample_indices[i - len(top_k_indices)]
            else:
                sample_idx = top_k_indices[i]

            for j, d in enumerate(all_data):
                ax = axes[i, j]
                current_data = d
                if is_ood:
                    current_data = ood_rows[j]
                    if (
                        current_data is None
                        or sample_idx >= current_data["logits"].shape[1]
                    ):
                        ax.axis("off")
                        continue

                print(f"Plotting sample {sample_idx} for model {d['model_name']} (OOD: {is_ood}) shape {current_data['logits'].shape}...")
                if "t_max" not in current_data:
                    current_data["t_max"] = 1.0  # Default for QWEN or DE

                logits_sample = current_data["logits"][:, sample_idx, :]  # (T, classes)
                probs_sample = softmax(logits_sample, axis=-1)
                labels = current_data["labels"]

                # Handle potentially multiple correct classes (list or int)
                label_val = labels[sample_idx]
                if isinstance(label_val, (list, np.ndarray)):
                    correct_classes = [int(c) for c in label_val]
                elif label_val is None or (
                    isinstance(label_val, (int, np.integer)) and label_val < 0
                ):
                    correct_classes = []
                else:
                    correct_classes = [int(label_val)]

                mi = mutual_information(probs_sample)
                ent = mean_entropy(probs_sample)

                row_type = "OOD" if is_ood else "Idx"
                title = f"{row_type}: {sample_idx} | MI: {mi:.3f} | Ent: {ent:.3f}"
                if i == 0:
                    title = f"{d['model_name']}\n{title}"
                ax.set_title(title, fontsize=9)

                if args.plot_type == "heatmap":
                    # Use pcolormesh with physical time spacing
                    X = np.linspace(0, current_data["t_max"], probs_sample.shape[0] + 1)
                    Y = np.arange(probs_sample.shape[1] + 1)
                    mesh = ax.pcolormesh(
                        X,
                        Y,
                        probs_sample.T,
                        cmap="viridis",
                        edgecolors="face",
                        shading="auto",
                    )

                    if j == num_models - 1:
                        plt.colorbar(mesh, ax=ax)

                    if d.get("run_id") == "BASE":
                        ax.set_xticks([])
                        # Only keep labels if it is the first column
                        if j > 0:
                            ax.set_yticklabels([])
                        else:
                            ax.set_yticks(np.arange(probs_sample.shape[-1]) + 0.5)
                            ax.set_yticklabels(
                                [chr(65 + c) for c in range(probs_sample.shape[-1])],
                                rotation=0,
                            )
                            for tick in ax.get_yticklabels():
                                class_idx = ord(tick.get_text()) - 65
                                if class_idx in correct_classes:
                                    tick.set_color("red")
                                    tick.set_fontweight("bold")
                    else:
                        ax.set_yticks(np.arange(probs_sample.shape[-1]) + 0.5)
                        ax.set_yticklabels(
                            [chr(65 + c) for c in range(probs_sample.shape[-1])],
                            rotation=0,
                        )
                        for tick in ax.get_yticklabels():
                            class_idx = ord(tick.get_text()) - 65
                            if class_idx in correct_classes:
                                tick.set_color("red")
                                tick.set_fontweight("bold")

                        ax.set_xticks([])  # Hide x-ticks by default
                        if i == total_rows - 1:
                            # X-ticks at 0.5 intervals from 0 to t_max
                            tick_vals = np.arange(0, current_data["t_max"] + 1e-5, 0.5)
                            ax.set_xticks(tick_vals)
                            ax.set_xticklabels([f"{val:.1f}" for val in tick_vals])
                            ax.set_xlim(0, current_data["t_max"])
                else:
                    t_vals = np.linspace(
                        0, current_data["t_max"], probs_sample.shape[0]
                    )
                    colors = plt.cm.Set2(np.linspace(0, 1, probs_sample.shape[-1]))
                    for c in range(probs_sample.shape[-1]):
                        if c in correct_classes:
                            ax.plot(
                                t_vals,
                                probs_sample[:, c],
                                color=colors[c],
                                lw=3,
                                zorder=3,
                            )
                        else:
                            ax.plot(
                                t_vals,
                                probs_sample[:, c],
                                color=colors[c],
                                lw=2,
                                alpha=1.0,
                            )
                    ax.set_ylim(0, 1.05)
                    ax.set_xlim(0, current_data["t_max"])
                    if i == total_rows - 1:
                        tick_vals = np.arange(0, current_data["t_max"] + 1e-5, 0.5)
                        ax.set_xticks(tick_vals)
                        ax.set_xticklabels([f"{val:.1f}" for val in tick_vals])

                if i == total_rows - 1:
                    ax.set_xlabel("t")

        plt.suptitle(
            f"Dataset: {ds_name} | Main rows: {len(top_k_indices)} | OOD rows: {num_ood_samples}",
            fontsize=12,
            y=1.02,
        )
        plt.tight_layout()
        output_path = f"funct_smooth/smootheness_{ds_name}_{args.plot_type}.pdf"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved {output_path}")
        plt.close()


if __name__ == "__main__":
    main()