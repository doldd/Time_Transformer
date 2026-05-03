#!/usr/bin/env python3
import os
import argparse
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns


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
    parser.add_argument("--input-dir", default="funct_smooth/logits_for_plotting")
    parser.add_argument(
        "--load-wandb", action="store_true", help="Load config from W&B if available"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Directory {args.input_dir} does not exist.")
        return

    files = [f for f in os.listdir(args.input_dir) if f.endswith(".npz")]
    if not files:
        print("No .npz files found.")
        return

    datasets = {}
    for f in files:
        ds_name = f.split("_seed_")[0]
        if ds_name not in datasets:
            datasets[ds_name] = []
        datasets[ds_name].append(f)

    if args.load_wandb:
        import wandb

        api = wandb.Api()

    for ds_name, ds_files in datasets.items():
        all_data = []
        for f in ds_files:
            path = os.path.join(args.input_dir, f)
            d = dict(np.load(path))
            # Extract model_name from filename: {ds}_seed_{seed}_{model_name}_{run_id}.npz
            parts = f.replace(".npz", "").split("_")
            model_name = parts[-2]
            d["model_name"] = model_name

            if args.load_wandb and "wandb_path" in d:
                try:
                    run_path = str(d["wandb_path"])
                    print(f"Loading W&B config for {run_path}...")
                    run = api.run(run_path)
                    d["wandb_config"] = run.config
                except Exception as e:
                    print(f"Error loading W&B run {d.get('wandb_path')}: {e}")

            all_data.append(d)

        # probs: (models, T, samples, classes)
        # Compute BMA entropy per sample to select top K
        # BMA entropy = entropy(mean(probs over T))

        sample_entropies = []
        for d in all_data:
            logits = d["logits"]  # (T, samples, classes)
            probs = softmax(logits, axis=-1)
            avg_probs = np.mean(probs, axis=0)  # (samples, classes)
            ent = -np.sum(avg_probs * np.log(avg_probs + 1e-10), axis=-1)
            sample_entropies.append(ent)

        # Max entropy across all models for each sample
        max_ent_per_sample = np.max(np.stack(sample_entropies), axis=0)
        top_k_indices = np.argsort(max_ent_per_sample)[::-1][: args.k]

        num_models = len(all_data)
        fig, axes = plt.subplots(
            args.k, num_models, figsize=(4 * num_models, 3 * args.k), squeeze=False
        )

        t_max = all_data[0]["t_max"]
        n_samples = all_data[0]["n_samples"]
        t_vals = np.linspace(0, t_max, n_samples)

        for i, idx in enumerate(top_k_indices):
            for j, d in enumerate(all_data):
                ax = axes[i, j]
                logits_sample = d["logits"][:, idx, :]  # (T, classes)
                probs_sample = softmax(logits_sample, axis=-1)
                labels = d["labels"]
                correct_class = int(labels[idx])

                mi = mutual_information(probs_sample)
                ent = mean_entropy(probs_sample)

                title = f"Idx: {idx} | MI: {mi:.3f} | Ent: {ent:.3f}"
                if i == 0:
                    title = f"{d['model_name']}\n{title}"
                ax.set_title(title, fontsize=9)

                if args.plot_type == "heatmap":
                    sns.heatmap(
                        probs_sample.T,
                        ax=ax,
                        cmap="viridis",
                        cbar=(j == num_models - 1),
                    )
                    ax.set_yticks(np.arange(probs_sample.shape[-1]) + 0.5)
                    ax.set_yticklabels(
                        [chr(65 + c) for c in range(probs_sample.shape[-1])], rotation=0
                    )
                    for tick in ax.get_yticklabels():
                        if ord(tick.get_text()) - 65 == correct_class:
                            tick.set_color("red")
                            tick.set_fontweight("bold")
                else:
                    colors = plt.cm.tab10(np.linspace(0, 1, probs_sample.shape[-1]))
                    for c in range(probs_sample.shape[-1]):
                        if c == correct_class:
                            ax.plot(
                                t_vals, probs_sample[:, c], color="red", lw=2, zorder=3
                            )
                        else:
                            ax.plot(
                                t_vals, probs_sample[:, c], color=colors[c], alpha=0.6
                            )
                    ax.set_ylim(0, 1.05)

                if i == args.k - 1:
                    ax.set_xlabel("t")

        plt.suptitle(
            f"Dataset: {ds_name} (Top-{args.k} uncertain samples)", fontsize=12, y=1.02
        )
        plt.tight_layout()
        output_path = f"funct_smooth/smootheness_{ds_name}_{args.plot_type}.pdf"
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved {output_path}")
        plt.close()


if __name__ == "__main__":
    main()
