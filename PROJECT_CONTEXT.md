# Project Context: PtoC (Point-to-Curve) LoRA Fine-Tuning

This project implements and evaluates the **Point-to-Curve (PtoC)** framework for fine-tuning Large Language Models (LLMs), specifically the Qwen2.5 7B model. The method discovers continuous, low-loss valleys connecting independent weight optima in the LoRA parameter space.

## Core Concepts

- **LoRA Subspaces:** Instead of navigating the full parameter space, we optimize continuous paths strictly within the Low-Rank Adaptation (LoRA) modules.
- **Segmented Bézier Curves:** The path is parameterized as a piecewise (segmented) Bézier curve. Each segment has a degree ($m$ or `SegDeg`) and the total path has a number of segments ($\nseg$ or `N_Seg`).
- **Anchoring ($C_{\text{fix}}$):** A binary mask determines which control points are fixed (anchors from pre-trained point-estimate models) and which are trainable (intermediate points optimized to find the low-loss path).
- **Bayesian Model Averaging (BMA):** At inference, we sample along the continuous curve to perform zero-overhead ensembling, improving uncertainty calibration and generalization.

## Mathematical Notation Summary

- **Collective Parameter Space ($\thetalora$):** The global set of all LoRA parameters across all layers.
- **Global Curve ($\theta(t)$):** Defined as $\theta(t) = \sum_{i=0}^{m} b_i(t) \Phi_i$, where $\Phi_i$ are global control points.
- **Segmented Mapping:** For the $k$-th segment, the local curve is defined by control points $(\Phi_{(k-1)m}, \dots, \Phi_{km})$ with local coordinate $\tau \in [0, 1]$.
- **Continuity:** The path is $C^0$ continuous at anchor points, allowing it to navigate non-convex landscapes more effectively than a single high-degree curve.

## Implementation Details

- **Code-to-Paper Mapping:**
    - `cp_fix` $\rightarrow$ $C_{\text{fix}}$ (Binary mask for control points).
    - `N_Seg` $\rightarrow$ $\nseg$ (Number of segments).
    - `SegDeg` $\rightarrow$ $m$ (Degree of each Bézier segment) ("curve_segment" hyper parameter if not False. If false degree is len(cp_fix)-1).
    - `lora_mode` $\rightarrow$ Specifies whether $A$, $B$, or a scaling matrix $S$ is time-varying.
- **Flatness:** We incorporate noise-based regularization inspired by **Flat-LoRA (Liu et al., 2025)** to ensure the path remains in robust, flat regions of the loss landscape.

## Current Manuscript Status (ICML 2026)

- **Main File:** `main.tex` in the Overleaf project "Default Project".
- **Bib File:** `example_paper.bib` containing key citations (Liu et al. 2025, Hu et al. 2022, Dold et al. 2025).
- **Key Results:** Segmented curves (e.g., $\nseg=6, m=2$) outperform standard Deep Ensembles on benchmarks like Winogrande_s.

## Typical Developer Workflow

1.  **Understand:** Read `src/subspace_curve.py` for curve math and `qwen_fineTuning_PtoC_fixCps.py` for training logic.
2.  **Verify:** Run `pytest` for unit tests on subspace math.
3.  **Sync:** Ensure Overleaf `main.tex` reflects the latest experimental findings from W&B sweeps.
