# Related Work and Project Context: LoraSubspace

[TOC]

## Current Research Focus
This project explores **LoraSubspace**, a method for Bayesian Low-Rank Adaptation (LoRA). The central hypothesis is that adapting in a low-rank subspace while considering the full parameter space's geometry leads to better generalization and uncertainty quantification.

### Implementation Note: `lora_rho`
Our implementation includes a hyperparameter `lora_rho`. When `lora_rho > 0`, it effectively implements the noise injection strategy suggested by **Flat-LoRA (Li et al., 2025)**. 

**Mathematical Definition:**
$$ W = W_0 + AB + \delta W $$
where $\delta W = \text{lora\_rho} \times \mathcal{N}(0, I)$ for an $n \times m$ matrix $W$. 
This perturbation on top of the full matrix $W_0$ in the LoRA adapter aims to identify adaptations situated in flat regions of the full parameter space, improving both in-domain and out-of-domain generalization.

---

## Technical Context: Curve Training & `cp_fix`
The repository implements Bézier-curve parameterized neural networks, where the `cp_fix` configuration controls the pretraining and fixing of control points:

- **Izmailov Approach (`cp_fix = [1, 0, 1]`):** The first and last control points are pretrained and fixed, and the curve (middle points) is subsequently optimized.
- **Open Curve Segment (`cp_fix = [0, 0, 0]`):** Represents a quadratic Bézier curve where no parameters are pretrained/fixed. This "open segment" approach is described in **Dold et al. (2024)**.
- **Improved Sampling (Dold et al., 2025):** Focuses on optimizing the curve and improving sampling performance via natural priors, though the sampling optimization itself is not the primary focus of the current `lora_rho` experiments.

---

## Literature Summary (Zotero: `MM7LCWGM`)

### 1. Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape (Li et al., 2025)
- **Key Insight:** Solutions that appear flat in the LoRA subspace can be sharp in the full parameter space, hurting generalization.
- **Method:** Uses a Bayesian expectation loss and random weight perturbations (RWP) to find flatter regions without the overhead of SAM.
- **Mathematical Implementation (Refined):**
  The random perturbation $\epsilon_W$ is sampled as:
  $$ (\epsilon_W)_{i,j} \sim \mathcal{N}\left(0, \frac{\sigma^2}{n} \|W'_{i,:}\|^2_2\right) $$
  where $n$ is the input dimension, $W'$ is the merged weight matrix, and $\sigma$ is the perturbation strength (typically 0.05 or 0.1).
- **Key Advantage:** By scaling the noise by the weight magnitude and $1/n$, the variance increase in the forward pass is independent of the layer width. Memory efficiency is achieved by storing only the random seed and row-wise filter norms.
- **Relevance:** Directly validates our use of noise injection for robustness via `lora_rho`.

### 2. Bayesian Semi-structured Subspace Inference (Dold et al., 2024)
- **Relevance to Project:** Describes curve training for **semi-structured models** using **quadratic Bézier curves**.
- **Technical Detail:** The Bézier curve $B_\Lambda(t)$ is defined by control points $p_0, \dots, p_k$ as:
  $$ B_\Lambda(t) = \sum_{l=0}^k \binom{k}{l} (1-t)^{k-l} t^l p_l $$
- **Context:** Our `cp_fix = [0, 0, 0]` configuration implements the "single-stage" training of such open segments, as opposed to the two-stage approach of Izmailov (fixed endpoints).
- **Foundational Insight:** The subspace construction must be aware of the "structured" model part ($x^\top \theta$), which can significantly impact the loss landscape of the DNN component.

### 3. Paths and Ambient Spaces in Neural Loss Landscapes (Dold et al., 2025)
- **Note:** Original PDF was inaccessible (404/Page not found). 
- **Inferred Focus:** Explores curve optimization and loss tunnels to improve sampling performance via natural priors.

### 4. Scalable Bayesian LoRA via Stochastic Variational Subspace Inference (Samplawski et al., 2025)
- **Method:** ScalaBL performs Bayesian inference in an $r$-dimensional subspace (where $r$ is the LoRA rank).
- **Technical Detail:** Repurposes LoRA matrices $A$ and $B$ as projection matrices for a latent vector $s \in \mathbb{R}^r$:
  $$ W = W_0 + B \text{diag}(s) A $$
- **Relevance:** Achieves extreme parameter efficiency (requiring only $2r$ additional variational parameters per layer) compared to BLoB (which requires $rd$ parameters). Scalable to 32B+ parameter models.

### 5. BLoB: Bayesian Low-Rank Adaptation by Backpropagation (Wang et al., 2024)
- **Method:** Performs Stochastic Variational Inference (SVI) directly over the LoRA matrix $A$.
- **Formula:** $W_t = W_0 + B(A_\mu + A_\sigma \cdot \epsilon_t)$.
- **Relevance:** A key baseline that improves over Laplace LoRA but at a higher parameter cost ($1.4\times$ LoRA).

### 6. Subspace Inference for Bayesian Deep Learning (Izmailov et al., 2020)
- **Foundation:** The theoretical backbone for subspace-based approaches. 
- **Codebase Mapping:** Corresponds to the `cp_fix = [1, 0, 1]` configuration where the endpoints are pretrained and fixed, and a quadratic Bézier curve is used to find low-loss paths between them.
- **Projection:** $W_t = W_0 + Pz_t$, where $P$ is a projection matrix and $z_t$ is a low-dimensional latent vector.
