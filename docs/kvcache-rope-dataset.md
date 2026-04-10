# KV Cache RoPE Feature Distribution Analysis

This document outlines the visual analysis of pre-RoPE and post-RoPE geometric distributions of Query (Q) and Key (K) embeddings across different transformer layers and attention heads.

## Visualizing Pre-RoPE and Post-RoPE Distributions

In Large Language Models (such as the LLaMA/Qwen architectures), attention mechanisms apply **Rotary Position Embeddings (RoPE)** to inject relative positional information into the Q and K embeddings before the dot product is computed. 

Visualizing these vectors in a 2D complex plane reveals a profound geometric property that forms the foundational basis for compression algorithms like TriAttention.

### The Geometric Phenomenon

1. **Pre-RoPE Concentration (The Semantic Core)**
   * Before RoPE is applied, the query ($q_{pre}$) and key ($k_{pre}$) vectors are pure semantic representations.
   * If we plot a specific complex frequency component (e.g., dimensions `0` and `1` as $Re + i \cdot Im$) for thousands of tokens, we observe that for a given attention head, the vectors are **not uniformly distributed**.
   * Instead, they are highly clustered and **concentrated around a stable expectation center** ($\mathbb{E}[q]$ and $\mathbb{E}[k]$) with a specific amplitude and initial phase angle $\theta$.

2. **Post-RoPE Rotation (The Position Arc)**
   * RoPE applies a rotation matrix to these vectors based on their token position index $m$ and a base frequency $\omega_f$. Mathematically, this is equivalent to multiplying the complex vector by $e^{i m \omega_f}$.
   * Visually, this takes the dense "semantic core" from the Pre-RoPE space and smears it along a circular arc around the origin. The radius of the arc corresponds to the amplitude of the original cluster.

### Visualization Script

To comprehensively analyze this phenomenon across all layers and heads, we use `scripts/plot_all_heads.py`.

This script processes tensor dumps (e.g., `layer_05.pt`) containing:
* `pre_rope_q` and `post_rope_q` (Shape: `[seq_len, num_q_heads, head_dim]`)
* `pre_rope_k` and `post_rope_k` (Shape: `[seq_len, num_kv_heads, head_dim]`)

#### Output Structure

The script generates grid-based subplot images separating Q and K heads to prevent visual clutter, saving them into organized directories:

* **Q Heads (`docs/assets/rope_distributions/q_heads/`)**: 
  Contains images like `layer_05_q.png`. For a model with 32 query heads, it generates a 4x8 grid of subfigures. Each subfigure overlays the Pre-RoPE distribution (Blue) and the Post-RoPE distribution (Red) for a single query head.
* **K Heads (`docs/assets/rope_distributions/k_heads/`)**:
  Contains images like `layer_05_k.png`. For a model with Grouped-Query Attention (GQA) utilizing 4 KV heads, it generates a 1x4 grid, visually demonstrating the clustering behavior of the shared keys.

### Usage

Ensure you have the required dependencies (`matplotlib`, `seaborn`, `torch`) installed:

```bash
pip install seaborn matplotlib torch
```

Run the batch plotting script from the repository root:

```bash
python scripts/plot_all_heads.py
```

The script automatically:
1. Iterates through all `.pt` layer dumps in `data/kvcache-rope/`.
2. Samples 10,000 tokens per layer to prevent severe overplotting.
3. Sets a strict `aspect='equal'` ratio and removes axes lines to maintain a clean, scientific aesthetic.
4. Uses extreme low alpha blending (`alpha=0.1`) without edges to naturally reveal data density cores through color accumulation.
5. Saves the high-resolution grid outputs to the `docs/assets/rope_distributions/` directory.