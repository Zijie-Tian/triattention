# TriAttention for Prefill: Sparse Pre-RoPE KV Cache Compression

TriAttention is originally designed as a KV cache compression method for the autoregressive decode phase of Large Language Models (LLMs). This document details how to extend the TriAttention Pre-RoPE geometric compression paradigm to the **Prefill phase**, effectively reducing the $O(N^2)$ attention computation to a highly sparse, $O(N^2/B)$ analytical estimation.

## 1. Theoretical Background: TriAttention in Decode

In the standard decode phase, generating the $N$-th token requires computing the attention between the current query $q_N$ and all historical keys $k_j$ ($j \le N$). With Rotary Position Embedding (RoPE) applied, the exact dot product is:
$$ A_{N, j} = \langle q_N \otimes R(N), k_j \otimes R(j) \rangle $$

By decomposing each head dimension $D$ into $D/2$ complex frequency components $f$, this is equivalent to:
$$ A_{N, j} = \sum_{f} \|q_{N, f}\| \|k_{j, f}\| \cos\Big( \omega_f (N - j) + \theta_{q_{N, f}} - \theta_{k_{j, f}} \Big) $$
where $\theta$ is the initial phase angle of the vector in the Pre-RoPE space.

**TriAttention's Core Insight**: In the Pre-RoPE space, queries from the same layer are highly clustered around a fixed expectation vector $\mathbb{E}[q]$. Therefore, we can substitute the unknown $q_N$ with the offline-calibrated $\mathbb{E}[q]$, allowing us to analytically resolve the distance preference:
$$ \tilde{A}_{N, j} = \sum_{f} \|\mathbb{E}[q_f]\| \|k_{j, f}\| \cos\Big( \omega_f (N - j) + \underbrace{\theta_{\mathbb{E}[q_f]} - \theta_{k_{j, f}}}_{\phi_f} \Big) + \text{MLR}_{extra} $$
This allows us to evaluate the importance of historical KV tokens in the decode phase *without* explicitly applying RoPE rotations or high-dimensional dot products.

## 2. The Prefill Challenge: The Failure of 2D Blocking

In the Prefill phase, we must compute the $N \times N$ attention matrix. To avoid $O(N^2)$ complexity in sparse prefill algorithms, the most intuitive approach is to divide both $Q$ and $K$ into $B \times B$ macro-blocks (e.g., $16 \times 16$), and use the block means $\bar{Q}_{block}$ and $\bar{K}_{block}$ to estimate an upper bound for the block's attention score:

$$ \text{Score}_{approx}(I, J) = \sum_f \|\bar{Q}_{I, f}\| \|\bar{K}_{J, f}\| \cos\big(\omega_f (\bar{i} - \bar{j}) + \phi_{\bar{Q}_I} - \phi_{\bar{K}_J}\big) $$

**Why 2D Blocking Fails:**
In real-world empirical validation (tested on `layer_05.pt` with 32K context), the Recall@10% for 2D block estimation is **only ~37%**.
LLM attention distributions are extremely heavy-tailed. Within a $16$-token Key block, there might be only **1** critical "Heavy Hitter" token (with a massive $\|k_j\|$ norm) alongside 15 noise tokens. When we calculate the block mean $\bar{K}_{J, f} = \frac{1}{B}\sum k_{j, f}$, this sharp Heavy Hitter signal is diluted by $1/16$. Since the Softmax mechanism is dominated by maximum values, the mean vector completely fails to represent the true potential of the block.

Attempting to use bounding boxes (e.g., Quest-style $K_{max}, K_{min}$) in 128-dimensional space also fails because independent dimension maximums rarely co-occur in the same token, creating a "frankenstein" vector that drastically overestimates noise blocks.

## 3. The Ultimate Solution: Query-Pooling + Exact Key ($1 \times B$ Reduction)

While Key singularities (Heavy Hitters) must absolutely never be averaged, **adjacent Queries in the prefill sequence have highly consistent semantic intents**.
Queries at position $i$ and $i+1$ generally look for the exact same contextual clues in the historical keys.

Based on this insight, we derive the **Query-Pooling + Exact Key** paradigm.

### 3.1 Mathematical Derivation

We divide the $N$ queries into $N/B$ blocks. For a block $I$ containing $\{i, i+1, \dots, i+B-1\}$ with center position $\bar{i}$, we pool the queries in the Pre-RoPE space to obtain the block's intent representative vector:
$$ \bar{q}_I = \frac{1}{B} \sum_{x \in I} q_{pre, x} $$
*(Crucially, this pooling must occur in the Pre-RoPE space. Averaging in Post-RoPE space causes catastrophic phase cancellation/destructive interference).*

To determine which keys block $I$ should attend to, we **do not block the Keys**. Instead, we match the representative vector $\bar{q}_I$ directly against all global, exact $k_{pre, j}$ tokens. We treat $\bar{q}_I$ exactly as the stable expectation vector $\mathbb{E}[q]$ from the Decode phase TriAttention formulation:

$$ \text{BlockScore}(I, j) = \sum_{f} \|\bar{q}_{I, f}\| \|k_{pre, j, f}\| \cos\Big( \omega_f (\bar{i} - j) + \theta_{\bar{q}_{I, f}} - \theta_{k_{pre, j, f}} \Big) + \text{MLR}_{extra} $$

Where:
1. $\|\bar{q}_{I, f}\|$ is the pooled amplitude of the Query block at frequency $f$ (representing intent strength).
2. $\|k_{pre, j, f}\|$ is the **exact frequency amplitude** of the $j$-th Key (perfectly preserving sharp Heavy Hitter signals).
3. The phase difference $\phi_{I, j, f} = \theta_{\bar{q}_{I, f}} - \theta_{k_{pre, j, f}}$ reflects the semantic angle between the local block intent and the specific Key.
4. $\omega_f (\bar{i} - j)$ is the exact RoPE distance decay preference.

### 3.2 Complexity Analysis

- **Traditional Exact Attention**: Computes rotations and dot products for every $(q_i, k_j)$ pair $\rightarrow O(N^2)$ complexity.
- **Query-Pooling + Exact Key**:
  1. Compute $\bar{q}_I$: $O(N)$
  2. For $N/B$ Query blocks, evaluate $N$ Keys using the 1D TriAttention analytical trigonometric formula.
  3. Total Evaluation Complexity: $O(\frac{N}{B} \times N) = \mathbf{O(\frac{N^2}{B})}$.

For $B=16$, the scoring computation is **reduced by 93.75%**, completely avoiding dense RoPE matrix rotations during inference.

## 4. Empirical Validation

On a 32K context sequence (`layer_05.pt`), generating the sparse mask using this $O(N^2/16)$ method (retaining the top 10% KV blocks) compared against the $O(N^2)$ ground-truth FlashAttention mask yields:

*   **Row-wise Recall@10%:** `93.82%`
    *(We successfully capture ~94% of the true top-10% most critical Key blocks).*
*   **Attention Score Recovery:** `99.62%`
    *(The sum of the exact attention scores of the blocks we selected reaches 99.62% of the theoretical maximum sum. Mathematically, this is equivalent to near-lossless compression).*

## 5. Engineering Implementation Guidelines

To integrate this TriAttention extension into systems like vLLM for ultra-fast long-prompt prefill:

1. **Query Aggregation**: Aggregate prompt Query vectors with a stride of $B$ (e.g., 16 or 32) using simple vector addition **only in the Pre-RoPE space**.
2. **Analytical Scoring**: Invoke a highly optimized CUDA kernel (similar to `score_keys_for_round`) passing $\bar{q}_{I}$ and all historical $K_{pre}$.
3. **Sparse Mask Generation**: Output the KV cache indices that each Query block should retain, generating a Block-Sparse mask.
4. **Sparse Attention**: Dispatch the final computation to a standard Sparse FlashAttention kernel (e.g., BlockSparseAttention or Triton block-mask kernels).

This fusion of system I/O dimensionality reduction and pure mathematical analytical resolution positions TriAttention as an end-to-end framework for the entire lifecycle of long-context LLMs.