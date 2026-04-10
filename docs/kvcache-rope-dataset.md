# kvcache-rope 数据集使用指南

## 概述

kvcache-rope 数据集包含各模型在不同 context length 下推理时的 pre/post RoPE QKV tensor 数据，用于分析 RoPE 位置编码对 KV cache 的影响，以及稀疏注意力策略的设计与验证。

## 数据存储位置

**阿里云盘路径**: `/data/COMPASS/kvcache-rope/`

本地缓存路径: `results/kvcache-rope/`

## 数据清单

| 模型 | 层数 | Context Lengths | 文件数 | 数据量 |
|------|------|----------------|--------|--------|
| GLM-4-9B | 40 | 16k, 32k, 64k, 128k | 160 | ~177 GB |
| Llama-3.1-8B | 32 | 16k, 32k, 64k | 96 | ~77 GB |
| Qwen2.5-7B | 28 | 16k, 32k, 64k | 84 | ~51 GB |

## 数据格式

每个 `.pt` 文件包含以下字段：

```python
{
    'pre_rope_q': tensor,    # [seq_len, num_heads, head_dim] - RoPE前的Q
    'pre_rope_k': tensor,    # [seq_len, num_kv_heads, head_dim] - RoPE前的K
    'post_rope_q': tensor,   # [seq_len, num_heads, head_dim] - RoPE后的Q
    'post_rope_k': tensor,   # [seq_len, num_kv_heads, head_dim] - RoPE后的K
    'v': tensor,             # [seq_len, num_kv_heads, head_dim] - V值
    'positions': tensor,     # [seq_len] - 位置索引
}
```

## 文件命名规则

```
layer_XX.pt              # GPU-only 模式
layer_XX_chunk_MMMM.pt   # Chunked prefill 模式
```

例如：`layer_05.pt` 表示第 5 层的数据。

## 下载数据

### 使用 aliyunpan 下载

```bash
# 下载 Llama-3.1-8B 数据
aliyunpan download /data/COMPASS/kvcache-rope/llama-3.1-8b ./results/kvcache-rope/

# 下载 GLM-4-9B 数据
aliyunpan download /data/COMPASS/kvcache-rope/glm-4-9b ./results/kvcache-rope/

# 下载 Qwen2.5-7B 数据
aliyunpan download /data/COMPASS/kvcache-rope/qwen2.5-7b ./results/kvcache-rope/
```

### 查看云盘内容

```bash
aliyunpan ll /data/COMPASS/kvcache-rope/
aliyunpan ll /data/COMPASS/kvcache-rope/llama-3.1-8b/
```

## 代码加载示例

### Python 基本加载

```python
import glob
import torch

# 自动查找 kvcache-rope 数据
candidates = glob.glob("results/kvcache-rope/**/layer_05.pt", recursive=True)
if not candidates:
    print("ERROR: layer_05.pt not found. Download from aliyunpan first.")
    sys.exit(1)

data = torch.load(candidates[0], map_location="cpu")

# 提取 Q, K, V
Q_all = data["post_rope_q"].float()  # [S, n_heads, D]
K_all = data["post_rope_k"].float()   # [S, n_kv_heads, D]
V_all = data["v"].float()             # [S, n_kv_heads, D]
positions = data["positions"]         # [S]

print(f"Loaded: {candidates[0]}")
print(f"  seq_len={S}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, D={D}")
```

### 提取特定 chunk 的数据

```python
def extract_chunk(Q_all, K_all, chunk_idx, chunk_size=4096, kv_head=0):
    """提取给定 chunk 的 Q, K 对用于测试"""
    S = Q_all.shape[0]
    n_kv_heads = K_all.shape[1]
    gqa_ratio = Q_all.shape[1] // n_kv_heads

    q_start = chunk_idx * chunk_size
    q_end = min(q_start + chunk_size, S)
    k_end = q_end

    # Q: 第 kv_head 个 KV head 对应的所有 Q heads (GQA)
    Q = Q_all[q_start:q_end, kv_head * gqa_ratio, :].contiguous()
    # K: 前 kv_head 个 KV head 的数据
    K = K_all[:k_end, kv_head, :].contiguous()

    return Q, K, q_start
```

## 使用场景

### 1. BLASST Block Mask 测试

```bash
# 稀疏度分析
python tests/test_blockmask_sparsity.py

# 指定数据路径
python tests/test_blockmask_sparsity.py results/kvcache-rope/llama-3.1-8b/32k/layer_05.pt

# 自定义参数
python tests/test_blockmask_sparsity.py --lambdas 0.1 0.001 0.0001 --chunk-size 2048
```

### 2. CPU 算子性能测试

```bash
# 性能基准测试
python tests/test_blockmask_perf.py

# 自定义 lambda 和迭代次数
python tests/test_blockmask_perf.py --lambda 0.001 --warmup 5 --iters 20
```

### 3. 正确性验证

```bash
python tests/test_qk_blockmask_torch.py
```

## 1M Context 数据估算

如需 1M 上下文长度的数据，单层大小估算如下：

| 模型 | 层数 | 单层大小 | 全模型大小 |
|------|------|---------|----------|
| GLM-4-9B | 40 | ~19 GB | ~760 GB |
| Llama-3.1-8B | 32 | ~22 GB | ~704 GB |
| Qwen2.5-7B | 28 | ~17 GB | ~476 GB |

理论计算（以 GLM-4 为例）：
- 1M 上下文 = 1,048,576 tokens
- pre_rope_q: 1M × 32heads × 128dim × 2bytes ≈ 8.6 GB
- pre_rope_k: 1M × 2kv_heads × 128dim × 2bytes ≈ 0.54 GB
- post_rope_q: 1M × 32heads × 128dim × 2bytes ≈ 8.6 GB
- post_rope_k: 1M × 2kv_heads × 128dim × 2bytes ≈ 0.54 GB
- v: 1M × 2kv_heads × 128dim × 2bytes ≈ 0.54 GB
- positions: 1M × 4bytes ≈ 4 MB
- **单层总计: ~18.8 GB**

## 相关文档

- [BLASST Block-Mask 测试指南](./blasst-blockmask-test.md)
- [BLASST 性能分析](./blasst-performance.md)
- [RoPE 数据收集流程](./rope-data-collection.md)

## 注意事项

1. **GPU-only 模式限制**: 128k 上下文在 80GB A100 上会 OOM，需要使用 CPU offload 模式
2. **存储空间**: 下载前确保有足够磁盘空间
3. **数据完整性**: 下载后建议验证文件数量是否完整
