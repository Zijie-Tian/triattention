# TriAttention 推理性能验证计划

## 1. 研究目标

验证 TriAttention KV cache 压缩方案在真实模型（Llama-3.1-8B-Instruct）上的推理性能是否可以得到保证，即压缩后的模型在长文本数学推理任务（AIME24）上的准确率是否接近全量 KV cache（fullkv）的基线。

## 2. 核心问题

TriAttention 的核心假设是：**频率域评分可以准确估计每个历史 token 对未来 attention 的贡献，从而只保留最重要的 token。** 本计划旨在系统性地验证这一假设在真实模型和真实任务上的有效性。

## 3. 验证策略（分层）

采用四层验证策略，从底层公式正确性到顶层 benchmark 性能逐层推进：

### Layer 1: 公式等价性验证（Foundation）

**目标**：确认 PyTorch fallback 和 Triton kernel 两个版本的评分公式输出一致。

**方法**：
- 用真实模型的 KV cache 作为输入
- 同时运行 `compute_scores_pytorch` 和 `compute_scores_triton`
- 比较输出差异（`allclose` tolerance = 1e-3）

**通过标准**：两个版本的分数差异 < 0.1%。

**文件**：`tests/test_scoring_equivalence.py`

### Layer 2: Attention Pattern 对比（Precision）

**目标**：比较 TriAttention 压缩后的 attention 分布与 fullkv 的分布是否一致。

**方法**：
- 对同一个 prompt，分别用 fullkv 和 triattention（budget=2048）运行
- 提取每层的 attention weights
- 计算 top-k overlap：`len(set(keep_indices) & set(fullkv_topk)) / k`

**通过标准**：top-k overlap > 85%（说明评分公式在正确识别重要 token）。

**文件**：`tests/test_attention_pattern.py`

### Layer 3: 可控短文本验证（Debuggability）

**目标**：用已知答案的短 prompt，手动检查 `keep_indices` 是否合理。

**方法**：
- Prompt: `"The capital of France is"`（期望输出 `"Paris"`）
- 设置 budget=8，观察保留的 token 是否包含语义关键词（France, capital, is）
- 对比不同 budget（8, 16, 32, 64）下的保留模式

**通过标准**：关键词 token 的保留率 > 80%。

**文件**：`tests/test_controllable_prompt.py`

### Layer 4: AIME24 Benchmark 验证（End-to-End）

**目标**：在真实数学推理任务上验证压缩无损性。

**方法**：
- **快速迭代**：AIME24 前 3 题，用于开发迭代中的快速反馈（~30-60s）
- **完整验证**：AIME24 全部 30 题，用于最终报告（~10-15min）
- 对比指标：fullkv baseline vs triattention budget=2048

**通过标准**：
- 快速迭代：前 3 题 pass@1 与 fullkv 差异 <= 1/3（即最多错 1 题）
- 完整验证：30 题 pass@1 与 fullkv 差异 <= 3 题（约 10% 容错）

**文件**：
- `tests/test_aime24_quick.py`（前 3 题）
- `tests/test_aime24_full.py`（完整 30 题）

## 4. 实现计划

### 4.1 模型与数据

| 项目 | 配置 |
|------|------|
| 模型 | Llama-3.1-8B-Instruct（`~/models/Llama-3.1-8B-Instruct`） |
| 设备 | GPU1（CUDA_VISIBLE_DEVICES=1） |
| 数据 | AIME24（`HuggingFaceH4/aime_2024`） |
| 校准 | 复用 `tests/outputs/llama31_8b_stats.pt` 或重新生成 |

### 4.2 文件结构

```
tests/
├── test_scoring_equivalence.py      # Layer 1: PyTorch vs Triton 等价性
├── test_attention_pattern.py         # Layer 2: Attention 分布对比
├── test_controllable_prompt.py       # Layer 3: 可控短文本验证
├── test_aime24_quick.py              # Layer 4a: AIME24 前 3 题快速验证
├── test_aime24_full.py               # Layer 4b: AIME24 完整 30 题验证
└── outputs/
    ├── llama31_8b_stats.pt           # 校准统计（复用已有）
    ├── fullkv_aime24_results.jsonl   # fullkv 基线结果
    └── triattention_aime24_results.jsonl  # triattention 结果
```

### 4.3 关键实现细节

#### 4.3.1 不影响现有实现

所有验证代码都在 `tests/` 下独立编写，不修改：
- `triattention/methods/triattention.py`
- `triattention/vllm/core/scoring.py`
- `triattention/vllm/core/kernels/triton_scoring.py`

如需复用逻辑，通过 import 引用，不做 inline 修改。

#### 4.3.2 Layer 2: Attention Pattern 对比

需要 hook 模型获取 attention weights：

```python
# 在 model.generate 时设置 output_attentions=True
outputs = model.generate(
    inputs,
    output_attentions=True,
    return_dict_in_generate=True,
)
# 提取 attention 并对比 keep_indices 和 attention top-k
```

#### 4.3.3 Layer 4: AIME24 快速验证

复用 `triattention/evaluation/` 的评估框架，但限制样本数：

```python
from triattention.evaluation.data_loader import load_dataset
from triattention.evaluation.evaluate import evaluate

samples = load_dataset("aime24")[:3]  # 只取前 3 题
# 运行生成 + 评估
```

### 4.4 指标与基准

| 层级 | Metric | Baseline | Target |
|------|--------|----------|--------|
| L1 | PyTorch/Triton max diff | - | < 1e-3 |
| L2 | Top-k overlap | - | > 85% |
| L3 | 关键词保留率 | - | > 80% |
| L4a | AIME24 前 3 题 pass@1 | fullkv | diff <= 1/3 |
| L4b | AIME24 30 题 pass@1 | fullkv | diff <= 3 |

## 5. 执行顺序

```
Week 1
├── Day 1-2: 实现 Layer 1 (scoring equivalence)
├── Day 3-4: 实现 Layer 2 (attention pattern)
└── Day 5: 实现 Layer 3 (controllable prompt)

Week 2
├── Day 1-3: 实现 Layer 4a (AIME24 quick, 前 3 题)
└── Day 4-5: 实现 Layer 4b (AIME24 full, 30 题) + 撰写报告
```

## 6. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| AIME24 数据下载失败 | 无法验证 | 使用本地缓存或备用数据集 |
| GPU1 显存不足 | 模型加载失败 | 使用 FP16 或 4-bit 量化 |
| fullkv baseline 运行时间过长 | 阻塞迭代 | 复用已有的 fullkv 结果（如有） |
| 校准统计不匹配 | 评分错误 | 重新运行 calibration |

## 7. 验证命令

### 快速验证（开发迭代）

```bash
CUDA_VISIBLE_DEVICES=1 \
  PYTHONPATH=/mnt/data/tzj/Code/triattention \
  python tests/test_aime24_quick.py \
  --model ~/models/Llama-3.1-8B-Instruct \
  --stats tests/outputs/llama31_8b_stats.pt \
  --budget 2048
```

输出：单个数字（0, 1, 2, 或 3），表示前 3 题的正确数。

### 完整验证（最终报告）

```bash
CUDA_VISIBLE_DEVICES=1 \
  PYTHONPATH=/mnt/data/tzj/Code/triattention \
  python tests/test_aime24_full.py \
  --model ~/models/Llama-3.1-8B-Instruct \
  --stats tests/outputs/llama31_8b_stats.pt \
  --budget 2048
```

输出：单个数字（0-30），表示 30 题的正确数。

## 8. 附录：参考命令

### 运行现有 AIME24 实验框架

```bash
# fullkv baseline
python scripts/cli.py run-one \
  --model Qwen3-8B \
  --dataset aime24 \
  --method fullkv

# triattention
python scripts/cli.py run-one \
  --model Qwen3-8B \
  --dataset aime24 \
  --method triattention \
  --budget 2048

# 评估
python triattention/evaluation/evaluate.py \
  --file_path outputs/merged/results.jsonl \
  --data_name aime24
```
