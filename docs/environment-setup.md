# TriAttention 环境配置指南

本文档描述如何在新的机器上配置 TriAttention 开发环境。

## 硬件要求

- **GPU**: NVIDIA GPU (推荐 A100 或更高, 至少 24GB 显存)
- **内存**: 32GB+
- **磁盘**: 50GB+ 可用空间
- **CUDA Driver**: >= 12.1

## 环境创建

```bash
# 1. 创建 conda 环境
mamba create -n triattention python=3.10 -y
# 或
conda create -n triattention python=3.10 -y

# 2. 激活环境
conda activate triattention
```

## 安装依赖

### 1. PyTorch (CUDA 12.1)

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

### 2. Flash Attention (推荐)

Flash Attention 提供显著的加速效果，但需要 CUDA 编译:

```bash
pip install flash-attn --no-build-isolation
```

### 3. 核心依赖

```bash
pip install \
  transformers==5.5.1 \
  datasets==4.8.4 \
  huggingface-hub==1.9.2 \
  accelerate==1.13.0 \
  einops==0.8.2 \
  sentencepiece==0.2.1 \
  pyyaml==6.0.3 \
  tqdm==4.67.3 \
  matplotlib==3.10.8 \
  regex \
  pebble==5.2.0 \
  sympy==1.14.0 \
  scipy==1.15.3 \
  latex2sympy2==1.9.1 \
  word2number==1.1 \
  triton==3.0.0 \
  antlr4-python3-runtime==4.7.2 \
  numpy
```

### 4. 安装项目

```bash
cd /path/to/triattention
pip install -e .
```

## 依赖版本参考

| 包 | 版本 | 说明 |
|-----|------|------|
| Python | 3.10 | |
| torch | 2.4.0+cu121 | CUDA 12.1 |
| transformers | 5.5.1 | |
| flash-attn | 2.8.3 | 可选，但推荐安装 |
| datasets | 4.8.4 | |
| accelerate | 1.13.0 | |
| einops | 0.8.2 | |
| scipy | 1.15.3 | |
| sympy | 1.14.0 | |

## 可选组件

### vLLM (生产推理服务器)

```bash
pip install vllm
```

### MLX (Apple Silicon Mac)

```bash
pip install mlx mlx-lm
```

## 验证安装

```bash
conda run -n triattention python -c "
import torch
import transformers
import flash_attn
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'transformers: {transformers.__version__}')
print(f'flash-attn: {flash_attn.__version__}')
"
```

预期输出:
```
PyTorch: 2.4.0+cu121
CUDA available: True
transformers: 5.5.1
flash-attn: 2.8.3
```

## 快速测试

```bash
conda activate triattention

python scripts/cli.py run-one \
    --model Qwen3-8B \
    --dataset aime24 \
    --method triattention \
    --budget 2048
```

## 常见问题

### 1. 多 GPU 选择

使用 `--gpus` 参数指定 GPU:

```bash
python scripts/dispatch.py --config <config.yaml> --gpus 0,1
```

### 2. 模型下载位置

模型默认下载到 `~/models/` 目录。确保有足够磁盘空间:

```bash
# 检查可用空间
df -h ~

# 默认下载路径
ls ~/models/
```

### 3. CUDA 版本不匹配

如果遇到 CUDA 版本问题，检查:

```bash
# 查看驱动版本
nvidia-smi

# 查看 nvcc 版本
nvcc --version

# 查看 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"
```

确保 CUDA Driver >= 12.1。

### 4. flash-attn 编译失败

如果 flash-attn 编译失败:
1. 确保安装了 CUDA Toolkit: `conda install cuda-toolkit -c nvidia`
2. 设置 PATH: `export PATH=/usr/local/cuda/bin:$PATH`
3. 尝试使用预编译版本或跳过此依赖
