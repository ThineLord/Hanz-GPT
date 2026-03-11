# Hanz-GPT (Fork of nanoGPT)

**English** | [中文](#中文版本)

A minimal but practical GPT training/finetuning repo, forked from [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).
This fork keeps the “simple + readable” core, while making it easier to run on Mac (MPS) and organizing checkpoints.

---

## Key features

- Minimal training loop (`train.py`) + GPT model definition (`model.py`)
- Easy to hack for new datasets / finetuning
- Checkpointing & resume training
- Basic text sampling (`sample.py`)
- Runs on CUDA, CPU, and MPS (macOS) — **recommended dtype for MPS is float32**

---

## What’s different from upstream (this fork)

- Device configuration updated for **MPS-friendly run settings** (dtype pinned to `float32` for MPS workflows)
- Added `checkpoints/` directory for archived checkpoints
- README cleanup + bilingual documentation

> Note: if you change device/dtype/compile for different hardware, update your configs accordingly.

---

## Repository structure (high level)

- `train.py` – main training script
- `sample.py` – text generation from checkpoint/pretrained weights
- `model.py` – GPT model definition + utilities
- `configurator.py` – “poor man’s configurator” for CLI/config overrides
- `config/` – optional config presets (extend this folder to make runs reproducible)
- `checkpoints/` – archived checkpoints (avoid accidental overwrite of `out/ckpt.pt`)
- `data/` – dataset preparation/processed artifacts (depends on your dataset choice)
- `assets/` – images, figures, etc.

---

## Setup

### 1) Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell
```

### 2) Install dependencies
If you have a requirements file:
```bash
pip install -r requirements.txt
```

If not, install core packages manually:
```bash
pip install torch numpy
```

---

## Prepare your dataset

This repo follows the nanoGPT philosophy: you preprocess your dataset into a tokenized binary format (e.g., `train.bin`, `val.bin`, plus `meta.pkl`), then train.

- Create your processed dataset under: `data/<dataset_name>/`
  - `data/<dataset_name>/train.bin`
  - `data/<dataset_name>/val.bin`
  - `data/<dataset_name>/meta.pkl` (optional but recommended)
- Set `dataset = '<dataset_name>'` via:
  - command line overrides, or
  - a config file in `config/`, or
  - directly in the script if you’re just experimenting

---

## Training

### Single GPU (CUDA)
```bash
python train.py --batch_size=32 --compile=False
```

### macOS (MPS)
MPS usually works best with float32 and often needs `compile=False`:
```bash
python train.py device=mps dtype=float32 compile=False
```

### Distributed Data Parallel (DDP) example
```bash
torchrun --standalone --nproc_per_node=4 train.py
```

---

## Checkpoints & archival

Default checkpoints land in `out/ckpt.pt`.
This fork also supports archiving into `checkpoints/` to avoid overwriting useful checkpoints by mistake.

> Best practice: avoid committing massive weight files into Git history unless you are intentionally using Git LFS.

---

## Sampling / text generation

After training finishes and you have a checkpoint:

```bash
python sample.py --out_dir out --num_samples 5 --start "Hello"
```

If you initialize from OpenAI GPT-2 weights, check your `sample.py` arguments and tokenizer setup (depends on `meta.pkl` vs GPT-2 tokenizer).

---

## Troubleshooting

- **Mac MPS issues**: try `dtype=float32` and `compile=False`.
- **Out of memory**: reduce `batch_size`, or use gradient accumulation with config overrides.
- **Dataset mismatch**: ensure `meta.pkl` and tokenization are consistent, and your `vocab_size` is correct.

---

## License & credits

- Upstream project: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (MIT)
- This fork: Hanz-GPT (MIT)

---

# 中文版本

**中文** | [English](#hanz-gpt-fork-of-nanogpt)

一个极简但实用的 GPT 训练/微调仓库，fork 自 [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT).
在保留“代码短、可读性高”的前提下，这个 fork 更强调 Mac（MPS）运行体验，并对 checkpoint 做了更清晰的整理。

---

## 核心功能

- 极简训练脚本（`train.py`） + GPT 模型定义（`model.py`）
- 方便二次开发：新数据集、改损失函数、微调都容易
- checkpoint 保存与恢复训练
- 基础文本采样（`sample.py`）
- 支持 CUDA、CPU、MPS（macOS）— **MPS 推荐使用 float32**

---

## 与上游 nanoGPT 的差异（这个 fork 的改动点）

- 设备配置调整，更偏向 **MPS 友好**（MPS 工作流建议 `dtype=float32`）
- 新增 `checkpoints/` 目录，用于归档 checkpoint，避免 `out/ckpt.pt` 被意外覆盖
- README 清理 + 中英文文档

> 提醒：不同硬件对 device/dtype/compile 支持不同，建议用 config 文件管理你的实验参数。

---

## 仓库目录结构（高层概览）

- `train.py` — 主训练脚本
- `sample.py` — 从 checkpoint/pretrained 生成文本
- `model.py` — GPT 模型定义与工具函数
- `configurator.py` — 非常简陋但好用的配置覆盖器（CLI/config）
- `config/` — 可选配置预设（用来让跑实验更可复现）
- `checkpoints/` — 归档 checkpoint（避免被覆盖）
- `data/` — 数据集预处理/结果产出目录（取决于你用什么数据）
- `assets/` — 图片、图表等

---

## 环境准备

### 1）虚拟环境（推荐）
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell
```

### 2）安装依赖
有 requirements：
```bash
pip install -r requirements.txt
```

没有的话，至少安装：
```bash
pip install torch numpy
```

---

## 数据集准备

nanoGPT 的常规流程是：把数据集预处理成二进制（例如 `train.bin`、`val.bin`，以及 `meta.pkl`），再开始训练。

把数据放到：`data/<dataset_name>/` 下，例如：
- `data/<dataset_name>/train.bin`
- `data/<dataset_name>/val.bin`
- `data/<dataset_name>/meta.pkl`（可选但推荐）

然后通过 command line / config 文件 / 简单地修改脚本来设置：
```python
dataset = '<dataset_name>'
```

---

## 训练

### 单卡 CUDA
```bash
python train.py --batch_size=32 --compile=False
```

### macOS（MPS）
MPS 通常更稳的是 float32，并且经常需要关闭 compile：
```bash
python train.py device=mps dtype=float32 compile=False
```

### DDP 示例
```bash
torchrun --standalone --nproc_per_node=4 train.py
```

---

## Checkpoint 与归档

默认会保存到 `out/ckpt.pt`。
为了避免覆盖有用的 checkpoint，这个 fork 还用 `checkpoints/` 做归档。

> 强烈建议：不要把巨大的权重文件直接提交到 Git 历史里，除非你明确在用 Git LFS 这类机制。

---

## 文本采样 / 推理

有 checkpoint 后可以这样生成：
```bash
python sample.py --out_dir out --num_samples 5 --start "你好"
```

如果你从 OpenAI GPT-2 权重初始化，注意 tokenization（`meta.pkl` vs GPT-2 tokenizer）对应一致。

---

## 常见问题

- MPS 各种问题：优先试 `dtype=float32`、`compile=False`
- 显存不够：降低 `batch_size`，或用 config 调整梯度累积
- `meta.pkl` 不匹配：确保词表、tokenizer、处理流程一致

---

## 协议与致谢

- 上游：karpathy/nanoGPT（MIT）
- 本仓库：Hanz-GPT（MIT)
