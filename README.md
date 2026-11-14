# Ryn

Ryn is a coding-model project by **bxane**. The active local training target is **Ryn-0.3B**, a MiniMind-based model sized for experiments on a 12 GB GPU. The repository also records **Ryn-9B** as a future multi-GPU target.

**Training status:** the model weights have not been trained or released. The repository is a reproducible starting point, not a claim that Ryn currently matches Codex, Qwen, Kimi, or DeepSeek. A 9B model's capability comes from its training data, token budget, and optimization as well as its architecture. Matching frontier coding models would require a large, carefully licensed corpus, substantial multi-GPU compute, instruction tuning, and independent evaluations.

## Architecture

`configs/ryn0_3b.json` defines a 296M-parameter dense model for a single-GPU pilot. `configs/ryn9b.json` defines the future 8.912B target: 38 MiniMind transformer blocks with 4,096 hidden dimensions, 32 attention heads, 8 KV heads, and a 14,336-wide MLP. Both use the 152,064-token Qwen2.5-Coder vocabulary. The training script derives the vocabulary size from the downloaded tokenizer.

MiniMind is included as a pinned Git submodule. Its model implementation is the architecture used by Ryn; the Ryn training entrypoint adds FSDP full sharding for multi-GPU training.

## Setup

```bash
git clone --recurse-submodules https://github.com/bxanedot/ryn.git
cd ryn
```

Install a PyTorch build that matches the CUDA version on your training host, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ryn 9B is intended for a multi-GPU host with enough aggregate memory for FSDP full sharding. The starter settings assume 8 GPUs with BF16 support, 1 sequence per GPU, and gradient accumulation. Adjust them for the available hardware.

## Prepare licensed code data

Ryn does not redistribute training data. Use code you own or datasets whose terms allow the intended model training and release. Record the source and its license when preparing a corpus:

```bash
python tools/prepare_code_data.py \
  --input /path/to/permissively-licensed-code \
  --output data/ryn-code-pretrain.jsonl \
  --license MIT
```

The script accepts a directory of source files or JSONL rows with a `text` field. It filters common generated/build directories and private-key or credential files, normalizes text, deduplicates exact matches, and writes a manifest beside the output. Review the dataset and its licenses before training; automatic filtering cannot establish that a corpus is safe or licensed.

For supervised fine-tuning, provide JSONL rows in MiniMind's conversation format:

```json
{"conversations":[{"role":"user","content":"Fix this off-by-one bug."},{"role":"assistant","content":"The loop should use `i < len(items)`."}]}
```

Only assistant messages contribute to the SFT loss.

## Train from scratch

Pretraining uses causal language modeling over code. Run from the repository root:

```bash
NPROC_PER_NODE=8
DATA=data/ryn-code-pretrain.jsonl
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train.py \
  --stage pretrain \
  --config configs/ryn9b.json \
  --data "$DATA" \
  --output-dir outputs/ryn9b-base \
  --steps 100000 \
  --max-seq-len 4096 \
  --batch-size 1 \
  --grad-accum 8
```

The command above is a starting configuration, not a complete frontier-scale training run. Increase the token budget and sequence length only after measuring memory and throughput. The current runtime cannot start that run: it exposes no GPU or PyTorch installation, and no remote GPU job tool is connected.

## Single-GPU pilot

For a 12 GB GPU such as an RTX 4070 Super, do not run the 9B configuration from scratch. Start with the separate 296M-parameter MiniMind model, then adjust sequence length and batch settings only after measuring peak VRAM:

```bash
torchrun --standalone --nproc_per_node=1 scripts/train.py \
  --stage pretrain \
  --config configs/ryn0_3b.json \
  --data data/ryn-code-pretrain.jsonl \
  --output-dir outputs/ryn0.3b-pilot \
  --steps 2000 \
  --max-seq-len 1024 \
  --batch-size 1 \
  --grad-accum 8 \
  --save-every 500
```

This pilot is a training-pipeline and data check, not a substitute for Ryn 9B or a frontier coding model.

After the pilot checkpoint is saved, instruction-tune it on a reviewed coding conversation dataset:

```bash
torchrun --standalone --nproc_per_node=1 scripts/train.py \
  --stage sft \
  --config configs/ryn0_3b.json \
  --data data/ryn-code-sft.jsonl \
  --init-from outputs/ryn0.3b-pilot \
  --output-dir outputs/ryn0.3b-instruct \
  --steps 1000 \
  --max-seq-len 1024 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 2e-5 \
  --save-every 250
```

Then instruction-tune the trained base checkpoint on a separately licensed coding conversation dataset:

```bash
torchrun --standalone --nproc_per_node=8 scripts/train.py \
  --stage sft \
  --config configs/ryn9b.json \
  --data data/ryn-code-sft.jsonl \
  --init-from outputs/ryn9b-base \
  --output-dir outputs/ryn9b-instruct \
  --steps 10000 \
  --max-seq-len 4096 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 2e-5
```

## License and provenance

Ryn-authored source code is licensed under GPL-3.0-only. MiniMind remains under its upstream Apache-2.0 license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The license of any released checkpoint must be reviewed against its training data and other model components before distribution.
