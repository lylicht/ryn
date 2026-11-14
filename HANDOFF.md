# GPU session handoff: Ryn

Use this file to continue the Ryn work in a GPU-enabled Codex session.

## Project target

- Owner/author: **bxane**
- Repository: <https://github.com/bxanedot/ryn>
- Active local run: **ryn0.3b**. Long-term scale-up config: **ryn9b**.
- Goal: a coding-focused model built with the MiniMind implementation at <https://github.com/jingyaogong/minimind>.
- License: Ryn-authored code is GPL-3.0-only. MiniMind remains Apache-2.0; see `THIRD_PARTY_NOTICES.md`.
- Training status: **no Ryn weights have been trained or released**. Do not describe it as Codex/Astra-level or publish benchmark claims without actual results.

## Repository state from the preparing session

The project files in this repository are intended to include:

- `configs/ryn9b.json`: 4,096 hidden size, 38 layers, 32 attention heads, 8 KV heads, 14,336 MLP size, 32,768 positional limit, Qwen2.5-Coder tokenizer.
- Approximate dense parameter count: **8.912B**, computed with tied input/output embeddings and the configured 152,064-token vocabulary.
- `configs/ryn0_3b.json`: 296M-parameter MiniMind model for the user's single RTX 4070 Super (12 GB VRAM) from-scratch pilot.
- `configs/ryn0_2b.json`: 217M-parameter smaller fallback if the 296M pilot exceeds measured VRAM.
- `upstream/minimind`: pinned Git submodule at upstream revision `233d00fa068ec2b8c0d4f07b703934d6391a5a73`.
- `tools/prepare_code_data.py`: local source/JSONL preparation with deduplication, basic credential filtering, and a license manifest.
- `scripts/train.py`: MiniMind model training with PyTorch FSDP full sharding; supports code pretraining and assistant-only SFT.
- `README.md`, `docs/training-plan.md`, `THIRD_PARTY_NOTICES.md`, and `LICENSE` document scope and provenance.

The preparing session ran in a CPU-only container with Python 3.12, no PyTorch, and no visible NVIDIA device. Its Hugging Face GPU Jobs tool was unavailable. No training run was started. The available GitHub connection had write permission for `bxanedot/ryn`.

## Hardware note

The user mentioned a machine with 32 GB system RAM and an RTX 4070 Super. That GPU has 12 GB VRAM. It is suitable for inference with quantized models and small experiments; it cannot train this 8.9B model from scratch. FSDP with a single GPU does not split the model across devices and will not solve the VRAM limit.

The user has chosen the 0.3B from-scratch model as the active local target; `ryn9b` remains a future scale-up. Before running any training command, verify the actual GPU host with `nvidia-smi` and `torch.cuda`.

QLoRA on an existing coding model could provide stronger results, but it would change the selected from-scratch plan. Ask before switching approaches.

Do not silently change the requested approach or model size. To train `ryn9b` from scratch, arrange a multi-GPU host with sharded optimizer/model state, substantial storage, a large rights-cleared corpus, and a compute budget. The current starter script requires a CUDA host with BF16 support; its throughput and memory limits still need profiling.

## Next session steps

1. Open or clone `bxanedot/ryn`, then run `git submodule update --init --recursive`.
2. Read this file, `README.md`, `configs/ryn9b.json`, and `scripts/train.py` before changing the training plan.
3. Check the available GPU, VRAM, system RAM, CUDA, disk space, and checkpoint storage. On a 4070 Super, do not launch the `ryn9b` pretraining command.
4. Run the selected roughly 0.3B from-scratch pilot on the 4070 Super using `configs/ryn0_3b.json` and the exact one-GPU command in the README. If it exceeds measured VRAM, reduce sequence length and then fall back to `configs/ryn0_2b.json`. This small model can learn basic coding patterns; it will not match Codex-level capability. Ask before switching to QLoRA adaptation of an existing coding model.
5. Keep training data out of Git. Use only data with documented training rights; preserve the generated `.manifest.json` and corpus provenance.
6. Start with a short pilot on the selected hardware. Record GPU type, batch size, sequence length, tokens/sec, peak VRAM, loss, and checkpoint size before scaling.
7. Save weights outside normal Git history (for example, a model artifact store or Git LFS after confirming destination and access). Add evaluation results and exact run metadata before calling any checkpoint a release.

## Plugins and tools

The preparing session found no callable Ponytail or Superpower plugin and no callable remote GPU job tool. It did have a GitHub integration. Re-check the tools available in the next session; do not assume those plugins or GPU access carried over.
