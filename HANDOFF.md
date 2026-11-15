# Ryn training handoff

## Current project state

- Repository: <https://github.com/bxanedot/ryn>
- Model: Ryn Assistant 300M, configured for **295,825,920 parameters** in `configs/ryn0_3b.json`.
- Paused status recorded **2026-11-15 20:18 CET**: latest durable exact-resume checkpoint is step 2,700 (123,699,200 / 1,000,000,000 lifetime tokens, 12.37%). A resume attempt logged step 2,800 but stopped before a new checkpoint was written; those 100 updates were lost.
- Prepared pretraining and SFT data plus the step 2,700 checkpoint are kept outside Git in `../ryn-data`; a read-only snapshot is in `../ryn-preservation/RynData-20261115-1830`.
- Architecture: MiniMind submodule pinned at `233d00fa068ec2b8c0d4f07b703934d6391a5a73`.
- Tokenizer: `Qwen/Qwen2.5-0.5B-Instruct`, tokenizer only. Model weights initialize randomly.
- From-scratch pretraining has run on the local RTX 4070 SUPER; no finished model has been exported or released.
- `scripts/train.py` is a single-process CUDA trainer with BF16 autocast, gradient accumulation, dense-model activation checkpointing, fused AdamW, packed memory-mapped token blocks, per-language validation loss, and versioned exact-resume checkpoints.
- `tools/prepare_text_data.py` validates text or SFT JSONL, deduplicates, flags common secrets and personal data patterns, creates deterministic train/validation JSONL, records provenance and optionally counts tokenizer tokens.
- `scripts/chat.py` loads a saved Ryn checkpoint for local inference.

## GPU-host procedure

1. Use WSL2 and install CUDA-enabled PyTorch for the installed NVIDIA driver. Confirm `nvidia-smi` and `torch.cuda.is_available()`.
2. Read `README.md`, `docs/assistant-behavior.md`, `configs/ryn0_3b.json`, and the pinned MiniMind model source.
3. Prepare reviewed, licensed, language-tagged text and SFT data outside Git. Keep the generated manifests and record-level source/license fields.
4. Run the CPU-compatible tests with `python -m unittest discover -s tests -v`.
5. Consult the measured GPU benchmarks and training progress in `README.md`; benchmark again only when changing hardware or training settings.
6. Check free disk space for the token cache and multiple full optimizer checkpoints before starting the one-billion-token run.
7. Pretrain, resume from the latest checkpoint, and create later SFT or additional pretraining runs using the documented commands in `README.md`.
8. Evaluate by language and task before claiming multilingual ability. The trainer reports validation loss, not task competence.

## Safety and limits

- Training is never started by setup, tests, or the benchmark command unless the benchmark is explicitly invoked.
- Training data and model checkpoints stay outside ordinary Git history.
- Exact resume requires the same corpus, tokenizer, architecture, sequence length, batch/accumulation settings, optimizer settings, and total step/token target. Use `--init-from` for a new training phase or changed data.
- Checkpoint directories are versioned and not overwritten. The Python trainer handles KeyboardInterrupt; when stopping through its PowerShell supervisor, verify a `Training stopped safely` log message and updated `latest.json`. If they are absent, resume from the latest durable checkpoint.
- Secret and PII filters are heuristic; review data and rights manually.
- The configured dense MiniMind model is the supported training path. Multi-GPU training and MoE activation checkpointing are not implemented.
- Do not publish checkpoints or announce a release without owner authorization.
