# GPU session handoff: Ryn multilingual assistant

Continue work in a GPU-enabled session by opening this repository and reading this file first.

## Goal and current state

- Owner: **bxane**
- Repository: <https://github.com/bxanedot/ryn>
- Model: **Ryn Assistant 300M**, `configs/ryn0_3b.json` (296,034,048 parameters, about 0.3B).
- Purpose: multilingual general assistant for everyday work, planning, writing, learning, summarizing, and organization.
- Framework: MiniMind, pinned at `upstream/minimind` revision `233d00fa068ec2b8c0d4f07b703934d6391a5a73`.
- Tokenizer: `Qwen/Qwen2.5-0.5B-Instruct` tokenizer only; Ryn weights initialize randomly.
- Status: **no Ryn weights have been trained or released**. Do not claim broad language support or Codex-level ability without evaluations.
- Optional web use comes from a runtime search/fetch tool. It is not part of the model weights.

The preparation environment was CPU-only, with Python 3.12, no PyTorch, and no visible NVIDIA device. No training job was started. The user has a separate machine with 32 GB system RAM and an RTX 4070 Super (12 GB VRAM).

## Next steps on the GPU host

1. Confirm `nvidia-smi`, VRAM, disk, CUDA, and `torch.cuda.is_available()` in the new session.
2. Clone/open `bxanedot/ryn`, run `git submodule update --init --recursive`, then read `README.md`, `docs/assistant-behavior.md`, and `configs/ryn0_3b.json`.
3. Use only licensed, reviewed, language-tagged data. Start with a small multilingual corpus and keep its manifest outside Git.
4. Run the short single-GPU pretraining command and then the SFT command in the README. Reduce sequence length if VRAM is tight. Record peak VRAM, tokens/sec, loss, steps, and data token counts.
5. Evaluate by language and task before making capability claims. Include organization/planning tasks and factuality checks.
6. If internet tools are available in the host, verify how Ryn is expected to call them, include citations in responses, and test the unavailable-tool behavior. Otherwise, keep web use disabled and have Ryn state that limitation.
7. Keep weights and training data out of ordinary Git. Publish checkpoints only after reviewing data/model licenses and adding run details.

## Behavior spec

Follow [docs/assistant-behavior.md](docs/assistant-behavior.md): understand the goal and constraints, make an actionable plan, act only through available and authorized tools, verify the result, and report what remains uncertain. Reply in the user's language where possible. Do not invent deadlines, facts, or actions.

## Expected limits

This 296M-parameter model is a small training project. Its size permits experiments on the 4070 Super; it does not imply strong reasoning, reliable organization, or support for every language. Quality depends on data, training time, instruction tuning, and per-language evaluations. Internet access requires runtime tools and citations; the base model has no live web access.
