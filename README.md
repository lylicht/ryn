# Ryn: multilingual 300M assistant

Ryn is a from-scratch multilingual general assistant built with the pinned MiniMind architecture. `configs/ryn0_3b.json` defines **295,825,920 parameters** with tied embeddings and the Qwen2.5 tokenizer. The tokenizer is used for text only; no Qwen model weights are used. CUDA pretraining resumed from durable checkpoint 70,000; the final fine-tuned model has not yet been evaluated or exported to Desktop.

The pinned tokenizer has 151,665 entries. Both model configs and the trainer validate this vocabulary and its resulting parameter count; the prior config estimate used a stale vocabulary size.

Each checkpoint stores the model weights as one `model.safetensors` file; loading the model also requires its config and tokenizer files, which belong alongside that single weight file in the final Desktop package.

Ryn is intended for everyday writing, explanations, planning, learning, summaries, and organization. A compact 300M model is an experiment, not evidence of strong reasoning or broad language support. Evaluate it by language and task before making capability claims. Optional live web access must come from a host tool; it is not in the model weights.

## Training record

All times below use **Europe/Zurich (CET, UTC+01:00)**.

| Milestone | Date and time |
| --- | --- |
| First GPU training attempt | 2026-11-15 01:57:19 CET |
| From-scratch run used for the current model lineage | 2026-11-15 02:21:04 CET |
| Memory-lighter continuation launched from checkpoint 8,900 | 2026-11-15 12:58:08 CET |
| Continuation resumed from durable step 1 after checkpoint-save fix | 2026-11-15 13:20:06 CET |
| Exact resume regrouped to batch 2 / accumulation 16 from step 200 | 2026-11-15 13:39:25 CET |
| Resumed batch 2 / accumulation 16 after step 607 OOM | 2026-11-15 13:58:56 CET |
| Fell back to batch 1 / accumulation 32 from step 1,000 after step 1,015 OOM | 2026-11-15 14:19:39 CET |
| Retried batch 1 / accumulation 32 from step 1,000 after step 1,090 OOM | 2026-11-15 14:31:24 CET |
| Saved durable checkpoint 1,100 after the step 1,090 retry | 2026-11-15 14:44:20 CET |
| Resumed from checkpoint 1,100 with combined-token progress reporting | 2026-11-15 14:49:55 CET |
| Resumed batch 2 / accumulation 16 from step 1,100 for higher measured throughput | 2026-11-15 14:54:37 CET |
| Saved durable checkpoint 1,200 with corrected combined-token ETA | 2026-11-15 15:01:17 CET |
| Completed a 20-step batch 4 / accumulation 8 benchmark | 2026-11-15 15:07 CET |
| Resumed production training from step 1,200 at batch 4 / accumulation 8 | 2026-11-15 15:09:12 CET |
| Completed a 20-step batch 8 / accumulation 4 benchmark | 2026-11-15 15:12 CET |
| Resumed production training from step 1,300 at batch 8 / accumulation 4 | 2026-11-15 15:13:59 CET |
| Saved durable checkpoint 1,400 at batch 8 / accumulation 4 | 2026-11-15 15:15:48 CET |
| Saved durable checkpoint 1,500 at batch 8 / accumulation 4 | 2026-11-15 15:17:42 CET |
| Saved durable checkpoint 1,600 at batch 8 / accumulation 4 | 2026-11-15 15:19:10 CET |
| Saved durable checkpoint 1,700 at batch 8 / accumulation 4 | 2026-11-15 15:20:42 CET |
| Saved durable checkpoint 1,800 at batch 8 / accumulation 4 | 2026-11-15 15:22:10 CET |
| Saved durable checkpoint 1,900 at batch 8 / accumulation 4 | 2026-11-15 15:23:48 CET |
| Saved durable checkpoint 2,000 at batch 8 / accumulation 4 | 2026-11-15 15:25:28 CET |
| Saved durable checkpoint 2,100 at batch 8 / accumulation 4 | 2026-11-15 15:26:59 CET |
| Disabled automatic old-checkpoint deletion and resumed from step 2,100 | 2026-11-15 15:28:06 CET |
| Saved checkpoint 2,200 while retaining all four current 256-token checkpoints | 2026-11-15 15:29:52 CET |
| Saved checkpoint 2,300 while retaining all five current 256-token checkpoints | 2026-11-15 15:31:35 CET |
| Saved checkpoint 2,400 while retaining all six current 256-token checkpoints | 2026-11-15 15:33:09 CET |
| Saved checkpoint 2,500 while retaining all seven current 256-token checkpoints | 2026-11-15 15:34:39 CET |
| Saved checkpoint 2,600 while retaining all eight current 256-token checkpoints | 2026-11-15 15:36:19 CET |
| Saved checkpoint 2,700 while retaining all nine current 256-token checkpoints | 2026-11-15 15:37:47 CET |
| Resumed the 256-token continuation from durable checkpoint 2,700 after disk recovery and Python 3.11.9 restoration; changed checkpoint cadence to every 15,000 steps | 2026-11-15 18:02:59 CET |
| Reached live step 2,800 on CUDA; first post-resume 100-step interval measured 7,990.9 tokens/second and loss 4.4703 | 2026-11-15 18:05 CET |
| Reached live step 3,000 on CUDA; interval loss 4.4759 at 6,848.1 tokens/second, combined progress 12.62% | 2026-11-15 18:09 CET |
| Latest logged step 3,100; interval loss 4.4494 at 6,607.3 tokens/second, combined progress 12.70% | 2026-11-15 18:12 CET |
| Latest logged step 3,200; interval loss 4.4566 at 7,477.0 tokens/second, combined progress 12.78% | 2026-11-15 18:13 CET |
| Latest logged step 3,300; interval loss 4.5185 at 8,373.9 tokens/second, combined progress 12.86% | 2026-11-15 18:14 CET |
| Latest logged step 3,400; interval loss 4.4911 at 8,757.1 tokens/second, combined progress 12.94% | 2026-11-15 18:16 CET |
| Latest logged step 3,500; interval loss 4.4799 at 7,636.0 tokens/second, combined progress 13.03% | 2026-11-15 18:18 CET |
| Latest logged step 3,600; interval loss 4.4863 at 7,342.9 tokens/second, combined progress 13.11% | 2026-11-15 18:19 CET |
| Latest logged step 3,700; interval loss 4.4596 at 7,722.6 tokens/second, combined progress 13.19% | 2026-11-15 18:22 CET |
| Latest logged step 3,800; interval loss 4.4505 at 7,692.0 tokens/second, combined progress 13.27% | 2026-11-15 18:23 CET |
| Latest logged step 3,900; interval loss 4.4372 at 8,070.9 tokens/second, combined progress 13.35% | 2026-11-15 18:25 CET |
| Latest logged step 4,000; interval loss 4.4451 at 8,043.8 tokens/second, combined progress 13.43% | 2026-11-15 18:27 CET |
| Latest logged step 4,100; interval loss 4.4264 at 8,002.0 tokens/second, combined progress 13.52% | 2026-11-15 18:28 CET |
| Latest logged step 4,200; interval loss 4.4318 at 7,904.2 tokens/second, combined progress 13.60% | 2026-11-15 18:30 CET |
| Latest log reached step 5,200 (14.42% lifetime tokens); no checkpoint was written after step 2,700 | 2026-11-15 18:48:04 CET |
| Pretraining paused for today; preservation snapshot and read-only ACL recorded | 2026-11-15 19:09 CET |
| Resumed exact training from checkpoint 2,700; validated 295,825,920 parameters and tokenizer vocabulary 151,665 | 2026-11-15 19:12:42 CET |
| First resumed log reached step 2,800 (12.45% cumulative tokens; 5,892.8 tokens/second) | 2026-11-15 19:15:26 CET |
| Paused after the resume attempt; latest durable checkpoint is step 2,700 | 2026-11-15 20:18 CET |
| Pushed private code backup branch codex/ryn-training-system at commit 491d5d | 2026-11-15 20:20:20 CET |
| Uploaded hash-verified prepared-data and exact-resume bundle to private draft GitHub release | 2026-11-15 20:28:38 CET |
| Resumed CUDA pretraining from exact durable checkpoint 2,700 after the remote backup | 2026-11-15 20:35:47 CET |
| Latest resumed log reached step 2,800 (12.45% cumulative tokens; 9,189.9 tokens/second; checkpoint 2,700 remains durable) | 2026-11-15 20:39:20 CET |
| Latest log reached step 3,000 (12.62% cumulative tokens; 9,470.5 tokens/second; checkpoint 2,700 remains durable) | 2026-11-15 20:42:15 CET |
| Latest resumed log reached step 3,200 (12.78% cumulative tokens; 9,342.2 tokens/second; checkpoint 2,700 remains durable) | 2026-11-15 20:43:43 CET |
| Latest resumed log reached step 3,500 (13.03% cumulative tokens; 8,502.9 tokens/second; checkpoint 2,700 remains durable) | 2026-11-15 20:48:17 CET |
| Final pretraining, SFT, evaluation, and Desktop export completed | Pending |

At 2026-11-15 14:51 CET, the latest verified durable checkpoint was step 1,100 in the 256-token continuation, initialized from the step 8,900 weights of the prior 512-token phase. The active run at that time used batch size 1 and gradient accumulation 32 (8,192 tokens per optimizer update), and resumed the optimizer and scheduler from step 1,100 at 14:49:55 CET. Combined pretraining progress was **110,592,000 / 1,000,000,000 tokens (11.06%)**. That 100-step interval had loss 4.5524 at 1,106.8 tokens/second, 36% GPU utilization, and 5.42 GiB peak allocated VRAM. An earlier attempt to save step 100 failed with a Windows access-denied error on the atomic directory rename, leaving step 1 as the latest checkpoint. I added a bounded rename retry and a test for transient pre- and post-rename denials. The training reporter now accepts `--prior-tokens` so the one-billion-token estimate includes earlier phases; the supervisor passes 101,580,800 prior tokens. All 11 CPU tests and the PowerShell parser check passed. The supervisor restarted from step 1 at 13:20:06 CET and saved step 100 at 13:28:41 CET, then resumed step 200 with the new microbatch grouping at 13:39:25 CET. An OOM at step 607 reloaded step 600 at 13:58:56 CET; the retry resumed at batch 2 / accumulation 16 and reached step 1,000. A second OOM at step 1,015 triggered the batch 1 / accumulation 32 fallback, which reloaded step 1,000 at 14:19:39 CET. An OOM at step 1,090 caused another reload from step 1,000 at 14:31:24 CET, then the retry saved step 1,100 at 14:44:20 CET. I interrupted only after that complete checkpoint to apply the progress-reporting fix and resumed from it at 14:49:55 CET; no model or optimizer progress was discarded. CUDA OOMs occurred at steps 3,507, 3,585, 3,705, 5,109, 5,511, 6,108, 7,306, 8,007, 8,331, 8,707, and 8,907 in the 512-token phase. A cuBLAS failure at step 6,301 exposed an invalid PowerShell log-tail parameter combination, which was fixed. The corrected supervisor loaded step 6,300 and reached step 7,300; it reloaded step 7,300 after the step 7,306 OOM, then reloaded step 8,000 after the step 8,007 OOM and reached step 8,300; after the step 8,331 OOM it reloaded step 8,300 and reached step 8,700; after the step 8,707 OOM it reloaded step 8,700 and reached step 8,900. Five retries at step 8,907 exhausted the 512-token supervisor; its latest durable checkpoint remains step 8,900, which initialized the new phase. Throughput and remaining-time estimates vary with GPU load. The prior run and log are `../ryn-data/runs/ryn-0.3b-base-final-512` and `../ryn-data/pretrain-continuation.log`; the active run and log are `../ryn-data/runs/ryn-0.3b-base-final-256` and `../ryn-data/pretrain-continuation-256.log`.

As of 2026-11-15 14:55 CET, the latest durable checkpoint is still step 1,100 and combined pretraining remains **110,592,000 / 1,000,000,000 tokens (11.06%)**. The trainer has resumed on CUDA from that checkpoint at batch size 2 / accumulation 16, with `--prior-tokens 101580800`; the current process and exact resume settings are verified. This grouping previously measured 2,502.9 tokens/second and a 5.78 GiB peak, which corresponds to about 98.7 hours remaining at that old measured rate; the fresh corrected ETA will be available after the next 100-step log interval. The supervisor retains its automatic fallback to batch size 1 / accumulation 32 after repeated OOMs.

At 2026-11-15 15:02 CET, the latest durable checkpoint was step 1,200 and combined pretraining was **111,411,200 / 1,000,000,000 tokens (11.14%)**. The batch 2 / accumulation 16 interval reported loss 4.5833, 2,148.2 tokens/second, 45% GPU utilization, and 5.78 GiB peak allocated VRAM; the corrected one-billion-token estimate was 413,645 seconds (about 114.9 hours) at that measured rate. The log confirms `lifetime_tokens=111411200` and calculates its 1B ETA using the 101,580,800 tokens from earlier phases. Checkpoint 1,200 was saved at 15:01:17 CET.

As of 2026-11-15 15:10 CET, the latest durable checkpoint remains step 1,200 and combined pretraining remains **111,411,200 / 1,000,000,000 tokens (11.14%)**. A 20-step batch 4 / accumulation 8 benchmark on the RTX 4070 SUPER measured 5,295.8 tokens/second over steps 11–20, with 68% GPU utilization and 6.51 GiB peak allocated VRAM. Windows reported 10.2 GiB of 12.3 GiB in use during the benchmark. Production training resumed from step 1,200 at batch 4 / accumulation 8 at 15:09:12 CET; the live run is on CUDA with about 9.7 GiB in use. The supervisor now falls back from batch 4 to batch 2 after an OOM, then to batch 1 after repeated batch-2 OOMs. The benchmark rate projects about 47 hours to one billion tokens, but production throughput and memory headroom still need confirmation over a longer interval.

As of 2026-11-15 15:15 CET, the latest durable checkpoint is step 1,300 and combined pretraining is **112,230,400 / 1,000,000,000 tokens (11.22%)**. A 20-step batch 8 / accumulation 4 benchmark measured 7,281.8 tokens/second over steps 11–20, with 94% GPU utilization and 8.23 GiB peak allocated VRAM; Windows reported 11.6 GiB of 12.3 GiB in use during that benchmark. Its run was disposable and did not change or count toward durable model training. Production resumed from step 1,300 at batch 8 / accumulation 4 at 15:13:59 CET and is active on CUDA; current reported memory use is about 11.4 GiB. The supervisor immediately falls back to batch 4 after a batch-8 OOM, then batch 2 and batch 1 after further OOMs. The benchmark rate implies about 34 hours to the billion-token target from the current durable count; this remains an estimate until a production interval completes, and the low memory headroom may trigger fallback.

As of 2026-11-15 15:17 CET, the latest durable checkpoint is step 1,400 and combined pretraining is **113,049,600 / 1,000,000,000 tokens (11.30%)**. The first 100-step production interval at batch 8 / accumulation 4 reported loss 4.5431, 8,181.8 tokens/second, 80% GPU utilization, and 8.24 GiB PyTorch peak allocation; its corrected ETA to one billion tokens is 108,406 seconds (about 30.1 hours) at that measured rate. The system reported 11.76 GiB of 12.28 GiB GPU memory in use near the checkpoint save, leaving little desktop-memory headroom. Checkpoint 1,400 was saved at 15:15:48 CET, and the supervisor remains live with staged OOM fallback to batch 4, then 2, then 1.

As of 2026-11-15 15:19 CET, the latest durable checkpoint is step 1,500 and combined pretraining is **113,868,800 / 1,000,000,000 tokens (11.39%)**. The second batch 8 / accumulation 4 production interval reported loss 4.5456, 7,166.3 tokens/second, 77% GPU utilization, and 8.24 GiB PyTorch peak allocation; the corrected one-billion-token estimate is 123,653 seconds (about 34.3 hours) at that rate. System GPU memory was about 11.3 GiB of 12.28 GiB at the latest check. Checkpoint 1,500 was saved at 15:17:42 CET, and the supervisor is continuing with staged fallback on OOM.

As of 2026-11-15 15:21 CET, the latest durable checkpoint is step 1,600 and combined pretraining is **114,688,000 / 1,000,000,000 tokens (11.47%)**. The latest batch 8 / accumulation 4 interval reported loss 4.5310, 9,261.4 tokens/second, 76% GPU utilization, and 8.24 GiB PyTorch peak allocation; the corrected estimate to one billion tokens is 95,591 seconds (about 26.6 hours) at that rate. Checkpoint 1,600 was saved at 15:19:10 CET; batch-8 training remains active and the supervisor retains its lower-batch fallbacks.

As of 2026-11-15 15:22 CET, the latest durable checkpoint is step 1,700 and combined pretraining is **115,507,200 / 1,000,000,000 tokens (11.55%)**. The latest batch 8 / accumulation 4 interval reported loss 4.5321, 8,930.2 tokens/second, 84% GPU utilization, and 8.24 GiB PyTorch peak allocation; its corrected estimate to one billion tokens is 99,045 seconds (about 27.5 hours) at that rate. Checkpoint 1,700 was saved at 15:20:42 CET, and production training remains active.

As of 2026-11-15 15:24 CET, the latest durable checkpoint is step 1,800 and combined pretraining is **116,326,400 / 1,000,000,000 tokens (11.63%)**. The latest batch 8 / accumulation 4 interval reported loss 4.5252, 9,276.1 tokens/second, 77% GPU utilization, and 8.24 GiB PyTorch peak allocation; the corrected one-billion-token ETA is 95,264 seconds (about 26.5 hours) at this interval's rate. Checkpoint 1,800 was saved at 15:22:10 CET, and the CUDA supervisor is continuing with staged OOM fallback.

As of 2026-11-15 15:25 CET, the latest durable checkpoint is step 1,900 and combined pretraining is **117,145,600 / 1,000,000,000 tokens (11.71%)**. The latest batch 8 / accumulation 4 interval reported loss 4.5396, 8,394.0 tokens/second, 69% GPU utilization, and 8.24 GiB PyTorch peak allocation; the corrected estimate to one billion tokens is 105,177 seconds (about 29.2 hours) at this rate. Checkpoint 1,900 was saved at 15:23:48 CET, and the supervised CUDA run continues under the staged OOM fallback.

As of 2026-11-15 15:27 CET, the latest durable checkpoint is step 2,000 and combined pretraining is **117,964,800 / 1,000,000,000 tokens (11.80%)**. The latest batch 8 / accumulation 4 interval reported loss 4.5049, 8,228.0 tokens/second, 49% GPU utilization, and 8.24 GiB PyTorch peak allocation; the corrected estimate to one billion tokens is 107,199 seconds (about 29.8 hours) at this interval's rate. Checkpoint 2,000 was saved at 15:25:28 CET and the CUDA supervisor remains active.

As of 2026-11-15 15:29 CET, the latest durable checkpoint is step 2,100 and combined pretraining is **118,784,000 / 1,000,000,000 tokens (11.88%)**. Its last 100-step interval reported loss 4.5276, 8,790.8 tokens/second, 76% GPU utilization, and 8.24 GiB peak allocated VRAM; the corrected ETA was 100,243 seconds (about 27.8 hours) at that rate. I removed automatic deletion of old versioned checkpoints and changed the regression test to verify all saved versions remain. All 11 CPU tests and the PowerShell parser check passed. Training resumed on CUDA from step 2,100 at batch 8 / accumulation 4 at 15:28:06 CET. The prior saver had already removed older checkpoints; only steps 1,900–2,100 remain in the active 256-token run and steps 8,700–8,900 remain in the 512-token run. This fix prevents further automatic removal; deleted checkpoint versions could not be recovered.

As of 2026-11-15 15:31 CET, the latest durable checkpoint is step 2,200 and combined pretraining is **119,603,200 / 1,000,000,000 tokens (11.96%)**. Its interval reported loss 4.5127, 8,352.5 tokens/second, 78% GPU utilization, and 8.24 GiB peak allocated VRAM; the corrected one-billion-token ETA was 105,405 seconds (about 29.3 hours) at that rate. Step 2,200 was saved at 15:29:52 CET, and the checkpoint directory now retains steps 1,900, 2,000, 2,100, and 2,200 together. CUDA training is still active.

As of 2026-11-15 15:33 CET, the latest durable checkpoint is step 2,300 and combined pretraining is **120,422,400 / 1,000,000,000 tokens (12.04%)**. Its batch 8 / accumulation 4 interval reported loss 4.5411, 8,131.0 tokens/second, 87% GPU utilization, and 8.24 GiB PyTorch peak allocation; the corrected ETA was 108,176 seconds (about 30.0 hours) at that measured rate. Checkpoint 2,300 was saved at 15:31:35 CET. The active run retains all five checkpoints from steps 1,900 through 2,300, confirming that the saver no longer deletes older versions.

As of 2026-11-15 15:35 CET, the latest durable checkpoint is step 2,400 and combined pretraining is **121,241,600 / 1,000,000,000 tokens (12.12%)**. Its batch 8 / accumulation 4 interval reported loss 4.4938, 8,513.9 tokens/second, 94% GPU utilization, and 8.24 GiB PyTorch peak allocation; the corrected estimate to one billion tokens is 103,214 seconds (about 28.7 hours) at this interval's rate. Checkpoint 2,400 was saved at 15:33:09 CET. Six current checkpoint directories (steps 1,900–2,400) are retained and training continues on CUDA.

As of 2026-11-15 15:36 CET, the latest durable checkpoint is step 2,500 and combined pretraining is **122,060,800 / 1,000,000,000 tokens (12.21%)**. The latest interval reported loss 4.4598, 9,118.0 tokens/second, 83% GPU utilization, and 8.24 GiB PyTorch peak allocation; the corrected ETA to one billion tokens is 96,286 seconds (about 26.7 hours) at that rate. Checkpoint 2,500 was saved at 15:34:39 CET. Seven consecutive checkpoint directories, steps 1,900–2,500, are present.

As of 2026-11-15 15:38 CET, the latest durable checkpoint is step 2,600 and combined pretraining is **122,880,000 / 1,000,000,000 tokens (12.29%)**. The latest interval reported loss 4.5133, 8,157.6 tokens/second, 75% GPU utilization, and 8.24 GiB PyTorch peak allocation; its corrected ETA to one billion tokens is 107,522 seconds (about 29.9 hours) at that rate. Checkpoint 2,600 was saved at 15:36:19 CET. Eight consecutive checkpoint directories (1,900–2,600) are retained, and the CUDA supervisor remains active.

As of 2026-11-15 15:39 CET, the latest durable checkpoint is step 2,700 and combined pretraining is **123,699,200 / 1,000,000,000 tokens (12.37%)**. The latest interval reported loss 4.4837, 9,347.3 tokens/second, 77% GPU utilization, and 8.24 GiB PyTorch peak allocation; the corrected estimate to one billion tokens is 93,749 seconds (about 26.0 hours) at that rate. Checkpoint 2,700 was saved at 15:37:47 CET. Nine consecutive checkpoint directories (1,900–2,700) are retained, and training remains live on CUDA.

The first held-out per-language evaluation at step 5,000 (32 validation blocks per language) reported next-token losses: English 5.20, Spanish 4.76, German 4.77, French 4.34, Russian 4.52, Arabic 4.80, Hindi 1.67, Japanese 4.94, and Chinese 6.99. These are early-training loss measurements, not task-quality scores.

| Hardware and runtime | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12 GiB VRAM (12,282 MiB) |
| CPU | AMD Ryzen 9 7900X3D, 12 cores / 24 threads |
| System memory | 31.2 GiB RAM |
| Operating system | Windows 11 Pro |
| Training runtime | PyTorch 2.10.0+cu128, CUDA 12.8, NVIDIA driver 621.14 |

## Install

For a Windows PC with an NVIDIA GPU, use WSL2 and keep the repository and active training data in the Linux filesystem for faster I/O:

```bash
# In PowerShell, if WSL2 is not installed yet:
wsl --install -d Ubuntu

# In Ubuntu / WSL2:
git clone --recurse-submodules https://github.com/bxanedot/ryn.git ~/src/ryn
cd ~/src/ryn
python3 -m venv .venv
source .venv/bin/activate
```

Use the [official PyTorch install selector](https://pytorch.org/get-started/locally/) to install a CUDA-enabled build compatible with WSL2 and the installed NVIDIA driver, then install Ryn's remaining dependencies:

```bash
pip install -r requirements.txt
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA GPU")'
```

The training script requires CUDA and BF16 support by default. It uses a single process, fused AdamW, BF16 autocast, scaled dot product attention, and activation checkpointing. `configs/ryn0_2b.json` is a smaller fallback if the 300M model does not fit. No multi-GPU launcher is needed.

## Prepare data

Keep raw data and prepared JSONL outside Git. Use sources whose terms permit model training. The preparation tool validates format, filters duplicate examples and suspicious secret/PII patterns, creates a deterministic train/validation split, tags each row with source, language, and license, and writes a provenance manifest. Supply the same tokenizer used by training to count its actual tokens:

The included fetcher streams 1.3B tokenizer-counted tokens: 50% English from FineWeb's `sample-10BT` and equal shares of Spanish, French, German, Russian, Arabic, Hindi, Japanese, and Chinese from FineWeb2. It pins both dataset revisions and writes row-level source/license tags plus a source manifest. Both dataset cards state ODC-By 1.0; FineWeb also refers users to Common Crawl's Terms of Use, and the underlying crawled pages retain their own rights and privacy limitations. The subsequent filters are not a rights or privacy audit.

```bash
python tools/fetch_general_corpus.py --output ~/data/ryn/fineweb-raw.jsonl
```

The English sample and multilingual subsets are documented in the [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) and [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) cards. Review the output manifests before starting training.

```bash
python tools/prepare_text_data.py \
  --input ~/data/ryn/raw-pretrain \
  --output ~/data/ryn/pretrain.jsonl \
  --format text \
  --language en \
  --license "CC-BY-4.0 and public-domain sources; reviewed 2026-11-15" \
  --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
  --validation-fraction 0.05
```

The tool also writes `pretrain.validation.jsonl` and `pretrain.jsonl.manifest.json`. Large JSONL files stream by record; plain-text files stream into chunks up to `--max-bytes`. Existing row-level `source` and `license` values are preserved; missing values use the input filename and `--license`. For multilingual corpora, include a `language` field in each JSONL row. Check the manifest's per-language record and token counts; use `--language-weights '{"en":1,"es":1,"ar":1}'` in training to sample those languages evenly. For a single plain-text directory, pass the directory and `--language` once per language, or build one language-tagged JSONL corpus first.

For supervised fine-tuning, the OASST1 exporter reconstructs reviewed conversation paths from the Apache-2.0 dataset, pins its revision, and keeps eight languages used in Ryn's pretraining mix:

```bash
python tools/prepare_oasst_sft.py --output ~/data/ryn/oasst1-raw.jsonl
python tools/prepare_text_data.py \
  --input ~/data/ryn/oasst1-raw.jsonl \
  --output ~/data/ryn/oasst1-sft.jsonl \
  --format sft \
  --license "Apache-2.0; OpenAssistant/oasst1" \
  --tokenizer Qwen/Qwen2.5-0.5B-Instruct
```

The OASST1 dataset card describes human-generated, annotated message trees and its train/validation files. The exporter uses its train split and the preparation tool creates a deterministic held-out split.

The prepared snapshot contains **44,225 training** and **2,355 validation** records across eight languages. English and Spanish dominate; Arabic has 28 training records and Japanese 200, so the SFT data alone does not support balanced quality claims for those languages.

For any other supervised fine-tuning source, prepare a JSONL of MiniMind-style conversations with `--format sft`. Each row needs `conversations` or `messages`, and each conversation needs at least one assistant response. Only assistant message tokens contribute to training loss.

```json
{"language":"en","conversations":[{"role":"user","content":"Help me plan these tasks."},{"role":"assistant","content":"Start with the task that has the earliest deadline."}]}
```

```bash
python tools/prepare_text_data.py \
  --input ~/data/ryn/sft-source.jsonl \
  --output ~/data/ryn/sft.jsonl \
  --format sft \
  --license "Reviewed licensed instruction data" \
  --tokenizer Qwen/Qwen2.5-0.5B-Instruct
```

The filters are heuristic. They can miss secrets or personal information and can flag harmless numbers or addresses. Review sources and outputs before training. Do not use the tool as proof of data rights.

## Benchmark the 4070 Super

This measures real training steps using the supplied training corpus and does not save model checkpoints. It reports measured input tokens per second, peak allocated VRAM, sampled GPU utilization, loss, steps remaining, estimated time to one billion tokens, and corpus token count.

```bash
python scripts/train.py \
  --stage pretrain \
  --config configs/ryn0_3b.json \
  --data ~/data/ryn/pretrain.jsonl \
  --benchmark \
  --steps 20 \
  --max-seq-len 1024 \
  --batch-size 1 \
  --grad-accum 8
```

The benchmark does not start automatically. Change sequence length or batch size after measuring if CUDA reports out of memory. The script reports an OOM and suggests a shorter sequence or smaller batch; increasing accumulation keeps the effective batch size similar.

## Stage A: pretraining

The token cache under `.ryn-cache/` is built once per dataset, tokenizer, stage, and sequence length. It streams JSONL, packs consecutive tokens into fixed-size blocks, and uses memory-mapped files rather than loading the full corpus into Python objects. The cache can require several GB for a billion-token corpus; keep enough free disk space.

Start the one-billion-token run only when you are ready for the long local job:

```bash
python scripts/train.py \
  --stage pretrain \
  --config configs/ryn0_3b.json \
  --data ~/data/ryn/pretrain.jsonl \
  --validation-data ~/data/ryn/pretrain.validation.jsonl \
  --output-dir outputs/ryn-assistant-300m-base \
  --target-tokens 1000000000 \
  --max-seq-len 1024 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 1e-4 \
  --warmup-steps 2000 \
  --save-every 5000 \
  --eval-every 5000
```

The actual throughput and finish time depend on data and hardware; use measured output rather than advance estimates. Checkpoints are atomic, versioned directories containing weights, tokenizer, AdamW state, scheduler, progress, RNG state, and run settings. Each full checkpoint is several GB, so choose `--save-every` with available disk space in mind. The Python trainer has a KeyboardInterrupt handler that attempts to save its last completed optimizer step. When stopping through the PowerShell supervisor, verify the `Training stopped safely` message and updated `latest.json`; if either is absent, resume only from the last durable checkpoint.

## Resume and continue

Resume the exact run from its newest checkpoint by repeating the original settings and adding `--resume`:

```bash
python scripts/train.py \
  --stage pretrain \
  --config configs/ryn0_3b.json \
  --data ~/data/ryn/pretrain.jsonl \
  --validation-data ~/data/ryn/pretrain.validation.jsonl \
  --output-dir outputs/ryn-assistant-300m-base \
  --target-tokens 1000000000 \
  --max-seq-len 1024 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 1e-4 \
  --warmup-steps 2000 \
  --save-every 5000 \
  --eval-every 5000 \
  --resume
```

Exact resume rejects a changed model, corpus, tokenizer, sequence length, optimizer setup, or token target. To add data or change the training setup, start a new run from saved weights with `--init-from`; that creates fresh optimizer and scheduler state and leaves the original run intact.

## Stage B: supervised fine-tuning

Start a separate run from the latest base weights. Only assistant responses are trained; verify the examples and languages before starting.

```bash
python scripts/train.py \
  --stage sft \
  --config configs/ryn0_3b.json \
  --data ~/data/ryn/sft.jsonl \
  --validation-data ~/data/ryn/sft.validation.jsonl \
  --init-from outputs/ryn-assistant-300m-base \
  --output-dir outputs/ryn-assistant-300m-instruct \
  --steps 5000 \
  --max-seq-len 1024 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 2e-5 \
  --warmup-steps 200 \
  --save-every 1000
```

Further pretraining or fine-tuning sessions can use `--init-from` with a previous run directory or checkpoint. They get new versioned output directories and fresh optimizer state.

## Test and chat

CPU-compatible tests cover model shape and parameter count, config validation, synthetic packed data, assistant-only SFT masks, training loss/backward, checkpoint save/restore, and inference generation. They do not require downloading a large dataset or starting training:

```bash
python -m unittest discover -s tests -v
```

Chat with the latest checkpoint in a run directory:

```bash
python scripts/chat.py --model outputs/ryn-assistant-300m-instruct
python scripts/chat.py --model outputs/ryn-assistant-300m-instruct --prompt "Help me plan my day."
```

## Limitations and next steps

- This trainer is single-GPU. It does not use DDP or FSDP.
- Activation checkpointing covers the configured dense model. MoE is not enabled by the active config.
- Validation currently reports next-token loss overall and by language; task quality, factuality, and instruction-following still need separate evaluation sets.
- The preparation filters are not a complete privacy or license audit.
- Pretraining is being measured on an RTX 4070 SUPER. Language-by-language task quality, factuality, and instruction following still need evaluation after training.

**Earlier run snapshot (2026-11-15; superseded by the current run below):** The training corpus and validation split were prepared. CUDA pretraining resumed from exact durable checkpoint 2,700 at **2026-11-15 20:59:26 CET (Europe/Zurich)**; startup verified 295,825,920 parameters, tokenizer vocabulary 151,665, and CUDA execution. As of **21:12 CET**, the restarted run logged step 3,500: loss 4.4798, 9,290.9 input tokens/second, 83% sampled GPU utilization, and 8.24 GiB peak allocated VRAM. Logged lifetime progress is 130,252,800 / 1,000,000,000 tokens (13.03%); the rate-based ETA is 93,612 seconds (about 26 hours). Steps 2,800–3,500 in this run are not yet durable; `latest.json` remains at checkpoint 2,700, 123,699,200 tokens (12.37%), with the next scheduled save at step 15,000. The previous session had logged step 3,800 but did not save steps 2,800–3,800; those updates were lost when it stopped. See the Hardware section for the RTX 4070 SUPER, Ryzen 9 7900X3D, 31.2 GiB RAM, Windows 11 Pro, and CUDA/PyTorch versions. The code is backed up on private GitHub branch `codex/ryn-training-system`; private draft recovery release [ryn-backup-2026-11-15-step-2700](https://github.com/bxanedot/ryn/releases/tag/untagged-c4672d805125747a1c01) contains the exact-resume checkpoint and prepared pretraining/SFT data with SHA-256 manifests. Raw source downloads, rebuildable caches, logs, and older checkpoints are omitted, so it is a focused recovery bundle rather than the full 67.4 GiB tree. The local snapshot at `../ryn-preservation/RynData-20261115-1830` is Read & Execute only for the current account; the live `../ryn-data` tree is writable while training and should be relocked when paused. Both local copies are on C: and do not protect against drive failure. Earlier, raw and prepared JSONL files, four incomplete cache directories, and unused 512- and 1,024-token caches were compressed in place with unchanged SHA-256; the active 256-token cache remains uncompressed. Older compressed caches must be decompressed or rebuilt before memory-mapped reuse. After pretraining, run supervised fine-tuning, evaluate held-out loss and representative prompts by language, export the finished model package to Desktop, then replace the pending completion row above with the verified finish time and artifact path.

## Continued training run

- Restarted: **2026-11-16 00:39:17 CET (Europe/Zurich)** from the last durable 256-token checkpoint, step **2,700** (123,699,200 lifetime tokens; **12.37%** of the 1B-token target). The previous log reached step 5,000 (142,540,800 lifetime tokens), but `latest.json` remained at 2,700; steps 2,701–5,000 were repeated.
- Durable checkpoint: **step 5,000**, saved **2026-11-16 01:11 CET**; 142,540,800 / 1,000,000,000 lifetime tokens (**14.25%**), latest logged loss 4.3710, throughput 10,051 tokens/sec, peak VRAM 8.24 GiB. The rate-based estimate was 85,306 seconds (23.7 hours) remaining for pretraining only; SFT, evaluation, and Desktop export remain after it.
- Private GitHub recovery: the step-2,700 full-data bundle and step-5,000 incremental exact-resume bundle are in [the private draft release](https://github.com/bxanedot/ryn/releases/tag/untagged-c4672d805125747a1c01). GitHub-reported digests and sizes for both step-5,000 parts match the local SHA-256 manifest.
- Durable checkpoint: **step 10,000**, saved **2026-11-16 02:20 CET**; 183,500,800 / 1,000,000,000 lifetime tokens (**18.35%**), loss 4.2364, throughput 10,005 tokens/sec, peak VRAM 8.24 GiB. Validation loss by language is in `../ryn-data/pretrain-continuation-256.log`; the latest estimate was 81,605 seconds (22.7 hours) for pretraining only.
- The private draft release now includes exact-resume recovery bundles for steps **2,700, 5,000, and 10,000**: [Ryn 0.3B recovery release](https://github.com/bxanedot/ryn/releases/tag/untagged-c4672d805125747a1c01). GitHub-reported digests and sizes for step-10,000 parts match the local SHA-256 manifest.
- Durable checkpoint: **step 15,000**, saved **2026-11-16 03:29 CET**; 224,460,800 / 1,000,000,000 lifetime tokens (**22.45%**), loss 4.1457, throughput 9,727 tokens/sec, peak VRAM 8.24 GiB. The rate-based estimate was 79,730 seconds (22.1 hours) for pretraining only. Validation loss improved in all nine tracked languages; exact values are in the log.
- The private draft [recovery release](https://github.com/bxanedot/ryn/releases/tag/untagged-c4672d805125747a1c01) now contains steps **2,700, 5,000, 10,000, and 15,000**. GitHub-reported digests and sizes for step-15,000 parts match the local SHA-256 manifest.
- Durable checkpoint: **step 65,000**, saved **2026-11-16 14:35:12 CET**; 634,060,800 / 1,000,000,000 lifetime tokens (**63.41%**), loss 3.7492, throughput 10,500 tokens/sec, peak VRAM 8.24 GiB. The rate-based estimate then was 34,850 seconds (9.7 hours) for pretraining only.
- The same private draft [recovery release](https://github.com/bxanedot/ryn/releases/tag/untagged-c4672d805125747a1c01) now includes the step-65,000 incremental exact-resume bundle. GitHub-reported sizes and SHA-256 digests for both archive parts match the local manifest; the release remains private and unpublished.
- Durable checkpoint: **step 70,000**, saved **2026-11-16 15:47:21 CET**; 675,020,800 / 1,000,000,000 lifetime tokens (**67.50%**), loss 3.7185, throughput 9,055 tokens/sec, peak VRAM 8.24 GiB. The rate-based estimate then was 35,889 seconds (10.0 hours) for pretraining only.
- The same private draft [recovery release](https://github.com/bxanedot/ryn/releases/tag/untagged-c4672d805125747a1c01) now includes the step-70,000 incremental exact-resume bundle. GitHub-reported sizes and SHA-256 digests for both archive parts match the local manifest; the release remains private and unpublished.
- The same private draft [recovery release](https://github.com/bxanedot/ryn/releases/tag/untagged-c4672d805125747a1c01) now includes the step-75,000 incremental exact-resume bundle. GitHub-reported sizes and SHA-256 digests for both archive parts match the local manifest; the release remains private and unpublished.
- Paused **2026-11-16 16:40 CET (Europe/Zurich)**. The log reached step **73,400** (702,873,600 lifetime tokens; **70.29%**), but `latest.json` still points to the durable step-70,000 checkpoint (**67.50%**). Resume from step 70,000; steps 70,001–73,400 will be repeated. The step-70,000 recovery bundle is present in the private draft release above.
- Resumed **2026-11-16 20:31:58 CET (Europe/Zurich)** from durable checkpoint **70,000** (675,020,800 lifetime tokens; **67.50%**). The resumed log has reached step **70,100** (675,840,000 lifetime tokens; **67.58%**), with loss **3.7190**, **8,665 tokens/sec**, **93%** sampled GPU utilization, and **8.24 GiB** peak allocated VRAM. The latest durable checkpoint remains step 70,000; the run is repeating the unsaved steps 70,001–73,400 before continuing.
- Live progress observed **2026-11-16 20:52 CET (Europe/Zurich)**: step **71,300**, 685,670,400 / 1,000,000,000 lifetime tokens (**68.57%**), loss **3.6861**, **8,762 tokens/sec**, **60%** sampled GPU utilization, and **8.24 GiB** peak allocated VRAM. This is log progress; the latest durable checkpoint remains step **70,000** (**67.50%**).
- Durable checkpoint: **step 75,000**, saved **2026-11-16 21:45:56 CET**; 715,980,800 / 1,000,000,000 lifetime tokens (**71.60%**), loss **3.6913**, throughput **9,638 tokens/sec**, and peak VRAM **8.24 GiB**. Validation loss by language: Arabic 3.6083, Chinese 5.6350, German 3.7398, English 4.2416, French 3.2182, Hindi 1.3804, Japanese 3.7574, Russian 3.2180, and Spanish 3.7063.
- Restarted from durable step **75,000** at **2026-11-16 21:47:42 CET (Europe/Zurich)**. Startup verified **295,825,920 parameters** on CUDA; the active launcher now saves every **15,000 optimizer steps** and evaluates every **5,000**. C: had about **25.1 GiB free** at restart. Existing Ryn checkpoints and training data are retained.
- Save interval was **5,000 optimizer steps** through step 75,000; the current 15,000-step cadence reserves C: space for SFT checkpoints and Desktop export, at the cost of a larger possible replay window. Only C: is available for storage; it had **169 GiB free** at the 00:39 restart and about **28.4 GiB free** at the 20:31 restart.
- Hardware: NVIDIA GeForce RTX 4070 SUPER 12 GiB, AMD Ryzen 9 7900X3D, 31.2 GiB RAM, Windows 11 Pro, CUDA/PyTorch as recorded in Hardware above.
- Completion time and the verified Desktop artifact path will be recorded here after pretraining, SFT, evaluation, and export are finished.

## License and provenance

Ryn-authored source code is GPL-3.0-only. The MiniMind submodule remains under Apache-2.0; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Choose a checkpoint license only after reviewing training data and model components.
