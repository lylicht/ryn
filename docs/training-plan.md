# Ryn 9B training plan

Ryn's goal is a capable coding model, trained from random weights on a documented corpus. This plan is intentionally staged so quality can be measured before scaling up.

1. **Pretraining:** code and technical text from sources with explicit training rights; remove duplicates, credentials, generated artifacts, and low-quality boilerplate. Track token counts and source licenses.
2. **Instruction tuning:** high-quality coding conversations covering code generation, debugging, explanations, refactoring, tests, and tool-call formats. Train assistant targets only.
3. **Preference tuning:** collect reviewed comparisons for correctness, security, test quality, and instruction following. Apply DPO only after a stable SFT checkpoint exists.
4. **Evaluation:** publish exact dataset versions, token count, training steps, hardware, and results. Evaluate on held-out coding tasks with contamination checks and human review.

## Compute and quality limits

The dense 8.9B target is far beyond the MiniMind project's small educational runs. FSDP full sharding is required for practical multi-GPU training; the starter settings have not been profiled. Pretraining to frontier coding quality would require a large token budget and extensive compute. No model checkpoint or benchmark result is included until a real run produces it.

## Training data policy

Do not assume that publicly reachable code is licensed for model training. Use code owned by bxane or datasets with explicit terms compatible with the intended model release. Keep raw training data outside this repository, retain source records and licenses, and review generated/sensitive content before training.
