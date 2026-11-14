# Ryn 300M multilingual assistant training plan

Ryn's active target is a roughly 300M-parameter general assistant trained from random weights with the MiniMind architecture. Start with a short run on the user's single GPU, then measure before scaling.

1. **Corpus preparation:** gather rights-cleared multilingual text with explicit language tags. Include natural language, reference material, and high-quality examples in the languages intended for support. Deduplicate, filter credentials, and retain source/license manifests.
2. **Pretraining:** train next-token prediction on the reviewed corpus. Track total tokens and a held-out validation loss by language.
3. **Instruction tuning:** cover writing, planning, organizing, explanations, summaries, translation, and everyday work. Include the plan/act/verify workflow in `assistant-behavior.md`; mask user/system text and learn assistant turns only.
4. **Tool-aware behavior:** train examples that distinguish available tools from unavailable ones. Internet search and fetching must be supplied by the runtime; the base weights cannot access the live web.
5. **Evaluation:** report language-by-language results, task success, factuality, instruction following, and citations for browsed answers. Publish exact data versions, token counts, hardware, steps, and known limitations with each checkpoint.

The 300M model is a small experiment. Its size alone does not establish reliable general-assistant or broad multilingual quality. No checkpoint should be described as capable until it has been trained and evaluated.
