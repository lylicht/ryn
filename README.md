# Ryn: multilingual 300M assistant

Ryn is a small, multilingual general assistant by **bxane**, built with the MiniMind architecture. The active model is `ryn-assistant-300m`: the exact configuration has **296,034,048 parameters**, or about 0.3 billion. It is aimed at everyday work, planning, writing, learning, and organization rather than specialized coding.

The model is initialized from random weights. The Qwen2.5 tokenizer is used only to tokenize text; no Qwen model weights are used. **No Ryn checkpoint has been trained or released.** A 300M model is a compact experiment and should not be expected to match current commercial assistants without substantial data, training, and evaluation.

## Intended behavior

Ryn should reply in the user's language when possible, help with writing and explanations, organize tasks and projects, summarize information, compare options, and turn goals into practical next steps. The training plan prioritizes multilingual coverage and a consistent real-world workflow. See [the behavior plan](docs/assistant-behavior.md).

Language coverage depends on the training corpus. Include well-reviewed examples across the languages you want to support, tag them by language, and publish per-language evaluation results. Do not claim support for “any language” until it has been measured.

### Internet access

Internet access is an optional runtime capability. The model weights do not browse the web. A host application may connect a search/fetch tool; only then can Ryn retrieve current information. The host should show sources and dates, and Ryn should say when browsing is unavailable rather than imply it searched.

## Model configuration

`configs/ryn0_3b.json` sets the model architecture and tokenizer. `configs/ryn0_2b.json` is a smaller fallback if the 296M model exceeds the available GPU memory. The MiniMind source is pinned as `upstream/minimind`.

## Setup

```bash
git clone --recurse-submodules https://github.com/bxanedot/ryn.git
cd ryn
python -m venv .venv
source .venv/bin/activate
```

Install a CUDA-enabled PyTorch build that matches the GPU host, then install the remaining packages:

```bash
pip install -r requirements.txt
```

The user's RTX 4070 Super has 12 GB of VRAM. The short, single-GPU pilot below is the starting point; watch peak VRAM and reduce sequence length if needed.

## Prepare licensed multilingual data

Keep training data outside Git. Use sources whose terms permit the intended training and release. For a directory of text and source files, run one pass per language and merge the generated JSONL files. For existing JSONL, tag each row with its language, for example `en`, `es`, `ar`, `hi`, `zh`, or `ja`.

```bash
python tools/prepare_text_data.py \
  --input /path/to/licensed-english-corpus \
  --output data/english.jsonl \
  --language en \
  --license "MIT and public-domain sources"
```

JSONL rows need a `text` field; `language` is preserved when present and defaults to the `--language` value. The tool normalizes line endings, deduplicates exact text, filters common generated directories and some credential patterns, and writes a provenance manifest. Review all sources and outputs manually; the filters do not establish licensing or remove every secret.

For supervised fine-tuning, use MiniMind conversation JSONL. Include examples for planning, organizing, explaining, writing, and summarizing in each target language:

```json
{"conversations":[{"role":"user","content":"Help me organize these tasks for today: reply to Sam, prepare the draft, and submit the form by 16:00."},{"role":"assistant","content":"Plan:\n1. Submit the form first; it has a 16:00 deadline.\n2. Prepare the draft.\n3. Reply to Sam.\nIf any task has a fixed duration or dependency, tell me and I can adjust the order."}]}
```

Only assistant messages contribute to SFT loss.

## Train the 300M model

Start with a short pretraining run on a reviewed multilingual corpus:

```bash
torchrun --standalone --nproc_per_node=1 scripts/train.py \
  --stage pretrain \
  --config configs/ryn0_3b.json \
  --data data/multilingual-pretrain.jsonl \
  --output-dir outputs/ryn-assistant-300m-base \
  --steps 2000 \
  --max-seq-len 1024 \
  --batch-size 1 \
  --grad-accum 8 \
  --save-every 500
```

Then instruction-tune on a reviewed, language-tagged assistant dataset:

```bash
torchrun --standalone --nproc_per_node=1 scripts/train.py \
  --stage sft \
  --config configs/ryn0_3b.json \
  --data data/multilingual-assistant-sft.jsonl \
  --init-from outputs/ryn-assistant-300m-base \
  --output-dir outputs/ryn-assistant-300m-instruct \
  --steps 1000 \
  --max-seq-len 1024 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 2e-5 \
  --save-every 250
```

These are pipeline pilots, not enough training to establish broad multilingual ability. Keep run metadata, token counts, per-language evaluations, and limitations with every checkpoint. Do not commit large weight files to ordinary Git history.

## License and provenance

Ryn-authored source code is GPL-3.0-only. The MiniMind submodule remains under Apache-2.0; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Choose a checkpoint license only after reviewing training data and model components.
