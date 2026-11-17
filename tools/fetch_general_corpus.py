#!/usr/bin/env python3
"""Stream a pinned English and multilingual FineWeb corpus to provenance-tagged JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from hf_dataset_runtime import configure_hf_dataset_runtime


FINEWEB_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
FINEWEB2_REVISION = "af9c13333eb981300149d5ca60a8e9d659b276b9"
SOURCES = [
    ("en", "HuggingFaceFW/fineweb", "sample-10BT", FINEWEB_REVISION, 0.50),
    ("spa_Latn", "HuggingFaceFW/fineweb-2", "spa_Latn", FINEWEB2_REVISION, 0.0625),
    ("fra_Latn", "HuggingFaceFW/fineweb-2", "fra_Latn", FINEWEB2_REVISION, 0.0625),
    ("deu_Latn", "HuggingFaceFW/fineweb-2", "deu_Latn", FINEWEB2_REVISION, 0.0625),
    ("rus_Cyrl", "HuggingFaceFW/fineweb-2", "rus_Cyrl", FINEWEB2_REVISION, 0.0625),
    ("arb_Arab", "HuggingFaceFW/fineweb-2", "arb_Arab", FINEWEB2_REVISION, 0.0625),
    ("hin_Deva", "HuggingFaceFW/fineweb-2", "hin_Deva", FINEWEB2_REVISION, 0.0625),
    ("jpn_Jpan", "HuggingFaceFW/fineweb-2", "jpn_Jpan", FINEWEB2_REVISION, 0.0625),
    ("cmn_Hani", "HuggingFaceFW/fineweb-2", "cmn_Hani", FINEWEB2_REVISION, 0.0625),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-tokens", type=int, default=1_300_000_000)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    manifest_path = output.with_suffix(output.suffix + ".source-manifest.json")
    if args.target_tokens < len(SOURCES) or any(path.exists() for path in (output, partial, manifest_path)):
        parser.error("target must be positive and output/partial/manifest paths must not already exist")

    output.parent.mkdir(parents=True, exist_ok=True)
    configure_hf_dataset_runtime(output.parent / ".hf-home")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    targets = [int(args.target_tokens * source[4]) for source in SOURCES]
    targets[-1] += args.target_tokens - sum(targets)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, cache_dir=Path(__file__).resolve().parents[1] / ".hf-cache")
    tokenizer_hash = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode("utf-8")).hexdigest()
    eos_id = tokenizer.eos_token_id
    results = []

    with partial.open("x", encoding="utf-8", newline="\n") as stream:
        for (language, repo, config, revision, _), target in zip(SOURCES, targets):
            print(f"fetching={language} dataset={repo}/{config}@{revision}", flush=True)
            dataset = load_dataset(repo, name=config, split="train", streaming=True, revision=revision, cache_dir=output.parent / ".hf-datasets-cache")
            count, records = 0, 0
            started = time.monotonic()
            next_log = 25_000_000
            for batch in dataset.iter(batch_size=128):
                texts = [text.strip() if isinstance(text, str) else "" for text in batch.get("text", [])]
                encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
                for text, token_ids in zip(texts, encoded):
                    if count >= target:
                        break
                    if not isinstance(text, str) or len(text) < 80 or len(text.encode("utf-8")) > 1_900_000:
                        continue
                    count += len(token_ids) + int(eos_id is not None and (not token_ids or token_ids[-1] != eos_id))
                    records += 1
                    stream.write(json.dumps({
                        "text": text,
                        "language": language.lower(),
                        "source": f"{repo}/{config}@{revision}",
                        "license": "ODC-By 1.0; Common Crawl Terms of Use",
                    }, ensure_ascii=False) + "\n")
                    if count >= next_log:
                        elapsed = max(time.monotonic() - started, 1e-6)
                        print(f"language={language} tokens={count:,}/{target:,} records={records:,} tokens_per_sec={count / elapsed:.0f}", flush=True)
                        next_log += 25_000_000
                stream.flush()
                if count >= target:
                    break
            if count < target:
                raise RuntimeError(f"{repo}/{config} ended at {count:,} of {target:,} tokenizer tokens")
            results.append({"repo": repo, "config": config, "revision": revision, "language": language, "target_tokens": target, "actual_tokens": count, "records": records})
            print(f"language={language} tokens={count:,}/{target:,} records={records:,}", flush=True)

    partial.replace(output)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": args.tokenizer,
        "tokenizer_sha256": tokenizer_hash,
        "target_tokens": args.target_tokens,
        "actual_tokens": sum(item["actual_tokens"] for item in results),
        "sources": results,
        "license_and_terms": "FineWeb/FineWeb2 ODC-By 1.0; use also subject to Common Crawl Terms of Use. Underlying crawled page content retains its own rights and privacy limits.",
        "dataset_cards": [
            "https://huggingface.co/datasets/HuggingFaceFW/fineweb",
            "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), "actual_tokens": manifest["actual_tokens"]}, indent=2))


if __name__ == "__main__":
    main()
