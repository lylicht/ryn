#!/usr/bin/env python3
"""Export reviewed OASST1 message paths as provenance-tagged Ryn SFT JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from hf_dataset_runtime import configure_hf_dataset_runtime


DATASET = "OpenAssistant/oasst1"
REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
LANGUAGES = {"en", "es", "fr", "de", "ru", "zh", "ja", "ar"}
ROLES = {"prompter": "user", "assistant": "assistant"}


def conversation_for(message_id: str, messages: dict[str, dict]) -> list[dict] | None:
    leaf = messages.get(message_id)
    if not leaf or leaf.get("role") != "assistant" or leaf.get("lang") not in LANGUAGES:
        return None
    language = leaf["lang"]
    chain = []
    seen = set()
    current = leaf
    while current:
        current_id = current.get("message_id")
        if current_id in seen or current.get("lang") != language or current.get("deleted") or current.get("review_result") is False:
            return None
        seen.add(current_id)
        role = ROLES.get(current.get("role"))
        text = current.get("text")
        if role is None or not isinstance(text, str) or not text.strip():
            return None
        chain.append({"role": role, "content": text.strip()})
        parent_id = current.get("parent_id")
        current = messages.get(parent_id) if parent_id else None
        if parent_id and current is None:
            return None
        if len(chain) > 32:
            return None
    chain.reverse()
    if not chain or chain[0]["role"] != "user" or chain[-1]["role"] != "assistant":
        return None
    if any(left["role"] == right["role"] for left, right in zip(chain, chain[1:])):
        return None
    return chain


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    manifest_path = output.with_suffix(output.suffix + ".source-manifest.json")
    if any(path.exists() for path in (output, partial, manifest_path)):
        parser.error("refusing to overwrite existing output, partial, or manifest")

    output.parent.mkdir(parents=True, exist_ok=True)
    configure_hf_dataset_runtime(output.parent / ".hf-home")
    from datasets import load_dataset

    dataset = load_dataset(DATASET, split="train", streaming=True, revision=REVISION, cache_dir=output.parent / ".hf-datasets-cache")
    messages = {}
    for row in dataset:
        if row.get("role") not in ROLES or row.get("lang") not in LANGUAGES or row.get("deleted") or row.get("review_result") is False:
            continue
        messages[row["message_id"]] = row

    counts = Counter()
    records = 0
    with partial.open("x", encoding="utf-8", newline="\n") as stream:
        for message_id, row in messages.items():
            if row.get("role") != "assistant":
                continue
            conversation = conversation_for(message_id, messages)
            if conversation is None:
                continue
            language = row["lang"]
            stream.write(json.dumps({
                "messages": conversation,
                "language": language,
                "source": f"{DATASET}@{REVISION}",
                "license": "Apache-2.0",
            }, ensure_ascii=False) + "\n")
            counts[language] += 1
            records += 1
        stream.flush()

    if not records:
        partial.unlink(missing_ok=True)
        raise RuntimeError("No usable assistant conversations were exported")
    partial.replace(output)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "revision": REVISION,
        "license": "Apache-2.0",
        "source_card": "https://huggingface.co/datasets/OpenAssistant/oasst1",
        "languages": sorted(LANGUAGES),
        "records": records,
        "records_by_language": dict(counts),
        "output": {"name": output.name, "sha256": sha256_file(output)},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), "records": records, "records_by_language": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
