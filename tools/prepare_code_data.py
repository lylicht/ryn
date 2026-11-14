#!/usr/bin/env python3
"""Prepare a local, license-documented code corpus for MiniMind pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


CODE_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".html",
    ".java", ".js", ".jsx", ".kt", ".lua", ".md", ".php", ".py", ".rb",
    ".rs", ".scala", ".sh", ".sql", ".swift", ".toml", ".ts", ".tsx",
    ".vue", ".xml", ".yaml", ".yml", ".json",
}
SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "dist", "build",
    "target", "vendor", "coverage", "checkpoints", "outputs",
}
SKIP_NAMES = {".env", ".env.local", "credentials.json", "secrets.json", "id_rsa", "id_ed25519"}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
)


def iter_input_files(source: Path) -> Iterator[Path]:
    if source.is_file():
        yield source
        return

    for path in source.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in CODE_EXTENSIONS or path.suffix.lower() == ".jsonl":
            yield path


def read_records(path: Path) -> Iterator[str]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = record.get("text") if isinstance(record, dict) else None
                if isinstance(value, str):
                    yield value
        return

    try:
        yield path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return


def contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Code directory or text JSONL file")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    parser.add_argument("--license", required=True, help="License or license policy for this input corpus")
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-bytes", type=int, default=2_000_000)
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    if not source.exists():
        parser.error(f"input does not exist: {source}")
    if not args.license.strip():
        parser.error("--license must document the source corpus terms")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts = {"files_seen": 0, "records_written": 0, "duplicates": 0, "filtered": 0}

    with args.output.open("w", encoding="utf-8") as output:
        for path in iter_input_files(source):
            counts["files_seen"] += 1
            if path.name.lower() in SKIP_NAMES or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
                counts["filtered"] += 1
                continue
            try:
                if path.stat().st_size > args.max_bytes:
                    counts["filtered"] += 1
                    continue
            except OSError:
                counts["filtered"] += 1
                continue

            for raw_text in read_records(path):
                text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
                if len(text) < args.min_chars or contains_secret(text):
                    counts["filtered"] += 1
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(digest)
                output.write(json.dumps({"text": text, "source": path.name}, ensure_ascii=False) + "\n")
                counts["records_written"] += 1

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_name": source.name,
        "license_or_policy": args.license,
        "output_name": args.output.name,
        "counts": counts,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "notice": "Review source rights and records manually; filtering is not a license or secret audit.",
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": str(manifest_path), **counts}, indent=2))


if __name__ == "__main__":
    main()
