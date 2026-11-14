#!/usr/bin/env python3
"""Prepare a license-documented multilingual text corpus for MiniMind."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


TEXT_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".html", ".java",
    ".js", ".jsx", ".kt", ".kts", ".lua", ".md", ".php", ".py", ".rb", ".rst",
    ".rs", ".scala", ".sh", ".sql", ".swift", ".toml", ".ts", ".tsx", ".txt", ".vue",
    ".xml", ".yaml", ".yml",
}
SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "dist", "build", "target",
    "vendor", "coverage", "checkpoints", "outputs", ".gradle", ".kotlin", "Pods", "DerivedData",
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
        if source.suffix.lower() == ".jsonl" or source.suffix.lower() in TEXT_EXTENSIONS:
            yield source
        return

    for path in source.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() == ".jsonl" or path.suffix.lower() in TEXT_EXTENSIONS or path.name in {"gradle.properties", "libs.versions.toml"}:
            yield path


def read_records(path: Path, default_language: str) -> Iterator[tuple[str, str]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    continue
                language = str(record.get("language") or default_language).strip().lower()
                yield record["text"], language
        return

    try:
        yield path.read_text(encoding="utf-8", errors="replace"), default_language
    except OSError:
        return


def contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Text/source directory or JSONL file")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    parser.add_argument("--language", default="und", help="Language tag for untagged input, e.g. en, es, ar")
    parser.add_argument("--license", required=True, help="License or license policy for the input corpus")
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-bytes", type=int, default=2_000_000)
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    if not source.exists():
        parser.error(f"input does not exist: {source}")
    if not args.license.strip():
        parser.error("--license must document the source corpus terms")
    if args.min_chars < 1 or args.max_bytes < 1:
        parser.error("min-chars and max-bytes must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_path = args.output.expanduser().resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    seen: set[str] = set()
    counts = {"files_seen": 0, "records_written": 0, "duplicates": 0, "filtered": 0}

    with output_path.open("w", encoding="utf-8") as output:
        for path in iter_input_files(source):
            if path.resolve() in {output_path, manifest_path}:
                continue
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

            for raw_text, language in read_records(path, args.language):
                text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
                if len(text) < args.min_chars or contains_secret(text):
                    counts["filtered"] += 1
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(digest)
                record = {"text": text, "source": path.name, "language": language}
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["records_written"] += 1

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_name": source.name,
        "focus": "multilingual general assistant",
        "default_language": args.language,
        "license_or_policy": args.license,
        "output_name": output_path.name,
        "counts": counts,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "notice": "Review source rights and records manually; filtering is not a license or complete secret audit.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "manifest": str(manifest_path), **counts}, indent=2))


if __name__ == "__main__":
    main()
