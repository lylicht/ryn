#!/usr/bin/env python3
"""Audit, deduplicate, split, and provenance-tag licensed Ryn training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".html", ".java",
    ".js", ".jsx", ".kt", ".kts", ".lua", ".md", ".php", ".py", ".rb", ".rst",
    ".rs", ".scala", ".sh", ".sql", ".swift", ".toml", ".ts", ".tsx", ".txt", ".vue",
    ".xml", ".yaml", ".yml",
}
SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "dist", "build", "target",
    "vendor", "coverage", "checkpoints", "outputs", ".ryn-cache", "__pycache__",
    ".gradle", ".kotlin", "Pods", "DerivedData",
}
SKIP_NAMES = {".env", ".env.local", "credentials.json", "secrets.json", "id_rsa", "id_ed25519"}
SENSITIVE_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"(?<!\w)(?:\+?\d[\d(). -]{8,}\d)(?!\w)"),
}


def iter_input_files(source: Path) -> Iterator[Path]:
    if source.is_file():
        if source.suffix.lower() == ".jsonl" or source.suffix.lower() in TEXT_EXTENSIONS:
            yield source
        return
    for path in source.rglob("*"):
        if path.is_file() and not any(part in SKIP_DIRS for part in path.parts):
            if path.suffix.lower() == ".jsonl" or path.suffix.lower() in TEXT_EXTENSIONS or path.name in {"gradle.properties", "libs.versions.toml"}:
                yield path


def iter_plain_text(path: Path, max_bytes: int) -> Iterator[str]:
    buffer = []
    buffer_bytes = 0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line_bytes = len(line.encode("utf-8"))
            if buffer and buffer_bytes + line_bytes > max_bytes:
                yield "".join(buffer)
                buffer, buffer_bytes = [], 0
            if line_bytes > max_bytes:
                # shortcut: oversized lines split at character boundaries; upgrade to paragraph chunks if this harms corpus quality.
                chunk_chars = max(1, max_bytes // 4)
                for start in range(0, len(line), chunk_chars):
                    if buffer:
                        yield "".join(buffer)
                        buffer, buffer_bytes = [], 0
                    yield line[start:start + chunk_chars]
            else:
                buffer.append(line)
                buffer_bytes += line_bytes
    if buffer:
        yield "".join(buffer)


def render_sft(row: dict, path: Path, line_number: int) -> tuple[list[dict], str]:
    messages = row.get("conversations", row.get("messages"))
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"SFT row in {path.name}:{line_number} needs a non-empty conversations/messages array")
    normalized = []
    rendered = []
    assistant_target = False
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Invalid SFT message in {path.name}:{line_number}")
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        role = message["role"]
        normalized.append({**message, "role": role, "content": content})
        rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        assistant_target |= role == "assistant" and bool(content.strip())
    if not assistant_target:
        raise ValueError(f"SFT row in {path.name}:{line_number} contains no assistant target text")
    return normalized, "".join(rendered)


def sensitive_kinds(text: str) -> list[str]:
    return [name for name, pattern in SENSITIVE_PATTERNS.items() if pattern.search(text)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def queue_record(path, source_root, args, rendered, language, normalized_row, pending, seen, stats, write_pending, train_stream, validation_stream, record_source=None, record_license=None):
    if len(rendered) < args.min_chars:
        stats["filtered_short"] += 1
        return
    kinds = sensitive_kinds(rendered)
    if kinds:
        stats["filtered_sensitive_records"] += 1
        for kind in kinds:
            stats[f"suspicious_{kind}"] += 1
        return
    digest = hashlib.sha256((language + "\0" + rendered).encode("utf-8")).hexdigest()
    if digest in seen:
        stats["duplicates"] += 1
        return
    seen.add(digest)
    bucket = int(digest[:12], 16) / 0xFFFFFFFFFFFF
    split = "validation" if stats["accepted"] and args.validation_fraction and bucket < args.validation_fraction else "train"
    stats["accepted"] += 1
    try:
        source_name = str(record_source).strip() if record_source else path.relative_to(source_root).as_posix()
    except ValueError:
        source_name = path.name
    record = {
        **normalized_row,
        "source": source_name,
        "language": language,
        "license": str(record_license or args.license).strip(),
    }
    pending.append((record, rendered, split))
    if len(pending) >= 128:
        write_pending(train_stream, validation_stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Text/source directory or JSONL file")
    parser.add_argument("--output", required=True, type=Path, help="Training JSONL output path")
    parser.add_argument("--format", choices=("text", "sft"), default="text")
    parser.add_argument("--language", default="und", help="Language tag for untagged input")
    parser.add_argument("--license", required=True, help="License or license policy for this corpus")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer name/path for exact token counts")
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-bytes", type=int, default=2_000_000, help="Maximum JSONL record or plain-text chunk size")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    validation_output = output.with_name(f"{output.stem}.validation{output.suffix}")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if not source.exists() or not source.is_dir() and not source.is_file():
        parser.error(f"input does not exist: {source}")
    if not args.license.strip():
        parser.error("--license must document source terms")
    if args.min_chars < 1 or args.max_bytes < 1 or not 0 <= args.validation_fraction < 1:
        parser.error("min-chars/max-bytes must be positive and validation-fraction must be in [0, 1)")
    if output in {source, validation_output, manifest_path}:
        parser.error("output files must not replace the input")
    paths = sorted(iter_input_files(source))
    if not paths:
        parser.error("input contains no supported text or JSONL files")
    targets = [output, manifest_path] + ([validation_output] if args.validation_fraction else [])
    if any(path.exists() for path in targets):
        parser.error("refusing to overwrite existing outputs: " + ", ".join(str(path) for path in targets if path.exists()))

    tokenizer = None
    tokenizer_sha256 = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            parser.error(f"transformers is required for token counting: {exc}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, cache_dir=ROOT / ".hf-cache")
        tokenizer_sha256 = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode("utf-8")).hexdigest()

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(fd)
    tmp_train = Path(tmp_name)
    if args.validation_fraction:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{validation_output.name}.", dir=output.parent)
        os.close(fd)
        tmp_val = Path(tmp_name)
    else:
        tmp_val = None
    stats = Counter()
    language_counts = defaultdict(Counter)
    token_counts = defaultdict(Counter)
    license_counts = Counter()
    seen: set[str] = set()
    pending: list[tuple[dict, str, str]] = []

    def write_pending(train_stream, validation_stream):
        if not pending:
            return
        encodings = tokenizer([item[1] for item in pending], add_special_tokens=False)["input_ids"] if tokenizer else [None] * len(pending)
        for (record, rendered, split), ids in zip(pending, encodings):
            stream = validation_stream if split == "validation" else train_stream
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats[f"{split}_records"] += 1
            language = record["language"]
            language_counts[split][language] += 1
            license_counts[record["license"]] += 1
            if ids is not None:
                count = len(ids)
                if args.format == "text":
                    count += int(tokenizer.bos_token_id is not None)
                    count += int(tokenizer.eos_token_id is not None and (not ids or ids[-1] != tokenizer.eos_token_id))
                token_counts[split][language] += count
        pending.clear()

    try:
        with tmp_train.open("w", encoding="utf-8") as train_stream:
            val_context = tmp_val.open("w", encoding="utf-8") if tmp_val else None
            try:
                for path in paths:
                    if path.name.lower() in SKIP_NAMES or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
                        stats["filtered_sensitive_filename"] += 1
                        continue
                    try:
                        path.stat()
                    except OSError:
                        stats["unreadable_files"] += 1
                        continue
                    if path.suffix.lower() == ".jsonl":
                        source_stream = path.open("r", encoding="utf-8", errors="replace")
                        with source_stream:
                            source_rows = ((line_number, line) for line_number, line in enumerate(source_stream, 1))
                            for line_number, line in source_rows:
                                if len(line.encode("utf-8")) > args.max_bytes:
                                    stats["filtered_oversize"] += 1
                                    continue
                                try:
                                    row = json.loads(line)
                                except json.JSONDecodeError:
                                    stats["invalid_json"] += 1
                                    continue
                                if not isinstance(row, dict):
                                    stats["invalid_format"] += 1
                                    continue
                                source_language = str(row.get("language") or args.language).strip().lower() or "und"
                                if args.format == "text":
                                    text = row.get("text")
                                    if not isinstance(text, str):
                                        stats["invalid_format"] += 1
                                        continue
                                    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
                                    normalized_row = {"text": text}
                                    rendered = text
                                else:
                                    try:
                                        messages, rendered = render_sft(row, path, line_number)
                                    except ValueError:
                                        stats["invalid_format"] += 1
                                        continue
                                    normalized_row = {"conversations": messages}
                                queue_record(
                                    path, source, args, rendered, source_language, normalized_row, pending, seen,
                                    stats, write_pending, train_stream, val_context,
                                    record_source=row.get("source"), record_license=row.get("license"),
                                )
                    elif args.format == "text":
                        for text in iter_plain_text(path, args.max_bytes):
                            text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
                            queue_record(path, source, args, text, args.language.strip().lower() or "und", {"text": text}, pending, seen, stats, write_pending, train_stream, val_context)
                    else:
                        stats["invalid_format"] += 1
                write_pending(train_stream, val_context)
            finally:
                if val_context:
                    val_context.close()

        if stats["train_records"] == 0:
            raise ValueError("No usable training records remained after validation and filtering")
        os.replace(tmp_train, output)
        if tmp_val and stats["validation_records"]:
            os.replace(tmp_val, validation_output)
        elif tmp_val:
            tmp_val.unlink(missing_ok=True)
        source_manifest = [{"name": path.relative_to(source).as_posix() if source.is_dir() else path.name, "sha256": sha256_file(path)} for path in paths]
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_name": source.name,
            "source_files": source_manifest,
            "format": args.format,
            "default_language": args.language,
            "license_or_policy": args.license.strip(),
            "tokenizer": args.tokenizer,
            "tokenizer_sha256": tokenizer_sha256,
            "validation_fraction": args.validation_fraction,
            "outputs": {
                "train": {"name": output.name, "sha256": sha256_file(output)},
                "validation": {"name": validation_output.name, "sha256": sha256_file(validation_output)} if validation_output.exists() else None,
            },
            "counts": dict(stats),
            "records_by_language": {split: dict(counts) for split, counts in language_counts.items()},
            "records_by_license": dict(license_counts),
            "tokenizer_tokens_by_language": {split: dict(counts) for split, counts in token_counts.items()} if tokenizer else None,
            "notice": "Suspicious patterns are filtered heuristically. Review source rights, provenance, samples, and outputs manually; filtering is not a license or complete privacy audit.",
        }
        tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        tmp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_manifest, manifest_path)
        print(json.dumps({"output": str(output), "validation_output": str(validation_output) if validation_output.exists() else None, "manifest": str(manifest_path), **dict(stats)}, indent=2))
    except BaseException:
        tmp_train.unlink(missing_ok=True)
        if tmp_val:
            tmp_val.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
