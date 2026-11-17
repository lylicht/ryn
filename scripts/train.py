#!/usr/bin/env python3
"""Single-GPU MiniMind pretraining and assistant fine-tuning for Ryn."""

from __future__ import annotations

import argparse
import array
import bisect
import glob
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import MethodType

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "minimind"))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


def expand_paths(values: list[str]) -> list[Path]:
    paths = sorted({Path(p).resolve() for value in values for p in glob.glob(value)})
    if not paths:
        paths = [Path(p).expanduser().resolve() for p in values if Path(p).exists()]
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
        elif path.is_file():
            files.append(path)
    files = sorted(set(files))
    if not files:
        raise ValueError(f"No JSONL files found in: {', '.join(values)}")
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_signature(paths: list[Path]) -> list[dict[str, object]]:
    return [{"path": str(path), "sha256": sha256_file(path)} for path in paths]


def tokenizer_signature(tokenizer) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    payload = backend.to_str().encode("utf-8") if backend is not None else repr(tokenizer.get_vocab()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_model_config(raw: dict, vocab_size: int) -> None:
    required = (
        "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "intermediate_size", "max_position_embeddings", "rope_theta",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Model config is missing: {', '.join(missing)}")
    if any(int(raw[key]) < 1 for key in required[:-1]) or float(raw["rope_theta"]) <= 0:
        raise ValueError("Model dimensions and rope_theta must be positive")
    if raw["hidden_size"] != raw["num_attention_heads"] * raw["head_dim"]:
        raise ValueError("hidden_size must equal num_attention_heads * head_dim for MiniMind")
    if raw["num_attention_heads"] % raw["num_key_value_heads"]:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    expected_vocab = raw.get("tokenizer_vocab_size_expected")
    if expected_vocab is not None and vocab_size != expected_vocab:
        raise ValueError(f"Tokenizer has {vocab_size} tokens; config expects {expected_vocab}")


def build_config(raw: dict, tokenizer) -> MiniMindConfig:
    validate_model_config(raw, len(tokenizer))
    return MiniMindConfig(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        use_moe=bool(raw.get("use_moe", False)),
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        intermediate_size=raw["intermediate_size"],
        max_position_embeddings=raw["max_position_embeddings"],
        rope_theta=raw["rope_theta"],
        tie_word_embeddings=raw.get("tie_word_embeddings", True),
    )


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_path_from(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if (path / "checkpoints").is_dir() and latest_checkpoint(path):
        path = latest_checkpoint(path)
    elif (path / "latest.json").is_file():
        latest = json.loads((path / "latest.json").read_text(encoding="utf-8"))["checkpoint"]
        path = path / latest
    if (path / "model").is_dir():
        return path / "model"
    if (path / "config.json").is_file():
        return path
    raise ValueError(f"Could not find a MiniMind model in {path}")


def tokenizer_path_from(model_path: str | Path) -> Path:
    model_path = Path(model_path).expanduser().resolve()
    for parent in model_path.parents:
        candidate = parent / "tokenizer"
        if candidate.is_dir():
            return candidate
    return model_path


def read_records(paths: list[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object in {path}:{line_number}")
                yield path, line_number, row


def sft_segments(row: dict, path: Path, line_number: int) -> tuple[str, list[tuple[str, bool]]]:
    messages = row.get("conversations", row.get("messages"))
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"SFT row in {path}:{line_number} needs a non-empty conversations/messages array")
    language = str(row.get("language", "und")).strip().lower() or "und"
    segments = []
    has_target = False
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Invalid SFT message in {path}:{line_number}")
        role = message["role"]
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        segments.append((f"<|im_start|>{role}\n{content}<|im_end|>\n", role == "assistant"))
        has_target |= role == "assistant" and bool(content)
    if not has_target:
        raise ValueError(f"SFT row in {path}:{line_number} contains no assistant target text")
    return language, segments


def safe_language(value: str) -> str:
    clean = re.sub(r"[^a-z0-9_-]", "_", value.lower())[:32]
    return clean or "und"


def _cache_key(paths: list[Path], tokenizer, stage: str, seq_len: int) -> tuple[str, dict]:
    manifest = {
        "sources": source_signature(paths),
        "tokenizer_sha256": tokenizer_signature(tokenizer),
        "tokenizer_name": getattr(tokenizer, "name_or_path", "custom"),
        "vocab_size": len(tokenizer),
        "stage": stage,
        "sequence_length": seq_len,
    }
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return key, manifest


def build_token_cache(paths: list[Path], tokenizer, stage: str, seq_len: int, cache_root: Path) -> Path:
    key, source_manifest = _cache_key(paths, tokenizer, stage, seq_len)
    vocab_size = len(tokenizer)
    cache = cache_root / key
    manifest_file = cache / "manifest.json"
    if manifest_file.is_file():
        existing = json.loads(manifest_file.read_text(encoding="utf-8"))
        if existing.get("fingerprint") == key:
            return cache

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{key}.tmp-", dir=cache_root))
    token_files: dict[str, object] = {}
    mask_files: dict[str, object] = {}
    buffers: dict[str, array.array] = defaultdict(lambda: array.array("i"))
    mask_buffers: dict[str, bytearray] = defaultdict(bytearray)
    block_counts: dict[str, int] = defaultdict(int)
    token_counts: dict[str, int] = defaultdict(int)
    text_batch: list[str] = []
    batch_meta: list[tuple[str, bool]] = []
    sft_rows: list[tuple[str, list[tuple[str, bool]]]] = []
    duplicate_hashes: set[str] = set()
    duplicate_count = 0
    rows_seen = 0

    def flush_language(language: str) -> None:
        buffer, mask = buffers[language], mask_buffers[language]
        while len(buffer) >= seq_len:
            ids = buffer[:seq_len]
            labels = mask[:seq_len]
            # MiniMind shifts labels internally, so each block overlaps its neighbor by one token.
            if stage == "pretrain" or any(labels[1:]):
                if language not in token_files:
                    token_files[language] = (temporary / f"tokens-{language}.i32").open("ab")
                    mask_files[language] = (temporary / f"mask-{language}.u8").open("ab")
                token_files[language].write(ids.tobytes())
                mask_files[language].write(bytes(labels))
                block_counts[language] += 1
            del buffer[: seq_len - 1]
            del mask[: seq_len - 1]
    def append_sequence(language: str, ids: list[int], labels: list[bool]) -> None:
        lang = safe_language(language)
        if len(ids) != len(labels):
            raise ValueError("Tokenizer returned mismatched ids and labels")
        if ids and (min(ids) < 0 or max(ids) >= vocab_size):
            raise ValueError(f"Tokenizer produced an out-of-range token id for language {lang}")
        token_counts[lang] += len(ids)
        buffers[lang].extend(ids)
        mask_buffers[lang].extend(1 if label else 0 for label in labels)
        flush_language(lang)

    def flush_text_batch() -> None:
        nonlocal text_batch, batch_meta
        if not text_batch:
            return
        encoded = tokenizer(text_batch, add_special_tokens=False)["input_ids"]
        if len(encoded) != len(batch_meta):
            raise ValueError("Tokenizer returned a different number of rows than requested")
        for ids, (language, assistant) in zip(encoded, batch_meta):
            ids = list(ids)
            eos = tokenizer.eos_token_id
            if stage == "pretrain" and tokenizer.bos_token_id is not None:
                ids.insert(0, tokenizer.bos_token_id)
            if eos is not None and (not ids or ids[-1] != eos):
                ids.append(eos)
            append_sequence(language, ids, [assistant] * len(ids))
        text_batch, batch_meta = [], []

    def flush_sft_rows() -> None:
        nonlocal sft_rows
        if not sft_rows:
            return
        all_segments = [segment for _, segments in sft_rows for segment, _ in segments]
        encoded = tokenizer(all_segments, add_special_tokens=False)["input_ids"]
        index = 0
        for language, segments in sft_rows:
            ids, labels = [], []
            for _, assistant in segments:
                segment_ids = list(encoded[index])
                ids.extend(segment_ids)
                labels.extend([assistant] * len(segment_ids))
                index += 1
            append_sequence(language, ids, labels)
        sft_rows = []

    try:
        for path, line_number, row in read_records(paths):
            rows_seen += 1
            if stage == "pretrain":
                text = row.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Pretraining row in {path}:{line_number} needs non-empty text")
                language = safe_language(str(row.get("language", "und")))
                digest = hashlib.sha256((language + "\0" + text.strip()).encode("utf-8")).hexdigest()
                if digest in duplicate_hashes:
                    duplicate_count += 1
                    continue
                duplicate_hashes.add(digest)
                text_batch.append(text.strip())
                batch_meta.append((language, True))
                if len(text_batch) >= 256:
                    flush_text_batch()
            else:
                language, segments = sft_segments(row, path, line_number)
                rendered = "".join(segment for segment, _ in segments)
                digest = hashlib.sha256((safe_language(language) + "\0" + rendered).encode("utf-8")).hexdigest()
                if digest in duplicate_hashes:
                    duplicate_count += 1
                    continue
                duplicate_hashes.add(digest)
                sft_rows.append((language, segments))
                if len(sft_rows) >= 64:
                    flush_sft_rows()
        flush_text_batch()
        flush_sft_rows()
        for language in list(buffers):
            flush_language(language)
        for stream in [*token_files.values(), *mask_files.values()]:
            stream.flush()
            stream.close()
        total_blocks = sum(block_counts.values())
        if rows_seen == 0 or total_blocks == 0:
            raise ValueError("Dataset has no complete trainable sequences; add data or reduce --max-seq-len")
        manifest = {
            "fingerprint": key,
            **source_manifest,
            "rows_seen": rows_seen,
            "duplicate_rows_skipped": duplicate_count,
            "blocks_by_language": dict(block_counts),
            "tokens_by_language": dict(token_counts),
            "block_count": total_blocks,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        try:
            os.replace(temporary, cache)
        except FileExistsError:
            shutil.rmtree(temporary)
    except BaseException:
        for stream in [*token_files.values(), *mask_files.values()]:
            if not stream.closed:
                stream.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return cache


class PackedBlocks:
    """Read packed blocks through memory maps; only each requested batch reaches RAM."""

    def __init__(self, cache: Path, seq_len: int, stage: str):
        self.cache, self.seq_len, self.stage = cache, seq_len, stage
        self.manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        self.languages = sorted(self.manifest["blocks_by_language"])
        self.ranges = []
        self.tokens, self.masks = {}, {}
        start = 0
        for language in self.languages:
            count = self.manifest["blocks_by_language"][language]
            self.ranges.append((start, start + count, language))
            start += count
            self.tokens[language] = torch.from_file(
                str(cache / f"tokens-{language}.i32"), shared=False,
                size=count * seq_len, dtype=torch.int32,
            ).view(count, seq_len)
            if stage == "sft":
                self.masks[language] = torch.from_file(
                    str(cache / f"mask-{language}.u8"), shared=False,
                    size=count * seq_len, dtype=torch.uint8,
                ).view(count, seq_len)
        self.ends = [end for _, end, _ in self.ranges]
        self.language_counts = {lang: end - start for start, end, lang in self.ranges}
        self.corpus_tokens = sum(self.manifest["tokens_by_language"].values())

    def __len__(self):
        return self.ranges[-1][1] if self.ranges else 0

    def __getitem__(self, index: int):
        range_index = bisect.bisect_right(self.ends, index)
        start, _, language = self.ranges[range_index]
        local_index = index - start
        input_ids = self.tokens[language][local_index].to(torch.long)
        if self.stage == "pretrain":
            return input_ids, input_ids.clone()
        mask = self.masks[language][local_index].bool()
        labels = input_ids.masked_fill(~mask, -100)
        return input_ids, labels


def enable_activation_checkpointing(model) -> None:
    if model.config.use_moe:
        raise ValueError("Activation checkpointing is currently limited to the configured dense MiniMind model")
    base = model.model
    original_forward = base.forward

    def forward(this, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        if not this.training or past_key_values is not None or use_cache:
            return original_forward(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        hidden_states = this.dropout(this.embed_tokens(input_ids))
        if this.freqs_cos[0, 0] == 0:
            from model.model_minimind import precompute_freqs_cis
            cos, sin = precompute_freqs_cis(
                this.config.head_dim, this.config.max_position_embeddings,
                this.config.rope_theta, this.config.rope_scaling,
            )
            this.freqs_cos, this.freqs_sin = cos.to(hidden_states.device), sin.to(hidden_states.device)
        position_embeddings = (this.freqs_cos[:input_ids.shape[1]], this.freqs_sin[:input_ids.shape[1]])
        for layer in this.layers:
            def run_layer(states, block=layer):
                return block(states, position_embeddings, attention_mask=attention_mask)[0]
            hidden_states = checkpoint(run_layer, hidden_states, use_reentrant=False)
        hidden_states = this.norm(hidden_states)
        return hidden_states, [None] * len(this.layers), hidden_states.new_zeros(())

    base.forward = MethodType(forward, base)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint-*"))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(run_dir: Path, model, optimizer, scheduler, state: dict, step: int, tokenizer=None) -> Path:
    checkpoint_dir = run_dir / "checkpoints" / f"checkpoint-{step:08d}"
    if checkpoint_dir.exists():
        atomic_json(run_dir / "latest.json", {"checkpoint": str(checkpoint_dir.relative_to(run_dir))})
        return checkpoint_dir
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{checkpoint_dir.name}.tmp-", dir=checkpoint_dir.parent))
    try:
        model.save_pretrained(temporary / "model", safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(temporary / "tokenizer")
        torch.save({**state, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, temporary / "training_state.pt")
        for attempt in range(5):
            try:
                os.replace(temporary, checkpoint_dir)
                break
            except PermissionError:
                complete = (
                    (checkpoint_dir / "model" / "config.json").is_file()
                    and any((checkpoint_dir / "model").glob("*.safetensors"))
                    and (checkpoint_dir / "training_state.pt").is_file()
                    and (checkpoint_dir / "tokenizer").is_dir()
                )
                if complete:
                    shutil.rmtree(temporary, ignore_errors=True)
                    break
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)
        atomic_json(run_dir / "latest.json", {"checkpoint": str(checkpoint_dir.relative_to(run_dir))})
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return checkpoint_dir


def get_gpu_utilization() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True,
        )
        return result.stdout.splitlines()[0].strip() + "%"
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unavailable"


def evaluate(model, dataset: PackedBlocks, device, precision: str, limit: int) -> dict[str, float]:
    model.eval()
    result = {}
    with torch.inference_mode():
        for language, count in dataset.language_counts.items():
            start = next(start for start, _, name in dataset.ranges if name == language)
            loss_sum, token_count = 0.0, 0
            for index in range(start, min(start + count, start + limit)):
                input_ids, labels = dataset[index]
                input_ids, labels = input_ids.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)
                valid = int((labels[:, 1:] != -100).sum().item())
                if not valid:
                    continue
                context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and precision == "bf16" else nullcontext()
                with context:
                    output = model(input_ids=input_ids, labels=labels)
                loss_sum += float(output.loss) * valid
                token_count += valid
            if token_count:
                result[language] = loss_sum / token_count
    model.train()
    return result


def remaining_pretraining_tokens(total_tokens: int, prior_tokens: int = 0) -> int:
    return max(0, 1_000_000_000 - prior_tokens - total_tokens)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pretrain", "sft"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ryn0_3b.json")
    parser.add_argument("--data", nargs="+", required=True, help="JSONL files, directories, or glob patterns")
    parser.add_argument("--validation-data", nargs="+", default=None)
    parser.add_argument("--tokenizer", default=None, help="Defaults to tokenizer_name in the config")
    parser.add_argument("--init-from", default=None, help="Start a new phase from model weights only")
    parser.add_argument("--resume", nargs="?", const="latest", help="Resume exact state from latest checkpoint; an explicit path must be the latest checkpoint")
    parser.add_argument("--output-dir", type=Path, default=None, help="Required except in --benchmark mode")
    parser.add_argument("--benchmark", action="store_true", help="Measure a short run without saving model checkpoints")
    parser.add_argument("--steps", type=int, default=None, help="Optimizer steps (default: 100000 if --target-tokens is omitted)")
    parser.add_argument("--target-tokens", type=int, default=None, help="Stop after this many input token positions")
    parser.add_argument("--prior-tokens", type=int, default=0, help="Previously processed pretraining tokens, for combined 1B progress estimates")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=32)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--language-weights", default=None, help='JSON weights, e.g. \'{"en":1,"es":1,"ar":1}\'')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".ryn-cache")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps is not None and args.target_tokens is not None:
        raise ValueError("Choose either --steps or --target-tokens")
    if min(args.max_seq_len, args.batch_size, args.grad_accum, args.save_every, args.log_every, args.eval_every, args.eval_steps) < 1:
        raise ValueError("sequence length, batch size, accumulation, and intervals must be positive")
    if args.steps is not None and args.steps < 1 or args.target_tokens is not None and args.target_tokens < 1:
        raise ValueError("steps and target-tokens must be positive")
    if args.steps is None and args.target_tokens is None:
        args.steps = 20 if args.benchmark else 100000
    if args.prior_tokens < 0:
        raise ValueError("prior-tokens must be non-negative")
    if args.target_tokens is not None:
        args.steps = math.ceil(args.target_tokens / (args.max_seq_len * args.batch_size * args.grad_accum))
    if args.max_seq_len < 3 or args.warmup_steps < 0 or args.learning_rate <= 0 or args.grad_clip <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid sequence length, warmup, learning rate, or gradient clip")
    if not all(math.isfinite(value) for value in (args.learning_rate, args.weight_decay, args.grad_clip)):
        raise ValueError("learning-rate, weight-decay, and grad-clip must be finite")
    if args.stage == "sft" and not args.init_from and not args.resume:
        raise ValueError("SFT requires --init-from with a pretrained Ryn model")
    if args.resume and args.init_from:
        raise ValueError("Choose --resume for exact optimizer restoration or --init-from for weights only")
    if args.resume and args.output_dir is None:
        raise ValueError("--output-dir is required with --resume")
    if not args.benchmark and args.output_dir is None:
        raise ValueError("--output-dir is required unless --benchmark is set")
    if args.benchmark and args.resume:
        raise ValueError("--resume cannot be combined with --benchmark")

    train_paths = expand_paths(args.data)
    validation_paths = expand_paths(args.validation_data) if args.validation_data else []
    if set(train_paths) & set(validation_paths):
        raise ValueError("Training and validation inputs must use disjoint files")
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    if args.resume:
        run_dir = args.output_dir.expanduser().resolve()
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        selected = latest_checkpoint(run_dir) if args.resume == "latest" else Path(args.resume).expanduser().resolve()
        if selected is None or not selected.is_dir():
            raise ValueError(f"No checkpoint found for --resume {args.resume}")
        if latest_checkpoint(run_dir) != selected:
            raise ValueError("Resume from the latest checkpoint only; use --init-from to branch into a new run")
        tokenizer_path = selected / "tokenizer"
        if not tokenizer_path.is_dir():
            tokenizer_path = run_dir / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        cache_manifest = args.cache_dir.resolve() / run["settings"]["dataset"] / "manifest.json"
        tokenizer.name_or_path = json.loads(cache_manifest.read_text(encoding="utf-8"))["tokenizer_name"] if cache_manifest.is_file() else raw["tokenizer_name"]
        resume_model_path = selected / "model"
    elif args.init_from:
        init_model_path = model_path_from(args.init_from)
        tokenizer_path = tokenizer_path_from(init_model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        resume_model_path = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer or raw["tokenizer_name"], use_fast=True,
            cache_dir=args.cache_dir.expanduser().resolve() / "hf",
        )
        resume_model_path = None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = build_config(raw, tokenizer)
    if args.max_seq_len > config.max_position_embeddings:
        raise ValueError("--max-seq-len exceeds max_position_embeddings")
    if args.stage == "sft" and args.max_seq_len > 0 and args.init_from:
        init_config = json.loads((model_path_from(args.init_from) / "config.json").read_text(encoding="utf-8"))
        if init_config.get("vocab_size") != len(tokenizer):
            raise ValueError("The SFT tokenizer vocabulary does not match the pretrained checkpoint")

    train_cache = build_token_cache(train_paths, tokenizer, args.stage, args.max_seq_len, args.cache_dir.resolve())
    dataset = PackedBlocks(train_cache, args.max_seq_len, args.stage)
    language_weights = json.loads(args.language_weights) if args.language_weights else None
    if language_weights is not None:
        if not isinstance(language_weights, dict):
            raise ValueError("--language-weights must be a JSON object of positive weights")
        try:
            language_weights = {str(language): float(weight) for language, weight in language_weights.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("--language-weights must be a JSON object of finite positive weights") from exc
        if any(not math.isfinite(weight) or weight <= 0 for weight in language_weights.values()):
            raise ValueError("--language-weights must be a JSON object of finite positive weights")
    if args.target_tokens is not None:
        if language_weights:
            args.steps = math.ceil(args.target_tokens / (args.max_seq_len * args.batch_size * args.grad_accum))
        else:
            updates_per_epoch = math.ceil(len(dataset) / (args.batch_size * args.grad_accum))
            tokens_per_epoch = len(dataset) * args.max_seq_len
            args.steps = max(1, math.ceil(args.target_tokens / tokens_per_epoch) * updates_per_epoch)
    validation = None
    if validation_paths:
        validation = PackedBlocks(build_token_cache(validation_paths, tokenizer, args.stage, args.max_seq_len, args.cache_dir.resolve()), args.max_seq_len, args.stage)
    if language_weights:
        unknown = set(language_weights) - set(dataset.language_counts)
        if unknown:
            raise ValueError(f"Language weights must be positive and name present languages; unknown={sorted(unknown)}")

    if args.resume:
        run_dir = args.output_dir.expanduser().resolve()
        checkpoint_dir = selected
        dataset_signature = json.loads((train_cache / "manifest.json").read_text(encoding="utf-8"))["fingerprint"]
        settings = {
            "stage": args.stage, "config": raw, "dataset": dataset_signature,
            "validation_dataset": json.loads((validation.cache / "manifest.json").read_text(encoding="utf-8")).get("fingerprint") if validation else None,
            "sequence_length": args.max_seq_len, "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum, "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps, "weight_decay": args.weight_decay,
            "gradient_clip": args.grad_clip, "precision": args.precision,
            "activation_checkpointing": args.activation_checkpointing, "seed": args.seed,
            "steps": args.steps, "target_tokens": args.target_tokens,
            "prior_tokens": args.prior_tokens,
            "language_weights": language_weights,
        }
        if run["settings"] != settings:
            previous = {"prior_tokens": 0, **run["settings"]}
            regrouped = {
                **previous,
                "batch_size": settings["batch_size"],
                "gradient_accumulation": settings["gradient_accumulation"],
            }
            if (
                previous["batch_size"] * previous["gradient_accumulation"]
                != settings["batch_size"] * settings["gradient_accumulation"]
                or regrouped != settings
            ):
                raise ValueError("Exact resume requires the original model, data, and training settings; use --init-from for a new phase")
            run["settings"] = settings
            atomic_json(run_dir / "run.json", run)
    else:
        checkpoint_dir = None
        run_dir = args.output_dir.expanduser().resolve() if args.output_dir else None
        if run_dir and run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {run_dir}; use --resume or choose a new directory")
        settings = {
            "stage": args.stage, "config": raw,
            "dataset": json.loads((train_cache / "manifest.json").read_text(encoding="utf-8"))["fingerprint"],
            "validation_dataset": json.loads((validation.cache / "manifest.json").read_text(encoding="utf-8")).get("fingerprint") if validation else None,
            "sequence_length": args.max_seq_len, "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum, "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps, "weight_decay": args.weight_decay,
            "gradient_clip": args.grad_clip, "precision": args.precision,
            "activation_checkpointing": args.activation_checkpointing, "seed": args.seed,
            "steps": args.steps, "target_tokens": args.target_tokens,
            "prior_tokens": args.prior_tokens,
            "language_weights": language_weights,
        }

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Training and benchmarking require CUDA; run the CPU-compatible tests with python -m unittest discover -s tests")
    if device.type == "cuda" and args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested but this GPU does not support it; use --precision fp32")
    try:
        model = MiniMindForCausalLM.from_pretrained(resume_model_path, config=config) if resume_model_path else (
            MiniMindForCausalLM.from_pretrained(init_model_path, config=config) if args.init_from else MiniMindForCausalLM(config)
        )
        model.to(device)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print("CUDA out of memory while loading the model. Try configs/ryn0_2b.json or reduce --max-seq-len.", file=sys.stderr)
            raise SystemExit(2) from exc
        raise
    parameters = count_parameters(model)
    expected_parameters = raw.get("parameter_count_estimate")
    if expected_parameters is not None and parameters != expected_parameters:
        raise ValueError(f"Model has {parameters:,} parameters; config expects {expected_parameters:,}")
    print(f"model_parameters={parameters:,} tokenizer_vocab={len(tokenizer)} device={device}", flush=True)
    if args.activation_checkpointing:
        enable_activation_checkpointing(model)
    fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, fused=fused)
    total_steps = max(args.steps, 1)
    warmup = max(args.warmup_steps, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup, 1.0) * 0.5 * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps)),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if args.benchmark:
        run_dir = None
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        if checkpoint_dir is None:
            atomic_json(run_dir / "run.json", {"settings": settings, "parameter_count": parameters})
            tokenizer.save_pretrained(run_dir / "tokenizer")

    step, epoch, cursor, total_tokens, elapsed_offset = 0, 0, 0, 0, 0.0
    sampler = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    if checkpoint_dir:
        restored = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        step, epoch, cursor = restored["step"], restored["epoch"], restored["cursor"]
        total_tokens, elapsed_offset = restored["total_tokens"], restored.get("elapsed_seconds", 0.0)
        torch.set_rng_state(restored["torch_rng_state"])
        if device.type == "cuda" and restored.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(restored["cuda_rng_state"])
        if restored.get("sampler_rng_state") is not None:
            sampler.set_state(restored["sampler_rng_state"])
        if restored.get("python_rng_state") is not None:
            random.setstate(restored["python_rng_state"])

    started = time.monotonic()
    interval_started = started
    interval_tokens = interval_target_tokens = 0
    interval_loss = 0.0
    interval_steps = 0
    order = None
    order_epoch = None
    stable = {
        "step": step, "epoch": epoch, "cursor": cursor, "total_tokens": total_tokens,
        "elapsed_seconds": elapsed_offset, "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(), "sampler_rng_state": sampler.get_state(),
        "python_rng_state": random.getstate(),
    }
    if args.benchmark:
        args.steps = args.steps or 20

    def save_stable() -> Path | None:
        if not run_dir:
            return None
        state = {
            **stable,
            "run_settings": settings,
            "saved_at_unix": time.time(),
        }
        saved = save_checkpoint(run_dir, model, optimizer, scheduler, state, stable["step"], tokenizer)
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
        return saved

    optimizer_update_started = False

    def save_safe() -> Path | None:
        if optimizer_update_started:
            return latest_checkpoint(run_dir) if run_dir else None
        return save_stable()

    try:
        if not args.benchmark and checkpoint_dir is None:
            print(f"checkpoint={save_stable()}", flush=True)
            interval_started = time.monotonic()
        model.train()
        while ((total_tokens < args.target_tokens) if args.target_tokens is not None else (step < args.steps)):
            if language_weights:
                weighted_languages = sorted(language_weights)
                probabilities = torch.tensor([float(language_weights[lang]) for lang in weighted_languages])
                probabilities /= probabilities.sum()
                microbatches = []
                for _ in range(args.grad_accum):
                    selected_languages = torch.multinomial(probabilities, args.batch_size, replacement=True, generator=sampler)
                    indices = []
                    for language_index in selected_languages.tolist():
                        language = weighted_languages[language_index]
                        start = next(start for start, _, name in dataset.ranges if name == language)
                        indices.append(start + int(torch.randint(dataset.language_counts[language], (1,), generator=sampler).item()))
                    microbatches.append((indices, None))
            else:
                if cursor >= len(dataset):
                    epoch, cursor = epoch + 1, 0
                if order is None or order_epoch != epoch:
                    order_generator = torch.Generator(device="cpu").manual_seed(args.seed + epoch)
                    order = torch.randperm(len(dataset), generator=order_generator)
                    order_epoch = epoch
                remaining_batches = math.ceil((len(dataset) - cursor) / args.batch_size)
                group_size = min(args.grad_accum, remaining_batches)
                microbatches = []
                next_cursor = cursor
                for _ in range(group_size):
                    indices = order[next_cursor:min(next_cursor + args.batch_size, len(dataset))].tolist()
                    next_cursor += len(indices)
                    microbatches.append((indices, next_cursor))

            batches = []
            group_valid_tokens = 0
            for indices, next_position in microbatches:
                rows = [dataset[index] for index in indices]
                input_ids = torch.stack([row[0] for row in rows])
                labels = torch.stack([row[1] for row in rows])
                valid = int((labels[:, 1:] != -100).sum().item())
                if valid == 0:
                    continue
                batches.append((input_ids, labels, valid, next_position))
                group_valid_tokens += valid
            if not batches:
                raise ValueError("No trainable target tokens in the next gradient accumulation group")

            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            group_input_tokens = 0
            group_target_tokens = 0
            committed_cursor = cursor
            for input_ids, labels, valid, next_position in batches:
                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and args.precision == "bf16" else nullcontext()
                with context:
                    output = model(input_ids=input_ids, labels=labels)
                    loss = output.loss + output.aux_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {step + 1}")
                (loss * (valid / group_valid_tokens)).backward()
                group_loss += loss.detach() * valid
                group_input_tokens += input_ids.numel()
                group_target_tokens += valid
                del output, loss
                if next_position is not None:
                    committed_cursor = next_position

            clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer_update_started = True
            optimizer.step()
            scheduler.step()
            step += 1
            total_tokens += group_input_tokens
            if not language_weights:
                cursor = committed_cursor
            elapsed = elapsed_offset + time.monotonic() - started
            group_loss_value = float(group_loss) / group_target_tokens
            stable = {
                "step": step, "epoch": epoch, "cursor": cursor, "total_tokens": total_tokens,
                "elapsed_seconds": elapsed, "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(), "sampler_rng_state": sampler.get_state(),
                "python_rng_state": random.getstate(),
            }
            optimizer_update_started = False
            interval_tokens += group_input_tokens
            interval_target_tokens += group_target_tokens
            interval_loss += group_loss_value * group_target_tokens
            interval_steps += 1

            target_reached = args.target_tokens is not None and total_tokens >= args.target_tokens
            if step % args.log_every == 0 or step == 1 or step == args.steps or target_reached:
                now = time.monotonic()
                seconds = max(now - interval_started, 1e-9)
                tokens_per_second = interval_tokens / seconds
                remaining_tokens = remaining_pretraining_tokens(total_tokens, args.prior_tokens)
                remaining_steps = max(0, args.steps - step)
                eta_seconds = (
                    max(0, args.target_tokens - total_tokens) / max(tokens_per_second, 1e-9)
                    if args.target_tokens is not None
                    else remaining_steps * seconds / max(interval_steps, 1)
                )
                peak_vram = f"{torch.cuda.max_memory_allocated(device) / 1024**3:.2f}GiB" if device.type == "cuda" else "n/a"
                gpu_util = get_gpu_utilization() if device.type == "cuda" else "n/a"
                print(
                    f"stage={args.stage} step={step}/{args.steps} loss={interval_loss / max(interval_target_tokens, 1):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} tokens_per_sec={tokens_per_second:.1f} "
                    f"gpu_util={gpu_util} peak_vram={peak_vram} total_tokens={total_tokens} "
                    f"lifetime_tokens={args.prior_tokens + total_tokens} "
                    f"corpus_tokens={dataset.corpus_tokens} eta={eta_seconds:.0f}s "
                    f"to_1b_tokens={remaining_tokens / max(tokens_per_second, 1e-9):.0f}s",
                    flush=True,
                )
                interval_started = now
                interval_tokens = interval_target_tokens = interval_loss = interval_steps = 0

            if validation and (step % args.eval_every == 0 or step == args.steps or target_reached):
                metrics = evaluate(model, validation, device, args.precision, args.eval_steps)
                print("validation_loss_by_language=" + json.dumps(metrics, sort_keys=True), flush=True)
                model.train()
            if not args.benchmark and (step == 1 or step % args.save_every == 0 or step == args.steps or target_reached):
                saved = save_stable()
                print(f"checkpoint={saved}", flush=True)

            if args.target_tokens is not None and total_tokens >= args.target_tokens:
                break
    except KeyboardInterrupt:
        optimizer.zero_grad(set_to_none=True)
        sampler.set_state(stable["sampler_rng_state"])
        saved = save_safe()
        print(f"Training stopped safely after step {stable['step']}; checkpoint={saved}", flush=True)
        return
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower():
            optimizer.zero_grad(set_to_none=True)
            sampler.set_state(stable["sampler_rng_state"])
            if device.type == "cuda":
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                    print(
                        f"CUDA memory at OOM: allocated={torch.cuda.memory_allocated(device) / 1024**3:.2f}GiB "
                        f"reserved={torch.cuda.memory_reserved(device) / 1024**3:.2f}GiB "
                        f"free={free_bytes / 1024**3:.2f}GiB total={total_bytes / 1024**3:.2f}GiB",
                        file=sys.stderr, flush=True,
                    )
                except RuntimeError as stats_exc:
                    print(f"CUDA memory stats unavailable after OOM: {stats_exc}", file=sys.stderr, flush=True)
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
            saved = latest_checkpoint(run_dir) if run_dir else None
            print(
                f"CUDA out of memory at step {stable['step'] + 1}: {exc}. Last stable checkpoint: {saved}. "
                "Try a shorter --max-seq-len or smaller --batch-size; increase --grad-accum to keep the effective batch size.",
                file=sys.stderr, flush=True,
            )
            raise SystemExit(2) from exc
        raise
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        sampler.set_state(stable["sampler_rng_state"])
        saved = save_safe()
        print(f"Training failed at step {stable['step'] + 1}; last safe checkpoint={saved}", file=sys.stderr, flush=True)
        raise

    if args.benchmark:
        print(f"Benchmark complete: {total_tokens} input tokens; model parameters={parameters:,}", flush=True)
    else:
        saved = save_stable()
        if saved:
            print(f"checkpoint={saved}", flush=True)
        print(f"Training complete: step={step}, tokens={total_tokens}, latest={latest_checkpoint(run_dir)}", flush=True)


if __name__ == "__main__":
    main()
