#!/usr/bin/env python3
"""FSDP trainer for the MiniMind-based Ryn model."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "minimind"))
from model.model_minimind import MiniMindBlock, MiniMindConfig, MiniMindForCausalLM  # noqa: E402


class PretrainJsonl(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int):
        self.rows = load_dataset("json", data_files=path, split="train")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        text = str(self.rows[index]["text"])
        token_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        token_ids = [self.tokenizer.bos_token_id, *token_ids, self.tokenizer.eos_token_id]
        token_ids = token_ids[: self.max_length]
        pad_count = self.max_length - len(token_ids)
        input_ids = torch.tensor(token_ids + [self.tokenizer.pad_token_id] * pad_count, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class CodeSFTJsonl(Dataset):
    """SFT dataset that masks user/system tokens and learns assistant turns only."""

    def __init__(self, path: str, tokenizer, max_length: int):
        self.rows = load_dataset("json", data_files=path, split="train")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        messages = row.get("conversations", row.get("messages"))
        if not isinstance(messages, list):
            raise ValueError("SFT rows need a conversations or messages array")

        input_ids: list[int] = []
        labels: list[int] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            serialized = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            ids = self.tokenizer(serialized, add_special_tokens=False).input_ids
            input_ids.extend(ids)
            labels.extend(ids if role == "assistant" else [-100] * len(ids))

        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]
        if not any(label != -100 for label in labels):
            raise ValueError(f"SFT row {index} contains no assistant target tokens")
        pad_count = self.max_length - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * pad_count)
        labels.extend([-100] * pad_count)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def load_config(path: str, tokenizer) -> MiniMindConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    config = MiniMindConfig(
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
        tie_word_embeddings=raw["tie_word_embeddings"],
    )
    return config


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pretrain", "sft"), required=True)
    parser.add_argument("--config", default="configs/ryn0_3b.json")
    parser.add_argument("--data", required=True, help="MiniMind-compatible JSONL file or glob")
    parser.add_argument("--tokenizer", default=None, help="Defaults to the tokenizer in the model config")
    parser.add_argument("--init-from", default=None, help="Ryn checkpoint directory for SFT")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def save_model(model, tokenizer, output_dir: str, rank: int) -> None:
    output = Path(output_dir)
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = model.state_dict()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        model.module.save_pretrained(output, state_dict=state_dict, safe_serialization=True)
        tokenizer.save_pretrained(output)
        print(f"Saved Ryn checkpoint to {output}", flush=True)
    dist.barrier()


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU; no GPU is visible to this process")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires a GPU with BF16 support")
    if min(args.steps, args.grad_accum, args.batch_size, args.save_every) < 1:
        raise ValueError("steps, grad-accum, batch-size, and save-every must be positive")
    if args.max_seq_len < 3:
        raise ValueError("max-seq-len must be at least 3")
    if args.stage == "sft" and not args.init_from:
        raise ValueError("SFT requires --init-from with a pretrained Ryn checkpoint")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed + rank)
    device = torch.device("cuda", local_rank)

    raw_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tokenizer_name = args.tokenizer or raw_config["tokenizer_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = load_config(args.config, tokenizer)
    if args.max_seq_len > config.max_position_embeddings:
        raise ValueError("max-seq-len exceeds max_position_embeddings in the model config")

    if args.stage == "pretrain":
        dataset = PretrainJsonl(args.data, tokenizer, args.max_seq_len)
    else:
        dataset = CodeSFTJsonl(args.data, tokenizer, args.max_seq_len)
    if len(dataset) == 0:
        raise ValueError("training dataset is empty")

    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if args.init_from:
        base_model = MiniMindForCausalLM.from_pretrained(args.init_from, config=config)
    else:
        base_model = MiniMindForCausalLM(config)
    wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={MiniMindBlock})
    precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    model = FSDP(
        base_model,
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=precision,
        device_id=device,
        use_orig_params=True,
        sync_module_states=True,
        limit_all_gathers=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    warmup = max(args.warmup_steps, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup, 1.0)
        * 0.5
        * (1.0 + math.cos(math.pi * min(step, args.steps) / args.steps)),
    )

    step = 0
    epoch = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    while step < args.steps:
        sampler.set_epoch(epoch)
        epoch += 1
        for batch_index, (input_ids, labels) in enumerate(loader):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            group_start = (batch_index // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(loader) - group_start)
            output = model(input_ids=input_ids, labels=labels)
            loss = (output.loss + output.aux_loss) / group_size
            loss.backward()

            is_update_step = (batch_index + 1) % args.grad_accum == 0 or batch_index + 1 == len(loader)
            if not is_update_step:
                continue

            model.clip_grad_norm_(1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if rank == 0 and (step % 10 == 0 or step == 1):
                print(
                    f"stage={args.stage} step={step}/{args.steps} "
                    f"loss={loss.item() * group_size:.4f} lr={scheduler.get_last_lr()[0]:.3e}",
                    flush=True,
                )
            if step % args.save_every == 0 or step == args.steps:
                save_model(model, tokenizer, args.output_dir, rank)
            if step >= args.steps:
                break

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
