#!/usr/bin/env python3
"""Run a local prompt against a saved Ryn checkpoint."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from train import model_path_from, tokenizer_path_from
from model.model_minimind import MiniMindForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Run directory, checkpoint directory, or model directory")
    parser.add_argument("--prompt", default=None, help="One prompt; omit for an interactive chat")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.temperature <= 0:
        parser.error("max-new-tokens and temperature must be positive")

    model_path = model_path_from(args.model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_from(model_path), use_fast=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniMindForCausalLM.from_pretrained(model_path)
    model.to(device).eval()

    def reply(text: str) -> str:
        prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        if input_ids.shape[1] + args.max_new_tokens > model.config.max_position_embeddings:
            raise ValueError("Prompt plus max-new-tokens exceeds the model context length")
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.no_grad()
        with torch.inference_mode(), context:
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
            )
        return tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

    if args.prompt is not None:
        print(reply(args.prompt))
        return
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.lower() in {":q", ":quit", "exit"}:
            return
        if text:
            print(f"ryn> {reply(text)}")


if __name__ == "__main__":
    main()
