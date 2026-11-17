from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train import (  # noqa: E402
    PackedBlocks,
    build_config,
    build_token_cache,
    count_parameters,
    enable_activation_checkpointing,
    evaluate,
    remaining_pretraining_tokens,
    save_checkpoint,
    sft_segments,
    tokenizer_path_from,
    validate_model_config,
)

sys.path.insert(0, str(ROOT / "upstream" / "minimind"))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


class TinyTokenizer:
    name_or_path = "synthetic-tokenizer"
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __len__(self):
        return 32

    @property
    def backend_tokenizer(self):
        class Backend:
            @staticmethod
            def to_str():
                return "synthetic-tokenizer-v1"
        return Backend()

    def __call__(self, texts, add_special_tokens=False, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [[3 + ord(char) % 27 for char in text] for text in texts]}

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def tiny_config():
    return MiniMindConfig(
        hidden_size=16,
        num_hidden_layers=1,
        vocab_size=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        max_position_embeddings=16,
        tie_word_embeddings=True,
    )


class TrainingTests(unittest.TestCase):
    def test_remaining_pretraining_tokens_include_prior_phases(self):
        self.assertEqual(remaining_pretraining_tokens(9_011_200, 101_580_800), 889_408_000)
        self.assertEqual(remaining_pretraining_tokens(1_000_000_000), 0)
        self.assertEqual(remaining_pretraining_tokens(1_000_000_001), 0)

    def test_config_validation_and_exact_parameter_count(self):
        raw = json.loads((ROOT / "configs" / "ryn0_3b.json").read_text(encoding="utf-8"))
        for name in ("ryn0_3b.json", "ryn0_2b.json"):
            config_raw = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
            validate_model_config(config_raw, config_raw["tokenizer_vocab_size_expected"])
            tokenizer = type("Tokenizer", (), {
                "__len__": lambda self: config_raw["tokenizer_vocab_size_expected"],
                "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
            })()
            with torch.device("meta"):
                model = MiniMindForCausalLM(build_config(config_raw, tokenizer))
            self.assertEqual(count_parameters(model), config_raw["parameter_count_estimate"])
        raw["num_attention_heads"] = 10
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            validate_model_config(raw, raw["tokenizer_vocab_size_expected"])

    def test_pretrain_cache_packs_and_deduplicates_by_language(self):
        fixture = ROOT / "tests" / "fixtures" / "tiny_pretrain.jsonl"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.jsonl"
            rows = fixture.read_text(encoding="utf-8").splitlines()
            rows.append(rows[0])
            source.write_text("\n".join(rows) + "\n", encoding="utf-8")
            cache = build_token_cache([source], TinyTokenizer(), "pretrain", 16, Path(directory) / "cache")
            dataset = PackedBlocks(cache, 16, "pretrain")
            self.assertGreater(len(dataset), 0)
            self.assertEqual(dataset.manifest["duplicate_rows_skipped"], 1)
            self.assertEqual(set(dataset.languages), {"en", "es"})
            ids, labels = dataset[0]
            self.assertEqual(ids.shape, torch.Size([16]))
            self.assertTrue(torch.equal(ids, labels))
            losses = evaluate(MiniMindForCausalLM(tiny_config()), dataset, torch.device("cpu"), "fp32", 2)
            self.assertEqual(set(losses), {"en", "es"})
            self.assertTrue(all(torch.isfinite(torch.tensor(loss)) for loss in losses.values()))

    def test_sft_cache_masks_non_assistant_tokens(self):
        fixture = ROOT / "tests" / "fixtures" / "tiny_sft.jsonl"
        with tempfile.TemporaryDirectory() as directory:
            cache = build_token_cache([fixture], TinyTokenizer(), "sft", 16, Path(directory))
            dataset = PackedBlocks(cache, 16, "sft")
            ids, labels = dataset[0]
            self.assertEqual(ids.shape, torch.Size([16]))
            self.assertTrue((labels == -100).any())
            self.assertTrue((labels != -100).any())
            with self.assertRaisesRegex(ValueError, "no assistant target"):
                sft_segments({"conversations": [{"role": "user", "content": "question"}]}, fixture, 1)

    def test_tiny_model_loss_backward_and_inference(self):
        model = MiniMindForCausalLM(tiny_config())
        ids = torch.randint(3, 30, (2, 8))
        loss = model(input_ids=ids, labels=ids).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        output = model.generate(input_ids=ids[:1, :4], max_new_tokens=2, do_sample=False, top_k=0, top_p=1, temperature=1)
        self.assertEqual(output.shape, torch.Size([1, 6]))

        checkpointed = MiniMindForCausalLM(tiny_config())
        enable_activation_checkpointing(checkpointed)
        checkpoint_loss = checkpointed(input_ids=ids, labels=ids).loss
        checkpoint_loss.backward()
        self.assertTrue(torch.isfinite(checkpoint_loss))

    def test_versioned_checkpoint_restores_weights_and_optimizer_state(self):
        model = MiniMindForCausalLM(tiny_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        ids = torch.randint(3, 30, (1, 8))
        model(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()
        scheduler.step()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            rng_state = torch.get_rng_state()
            replace = os.replace
            denied_before_move = False
            denied_after_move = False

            def replace_checkpoint_then_report_denial(source, destination):
                nonlocal denied_before_move, denied_after_move
                if Path(destination).name == "checkpoint-00000001" and not denied_before_move:
                    denied_before_move = True
                    raise PermissionError("simulated transient Windows rename lock")
                replace(source, destination)
                if Path(destination).name == "checkpoint-00000001" and not denied_after_move:
                    denied_after_move = True
                    raise PermissionError("simulated Windows post-rename denial")

            with (
                mock.patch("train.os.replace", side_effect=replace_checkpoint_then_report_denial),
                mock.patch("train.time.sleep") as sleep,
            ):
                saved = save_checkpoint(
                    run_dir, model, optimizer, scheduler,
                    {"step": 1, "epoch": 0, "cursor": 1, "total_tokens": 8, "torch_rng_state": rng_state},
                    1, TinyTokenizer(),
                )
            self.assertTrue(denied_before_move)
            self.assertTrue(denied_after_move)
            sleep.assert_called_once_with(1)
            restored = torch.load(saved / "training_state.pt", map_location="cpu", weights_only=False)
            loaded_model = MiniMindForCausalLM.from_pretrained(saved / "model")
            self.assertEqual(restored["step"], 1)
            self.assertEqual(len(restored["optimizer"]["state"]), len(optimizer.state))
            self.assertTrue(torch.equal(model.lm_head.weight, loaded_model.lm_head.weight))
            loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=0.001)
            loaded_optimizer.load_state_dict(restored["optimizer"])
            loaded_scheduler = torch.optim.lr_scheduler.StepLR(loaded_optimizer, step_size=1)
            loaded_scheduler.load_state_dict(restored["scheduler"])
            self.assertEqual(loaded_scheduler.last_epoch, scheduler.last_epoch)
            self.assertTrue(torch.equal(restored["torch_rng_state"], rng_state))
            self.assertEqual(tokenizer_path_from(saved / "model"), saved / "tokenizer")
            self.assertEqual(Path(json.loads((run_dir / "latest.json").read_text())["checkpoint"]), Path("checkpoints") / "checkpoint-00000001")
            for step in range(2, 5):
                save_checkpoint(
                    run_dir, model, optimizer, scheduler,
                    {"step": step, "epoch": 0, "cursor": step, "total_tokens": step * 8, "torch_rng_state": rng_state},
                    step, TinyTokenizer(),
                )
            retained = sorted(path.name for path in (run_dir / "checkpoints").glob("checkpoint-*"))
            self.assertEqual(retained, [
                "checkpoint-00000001", "checkpoint-00000002", "checkpoint-00000003", "checkpoint-00000004",
            ])


if __name__ == "__main__":
    unittest.main()
