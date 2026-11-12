"""Reconstruct a runner checkpoint through the real PEFT loader on a tiny Thinker."""
import copy
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

import dpo_interface as interface
sys.path.insert(0, str(interface.SFT_INTERFACE))
from omni_components import load_processor, make_lora_model
from test_omni_components import tiny_native


def test_original_runner_reconstruction_and_frozen_reference(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    processor = load_processor(os.environ["OMNI_PROCESSOR_SNAPSHOT"])
    base = tiny_native(processor)
    trained = make_lora_model(copy.deepcopy(base))
    with torch.no_grad():
        for name, param in trained.named_parameters():
            if param.requires_grad:
                param.uniform_(-.01, .01)
    state = {"thinker." + n: p.detach().clone() for n, p in trained.named_parameters() if p.requires_grad}
    checkpoint = tmp_path / "tiny-epoch30.pth"
    torch.save({"epoch": 30, "model": state}, checkpoint)
    # Fixture-only hash: production continues to require the published epoch30 hash.
    monkeypatch.setattr(interface, "EPOCH30_RUNNER_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    monkeypatch.setattr(type(base), "from_pretrained", lambda *a, **k: copy.deepcopy(base))
    policy = interface.load_sft_model(os.environ["OMNI_PROCESSOR_SNAPSHOT"], checkpoint, device="cpu", trainable=True)
    reference = interface.load_sft_model(os.environ["OMNI_PROCESSOR_SNAPSHOT"], checkpoint, device="cpu", trainable=False)
    assert {"thinker." + n for n, p in policy.named_parameters() if p.requires_grad} == set(state)
    assert not any(p.requires_grad for p in reference.parameters())
    for name, param in policy.named_parameters():
        assert torch.equal(param, dict(reference.named_parameters())[name])
        if "thinker." + name in state:
            assert torch.equal(param, state["thinker." + name])
    policy.eval(); reference.eval(); trained.eval()
    ids = processor.tokenizer("A neutral sentence.", return_tensors="pt")["input_ids"]
    with torch.no_grad():
        expected = trained(input_ids=ids).logits
        assert torch.equal(policy(input_ids=ids).logits, expected)
        assert torch.equal(reference(input_ids=ids).logits, expected)
    from run_pipeline import export
    policy_path = tmp_path / "policy.pt"
    torch.save({"state": state}, policy_path)
    export(SimpleNamespace(adapter=checkpoint, checkpoint=policy_path, output=tmp_path / "export"))
    restored = torch.load(tmp_path / "export/checkpoint_000001_dpo_final.pth", weights_only=True)["model"]
    assert set(restored) == set(state)
    assert all(torch.equal(restored[k], state[k]) for k in state)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="verified original"):
        interface.verify_epoch30(checkpoint)


def test_other_checkpoint_is_not_accepted(tmp_path):
    p = tmp_path / "other.pth"
    p.write_bytes(b"unverified checkpoint")
    with pytest.raises(ValueError, match="verified original"):
        interface.verify_epoch30(p)
