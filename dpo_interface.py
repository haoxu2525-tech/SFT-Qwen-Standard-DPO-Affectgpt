"""Epoch30 SFT loading for candidate generation; no DPO training overrides."""
import hashlib
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent
AFFECTGPT = ROOT / "vendor/affectgpt/AffectGPT"
SFT_INTERFACE = ROOT / "vendor/sft/Experiment/SFT Base Model Training"
ADAPTER_REPO = "Sakie22Anona/SFT-Qwen2.5-Omni-7B-AffectGPT-LoRA-epochs-30-60"
ADAPTER_REVISION = "06e534a6f43782296b33e134c6d52ce324b100c8"


def add_sft_path():
    if str(SFT_INTERFACE) not in sys.path:
        sys.path.insert(0, str(SFT_INTERFACE))


def verify_epoch30(adapter):
    hashes = {
        "adapter_model.safetensors": "5e856ea4effd60ad10bc376f1a4515aa935d68c423128cc60e03aba5d964ad03",
        "adapter_config.json": "2262c2a442439d9b92c953fdcea84b0fa6067302e965f1d1083026bc3a07cbb0"}
    for name, expected in hashes.items():
        checksum = hashlib.sha256()
        with (Path(adapter) / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != expected:
            raise ValueError(f"Not the verified epoch30 export: {name}")


def load_sft_model(snapshot, adapter, device="cuda:0"):
    add_sft_path()
    from omni_components import load_thinker_config
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration
    from peft import PeftModel
    verify_epoch30(adapter)
    native = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(snapshot), config=load_thinker_config(snapshot), local_files_only=True,
        torch_dtype=torch.bfloat16)
    native.requires_grad_(False)
    model = PeftModel.from_pretrained(native, str(adapter), is_trainable=False)
    model.requires_grad_(False)
    return model.to(device)
