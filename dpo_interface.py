"""Native Omni data/model signatures around the unmodified official DPO trainer.

No DPO objective, probability computation, optimizer, scheduler, training loop,
gradient clipping or checkpoint writer is implemented here.
"""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent
DPO = ROOT / "vendor/dpo"
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


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Empty data or duplicate sample IDs")
    if any(row["dataset"] != "MERCaptionPlus" or row["split"] != "train" for row in rows):
        raise ValueError("Only original MERCaptionPlus training samples are accepted")
    return rows


def load_policy(snapshot, adapter, trainable=True, device="cuda:0"):
    add_sft_path()
    from omni_components import load_thinker_config
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration
    from peft import PeftModel
    verify_epoch30(adapter)
    native = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(snapshot), config=load_thinker_config(snapshot), local_files_only=True,
        torch_dtype=torch.bfloat16)
    native.requires_grad_(False)
    model = PeftModel.from_pretrained(native, str(adapter), is_trainable=trainable)
    if not trainable:
        model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (".lora_" not in name or ".model.layers." not in name):
            raise ValueError(f"Unexpected trainable parameter: {name}")
    return OmniModel(model.to(device))


class OmniModel(torch.nn.Module):
    """The official trainer sees a causal model; this interface supplies AV tensors."""
    def __init__(self, thinker):
        super().__init__()
        self.thinker = thinker
        self._media = None

    def train(self, mode=True):
        super().train(mode)
        native = self.thinker.get_base_model()
        native.visual.eval()
        native.audio_tower.eval()
        return self

    @contextlib.contextmanager
    def media_context(self, media):
        if self._media is not None:
            raise RuntimeError("Nested media context")
        self._media = media
        try:
            yield
        finally:
            self._media = None

    def forward(self, input_ids, attention_mask):
        if self._media is None:
            raise RuntimeError("Native audiovisual inputs were not supplied")
        add_sft_path()
        from omni_components import thinker_inputs
        keys = set().union(*(row.keys() for row in self._media))
        inputs = {key: torch.cat([row[key] for row in self._media], dim=0) for key in keys}
        inputs.update(input_ids=input_ids, attention_mask=attention_mask, use_audio_in_video=True)
        device = next(self.thinker.parameters()).device
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
        with autocast:
            return self.thinker(**thinker_inputs(self.thinker, inputs), use_cache=False)

    def state_dict(self, *args, **kwargs):
        # Checkpoint serialization only. The fixed base is reconstructed from its
        # pinned revision; all trained tensors retain the exact SFT wrapper keys.
        state = super().state_dict(*args, **kwargs)
        return type(state)((key, value) for key, value in state.items() if ".lora_" in key)


class PairData:
    """Only serializes rows; upstream get_batch_iterator owns shuffle/batching."""
    def __init__(self, rows, processor, dataset_root, max_length=4096):
        add_sft_path()
        from omni_components import encode_prompt
        self.rows = rows
        self.processor = processor
        self.root = Path(dataset_root).resolve()
        self.max_length = max_length
        self.encode = encode_prompt

    def get_dataset(self, name, split, **kwargs):
        if name != "affectgpt_preferences" or split != "train":
            raise ValueError("Only the unchanged MERCaptionPlus training membership is accepted")
        result = {}
        for row in self.rows:
            # Official data API keys by prompt. An opaque JSON routing key keeps
            # equal subtitle/question texts from collapsing distinct videos.
            key = json.dumps({"id": row["id"], "prompt": row["prompt"], "video": row["video"]},
                             ensure_ascii=False, sort_keys=True)
            result[key] = {"responses": [row["chosen"], row["rejected"]],
                           "pairs": [(0, 1)], "sft_target": row["ground_truth_answer"]}
        return result

    def tokenize(self, prompt, chosen, rejected, truncation_mode, tokenizer,
                 max_length, max_prompt_length):
        metadata = json.loads(prompt)
        source = (self.root / metadata["video"]).resolve()
        if not source.is_relative_to(self.root):
            raise ValueError("Video escapes the original dataset root")
        native = dict(self.encode(self.processor, metadata["prompt"], str(source)))
        prefix = native.pop("input_ids")[0].long()
        native.pop("attention_mask")
        native.pop("use_audio_in_video")
        item = {"media": native, "prompt": metadata["prompt"], "id": metadata["id"]}
        for label, answer in (("chosen", chosen), ("rejected", rejected)):
            # Both candidates use the existing AffectGPT answer delimiter.
            target = tokenizer(answer + "###", add_special_tokens=False)["input_ids"]
            target = torch.tensor(target + [tokenizer.eos_token_id], dtype=torch.long)
            ids = torch.cat((prefix, target))
            if len(ids) > self.max_length:
                raise ValueError(f"{metadata['id']} {label}: {len(ids)} > {self.max_length}; no truncation")
            item[label + "_input_ids"] = ids
            item[label + "_attention_mask"] = torch.ones_like(ids)
            item[label + "_labels"] = torch.cat((torch.full_like(prefix, -100), target))
        return item

    def collate_fn(self, tokenizer):
        def collate(rows):
            batch = {}
            for key in rows[0]:
                values = [row[key] for row in rows]
                if key.endswith(("_input_ids", "_attention_mask", "_labels")):
                    pad = -100 if key.endswith("_labels") else (tokenizer.pad_token_id if key.endswith("_input_ids") else 0)
                    values = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=pad)
                batch[key] = values
            return batch
        return collate


@contextlib.contextmanager
def data_bindings(preference_datasets, trainers, data):
    """Temporary data-interface bindings; original source files are untouched."""
    old_dataset = preference_datasets.get_dataset
    old_tokenize = preference_datasets.tokenize_batch_element
    old_collate = preference_datasets.get_collate_fn
    old_iterator = trainers.get_batch_iterator
    def iterator(*args, **kwargs):
        if kwargs.get("split") == "test":
            # MERCaptionPlus supplies no preference-validation split. Do not
            # manufacture one or feed benchmark labels into training diagnostics.
            return iter(())
        return preference_datasets.get_batch_iterator(*args, **kwargs)
    preference_datasets.get_dataset = data.get_dataset
    preference_datasets.tokenize_batch_element = data.tokenize
    preference_datasets.get_collate_fn = data.collate_fn
    trainers.get_batch_iterator = iterator
    try:
        yield
    finally:
        preference_datasets.get_dataset = old_dataset
        preference_datasets.tokenize_batch_element = old_tokenize
        preference_datasets.get_collate_fn = old_collate
        trainers.get_batch_iterator = old_iterator


def trainer_class(upstream):
    class OmniTrainer(upstream.BasicTrainer):
        def concatenated_forward(self, model, batch):
            # Same ordering as original concatenated_inputs: chosen then rejected.
            # Actual concatenation, forward, log-softmax, masks and sums are upstream.
            with model.media_context(batch["media"] + batch["media"]):
                return super().concatenated_forward(model, batch)
    return OmniTrainer
