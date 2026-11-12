"""Native Omni signatures around the unchanged official DPO implementation."""
import contextlib
import gzip
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
EPOCH30_RUNNER_SHA256 = "e7bc5a7aa4c77ae9b636a902dc03bb17db7ca08e2c50fc394f10142125abdea0"


def add_sft_path():
    if str(SFT_INTERFACE) not in sys.path:
        sys.path.insert(0, str(SFT_INTERFACE))


def verify_epoch30(adapter):
    adapter = Path(adapter)
    if adapter.is_file():
        checksum = hashlib.sha256()
        with adapter.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != EPOCH30_RUNNER_SHA256:
            raise ValueError("Not the verified original epoch30 runner checkpoint")
        return
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


def load_sft_model(snapshot, adapter, device="cuda:0", trainable=False):
    add_sft_path()
    from omni_components import load_thinker_config, make_lora_model
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration
    from peft import PeftModel
    verify_epoch30(adapter)
    native = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(snapshot), config=load_thinker_config(snapshot), local_files_only=True,
        torch_dtype=torch.bfloat16)
    native.requires_grad_(False)
    if Path(adapter).is_file():
        # Original SFT runner checkpoint: use its unchanged native LoRA factory.
        # The complete file hash above pins weights AND the recorded SFT config.
        model = make_lora_model(native)
        saved = torch.load(adapter, map_location="cpu", weights_only=True)
        if saved["epoch"] != 30:
            raise ValueError("Expected epoch30")
        state = {k.removeprefix("thinker."): v for k, v in saved["model"].items()}
        required = {n for n, p in model.named_parameters() if p.requires_grad}
        if set(state) != required:
            raise ValueError("Original checkpoint does not exactly cover the language LoRA")
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(".lora_" in k for k in result.missing_keys):
            raise ValueError("Original SFT LoRA reconstruction failed")
        for name, parameter in model.named_parameters():
            if name in state and not torch.equal(parameter.detach().cpu(), state[name]):
                raise ValueError(f"SFT reconstruction changed tensor values: {name}")
        del saved, state
    else:
        model = PeftModel.from_pretrained(native, str(adapter), is_trainable=trainable)
    if not trainable:
        model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (".lora_" not in name or ".model.layers." not in name):
            raise ValueError(f"Unexpected trainable parameter: {name}")
    return model.to(device)


def read_pairs(path):
    """Read supplied preferences, without judging, filtering or creating splits."""
    path = Path(path)
    if path.is_dir():
        from audit_preferences import rows_from
        rows = [row for row, _, _ in rows_from(path)]
    else:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Empty preference data or duplicate sample IDs")
    for row in rows:
        for key in ("prompt", "video", "chosen", "rejected"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{row['id']}: missing {key}; supply already labelled preferences")
        if row["chosen"] == row["rejected"]:
            raise ValueError(f"{row['id']}: identical answers do not specify a preference")
        if 'ground_truth_answer' in row and row['chosen'] != row['ground_truth_answer']:
            raise ValueError(f"{row['id']}: changed chosen answer")
    return rows


def membership_report(rows, original_ids):
    original_ids = set(original_ids)
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicated preference IDs')
    unknown = sorted(set(ids) - original_ids)
    if unknown:
        raise ValueError(f'Preference IDs outside original training annotations: {unknown}')
    if not ids:
        raise ValueError('No successful preference pairs supplied')
    return dict(original_count=len(original_ids), selected_count=len(ids),
                excluded_ids=sorted(original_ids-set(ids)))


class OmniModel(torch.nn.Module):
    """Supply native AV tensors to the original forward/generate call signatures.

    Parameters, train/eval mode and full state_dict use nn.Module unchanged.
    """
    def __init__(self, thinker):
        super().__init__()
        self.thinker = thinker
        self._media = None

    @contextlib.contextmanager
    def media_context(self, media):
        if self._media is not None:
            raise RuntimeError("Nested media context")
        self._media = media
        try:
            yield
        finally:
            self._media = None

    def native_inputs(self, input_ids, attention_mask):
        if not self._media or len(self._media) != input_ids.shape[0]:
            raise ValueError("Each batch row must have its corresponding native AV input")
        add_sft_path()
        from omni_components import thinker_inputs
        keys = set(self._media[0])
        if any(set(row) != keys for row in self._media):
            raise ValueError("Inconsistent native AV fields")
        inputs = {key: torch.cat([row[key] for row in self._media], dim=0) for key in keys}
        inputs.update(input_ids=input_ids, attention_mask=attention_mask, use_audio_in_video=True)
        return thinker_inputs(self.thinker, inputs)

    def precision_context(self):
        if next(self.thinker.parameters()).device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def forward(self, input_ids, attention_mask):
        with self.precision_context():
            return self.thinker(**self.native_inputs(input_ids, attention_mask))

    def generate(self, input_ids, attention_mask, **kwargs):
        with self.precision_context():
            return self.thinker.generate(**self.native_inputs(input_ids, attention_mask), **kwargs)


class PairData:
    """Native prompt serialization; upstream owns response tokens and batching."""
    def __init__(self, train_rows, eval_rows, processor, dataset_root, tokenize):
        add_sft_path()
        self.rows = {"train": train_rows, "test": eval_rows}
        self.processor = processor
        self.root = Path(dataset_root).resolve()
        from omni_components import encode_prompt
        self.encode = encode_prompt
        self.tokenize_response = tokenize

    def get_dataset(self, name, split, **kwargs):
        if name != "affectgpt_preferences" or split not in self.rows:
            raise ValueError(f"Unsupported preference data request: {name}/{split}")
        result = {}
        for row in self.rows[split]:
            # Routing identity only; it never becomes model text. Distinct videos
            # must not collapse when their original question/subtitle is equal.
            key = json.dumps({k: row[k] for k in ("id", "prompt", "video")},
                             ensure_ascii=False, sort_keys=True)
            result[key] = {"responses": [row["chosen"], row["rejected"]],
                           "pairs": [(0, 1)], "sft_target": row["chosen"]}
        return result

    def tokenize(self, prompt, chosen, rejected, truncation_mode, tokenizer,
                 max_length, max_prompt_length):
        metadata = json.loads(prompt)
        source = (self.root / metadata["video"]).resolve()
        if not source.is_relative_to(self.root):
            raise ValueError("Media escapes the original dataset root")
        native = dict(self.encode(self.processor, metadata["prompt"], str(source)))
        prefix = native.pop("input_ids")[0].tolist()
        prompt_mask = native.pop("attention_mask")[0].tolist()
        if native.pop("use_audio_in_video") is not True:
            raise ValueError('DPO must use the SFT/evaluation native synchronized AV encoder')
        # Keep full native AV tokens paired with their tensors and full answers.
        lengths = [len(tokenizer(answer, add_special_tokens=False)["input_ids"]) + 1
                   for answer in (chosen, rejected)]
        budget = max_length - len(prefix)
        if len(prefix) > max_prompt_length or budget < 1:
            raise ValueError(f"{metadata['id']}: native AV prompt alone exceeds the input budget; "
                             "media tokens cannot be sliced independently of media tensors")
        if max(lengths) > budget:
            raise ValueError(f"{metadata['id']}: full native input plus answer exceeds max_length; "
                             "no answer or media truncation is permitted")
        # Empty text prompt: the full prompt is already serialized above.
        # max_prompt_length=0 makes upstream's response budget exactly `budget`.
        # The capacity guard above ensures upstream never slices answers/EOS.
        item = self.tokenize_response("", chosen, rejected, truncation_mode,
                                      tokenizer, budget, 0)
        if item["chosen_input_ids"] == item["rejected_input_ids"]:
            raise ValueError(f"{metadata['id']}: chosen/rejected have identical retained tokens; "
                             "the preference cannot be learned at this length")
        for label in ("chosen", "rejected"):
            item[label + "_input_ids"] = prefix + item[label + "_input_ids"]
            item[label + "_attention_mask"] = prompt_mask + item[label + "_attention_mask"]
            item[label + "_labels"] = [-100] * len(prefix) + item[label + "_labels"]
            item[label] = metadata["prompt"] + item[label]
        item.update(prompt=metadata["prompt"], prompt_input_ids=prefix,
                    prompt_attention_mask=prompt_mask, media=native, id=metadata["id"])
        item["length_info"] = dict(prompt=len(prefix), chosen_before=lengths[0],
            rejected_before=lengths[1], chosen_after=len(item["chosen_input_ids"])-len(prefix),
            rejected_after=len(item["rejected_input_ids"])-len(prefix))
        return item


def trainer_class(upstream):
    class OmniTrainer(upstream.BasicTrainer):
        def concatenated_forward(self, model, batch):
            # The original concatenated_inputs orders chosen before rejected.
            with model.media_context(batch["media"] + batch["media"]):
                return super().concatenated_forward(model, batch)

        def get_batch_samples(self, batch):
            with contextlib.ExitStack() as stack:
                stack.enter_context(self.policy.media_context(batch["media"]))
                if self.config.loss.name in {"dpo", "ipo"}:
                    stack.enter_context(self.reference_model.media_context(batch["media"]))
                return super().get_batch_samples(batch)
    return OmniTrainer


@contextlib.contextmanager
def interfaces(preference_datasets, trainers, data):
    """Register data/model signatures only; restore registrations on exit."""
    old_dataset = preference_datasets.get_dataset
    old_tokenize = preference_datasets.tokenize_batch_element
    if hasattr(trainers, "OmniTrainer"):
        raise RuntimeError("OmniTrainer is already registered")
    preference_datasets.get_dataset = data.get_dataset
    preference_datasets.tokenize_batch_element = data.tokenize
    trainers.OmniTrainer = trainer_class(trainers)
    try:
        yield
    finally:
        preference_datasets.get_dataset = old_dataset
        preference_datasets.tokenize_batch_element = old_tokenize
        del trainers.OmniTrainer
