"""Native Omni signatures around the unchanged official DPO implementation."""
import contextlib
import ast
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
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


@lru_cache(maxsize=1)
def original_audio_loader():
    """Execute the pinned author's functions, without importing unrelated encoders."""
    import torchaudio
    package = importlib.util.find_spec("pytorchvideo")
    sampler_path = Path(next(iter(package.submodule_search_locations))) / "data/clip_sampling.py"
    sampler = runpy.run_path(str(sampler_path))["ConstantClipsPerVideoSampler"]
    source = AFFECTGPT / "my_affectgpt/models/ImageBind/data.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    namespace = dict(torch=torch, torchaudio=torchaudio,
                     ConstantClipsPerVideoSampler=sampler)
    for name in ("get_clip_timepoints", "load_audio"):
        node, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["load_audio"]


def encode_sampled_audio(processor, prompt, video, audio):
    """Original AffectGPT WAV sampling; native Omni video and separate audio inputs."""
    add_sft_path()
    from omni_components import native_messages, install_short_video_compatibility
    from qwen_omni_utils import process_mm_info
    install_short_video_compatibility()
    clips = original_audio_loader()([audio], "cpu", clips_per_video=8)[0]
    if clips.shape != (8, 1, 32000):
        raise ValueError(f"Original audio sampler returned {tuple(clips.shape)}: {audio}")
    messages = native_messages(prompt, video)
    # Discontinuous clips stay distinct; never invent synchronized AV timestamps.
    messages[1]["content"][1:1] = [
        {"type": "audio", "audio": clip[0].numpy()} for clip in clips]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos, video_kwargs = process_mm_info(
        messages, use_audio_in_video=False, return_video_kwargs=True)
    if len(audios) != 8 or len(videos) != 1:
        raise ValueError("Expected one video and the eight original audio clips")
    inputs = dict(processor(text=text, images=images, videos=videos, audio=audios,
        return_tensors="pt", add_special_tokens=False, padding=False,
        truncation=False, use_audio_in_video=False, fps=video_kwargs["fps"][0],
        do_resize=False))
    inputs["use_audio_in_video"] = False
    return inputs


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


def load_sft_model(snapshot, adapter, device="cuda:0", trainable=False):
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
    return model.to(device)


def read_pairs(path):
    """Read supplied preferences, without judging, filtering or creating splits."""
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Empty preference data or duplicate sample IDs")
    for row in rows:
        for key in ("prompt", "video", "audio", "chosen", "rejected"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{row['id']}: missing {key}; supply already labelled preferences")
        if row["chosen"] == row["rejected"]:
            raise ValueError(f"{row['id']}: identical answers do not specify a preference")
    return rows


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
        inputs.update(input_ids=input_ids, attention_mask=attention_mask, use_audio_in_video=False)
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
        self.encode = encode_sampled_audio
        self.tokenize_response = tokenize

    def get_dataset(self, name, split, **kwargs):
        if name != "affectgpt_preferences" or split not in self.rows:
            raise ValueError(f"Unsupported preference data request: {name}/{split}")
        result = {}
        for row in self.rows[split]:
            # Routing identity only; it never becomes model text. Distinct videos
            # must not collapse when their original question/subtitle is equal.
            key = json.dumps({k: row[k] for k in ("id", "prompt", "video", "audio")},
                             ensure_ascii=False, sort_keys=True)
            result[key] = {"responses": [row["chosen"], row["rejected"]],
                           "pairs": [(0, 1)], "sft_target": row["chosen"]}
        return result

    def tokenize(self, prompt, chosen, rejected, truncation_mode, tokenizer,
                 max_length, max_prompt_length):
        metadata = json.loads(prompt)
        source = (self.root / metadata["video"]).resolve()
        audio = (self.root / metadata["audio"]).resolve()
        if not source.is_relative_to(self.root) or not audio.is_relative_to(self.root):
            raise ValueError("Media escapes the original dataset root")
        native = dict(self.encode(self.processor, metadata["prompt"], str(source), str(audio)))
        prefix = native.pop("input_ids")[0].tolist()
        prompt_mask = native.pop("attention_mask")[0].tolist()
        native.pop("use_audio_in_video")
        # Keep native AV tokens paired with their tensors. Upstream truncates
        # response tokens using the space remaining after this native prefix.
        lengths = [len(tokenizer(answer, add_special_tokens=False)["input_ids"]) + 1
                   for answer in (chosen, rejected)]
        budget = max_length - len(prefix)
        if len(prefix) > max_prompt_length or budget < 1:
            raise ValueError(f"{metadata['id']}: native AV prompt alone exceeds the input budget; "
                             "media tokens cannot be sliced independently of media tensors")
        # Empty text prompt: the full prompt is already serialized above.
        # max_prompt_length=0 makes upstream's response budget exactly `budget`.
        # Its EOS append/slicing behavior is retained, including a clipped EOS.
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

    def preflight(self, tokenizer, max_length, max_prompt_length, report_path):
        """Scan all real rows before allocating either GPU model; never drop rows."""
        report = {"max_length": max_length, "max_prompt_length": max_prompt_length,
                  "rows": [], "errors": []}
        for split, rows in self.rows.items():
            for index, row in enumerate(rows):
                route = json.dumps({k: row[k] for k in ("id", "prompt", "video", "audio")})
                try:
                    item = self.tokenize(route, row["chosen"], row["rejected"], "keep_end",
                                         tokenizer, max_length, max_prompt_length)
                    info = item["length_info"]
                    report["rows"].append(dict(split=split, id=row["id"], **info,
                        total=info["prompt"]+max(info["chosen_after"], info["rejected_after"])))
                    del item
                except Exception as exc:
                    report["errors"].append(dict(split=split, id=row["id"],
                                                 error=f"{type(exc).__name__}: {exc}"))
                if (index + 1) % 100 == 0:
                    print(f"Length preflight {split}: {index+1}/{len(rows)}", flush=True)
        report["truncated_rows"] = sum(r["chosen_before"] != r["chosen_after"] or
                                       r["rejected_before"] != r["rejected_after"]
                                       for r in report["rows"])
        Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if report["errors"]:
            raise ValueError(f"Preflight found {len(report['errors'])} invalid rows; see {report_path}. "
                             "No GPU training started; no rows were silently excluded.")
        return report


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
