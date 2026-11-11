"""Epoch30 Omni LoRA -> official DPO -> original AffectGPT evaluation.

Linux/Python 3.10, existing Qwen SFT environment plus requirements-interface.txt.
Commands: prepare, fetch, candidates, annotate, train, export, infer, evaluate.
Training requires already labelled --pairs and --eval-pairs; no ranking or
split creation is performed. Official source files are pinned and unchanged.
annotate constructs synthetic rejected answers through Qwen's API; it is
project data preparation, not part of the original DPO algorithm.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import urllib.request
from urllib.parse import urlsplit

from dpo_interface import (DPO, AFFECTGPT, SFT_INTERFACE, ADAPTER_REPO,
    ADAPTER_REVISION, add_sft_path, verify_epoch30, read_pairs)


def enter_affectgpt(args, native=False):
    add_sft_path()
    from run_affectgpt_omni import prepare_execution
    execution = prepare_execution(AFFECTGPT, args.runtime, args.dataset_root)
    os.chdir(execution)
    sys.path.insert(0, str(execution))
    if native:
        from omni_components import install
        install(str(args.snapshot))
    return execution


def training_dataset(args, metadata_only=False):
    enter_affectgpt(args, native=not metadata_only)
    from omegaconf import OmegaConf
    from run_affectgpt_omni import REFERENCE_CONFIG
    from my_affectgpt.datasets.datasets.mercaptionplus_dataset import MERCaptionPlus_Dataset
    cfg = OmegaConf.load(REFERENCE_CONFIG)
    # The original constructor supports model_cfg=None for annotation-only use.
    # No tokenizer, model weights, or GPU are needed for API annotation.
    if metadata_only and cfg.datasets.mercaptionplus.get("ratio", 1) != 1:
        raise ValueError("Annotation-only loading requires the complete original dataset")
    return MERCaptionPlus_Dataset(dataset_cfg=cfg.datasets.mercaptionplus,
                                 model_cfg=None if metadata_only else cfg.model), cfg


def original_row(dataset, sample, label_type, root):
    qa = dataset.get_qa_pairs(dataset.dataset, label_type, sample)
    prompt = dataset.get_prompt_for_multimodal("textonly", sample["subtitle"], qa["question"])
    video = Path(dataset._get_video_path(sample)).resolve().relative_to(root).as_posix()
    audio = Path(dataset._get_audio_path(sample)).resolve().relative_to(root).as_posix()
    return dict(id=sample["name"], dataset="MERCaptionPlus", split="train", label_type=label_type,
        prompt=prompt, video=video, audio=audio, ground_truth_answer=qa["answer"], annotation=dict(sample))


def fetch(args):
    add_sft_path()
    from huggingface_hub import snapshot_download
    from omni_components import MODEL_ID, MODEL_REVISION
    cache = str(args.runtime / "cache/huggingface")
    snapshot = snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=cache)
    adapter = Path(snapshot_download(ADAPTER_REPO, revision=ADAPTER_REVISION,
        allow_patterns=["epoch-030/*"], cache_dir=cache)) / "epoch-030"
    verify_epoch30(adapter)
    print(f"--snapshot {snapshot}\n--adapter {adapter}")


def candidates(args):
    import numpy as np
    import torch
    from dpo_interface import load_sft_model
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset, cfg = training_dataset(args)
    from omni_components import AffectGPTThinker
    from my_affectgpt.conversation.conversation_video import Chat
    model = AffectGPTThinker(load_sft_model(args.snapshot, args.adapter)).eval()
    chat = Chat(model, cfg.model, device="cuda:0")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for sample in dataset.annotation:
            kind = dataset.get_cur_label_type(dataset.label_type_candidates, dataset.label_type)
            row = original_row(dataset, sample, kind, args.dataset_root)
            chat.postprocess_frame({"raw_frame": str(args.dataset_root / row["video"])})
            images = dict.fromkeys(("frame", "face", "audio", "image", "multi"))
            with torch.no_grad():
                row["candidates"] = [chat.answer_sample(prompt=row["prompt"], img_list=images,
                    num_beams=1, temperature=1, do_sample=True, top_p=0.9,
                    max_new_tokens=1200, max_length=2000) for _ in range(2)]
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()


NEGATIVE_MODEL = "qwen3.8-omni-flash"
# An explicit synthetic-negative instruction, not an upstream DPO prompt.
NEGATIVE_INSTRUCTION = (
    "Create one rejected answer for preference training. Inspect the supplied video "
    "and its embedded audio, the original question/subtitle, and the dataset's "
    "reference answer. The reference answer will be kept unchanged as the chosen "
    "answer. Write a plausible but substantively less accurate answer to the same "
    "question, relative to the reference and the audiovisual evidence. Use the "
    "same language and answer format as the reference, with comparable length. "
    "Do not merely paraphrase the reference or produce an equally correct answer. "
    "Return only the rejected answer, without a heading, explanation of the "
    "construction, or any statement that it is deliberately incorrect."
)


def qwen_negative(row, video_url, endpoint, api_key, max_tokens):
    """Official Chat Completions video_url + SSE protocol; no local AV decoding."""
    payload = dict(model=NEGATIVE_MODEL, modalities=["text"], stream=True,
                   stream_options={"include_usage": True}, messages=[
        {"role": "system", "content": NEGATIVE_INSTRUCTION},
        {"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "text", "text": row["prompt"]},
            {"type": "text", "text": "Dataset reference answer:\n" + row["ground_truth_answer"]}]}])
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    request = urllib.request.Request(endpoint + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    result = dict(content="", reasoning_content="", finish_reason=None, usage=None)
    complete = False
    # No automatic retry: a timed-out request may already have been billed.
    with urllib.request.urlopen(request, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                complete = True
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError("Qwen returned a streaming API error; no pair accepted")
            for key in ("id", "model", "created"):
                if key in chunk:
                    result[key] = chunk[key]
            if chunk.get("usage") is not None:
                result["usage"] = chunk["usage"]
            for choice in chunk.get("choices", []):
                if choice.get("index", 0) != 0:
                    raise ValueError("Expected exactly one API answer")
                delta = choice.get("delta", {})
                for key in ("content", "reasoning_content"):
                    result[key] += delta.get(key) or ""
                if choice.get("finish_reason") is not None:
                    result["finish_reason"] = choice["finish_reason"]
    result["stream_complete"] = complete
    return result


def video_digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def annotate(args):
    """Keep original targets; request one synthetic negative per original row."""
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise ValueError("max-tokens must be positive")
    endpoint = os.environ.get("DASHSCOPE_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not args.dry_run and (not api_key or urlsplit(endpoint).scheme != "https"):
        raise ValueError("Set DASHSCOPE_API_KEY and your region's HTTPS DASHSCOPE_BASE_URL")
    dataset, _ = training_dataset(args, metadata_only=True)
    random.seed(args.seed)
    rows = [original_row(dataset, sample,
        dataset.get_cur_label_type(dataset.label_type_candidates, dataset.label_type),
        args.dataset_root) for sample in dataset.annotation]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Empty annotations or duplicate sample IDs")
    rows = rows[:args.limit]
    urls = json.loads(args.video_urls.read_text(encoding="utf-8")) if args.video_urls else {}
    # Check every selected media reference before the first paid request.
    for row in rows:
        if not isinstance(row["ground_truth_answer"], str) or not row["ground_truth_answer"].strip():
            raise ValueError(f"{row['id']}: missing original answer")
        path = (args.dataset_root / row["video"]).resolve()
        if not path.is_relative_to(args.dataset_root) or not path.is_file():
            raise ValueError(f"{row['id']}: missing original video or path outside dataset root")
        if row["id"] in urls:
            if not isinstance(urls[row["id"]], str) or urlsplit(urls[row["id"]]).scheme != "https":
                raise ValueError(f"{row['id']}: video URL must be HTTPS")
        elif 4 * ((path.stat().st_size + 2) // 3) >= 10_000_000:
            raise ValueError(f"{row['id']}: exceeds API Base64 limit; supply an HTTPS URL "
                             "for the unchanged original video using --video-urls (JSON id -> URL)")
    settings = dict(model=NEGATIVE_MODEL, instruction=NEGATIVE_INSTRUCTION,
                    endpoint=endpoint, seed=args.seed, max_tokens=args.max_tokens)
    previous = read_pairs(args.output) if args.output.exists() and args.output.stat().st_size else []
    if len(previous) > len(rows):
        raise ValueError("Existing output contains more rows than this selection")
    for old, row in zip(previous, rows):
        if (any(old.get(k) != v for k, v in row.items()) or
                old["chosen"] != row["ground_truth_answer"] or
                old.get("annotation_api", {}).get("settings") != settings):
            raise ValueError(f"{row['id']}: resume data or API settings differ")
        media_hash = video_digest(args.dataset_root / row["video"])
        if old["annotation_api"].get("video_sha256") != media_hash:
            raise ValueError(f"{row['id']}: original video changed since annotation")
    print(json.dumps(dict(selected=len(rows), completed=len(previous),
                          remaining=len(rows)-len(previous), output=str(args.output))))
    if args.dry_run or len(previous) == len(rows):
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows[len(previous):], start=len(previous)+1):
            media_path = args.dataset_root / row["video"]
            media_hash = video_digest(media_path)
            video_url = urls.get(row["id"])
            if video_url is None:
                video_url = "data:video/mp4;base64," + base64.b64encode(media_path.read_bytes()).decode("ascii")
            result = qwen_negative(row, video_url, endpoint, api_key, args.max_tokens)
            answer = result["content"]
            valid = (result["stream_complete"] and result["finish_reason"] == "stop" and
                     bool(answer.strip()) and answer.strip() != row["ground_truth_answer"].strip())
            # These are integrity checks, not a semantic score or quality guarantee.
            row.update(chosen=row["ground_truth_answer"], rejected=answer if valid else "",
                annotation_api=dict(settings=settings, response=result,
                    video_sha256=media_hash,
                    transport="https_url" if row["id"] in urls else "base64"))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if not valid:
                raise ValueError(f"{row['id']}: incomplete, empty or identical answer; raw response "
                                 "saved with empty rejected. Resolve before resuming/training.")
            print(json.dumps(dict(completed=index, total=len(rows), id=row["id"], usage=result["usage"])))


def train(args):
    if sys.platform != "linux" or sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use Linux/Python 3.10 for the unchanged official DPO NumPy seed handling")
    if not args.preflight_only and (args.beta is None or not math.isfinite(args.beta) or args.beta <= 0):
        raise ValueError("beta must be finite and positive")
    if not args.preflight_only and args.adapter is None:
        raise ValueError("Training requires the verified epoch30 --adapter")
    if not math.isfinite(args.lr) or args.lr <= 0 or args.max_length < 2:
        raise ValueError("lr must be finite and positive; max-length must be at least 2")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rows, eval_rows = read_pairs(args.pairs), read_pairs(args.eval_pairs)
    dataset, _ = training_dataset(args, metadata_only=True)
    source = {sample["name"]: sample for sample in dataset.annotation}
    if {row["id"] for row in rows} != set(source):
        raise ValueError("Preference membership differs from original training annotations")
    for row in rows:
        expected = original_row(dataset, source[row["id"]], row["label_type"], args.dataset_root)
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Changed original annotation/prompt/media: {row['id']}")
        if "candidates" in row and sorted([row["chosen"], row["rejected"]]) != sorted(row["candidates"]):
            raise ValueError(f"{row['id']}: preferences differ from the supplied candidates")
    sys.path.insert(0, str(DPO))
    import torch
    from omegaconf import OmegaConf
    import preference_datasets
    import trainers
    import utils
    from omni_components import load_processor
    from dpo_interface import load_sft_model, OmniModel, PairData, interfaces
    for module in (preference_datasets, trainers, utils):
        if Path(module.__file__).resolve().parent != DPO:
            raise RuntimeError(f"Module collision with official DPO: {module.__file__}")
    # Import and call the actual official worker, including its W&B setup.
    spec = importlib.util.spec_from_file_location("official_dpo_entrypoint", DPO / "train.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    cfg = OmegaConf.load(DPO / "config/config.yaml")
    cfg.pop("defaults")
    cfg.model = OmegaConf.load(DPO / "config/model/blank_model.yaml")
    cfg.loss = OmegaConf.load(DPO / "config/loss/dpo.yaml")
    cfg.loss.beta = args.beta
    cfg.model.name_or_path = cfg.model.tokenizer_name_or_path = str(args.snapshot)
    cfg.model.policy_dtype = cfg.model.reference_dtype = "bfloat16"
    cfg.datasets, cfg.trainer = ["affectgpt_preferences"], "OmniTrainer"
    cfg.local_run_dir, cfg.local_dirs = str(args.output), [str(args.runtime / "cache")]
    cfg.exp_name = "epoch30_standard_dpo_affectgpt"
    cfg.lr = args.lr
    cfg.max_length, cfg.max_prompt_length = args.max_length, args.max_length - 1
    cfg.n_eval_examples = args.n_eval_examples
    cfg.sample_during_eval = args.sample_during_eval
    cfg.activation_checkpointing = args.gradient_checkpointing
    for name in ("batch_size", "eval_batch_size", "gradient_accumulation_steps"):
        if getattr(args, name) is not None:
            cfg[name] = getattr(args, name)
    if min(cfg.batch_size, cfg.eval_batch_size, cfg.gradient_accumulation_steps) < 1:
        raise ValueError("Batch sizes and gradient accumulation must be positive")
    if cfg.batch_size % cfg.gradient_accumulation_steps:
        raise ValueError("batch_size must be divisible by gradient_accumulation_steps")
    if len(rows) % cfg.batch_size or cfg.eval_every % cfg.batch_size:
        raise ValueError("batch_size must divide pair count and eval_every; no samples may be silently dropped")
    if len(eval_rows) < cfg.eval_batch_size:
        raise ValueError("Supply at least one complete preference-evaluation batch; evaluation is not disabled")
    if cfg.n_eval_examples < cfg.eval_batch_size or cfg.n_eval_examples % cfg.eval_batch_size:
        raise ValueError("n-eval-examples must be positive and divisible by eval-batch-size")
    if args.smoke_test:
        cfg.exp_name += "_longest_batch_20_step_smoke"
        cfg.n_epochs, cfg.n_examples = None, 20 * cfg.batch_size
    args.output.mkdir(parents=True, exist_ok=False)
    processor = load_processor(str(args.snapshot))
    data = PairData(rows, eval_rows, processor, args.dataset_root,
                    preference_datasets.tokenize_batch_element)
    report = data.preflight(processor.tokenizer, cfg.max_length, cfg.max_prompt_length,
                            args.output / "length-check.json")
    if args.preflight_only:
        print(f"Length preflight passed; truncated rows: {report['truncated_rows']}; no training started")
        return
    if args.smoke_test:
        # Real training rows only, selected after the complete data scan.
        # Upstream's n_examples loop repeats this longest batch for 20 updates.
        lengths = {r["id"]: r["total"] for r in report["rows"] if r["split"] == "train"}
        data.rows["train"] = sorted(rows, key=lambda r: lengths[r["id"]], reverse=True)[:cfg.batch_size]
        (args.output / "smoke-samples.json").write_text(json.dumps(
            {"purpose": "20-step diagnostic, not a full experiment",
             "ids": [r["id"] for r in data.rows["train"]]}, indent=2), encoding="utf-8")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires one visible CUDA GPU with BF16 support")
    OmegaConf.resolve(cfg)
    missing = OmegaConf.missing_keys(cfg)
    if missing:
        raise ValueError(f"Missing official configuration fields: {missing}")
    (args.runtime / "cache").mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = utils.get_local_dir(cfg.local_dirs)
    OmegaConf.save(cfg, args.output / "config.yaml")
    policy = OmniModel(load_sft_model(args.snapshot, args.adapter, trainable=True))
    reference = OmniModel(load_sft_model(args.snapshot, args.adapter, trainable=False))
    if cfg.activation_checkpointing:
        # BasicTrainer's upstream checkpointing switch only affects FSDP.
        # Apply the native Thinker API here, without replacing the DPO trainer.
        policy.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    utils.disable_dropout(policy)
    utils.disable_dropout(reference)
    with interfaces(preference_datasets, trainers, data):
        official.worker_main(0, 1, cfg, policy, reference)


def export(args):
    """Copy trained LoRA tensors into the existing SFT inference checkpoint format."""
    import torch
    from safetensors.torch import load_file
    verify_epoch30(args.adapter)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state"]
    original = load_file(str(args.adapter / "adapter_model.safetensors"))
    expected = {"thinker." + key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
                ".lora_B.weight", ".lora_B.default.weight"): value for key, value in original.items()}
    trained = {key: value for key, value in state.items() if ".lora_" in key}
    if set(trained) != set(expected) or any(trained[k].shape != expected[k].shape for k in expected):
        raise ValueError("Checkpoint does not contain the complete language LoRA")
    if not all(torch.isfinite(value).all() for value in trained.values()):
        raise ValueError("Nonfinite checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "checkpoint_000001_dpo_final.pth"
    torch.save({"model": trained, "epoch": 1}, destination)
    reloaded = torch.load(destination, map_location="cpu", weights_only=True)["model"]
    if set(reloaded) != set(trained) or not all(torch.equal(value, reloaded[key]) for key, value in trained.items()):
        raise RuntimeError("Checkpoint conversion changed tensor values")


def evaluate(args):
    cmd = [sys.executable, str(SFT_INTERFACE / "run_affectgpt_omni.py"), args.command,
           "--source", str(AFFECTGPT), "--runtime", str(args.runtime), "--dataset-root", str(args.dataset_root)]
    if args.command == "infer":
        cmd += ["--", "--options", "inference.test_epoch=1", f"inference.ckpt_root={args.checkpoints}"]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "fetch", "candidates", "annotate", "train", "infer", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--runtime", type=Path, required=True)
        if name != "fetch":
            sub.add_argument("--dataset-root", type=Path, required=True)
        if name in ("candidates", "train"):
            sub.add_argument("--snapshot", type=Path, required=True)
            sub.add_argument("--adapter", type=Path, required=name == "candidates")
            sub.add_argument("--output", type=Path, required=True)
        if name == "candidates":
            sub.add_argument("--seed", type=int, default=0)
        if name == "annotate":
            sub.add_argument("--output", type=Path, required=sys.platform != "win32",
                             default=Path("C:/SFT Qwen/qwen3.8-omni-flash-preferences.jsonl")
                             if sys.platform == "win32" else None)
            sub.add_argument("--seed", type=int, default=0)
            sub.add_argument("--limit", type=int, help="First N original rows; omit for the full dataset")
            sub.add_argument("--max-tokens", type=int, help="Optional API output limit including thinking")
            sub.add_argument("--video-urls", type=Path,
                             help="Optional JSON mapping sample IDs to HTTPS URLs of original videos")
            sub.add_argument("--dry-run", action="store_true", help="Check inputs without API calls/writes")
        if name == "train":
            sub.add_argument("--pairs", type=Path, required=True,
                             help="Original training rows with already labelled chosen/rejected")
            sub.add_argument("--eval-pairs", type=Path, required=True,
                             help="Explicitly supplied preference-validation rows; no automatic split")
            sub.add_argument("--beta", type=float, help="Required for training; not needed for --preflight-only")
            sub.add_argument("--lr", type=float, default=5e-6)
            sub.add_argument("--batch-size", type=int, default=1)
            sub.add_argument("--eval-batch-size", type=int, default=1)
            sub.add_argument("--gradient-accumulation-steps", type=int)
            sub.add_argument("--n-eval-examples", type=int, default=16)
            sub.add_argument("--sample-during-eval", action=argparse.BooleanOptionalAction, default=False)
            sub.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
            sub.add_argument("--max-length", type=int, default=4096,
                             help="Native prompt plus answer/EOS; long answers use original DPO truncation")
            sub.add_argument("--preflight-only", action="store_true",
                             help="Scan all train/eval lengths on CPU and exit before GPU model loading")
            sub.add_argument("--smoke-test", action="store_true",
                             help="After the full length scan, run 20 updates on the longest training batch")
        if name == "infer":
            sub.add_argument("--checkpoints", type=Path, required=True)
    sub = commands.add_parser("export")
    sub.add_argument("--checkpoint", type=Path, required=True)
    sub.add_argument("--adapter", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.command == "prepare":
        print(enter_affectgpt(args))
    elif args.command in ("infer", "evaluate"):
        evaluate(args)
    else:
        globals()[args.command](args)


if __name__ == "__main__":
    main()
