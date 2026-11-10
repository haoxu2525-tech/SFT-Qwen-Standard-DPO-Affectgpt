"""Epoch30 Omni -> official DPO -> original AffectGPT evaluation.

Linux/Python 3.10, existing Qwen SFT environment plus requirements-interface.txt.
Commands: prepare, fetch, candidates, rank, train, export, infer, evaluate.
Each command's --help lists its inputs. Official sources are pinned submodules.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

from dpo_interface import (DPO, AFFECTGPT, SFT_INTERFACE, ADAPTER_REPO,
    ADAPTER_REVISION, add_sft_path, verify_epoch30, read_rows)


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


def training_dataset(args):
    enter_affectgpt(args, native=True)
    from omegaconf import OmegaConf
    from run_affectgpt_omni import REFERENCE_CONFIG
    from my_affectgpt.datasets.datasets.mercaptionplus_dataset import MERCaptionPlus_Dataset
    cfg = OmegaConf.load(REFERENCE_CONFIG)
    return MERCaptionPlus_Dataset(dataset_cfg=cfg.datasets.mercaptionplus, model_cfg=cfg.model), cfg


def original_row(dataset, sample, label_type, root):
    qa = dataset.get_qa_pairs(dataset.dataset, label_type, sample)
    prompt = dataset.get_prompt_for_multimodal("textonly", sample["subtitle"], qa["question"])
    video = Path(dataset._get_video_path(sample)).resolve().relative_to(root).as_posix()
    return dict(id=sample["name"], dataset="MERCaptionPlus", split="train", label_type=label_type,
        prompt=prompt, video=video, ground_truth_answer=qa["answer"], annotation=dict(sample))


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
    from dpo_interface import load_policy
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset, cfg = training_dataset(args)
    from omni_components import AffectGPTThinker
    from my_affectgpt.conversation.conversation_video import Chat
    model = AffectGPTThinker(load_policy(args.snapshot, args.adapter, trainable=False).thinker).eval()
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


def rank(args):
    # Explicit data-preparation criterion: original EW-F1 vs original ovlabel.
    # This measures label agreement, not all reasoning details of a description.
    rows = read_rows(args.candidates)
    enter_affectgpt(args)
    from my_affectgpt.evaluation.ew_metric import (
        func_read_batch_calling_model, extract_openset_batchcalling, wheel_metric_calculation)
    predictions = {json.dumps([row["id"], i]): answer for row in rows
                   for i, answer in enumerate(row["candidates"])}
    if any(len(row["candidates"]) != 2 for row in rows):
        raise ValueError("Each sample must have two candidate answers")
    if args.output.exists():
        raise FileExistsError(args.output)
    llm, tokenizer, sampling = func_read_batch_calling_model("Qwen25")
    names, outputs = extract_openset_batchcalling(name2reason=predictions,
        llm=llm, tokenizer=tokenizer, sampling_params=sampling)
    labels = dict(zip(names, outputs))
    unresolved = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            row["candidate_labels"] = [labels[json.dumps([row["id"], i])] for i in range(2)]
            scores = [wheel_metric_calculation(
                name2gt={row["id"]: row["annotation"]["ovlabel"]},
                name2pred={row["id"]: label}, inter_print=False)[0] for label in row["candidate_labels"]]
            row["preference_scores"] = [float(score) if math.isfinite(score) else None for score in scores]
            row["chosen"] = row["rejected"] = None
            if all(math.isfinite(score) for score in scores) and scores[0] != scores[1] and row["candidates"][0] != row["candidates"][1]:
                winner = int(scores[1] > scores[0])
                row["chosen"], row["rejected"] = row["candidates"][winner], row["candidates"][1-winner]
            else:
                unresolved += 1  # preserve ties/nonfinite original scores; do not invent a winner
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Saved {len(rows)} rows; {unresolved} unresolved preferences. Training rejects unresolved rows.")


def train(args):
    if sys.platform != "linux" or sys.version_info[:2] != (3, 10):
        raise RuntimeError("Use Linux/Python 3.10 for the unchanged official DPO NumPy seed handling")
    if not math.isfinite(args.beta) or args.beta <= 0:
        raise ValueError("beta must be finite and positive")
    rows = read_rows(args.pairs)
    dataset, _ = training_dataset(args)
    source = {sample["name"]: sample for sample in dataset.annotation}
    if {row["id"] for row in rows} != set(source):
        raise ValueError("Preference membership differs from original training annotations")
    for row in rows:
        expected = original_row(dataset, source[row["id"]], row["label_type"], args.dataset_root)
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError(f"Changed original annotation/prompt/media: {row['id']}")
        if not row["chosen"] or not row["rejected"] or row["chosen"] == row["rejected"]:
            raise ValueError(f"Unresolved preference: {row['id']}; no samples were excluded")
        if sorted([row["chosen"], row["rejected"]]) != sorted(row["candidates"]):
            raise ValueError("Preference answers differ from generated candidates")
    sys.path.insert(0, str(DPO))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    import preference_datasets
    import trainers
    from utils import disable_dropout
    from omni_components import load_processor
    from dpo_interface import load_policy, PairData, data_bindings, trainer_class
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires one visible CUDA GPU with BF16 support")
    cfg = OmegaConf.load(DPO / "config/config.yaml")
    cfg.pop("defaults")
    cfg.model = OmegaConf.load(DPO / "config/model/blank_model.yaml")
    cfg.loss = OmegaConf.load(DPO / "config/loss/dpo.yaml")
    cfg.loss.beta = args.beta
    cfg.model.name_or_path = cfg.model.tokenizer_name_or_path = str(args.snapshot)
    cfg.model.policy_dtype = cfg.model.reference_dtype = "bfloat16"
    cfg.datasets = ["affectgpt_preferences"]
    cfg.local_run_dir, cfg.local_dirs = str(args.output), [str(args.runtime / "cache")]
    cfg.exp_name = "epoch30_standard_dpo_affectgpt"
    cfg.max_length = cfg.max_prompt_length = 4096
    # No invented preference validation split. Original benchmarks run separately.
    cfg.sample_during_eval = cfg.do_first_eval = cfg.wandb.enabled = False
    cfg.n_eval_examples = 0
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    if cfg.batch_size < 1 or cfg.gradient_accumulation_steps < 1 or cfg.batch_size % cfg.gradient_accumulation_steps:
        raise ValueError("batch_size must be a positive multiple of gradient_accumulation_steps")
    if len(rows) % cfg.batch_size or cfg.eval_every % cfg.batch_size:
        raise ValueError("batch_size must divide pair count and eval_every; original iterator drops incomplete batches")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.runtime / "cache").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, args.output / "config.yaml")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    policy = load_policy(args.snapshot, args.adapter, trainable=True)
    reference = load_policy(args.snapshot, args.adapter, trainable=False)
    disable_dropout(policy)
    disable_dropout(reference)
    data = PairData(rows, load_processor(str(args.snapshot)), args.dataset_root)
    with data_bindings(preference_datasets, trainers, data):
        trainer = trainer_class(trainers)(policy, cfg, cfg.seed, str(args.output), reference_model=reference)
        trainer.train()
        trainer.save()


def export(args):
    import torch
    from safetensors.torch import load_file
    verify_epoch30(args.adapter)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["state"]
    original = load_file(str(args.adapter / "adapter_model.safetensors"))
    expected = {"thinker." + key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
                ".lora_B.weight", ".lora_B.default.weight"): value for key, value in original.items()}
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
        raise ValueError("Checkpoint does not contain the complete language LoRA")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Nonfinite checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    destination = args.output / "checkpoint_000001_dpo_final.pth"
    torch.save({"model": state, "epoch": 1}, destination)
    reloaded = torch.load(destination, map_location="cpu", weights_only=True)["model"]
    if not all(torch.equal(value, reloaded[key]) for key, value in state.items()):
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
    for name in ("prepare", "fetch", "candidates", "rank", "train", "infer", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--runtime", type=Path, required=True)
        if name != "fetch":
            sub.add_argument("--dataset-root", type=Path, required=True)
        if name in ("candidates", "train"):
            sub.add_argument("--snapshot", type=Path, required=True)
            sub.add_argument("--adapter", type=Path, required=True)
        if name in ("candidates", "rank", "train"):
            sub.add_argument("--output", type=Path, required=True)
        if name == "candidates":
            sub.add_argument("--seed", type=int, default=0)
        if name == "rank":
            sub.add_argument("--candidates", type=Path, required=True)
            sub.add_argument("--criterion", choices=["affectgpt-ew-f1"], required=True,
                             help="Explicitly use original EW-F1 vs original labels; ties remain unresolved")
        if name == "train":
            sub.add_argument("--pairs", type=Path, required=True)
            sub.add_argument("--beta", type=float, required=True)
            sub.add_argument("--batch-size", type=int)
            sub.add_argument("--gradient-accumulation-steps", type=int)
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
