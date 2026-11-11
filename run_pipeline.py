"""Epoch30 SFT candidates and original AffectGPT evaluation.

DPO source is verbatim in vendor/dpo. Use its train.py, requirements.txt and
configuration directly. Native Omni audiovisual DPO training is not connected.
Commands: prepare, fetch, candidates, infer, evaluate.
"""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from dpo_interface import (AFFECTGPT, SFT_INTERFACE, ADAPTER_REPO,
    ADAPTER_REVISION, add_sft_path, verify_epoch30)


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


def evaluate(args):
    cmd = [sys.executable, str(SFT_INTERFACE / "run_affectgpt_omni.py"), args.command,
           "--source", str(AFFECTGPT), "--runtime", str(args.runtime), "--dataset-root", str(args.dataset_root)]
    if args.command == "infer":
        cmd += ["--", "--options", "inference.test_epoch=1", f"inference.ckpt_root={args.checkpoints}"]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "fetch", "candidates", "infer", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--runtime", type=Path, required=True)
        if name != "fetch":
            sub.add_argument("--dataset-root", type=Path, required=True)
        if name == "candidates":
            sub.add_argument("--snapshot", type=Path, required=True)
            sub.add_argument("--adapter", type=Path, required=True)
        if name == "candidates":
            sub.add_argument("--output", type=Path, required=True)
        if name == "candidates":
            sub.add_argument("--seed", type=int, default=0)
        if name == "infer":
            sub.add_argument("--checkpoints", type=Path, required=True)
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
