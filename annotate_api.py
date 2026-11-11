"""API-only annotation: execute original metadata methods without model imports.

Uses the existing run_pipeline instruction, row serialization and API parser.
Original source class/function bodies are compiled unchanged from the pinned
checkout; only irrelevant module imports are omitted. No torch or GPU needed.
Each response is saved to its own file before aggregation, including failures.
"""
import argparse
import ast
import base64
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import random
import re
import runpy
import subprocess
import time
from types import SimpleNamespace
import urllib.request
import urllib.error
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import requests
import yaml

HERE = Path(__file__).resolve().parent
SOURCE_REVISION = "95b9257758df29fa6481c138252cdd317cce1735"
SOURCE = HERE / "vendor/affectgpt/AffectGPT"


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = []
    found = set()
    for node in tree.body:
        node_names = {node.name} if isinstance(node, (ast.ClassDef, ast.FunctionDef)) else (
            {t.id for t in node.targets if isinstance(t, ast.Name)} if isinstance(node, ast.Assign) else set())
        if node_names & names:
            selected.append(node)
            found |= node_names & names
    if found != names:
        raise ValueError(f"Required original definitions missing: {names-found}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def original_rows(root, seed):
    revision = subprocess.check_output(["git", "-C", str(SOURCE.parent), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError("Original source revision changed")
    cfgpath = SOURCE/"train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml"
    cfg = yaml.safe_load(cfgpath.read_text(encoding="utf-8"))["datasets"]["mercaptionplus"]
    if cfg.get("ratio", 1) != 1:
        raise ValueError("Only the complete original membership is allowed")
    config = SimpleNamespace(**runpy.run_path(str(SOURCE/"config.py")))
    target = root/"mer2025-dataset"
    config.DATA_DIR["MERCaptionPlus"] = str(target)
    config.PATH_TO_RAW_VIDEO["MERCaptionPlus"] = str(target/"video")
    config.PATH_TO_RAW_AUDIO["MERCaptionPlus"] = str(target/"audio")
    config.PATH_TO_RAW_FACE["MERCaptionPlus"] = str(target/"openface_face")
    config.PATH_TO_TRANSCRIPTIONS["MERCaptionPlus"] = str(target/"subtitle_chieng.csv")
    ns = dict(os=os, pd=pd, np=np, re=re, random=random, config=config)
    definitions(SOURCE/"toolkit/utils/functions.py", {"string_to_list"}, ns)
    definitions(SOURCE/"my_affectgpt/datasets/datasets/base_dataset.py", {"BaseDataset"}, ns)
    definitions(SOURCE/"my_affectgpt/datasets/datasets/mercaptionplus_dataset.py", {"MERCaptionPlus_Dataset"}, ns)
    dataset = ns["MERCaptionPlus_Dataset"](dataset_cfg=SimpleNamespace(**cfg), model_cfg=None)
    random.seed(seed)
    rows = [original_row(dataset, sample,
            dataset.get_cur_label_type(dataset.label_type_candidates, dataset.label_type), root)
            for sample in dataset.annotation]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Empty or duplicated original membership")
    for row in rows:
        if not isinstance(row["ground_truth_answer"], str) or not row["ground_truth_answer"].strip():
            raise ValueError(f"Missing original answer: {row['id']}")
    return rows


def temporary_video(path, endpoint, key):
    # Official temporary-storage API; private, account/model-bound, expires after 48h.
    base = endpoint.split("/compatible-mode/")[0]
    response = requests.get(base+"/api/v1/uploads", params={"action": "getPolicy", "model": NEGATIVE_MODEL},
                            headers={"Authorization": "Bearer "+key}, timeout=(30, 120))
    if response.status_code != 200:
        raise RuntimeError(f"Temporary-upload policy failed: HTTP {response.status_code}")
    policy = response.json()["data"]
    if urlsplit(policy["upload_host"]).scheme != "https" or not urlsplit(policy["upload_host"]).hostname.endswith(".aliyuncs.com"):
        raise ValueError("Unexpected official upload host")
    object_key = policy["upload_dir"]+"/"+path.name
    fields = {"OSSAccessKeyId": (None, policy["oss_access_key_id"]), "Signature": (None, policy["signature"]),
              "policy": (None, policy["policy"]), "x-oss-object-acl": (None, policy["x_oss_object_acl"]),
              "x-oss-forbid-overwrite": (None, policy["x_oss_forbid_overwrite"]),
              "key": (None, object_key), "success_action_status": (None, "200")}
    with path.open("rb") as stream:
        fields["file"] = (path.name, stream)
        response = requests.post(policy["upload_host"], files=fields, timeout=(30, 180))
    if response.status_code != 200:
        raise RuntimeError(f"Temporary video upload failed: HTTP {response.status_code}")
    return "oss://"+object_key


def save_json(path, data):
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def input_rejected(record):
    error = record.get("annotation_api", {}).get("error", {})
    return error.get("http_status") == 400 and "data_inspection_failed" in error.get("body", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--wait-for-download", action="store_true")
    parser.add_argument("--continue-on-input-rejection", action="store_true",
                        help="Retain API content-rejection records and continue the remaining samples")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    args.dataset_root = args.dataset_root.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = args.output.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    definitions(HERE/"run_pipeline.py", {"original_row", "NEGATIVE_MODEL", "NEGATIVE_INSTRUCTION",
                                       "qwen_negative", "video_digest"}, globals())
    rows = original_rows(args.dataset_root, args.seed)
    args.output.with_suffix(".inputs.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    print(json.dumps(dict(original_samples=len(rows), seed=args.seed)), flush=True)
    if args.prepare_only:
        return
    if args.wait_for_download:
        receipt = args.dataset_root/"mer2025-dataset/video-download-complete.json"
        print("Waiting for the complete CRC-verified original video download", flush=True)
        while not receipt.exists():
            time.sleep(15)
        if json.loads(receipt.read_text()).get("samples") != len(rows):
            raise ValueError("Download membership mismatch")
    selected = rows[:args.limit]
    for row in selected:
        media = (args.dataset_root/row["video"]).resolve()
        if not media.is_relative_to(args.dataset_root) or not media.is_file():
            raise ValueError(f"Missing original media: {row['id']}")
    endpoint = os.environ["DASHSCOPE_BASE_URL"].rstrip("/")
    key = os.environ["DASHSCOPE_API_KEY"]
    if urlsplit(endpoint).scheme != "https":
        raise ValueError("HTTPS API endpoint required")
    settings = dict(model=NEGATIVE_MODEL, instruction=NEGATIVE_INSTRUCTION, endpoint=endpoint,
                    seed=args.seed, max_tokens=args.max_tokens)
    folder = args.output.with_suffix(".rows")
    folder.mkdir(exist_ok=True)
    expected = {r["id"]: r for r in selected}
    previous = {}
    for path in folder.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        row = expected.get(record["id"])
        if row is None:
            continue
        if any(record.get(k) != v for k, v in row.items()) or record.get("annotation_api", {}).get("settings") != settings:
            raise ValueError(f"Resume metadata changed: {record['id']}")
        if record["chosen"] != row["ground_truth_answer"]:
            raise ValueError(f"Changed chosen answer: {record['id']}")
        if not record.get("rejected") and not (args.continue_on_input_rejection and input_rejected(record)):
            raise ValueError(f"Saved unsuccessful attempt needs review: {record['id']}")
        if record["annotation_api"]["video_sha256"] != video_digest(args.dataset_root/row["video"]):
            raise ValueError("Original media changed")
        previous[record["id"]] = record
    print(json.dumps(dict(selected=len(selected), completed=len(previous), workers=args.workers)), flush=True)

    def annotate(row):
        path = args.dataset_root/row["video"]
        sha = video_digest(path)
        large = 4*((path.stat().st_size+2)//3) >= 10_000_000
        record = dict(row, chosen=row["ground_truth_answer"], rejected="")
        record["annotation_api"] = dict(settings=settings, video_sha256=sha,
                                         transport="temporary_oss" if large else "base64")
        try:
            video_url = temporary_video(path, endpoint, key) if large else "data:video/mp4;base64,"+base64.b64encode(path.read_bytes()).decode("ascii")
            result = qwen_negative(row, video_url, endpoint, key, args.max_tokens)
            record["annotation_api"]["response"] = result
            if (result["stream_complete"] and result["finish_reason"] == "stop" and result["content"].strip()
                    and result["content"].strip() != row["ground_truth_answer"].strip()):
                record["rejected"] = result["content"]
        except urllib.error.HTTPError as exc:
            record["annotation_api"]["error"] = dict(http_status=exc.code, body=exc.read(4096).decode(errors="replace"))
        except Exception as exc:
            record["annotation_api"]["error"] = dict(type=type(exc).__name__)
        save_json(folder/(row["id"]+".json"), record)
        return record

    pending = iter(row for row in selected if row["id"] not in previous)
    failed = False
    with args.output.open("w", encoding="utf-8") as journal:
        for row in selected:
            if row["id"] in previous:
                journal.write(json.dumps(previous[row["id"]], ensure_ascii=False)+"\n")
        journal.flush(); os.fsync(journal.fileno())
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            active = {pool.submit(annotate, row) for row in (next(pending, None) for _ in range(args.workers)) if row is not None}
            while active:
                complete, active = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in complete:
                    record = future.result()
                    journal.write(json.dumps(record, ensure_ascii=False)+"\n"); journal.flush(); os.fsync(journal.fileno())
                    previous[record["id"]] = record
                    failed |= not bool(record["rejected"]) and not (args.continue_on_input_rejection and input_rejected(record))
                    print(json.dumps(dict(saved=len(previous), total=len(selected), id=record["id"], valid=bool(record["rejected"]),
                                          usage=record["annotation_api"].get("response", {}).get("usage"))), flush=True)
                if not failed:
                    while len(active) < args.workers:
                        row = next(pending, None)
                        if row is None:
                            break
                        active.add(pool.submit(annotate, row))
    if failed:
        raise RuntimeError("Unsuccessful attempt saved; stopped before submitting more requests. Review before resuming.")
    temporary = args.output.with_suffix(".sorted.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(previous[row["id"]], ensure_ascii=False)+"\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, args.output)
    failed_ids = [row["id"] for row in selected if not previous[row["id"]].get("rejected")]
    save_json(args.output.with_suffix(".complete.json"), dict(samples=len(selected), original_samples=len(rows),
              processed_all=len(selected)==len(rows), successful=len(selected)-len(failed_ids),
              failed_ids=failed_ids, full_dataset=len(selected)==len(rows) and not failed_ids,
              sha256=video_digest(args.output), settings=settings,
              note="Synthetic negatives; integrity checks do not establish semantic preference quality."))


if __name__ == "__main__":
    main()
