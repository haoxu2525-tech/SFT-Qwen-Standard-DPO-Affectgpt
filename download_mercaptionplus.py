"""Download original MERCaption+ videos from pinned MER2025 split ZIP ranges.

No media conversion or annotation changes. Requires requests and an HF read
token in HF_TOKEN or ~/.cache/huggingface/token. Every extracted member is
checked against the archive's uncompressed size and CRC before publication.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import struct
import threading
import time
import zlib

import requests

REVISION = "dffb4cef9209fb44459edc16a0c845baaa3ed4d0"
BASE = f"https://huggingface.co/datasets/MERChallenge/MER2025/resolve/{REVISION}/"
PART_SIZE = 20_971_520_000
LAST_SIZE = 758_342_775
ANNOTATIONS = ("track2_train_mercaptionplus.csv", "track3_train_mercaptionplus.csv",
               "subtitle_chieng.csv", "README_AFTER_APPROVAL.md")


class Archive:
    def __init__(self, token):
        self.token = token
        self.urls = {}
        self.lock = threading.Lock()
        self.local = threading.local()

    def session(self):
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
        return self.local.session

    def url(self, disk, refresh=False):
        with self.lock:
            if disk not in self.urls or refresh:
                suffix = f"z{disk+1:02}" if disk < 11 else "zip"
                url = BASE + f"video/video_split.{suffix}?download=true"
                with self.session().get(url, headers={"Authorization": "Bearer " + self.token,
                        "Range": "bytes=0-0"}, timeout=(30, 120), stream=True) as response:
                    if response.status_code != 206:
                        raise RuntimeError(f"Archive authorization/range failed: HTTP {response.status_code}")
                    self.urls[disk] = response.url
            return self.urls[disk]

    def read(self, absolute, size):
        pieces = []
        while size:
            disk, offset = divmod(absolute, PART_SIZE)
            if disk > 11:
                raise ValueError("Read outside archive")
            total = PART_SIZE if disk < 11 else LAST_SIZE
            length = min(size, total-offset)
            if length <= 0:
                raise ValueError("Read outside final archive part")
            expected = f"bytes {offset}-{offset+length-1}/{total}"
            for attempt in range(6):
                try:
                    url = self.url(disk, refresh=attempt > 0)
                    with self.session().get(url, headers={"Range": f"bytes={offset}-{offset+length-1}"},
                                            timeout=(30, 180)) as response:
                        if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                            raise RuntimeError(f"Invalid range response: HTTP {response.status_code}")
                        block = response.content
                        if len(block) != length:
                            raise RuntimeError("Incomplete range")
                    break
                except (requests.RequestException, RuntimeError) as exc:
                    if attempt == 5:
                        # requests exceptions can contain signed URLs. Never log them.
                        raise RuntimeError(f"Range download failed on disk {disk}: {type(exc).__name__}") from None
                    time.sleep(min(2 ** attempt, 20))
            pieces.append(block)
            size -= length
            absolute += length
        return b"".join(pieces)


def central_entries(data):
    entries = []
    position = 0
    while position < len(data):
        v = struct.unpack_from("<4s6H3I5H2I", data, position)
        if v[0] != b"PK\x01\x02":
            raise ValueError("Invalid central directory")
        nl, xl, cl = v[10:13]
        name = data[position+46:position+46+nl].decode("utf-8")
        extra = data[position+46+nl:position+46+nl+xl]
        compressed, size, disk, offset = v[8], v[9], v[13], v[16]
        p = 0
        while p < len(extra):
            kind, length = struct.unpack_from("<HH", extra, p)
            payload = extra[p+4:p+4+length]
            p += 4+length
            if kind == 1:
                q = 0
                if size == 0xffffffff:
                    size = struct.unpack_from("<Q", payload, q)[0]; q += 8
                if compressed == 0xffffffff:
                    compressed = struct.unpack_from("<Q", payload, q)[0]; q += 8
                if offset == 0xffffffff:
                    offset = struct.unpack_from("<Q", payload, q)[0]; q += 8
                if disk == 65535:
                    disk = struct.unpack_from("<I", payload, q)[0]
        entries.append(dict(name=name, compressed=compressed, size=size, disk=disk,
                            offset=offset, crc=v[7], method=v[4], flags=v[3], filename_bytes=nl))
        position += 46+nl+xl+cl
    return entries


def prepare(root, archive):
    root.mkdir(parents=True, exist_ok=True)
    for name in ANNOTATIONS:
        response = archive.session().get(BASE+name, headers={"Authorization": "Bearer " + archive.token},
                                         timeout=(30, 180))
        if response.status_code != 200:
            raise RuntimeError(f"Annotation download failed: {name}, HTTP {response.status_code}")
        (root/name).write_bytes(response.content)
    tail = archive.read(11*PART_SIZE+LAST_SIZE-65536, 65536)
    end = tail.rfind(b"PK\x05\x06")
    if end < 0:
        raise ValueError("Missing ZIP end record")
    _, disk, cd_disk, _, _, cd_size, cd_offset, _ = struct.unpack_from("<4s4H2IH", tail, end)
    if disk != 11 or cd_size == 0xffffffff or cd_offset == 0xffffffff:
        raise ValueError("Unexpected pinned archive layout")
    directory = archive.read(cd_disk*PART_SIZE+cd_offset, cd_size)
    entries = central_entries(directory)
    with (root/ANNOTATIONS[0]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [r["name"] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate training IDs")
    requested = set(names)
    selected = [e for e in entries if e["name"].endswith(".mp4") and PurePosixPath(e["name"]).stem in requested]
    if len(selected) != len(names) or {PurePosixPath(e["name"]).stem for e in selected} != requested:
        raise ValueError("Archive membership does not cover every original training sample exactly once")
    for e in selected:
        if PurePosixPath(e["name"]).parts != ("video", PurePosixPath(e["name"]).name):
            raise ValueError("Unexpected member path")
        if e["flags"] & 1 or e["method"] not in (0, 8):
            raise ValueError("Unsupported archive encoding")
    (root/"mercaptionplus-video-manifest.json").write_text(json.dumps(selected), encoding="utf-8")
    (root/"download-source.json").write_text(json.dumps(dict(repo="MERChallenge/MER2025", revision=REVISION,
        central_sha256=hashlib.sha256(directory).hexdigest(), samples=len(selected),
        video_bytes=sum(e["size"] for e in selected),
        annotations_sha256={n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ANNOTATIONS}), indent=2))
    return selected


def verified(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    crc = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            crc = zlib.crc32(chunk, crc)
    return crc == entry["crc"]


def batches(entries):
    group = []
    start = stop = 0
    for entry in sorted(entries, key=lambda e: (e["disk"], e["offset"])):
        position = entry["disk"]*PART_SIZE+entry["offset"]
        # Local extra fields can differ from central extra fields: allow their full uint16 length.
        end = min(position+30+entry["filename_bytes"]+65535+entry["compressed"], 11*PART_SIZE+LAST_SIZE)
        if group and (position-stop > 128*1024 or end-start > 32*1024*1024):
            yield start, stop, group
            group = []
        if not group:
            start = position
        stop = max(stop, end)
        group.append(entry)
    if group:
        yield start, stop, group


def extract(batch, root, archive):
    start, stop, entries = batch
    blob = archive.read(start, stop-start)
    results = []
    for e in entries:
        position = e["disk"]*PART_SIZE+e["offset"]-start
        h = struct.unpack_from("<4s5H3I2H", blob, position)
        if h[0] != b"PK\x03\x04" or h[2] != e["flags"] or h[3] != e["method"]:
            raise ValueError("Local/central archive header mismatch")
        nl, xl = h[9:11]
        if blob[position+30:position+30+nl].decode("utf-8") != e["name"]:
            raise ValueError("Local/central filename mismatch")
        offset = position+30+nl+xl
        packed = blob[offset:offset+e["compressed"]]
        if len(packed) != e["compressed"]:
            raise ValueError("Incomplete compressed member")
        if e["method"] == 8:
            decoder = zlib.decompressobj(-15)
            data = decoder.decompress(packed, e["size"]+1)
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError("Invalid deflate member")
        else:
            data = packed
        if len(data) != e["size"] or zlib.crc32(data) != e["crc"]:
            raise ValueError("Original ZIP CRC/size verification failed")
        destination = root/e["name"]
        temp = destination.with_suffix(".mp4.part")
        with temp.open("wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, destination)
        results.append(dict(id=destination.stem, bytes=len(data), crc32=e["crc"],
                            sha256=hashlib.sha256(data).hexdigest()))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Download-only smoke check; omitted for full original membership")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    token = os.environ.get("HF_TOKEN") or (Path.home()/".cache/huggingface/token").read_text().strip()
    archive = Archive(token)
    entries = prepare(args.root, archive)
    selected = entries[:args.limit]
    (args.root/"video").mkdir(exist_ok=True)
    todo = [e for e in selected if not verified(args.root/e["name"], e)]
    done = len(selected)-len(todo)
    print(json.dumps(dict(total=len(selected), completed=done, video_bytes=sum(e["size"] for e in selected))), flush=True)
    started = time.monotonic()
    completed_bytes = 0
    last_report = 0
    with (args.root/"video-download-receipts.jsonl").open("a", encoding="utf-8") as receipts:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(extract, b, args.root, archive) for b in batches(todo)]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                for row in result:
                    receipts.write(json.dumps(row)+"\n")
                receipts.flush(); os.fsync(receipts.fileno())
                done += len(result)
                completed_bytes += sum(r["bytes"] for r in result)
                elapsed = time.monotonic()-started
                if elapsed-last_report >= 15 or done == len(selected):
                    print(json.dumps(dict(completed=done, total=len(selected), downloaded_video_bytes=completed_bytes,
                                          elapsed_seconds=round(elapsed), MB_per_second=round(completed_bytes/max(elapsed,1)/1e6,2))), flush=True)
                    last_report = elapsed
    receipt = dict(complete=True, revision=REVISION, samples=done, expected_samples=len(entries),
                   full_dataset=done == len(entries), video_bytes=sum(e["size"] for e in selected), crc_verified=True)
    filename = "video-download-complete.json" if receipt["full_dataset"] else "video-download-smoke.json"
    (args.root/filename).write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
