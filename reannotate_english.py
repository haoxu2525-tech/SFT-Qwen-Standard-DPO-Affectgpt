"""Regenerate only rejected answers containing Chinese, using original videos.

Original export remains unchanged. Each attempt is saved; publishing the repaired
export requires every selected row to pass the English-output guard.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time
import unicodedata

import annotate_api as api
from audit_preferences import CJK, rows_from
import download_mercaptionplus as download

ENGLISH_CONTEXT = (
    "Write the final rejected answer entirely in English, including any quoted dialogue "
    "and emotion labels. Do not include any Chinese characters or untranslated Chinese words. "
    "Match the reference answer's English answer format and approximate length. "
    "Return only the rejected answer."
)


def english_guard(text):
    if not text.strip() or not re.search(r'[A-Za-z]', text) or CJK.search(text):
        return False
    return not any(c.isalpha() and 'LATIN' not in unicodedata.name(c, '') for c in text)


def original_videos(root, selected, workers):
    token = os.environ.get('HF_TOKEN') or (Path.home()/'.cache/huggingface/token').read_text().strip()
    archive = download.Archive(token)
    entries = download.prepare(root, archive)
    ids = {r['id'] for r in selected}
    entries = [e for e in entries if Path(e['name']).stem in ids]
    assert len(entries) == len(ids)
    (root/'video').mkdir(exist_ok=True)
    todo = [e for e in entries if not download.verified(root/e['name'], e)]
    print(json.dumps(dict(stage='download',selected=len(entries),remaining=len(todo),
                          bytes=sum(e['size'] for e in entries))), flush=True)
    done = len(entries)-len(todo)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download.extract,b,root,archive) for b in download.batches(todo)]
        for f in as_completed(futures):
            done += len(f.result())
            print(json.dumps(dict(stage='download',completed=done,total=len(entries))),flush=True)
    assert all(download.verified(root/e['name'],e) for e in entries)
    api.save_json(root/'english-reannotation-download.json',dict(samples=len(entries),crc_verified=True,
        ids=sorted(ids),source_revision=download.REVISION))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--download-workers',type=int,default=8)
    parser.add_argument('--max-attempts',type=int,default=3)
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    if min(args.workers,args.download_workers,args.max_attempts)<1:parser.error('worker/attempt counts must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(args.output/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    all_rows=[r for r,_,_ in rows_from(args.source)]
    selected=[r for r in all_rows if CJK.search(r['rejected'])]
    ids=[r['id'] for r in selected]
    api.save_json(args.output/'selection.json',dict(total_successful=len(all_rows),selected=len(selected),ids=ids))
    if args.download:original_videos(args.dataset_root/'mer2025-dataset',selected,args.download_workers)
    api.definitions(api.HERE/'run_pipeline.py',{'NEGATIVE_MODEL','NEGATIVE_INSTRUCTION','qwen_negative','video_digest'},api.__dict__)
    endpoint=os.environ['DASHSCOPE_BASE_URL'].rstrip('/');key=os.environ['DASHSCOPE_API_KEY']
    original_call=api.qwen_negative
    api.qwen_negative=lambda row,url,endpoint,key,limit:original_call(row,url,endpoint,key,limit,context=ENGLISH_CONTEXT)
    folder=args.output/'rows';folder.mkdir(exist_ok=True)
    attempts=args.output/'attempts';attempts.mkdir(exist_ok=True)
    for row in selected:
        path=(args.dataset_root/row['video']).resolve()
        if not path.is_relative_to(args.dataset_root.resolve()) or not path.is_file():raise ValueError(f"Missing video: {row['id']}")
        if api.video_digest(path)!=row['annotation_api']['video_sha256']:raise ValueError(f"Original video changed: {row['id']}")
    settings=dict(model=api.NEGATIVE_MODEL,instruction=api.NEGATIVE_INSTRUCTION,endpoint=endpoint,
                  seed=0,max_tokens=None,english_context=ENGLISH_CONTEXT)
    def repair(original):
        row={k:v for k,v in original.items() if k not in ('chosen','rejected','annotation_api','language_reannotation')}
        saved=folder/(row['id']+'.json')
        if saved.exists():
            old=json.loads(saved.read_text())
            assert old['chosen']==original['chosen'] and old['language_reannotation']['original_rejected']==original['rejected']
            assert old['annotation_api']['settings']==settings
            if english_guard(old.get('rejected','')):return old
        history=[]
        for attempt in range(1,args.max_attempts+1):
            stamp=str(time.time_ns())
            result=api.annotate_one(row,args.dataset_root,endpoint,key,settings,attempts,1)
            meta=result['annotation_api']
            if result.get('rejected') and not english_guard(result['rejected']):
                meta['non_english_output']=result['rejected'];result['rejected']=''
                meta['error']=dict(type='EnglishOutputValidationError')
            result['language_reannotation']=dict(original_rejected=original['rejected'],
                original_annotation_api=original['annotation_api'],english_context=ENGLISH_CONTEXT)
            history.append(copy.deepcopy(meta));result['language_reannotation']['attempts']=history.copy()
            api.save_json(attempts/(row['id']+'.'+stamp+'.json'),result);api.save_json(saved,result)
            if result['rejected'] or api.input_rejected(result):break
            if attempt<args.max_attempts:time.sleep(10)
        return result
    completed={}
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    pending=iter(selected);pool=ThreadPoolExecutor(max_workers=args.workers)
    try:
        active={pool.submit(repair,row) for row in (next(pending,None) for _ in range(args.workers)) if row is not None}
        while active:
            done,active=wait(active,return_when=FIRST_COMPLETED)
            for f in done:
                r=f.result();completed[r['id']]=r
                print(json.dumps(dict(stage='annotation',completed=len(completed),total=len(selected),
                    id=r['id'],success=bool(r['rejected']),error=r['annotation_api'].get('error'))),flush=True)
            while len(active)<args.workers:
                row=next(pending,None)
                if row is None:break
                active.add(pool.submit(repair,row))
    except KeyboardInterrupt:
        print('Stopping submissions; waiting for current requests to save.',flush=True)
        return
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
    failed=[i for i,r in completed.items() if not r['rejected']]
    receipt=dict(selected=len(selected),successful=len(completed)-len(failed),failed_ids=failed,
                 all_english_replacements_ready=not failed,chosen_unchanged=True)
    api.save_json(args.output/'complete.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
