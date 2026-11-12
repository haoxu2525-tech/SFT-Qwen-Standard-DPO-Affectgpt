"""Read-only language and answer-style audit of saved preference pairs.

Only chosen/rejected are inspected, never API reasoning_content. Flags nominate
human review; they do not establish semantic correctness or change any answer.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import unicodedata

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]")
WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
MARKERS = {
    "explicit_negative": r"\b(?:rejected answer|negative sample|deliberately (?:incorrect|wrong)|intentionally (?:incorrect|wrong)|chosen answer|reference answer)\b",
    "assistant_refusal": r"\b(?:as an ai|i cannot (?:provide|assist|help)|i can't (?:provide|assist|help))\b",
    "contrast_phrase": r"\b(?:rather than|instead of|contrary to|not .{0,35} but)\b",
}


def rows_from(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if item["kind"] != "successful":
            continue
        path = directory / item["name"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Checksum mismatch: {path.name}")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                yield row, path.name, line_number


def profile(text):
    cjk = len(CJK.findall(text))
    latin = sum('a' <= c.lower() <= 'z' for c in text)
    other = Counter(unicodedata.name(c, "UNKNOWN").split()[0] for c in text
                    if c.isalpha() and not CJK.fullmatch(c) and 'LATIN' not in unicodedata.name(c, ""))
    category = ("chinese_majority" if cjk > latin else "mixed_with_chinese") if cjk else (
        "other_script_review" if other else "latin_script_no_chinese")
    return dict(category=category, chinese_characters=cjk, ascii_letters=latin,
                other_letter_scripts=dict(other), characters=len(text), words=len(WORDS.findall(text)))


def quantiles(values):
    values = sorted(values)
    return {"count": len(values), "mean": round(statistics.mean(values), 3),
            **{f"p{p}": values[round((len(values)-1)*p/100)] for p in (0, 50, 90, 95, 99, 100)}}


def style_features(text):
    info = profile(text)
    return [len(text), info["words"], info["chinese_characters"], text.count('\n'),
            text.count(','), text.count(';'), text.count(':'), text.count('!'), text.count('?'),
            text.count('"'), text.count('**'), text.count('—'), text.count('“') + text.count('”'),
            int(bool(re.search(r"(?m)^\s*(?:[-*]|\d+[.)])\s", text)))]


def style_probe(rows):
    """Pair-grouped holdout: both answers for an ID stay on the same side."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    results = {}
    for name, subset in [("all", rows), ("no_chinese", [r for r in rows if not CJK.search(r['chosen']+r['rejected'])])]:
        for kind in ("all", "description", "ovlabel"):
            selected = subset if kind == 'all' else [r for r in subset if r.get('label_type') == kind]
            if len(selected) < 50:
                continue
            train, test = train_test_split(list(range(len(selected))), test_size=.2, random_state=17)
            def arrays(indices):
                x, y = [], []
                for i in indices:
                    for field, label in [('chosen', 1), ('rejected', 0)]:
                        x.append(style_features(selected[i][field]));y.append(label)
                return np.array(x), np.array(y)
            x, y = arrays(train);xt, yt = arrays(test)
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=17))
            model.fit(x, y);scores = model.predict_proba(xt)[:, 1]
            results[name+'/'+kind] = dict(train_pairs=len(train), test_pairs=len(test),
                balanced_accuracy=round(float(balanced_accuracy_score(yt, scores >= .5)), 4),
                roc_auc=round(float(roc_auc_score(yt, scores)), 4))
    return {"method": "Length, script and punctuation only; logistic regression; pair-grouped 80/20 holdout; seed 17. Chance=0.5. This measures distinguishability, not semantic preference correctness.",
            "results": results}


def audit(source, output, with_style_probe=False):
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    summary = dict(samples=0, language=dict(chosen=Counter(), rejected=Counter()),
                   chinese_fields=dict(chosen=0, rejected=0), flagged_pairs=0)
    lengths = defaultdict(list);markers = defaultdict(Counter);ngrams = defaultdict(Counter)
    seen = set();rows = [];flagged = [];language = []
    for row, filename, line_number in rows_from(source):
        row = {k: row[k] for k in ('id', 'label_type', 'video', 'prompt', 'chosen', 'rejected')}
        assert row['id'] not in seen;seen.add(row['id']);rows.append(row);flags=[]
        record = {k:row.get(k) for k in ['id','label_type','video','prompt','chosen','rejected']}
        record.update(source_file=filename, source_line=line_number)
        profiles = {side:profile(row[side]) for side in ('chosen','rejected')}
        record['language_profiles'] = profiles
        for side in ('chosen','rejected'):
            text=row[side];info=profiles[side];summary['language'][side][info['category']]+=1
            if info['chinese_characters']:
                summary['chinese_fields'][side]+=1;flags.append(side+'_contains_chinese')
            if info['other_letter_scripts']:flags.append(side+'_other_letter_script')
            for measure in ('words','characters'):lengths[f"{row['label_type']}/{side}/{measure}"].append(info[measure])
            for name,pattern in MARKERS.items():
                if re.search(pattern,text,re.I):markers[row['label_type']+'/'+name][side]+=1;flags.append(side+'_'+name)
            tokens=WORDS.findall(text.lower())
            for n in (2,3):
                for gram in set(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)):
                    ngrams[row['label_type']+'/'+side][gram]+=1
        if row['chosen'].strip().casefold()==row['rejected'].strip().casefold():flags.append('identical_after_casefold')
        a,b=profiles['chosen']['characters'],profiles['rejected']['characters']
        if b/max(a,1)>2 or b/max(a,1)<.5:flags.append('character_length_ratio_outside_0.5_to_2')
        record['flags']=sorted(set(flags))
        if any(p['chinese_characters'] or p['other_letter_scripts'] for p in profiles.values()):language.append(record)
        if flags:flagged.append(record)
    summary.update(samples=len(rows),flagged_pairs=len(flagged),language_review_pairs=len(language),
                   no_detected_chinese_or_other_non_latin_script=len(rows)-len(language),
                   lengths={k:quantiles(v) for k,v in lengths.items()},marker_document_counts=dict(markers))
    lexical={}
    for kind in sorted({r['label_type'] for r in rows}):
        ca=ngrams[kind+'/chosen'];cb=ngrams[kind+'/rejected'];n=sum(r['label_type']==kind for r in rows)
        ranked=[]
        for phrase in ca.keys()|cb.keys():
            if ca[phrase]+cb[phrase]<100:continue
            ranked.append(dict(phrase=phrase,chosen=ca[phrase],rejected=cb[phrase],
                difference_percentage_points=round((cb[phrase]-ca[phrase])*100/n,3)))
        lexical[kind]=sorted(ranked,key=lambda r:abs(r['difference_percentage_points']),reverse=True)[:40]
    summary['phrase_imbalance']=lexical
    summary['notes']=["Only chosen/rejected were checked, not reasoning_content.",
        "Script detection is not a complete English language or grammar check. Latin-only text is not certified English.",
        "No annotation was translated, regenerated, removed or relabelled. Flagged text needs human audiovisual review.",
        "Phrase imbalance includes genuine emotion/content differences and is not by itself proof of a shortcut."]
    if with_style_probe:summary['style_probe']=style_probe(rows)
    for name,items in [('language_review.jsonl',language),('all_flags.jsonl',flagged)]:
        (output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items),encoding='utf-8')
    with (output/'language_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['id','label_type','problem_fields','chosen_chinese_characters','rejected_chinese_characters','chosen','rejected','source_file','source_line']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in language:
            writer.writerow(dict(id=r['id'],label_type=r['label_type'],problem_fields=';'.join(r['flags']),
                chosen_chinese_characters=r['language_profiles']['chosen']['chinese_characters'],
                rejected_chinese_characters=r['language_profiles']['rejected']['chinese_characters'],
                **{k:r[k] for k in ['chosen','rejected','source_file','source_line']}))
    rng=random.Random(17);sample=[];used=set()
    language_ids={r['id'] for r in language};flagged_ids={r['id'] for r in flagged}
    pools=[language,[r for r in flagged if r['id'] not in language_ids],[r for r in rows if r['id'] not in flagged_ids]]
    for pool in pools:
        for r in rng.sample(pool,min(30,len(pool))):
            if r['id'] not in used:
                sample.append({k:r.get(k) for k in ['id','label_type','video','prompt','chosen','rejected','flags']} | {'human_decision':'','human_notes':''});used.add(r['id'])
    (output/'manual_review_sample.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sample),encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['samples','language','chinese_fields','language_review_pairs','flagged_pairs','style_probe'] if k in summary},ensure_ascii=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('qwen3.8omniflash annotation'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--style-probe',action='store_true')
    args=parser.parse_args();audit(args.source,args.output,args.style_probe)
