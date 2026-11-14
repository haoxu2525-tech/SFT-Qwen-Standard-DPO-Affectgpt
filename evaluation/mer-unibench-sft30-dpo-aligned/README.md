# MER-UniBench: SFT epoch 30 and DPO final

Completed evaluation results for two fixed checkpoints under the same
`affectgpt-sampled-audio-face-omni-v2-original-generation` protocol.
These are our model results, not the AffectGPT authors' published results.

Both runs contain all 12,799 benchmark samples across nine datasets, with matching
sample IDs and zero empty predictions. SFT uses the original epoch-30 checkpoint
without DPO updates. DPO uses the final adapter after one epoch on 31,256 preference
pairs (3,907 optimizer updates). No per-dataset checkpoint selection was performed.

## Scores

Scores below are percentages from the original AffectGPT scorer. The first four
datasets use hit rate; the remaining five use F-score. Secondary accuracy,
precision and recall values are preserved in each run's `metrics.json`.

| Dataset | Metric | SFT epoch 30 | DPO final | DPO minus SFT (pp) |
|---|---|---:|---:|---:|
| MER2023 | Hit rate | 74.27 | 70.53 | -3.74 |
| MER2024 | Hit rate | 78.10 | 72.35 | -5.75 |
| MELD | Hit rate | 58.55 | 53.52 | -5.03 |
| IEMOCAPFour | Hit rate | 57.57 | 57.04 | -0.53 |
| CMUMOSI | F-score | 81.67 | 83.46 | +1.79 |
| CMUMOSEI | F-score | 75.34 | 76.25 | +0.91 |
| SIMS | F-score | 85.32 | 87.30 | +1.98 |
| SIMSv2 | F-score | 83.01 | 85.65 | +2.64 |
| OVMERDPlus | F-score | 61.49 | 59.90 | -1.59 |
| **Nine-dataset mean** | **Mixed main metrics** | **72.81** | **71.78** | **-1.03** |

The final difference subtracts the two scorer-reported means rounded to two
decimals. Averaging the displayed per-dataset differences before rounding gives
-1.04 pp due to rounding order. These are single-run point estimates; no
multi-seed variance or statistical significance is claimed.

## Files

- `comparison.csv` and `comparison.json`: dataset scores and differences.
- `sft-epoch30/` and `dpo-final/`: separate results for each checkpoint.
- `raw/<dataset>/*.npz`: all 22 original output files per run, copied byte-for-byte,
  including predictions, extracted emotion labels and derived sentiment labels.
- `predictions/<dataset>.jsonl`: readable exports with sample ID, model prediction,
  extracted labels and, where applicable, sentiment. Predictions are not rewritten.
- `metrics.json`: original scorer values in the 0-1 range.
- `score-summary.txt`: exact dataset and aggregate score lines from the scorer log.
  The scorer's word `best` refers to the only checkpoint supplied in each run.
- `protocol.json`: recorded protocol, generation defaults, source/code revision,
  checkpoint filename and SHA-256. Machine-specific checkpoint paths are omitted.
- `verification.json`: completion timestamp, counts and source-log hashes.
- `sha256.json`: checksums of all other files in this result directory.

## Evaluation provenance

Evaluation interface commit: `8d029e1c92bec3411a9bc62f7f5067971fcd7cd0`.
Pinned AffectGPT source: `95b9257758df29fa6481c138252cdd317cce1735`.

The original media functions supply eight 2-second audio clips at 16 kHz and eight
224 x 224 face crops. Qwen2.5-Omni receives these through its native audio/image
interface. Original question/subtitle prose, answer generation procedure and
scoring are retained. This aligns observed media and evaluation behavior; it does
not make Omni's architecture or internal representations identical to AffectGPT.

`protocol.json` records the inherited generation configuration. Explicit arguments
from the original `Chat.answer_sample` take precedence: temperature 1.0, top-p 0.9,
one beam, sampling enabled, minimum length 1, repetition penalty 1.0 and length
penalty 1.0. The original evaluation call supplies a 1,200-token output budget.
Inherited top-k is 20 and EOS IDs are `[151645, 151643]`.

The original Qwen2.5-7B-Instruct label extractor and AffectGPT metrics produce the
saved labels and scores. Historical continuous-audio/full-frame SFT scores and
the discarded v1 evaluation are separate protocols and are not included here.
The earlier SFT score of 73.99 is not a baseline for this comparison.

Model weights, benchmark media and full runtime logs are not part of this results
directory. The unchanged output NPZ files and readable exports provide the
sample-level evidence used by the scorer.
