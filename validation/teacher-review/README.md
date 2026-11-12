# Teacher review: implementation and remaining data review

These checks use annotation commit `802cb59f9d7b273f93dd82baea91319e0cc14962`.
The reports distinguish completed input/membership checks from unfinished human
review. No new preference labels or dataset splits were created by these audits.

## Native input and membership: verified

- All 31,256 successful MERCaptionPlus pairs were encoded with the shared SFT
  native processor, complete embedded audio, eight video frames and 224 x 224
  resolution. Maximum input length: 1,107 tokens. Maximum input plus an answer
  and EOS: 1,328 tokens. Zero encoding errors, truncations, or sequences over 4,096.
- `native-lengths.jsonl.gz` contains one length record per successful ID.
  `native-length-summary.json` contains distributions and longest IDs.
- 5,200 overlapping measurements from the two Pods matched exactly. Encoder,
  processor files and dependency versions matched (`native-environment.json`).
- Three real samples, including the longest, passed exact prompt/media tensor
  comparisons across SFT, DPO and evaluation input interfaces. See
  `sft-dpo-eval-input-equality.json`. This is a CPU input test without model
  weights; it does not establish GPU training memory requirements.
- The 71 IDs in `excluded-ids.txt` exactly match the existing failed records:
  66 audit refusals and five videos rejected as too short. They are excluded
  from DPO membership; original annotations and media were not deleted.

## Language and style: automatic audit completed, human review pending

All 31,256 current rejected answers contain no Chinese. One source chosen answer
still contains a Chinese label (`samplenew3_00063912`); it is identified in
`language-review.csv`. Its correction has not been approved or applied. Changing
only chosen would violate the original-answer consistency check.

`preference-audit-summary.json` checks final chosen/rejected text, lengths,
phrase frequencies, and a classifier using only length, script and punctuation.
The classifier's paired held-out balanced accuracy is 56.77% overall and 66.14%
for description pairs (chance: 50%). Mean description lengths are 108.04 words
for chosen and 117.664 for rejected. These are reasons to review style shortcuts,
not proof that the preference labels are wrong or that DPO will exploit them.
The 10,587 flagged pairs are review candidates, not rejected training records.

`manual-review-sample.jsonl` contains 61 samples covering language issues, other
flags and unflagged cases. Human decision/note fields are deliberately blank.
Reviewers should compare both answers against the original audiovisual evidence,
check that chosen is preferable, and check superficial wording/length cues.
No automatic filtering or semantic relabelling has been applied.

`all-field-chinese-summary.json` separately audits every string in all 12 uploaded
annotation artifacts at the stated commit. Chinese remains in API reasoning,
retry metadata and a few original labels; these fields are not serialized as
DPO model text. This all-field report must not be confused with the top-level
chosen/rejected audit or with a claim that every JSON string is English.

To regenerate the answer/style audit, install scikit-learn in the audit environment
and run:

```bash
python audit_preferences.py --source "qwen3.8omniflash annotation" \
  --output ./preference-review --style-probe
```

`manifest.json` records report checksums and the style-probe library versions.
CPU regression tests are `python -m unittest test_dpo_native_input test_annotation_recovery`.
