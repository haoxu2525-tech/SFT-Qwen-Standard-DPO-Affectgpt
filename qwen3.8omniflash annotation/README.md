# qwen3.8omniflash annotation

Qwen `qwen3.8-omni-flash` synthetic rejected answers for MERCaptionPlus.

- Successful preference pairs: **31,256** in `successful-*.jsonl.gz` (read all seven files).
- Failed records: **71** in `failed.jsonl`: 66 content-audit refusals (65 input, 1 output), and 5 videos rejected as too short. Failed rows have no accepted rejected answer.
- Archived manual retry records: `failure-attempt-history.jsonl.gz`; these are history, not additional samples.
- Counts and SHA-256 checksums: `manifest.json`.

Each successful row preserves the original `prompt`, `chosen`, `ground_truth_answer`, media references and generated `rejected`, plus API response metadata. The account-specific API endpoint is omitted. Videos and credentials are not included. These are synthetic negatives; completion and integrity checks do not establish semantic preference quality.

## English reannotation

The 1,331 successful rows whose rejected answers contained Chinese were regenerated
with `qwen3.8-omni-flash` using the same original videos and an English-only answer
instruction. All 31,256 current rejected answers contain no Chinese characters;
the 1,331 replacements also passed the Latin-letter script guard. The other 29,925
pairs and all 71 failed records are unchanged.

All chosen answers and original sample metadata are unchanged. One original chosen
answer (`samplenew3_00063912`) still contains the Chinese label `求助`; this update
does not modify that source annotation. Superseded Chinese rejected answers and their API attempt history have been
removed from the current export. Source annotations, prompts, and API reasoning
may still contain Chinese. Training should use the top-level `chosen` and
`rejected` fields. Earlier Git commits are unchanged.

`english-reannotation-validation.json` lists all replaced IDs and integrity checks.
Language-script checks do not establish grammatical correctness or preference quality.
