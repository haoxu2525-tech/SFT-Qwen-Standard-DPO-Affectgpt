# qwen3.8omniflash annotation

Qwen `qwen3.8-omni-flash` synthetic rejected answers for MERCaptionPlus.

- Successful preference pairs: **31,256** in `successful-*.jsonl.gz` (read all seven files).
- Failed records: **71** in `failed.jsonl`: 66 content-audit refusals (65 input, 1 output), and 5 videos rejected as too short. Failed rows have no accepted rejected answer.
- Archived manual retry records: `failure-attempt-history.jsonl.gz`; these are history, not additional samples.
- Counts and SHA-256 checksums: `manifest.json`.

Each successful row preserves the original `prompt`, `chosen`, `ground_truth_answer`, media references and generated `rejected`, plus API response metadata. The account-specific API endpoint is omitted. Videos and credentials are not included. These are synthetic negatives; completion and integrity checks do not establish semantic preference quality.
