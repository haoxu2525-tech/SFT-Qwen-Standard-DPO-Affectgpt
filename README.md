# SFT-Qwen-Standard-DPO-Affectgpt

Standard DPO from the verified epoch-30 Qwen2.5-Omni SFT adapter, using the
pinned official DPO loss/trainer and original AffectGPT supervision and evaluation.

## Native audiovisual training input

DPO directly calls the SFT module's `omni_components.encode_prompt`. It uses one
original video and its complete continuous embedded audio, the native chat
template, published `process_mm_info`, and `use_audio_in_video=True` in both the
processor and Thinker. The existing SFT settings remain eight video frames at
224 x 224, with the utility's returned FPS passed unchanged to the processor.
There is no separate WAV input or eight-audio-clip path in DPO. Original `audio`
paths remain provenance metadata; only the video container is read for media.

No new audio compression, FPS change, resolution change, or silent answer
truncation is applied. A length overflow is recorded and stops training. Any
training media-budget change needs an explicit experimental decision.
The published DPO source and all vendor files remain unchanged.

The full native-input scan of all 31,256 successful MERCaptionPlus pairs completed
with zero encoding errors, zero truncation and zero sequences over 4,096 tokens.
The maximum prompt length was 1,107 tokens; the maximum prompt plus answer and EOS
was 1,328 tokens (`samplenew3_00098643`). The 99th percentile total was 764 tokens.
Across the two Pods, 5,200 overlapping samples had identical counts,
and encoder source, processor files and dependency versions matched. These results
support keeping the SFT audiovisual representation without DPO-only compression.
This was an input-length check, not a GPU forward/backward memory benchmark.
Three real samples, including the longest input, also passed a CPU tensor-level
comparison across the SFT collator, DPO tokenization/model input adapter and
historical continuous-audio evaluation input adapter: prompt IDs, audiovisual
tensors, masks, grids and timing fields matched exactly. No model weights were
loaded for that check. This check does not describe the sampled evaluation
protocol below, authorized on 2026-11-12.

## AffectGPT-aligned evaluation inputs (2026-11-12)

The current decision keeps full continuous audio for SFT/DPO training, but
aligns evaluation media content with the pinned AffectGPT reference. This
supersedes the earlier requirement that training and evaluation media encodings
be identical. Both the SFT baseline and DPO checkpoint must be evaluated using
the same new protocol; previous continuous-audio/full-frame scores are separate
experiments and cannot serve as its baseline.

`run_pipeline.py infer/evaluate` now dispatches `reference_evaluation.py`.
It directly calls the original `load_audio(..., clips_per_video=8)` on original
dataset audio files (16 kHz, eight 2-second clips, original short-audio padding)
and `load_face(..., n_frms=8, height=224, width=224, sampling="uniform")` on
original face NPY files (original frame indices and short-video repetition).
No new face detector, audio sampler, full-frame fallback or continuous-audio
fallback is used. Missing required media fails instead of excluding a sample.

The same returned waveforms and face pixels are serialized as eight ordered
native audio items and eight ordered native image items in one Omni request,
with `use_audio_in_video=False`. No new continuous timeline or clip/frame
synchronization is invented. Omni's encoder, normalization, positional encoding,
chat serialization and fusion remain model-specific: this is matching observed
media content, not claiming identical internal features or a complete AffectGPT
architecture reproduction. Question/subtitle prose comes from the original
text-only branch as in the SFT interface; legacy model-specific media wrappers
are replaced by native roles/media tokens.

Original nine-dataset membership, `Chat.answer_sample`, text selection, decoding
parameters, stopping/answer parsing, Qwen2.5-7B-Instruct label extraction and
`evaluation.py` metrics remain in the pinned source. Single fixed checkpoints
are compared here. The original scorer's per-dataset best across seven SFT
epochs is a different checkpoint-selection protocol and must be labelled
separately. Matching media alone does not isolate training-audio improvements;
that claim requires a controlled training ablation.

Use separate **fresh** runtimes for SFT and DPO. Old continuous-media runtimes
and their input caches are rejected. Supply the original benchmark face NPY and
audio files, not just videos or a remuxed dataset overlay. Inference records the
protocol/checkpoint source and creates a checkpoint-directory symlink under the
name expected by the unchanged scorer, without copying or changing weights.
The epoch-30 SFT input below is the original runner `.pth` checkpoint directory;
the DPO input is the directory produced by `run_pipeline.py export`.

```bash
python run_pipeline.py infer --runtime ./runtime-eval-sft-sampled \
  --dataset-root /path/to/original-dataset \
  --checkpoints /path/to/sft-runner-checkpoints --checkpoint-epoch 30
python run_pipeline.py evaluate --runtime ./runtime-eval-sft-sampled \
  --dataset-root /path/to/original-dataset

python run_pipeline.py infer --runtime ./runtime-eval-dpo-sampled \
  --dataset-root /path/to/original-dataset \
  --checkpoints /path/to/exported-dpo --checkpoint-epoch 1
python run_pipeline.py evaluate --runtime ./runtime-eval-dpo-sampled \
  --dataset-root /path/to/original-dataset
```

CPU validation uses the actual original sampler functions, original dataset
raw-media path, pinned Omni processor and a tiny randomly initialized Thinker.
It checks waveform/pixel equality, short-input handling, resampling, forward and
generation compatibility, inherited Chat methods, unchanged generation arguments,
and dispatch to the original inference/scoring scripts. These checks do not
establish 7B GPU execution or new benchmark scores. Run with
`OMNI_PROCESSOR_SNAPSHOT=/path/to/pinned-processor python -m pytest test_reference_evaluation.py`.
The [validation record](validation/reference-evaluation/cpu-validation.json)
includes 23 passing CPU checks and three additional passing real MELD archive
samples. Set `MER_UNIBENCH_ARCHIVE` to enable those optional real-data tests.

The historical 7,978-token example was a long MELD evaluation video, not a
MERCaptionPlus training sample. It is not evidence that DPO training inputs need
compression. Lengths must be measured on the actual selected training pairs.

## Successful training subset

Pass `--pairs "qwen3.8omniflash annotation"` to read all seven checksummed
successful shards: **31,256 unique MERCaptionPlus pairs**. Failed files and retry
history are not loaded as pairs. The original dataset has 31,327 IDs; the 71
excluded records comprise 66 audit refusals and five videos rejected as too short.
Original annotations and media are retained; exclusion only affects DPO membership.

Membership validation requires a nonempty, duplicate-free subset of the original
IDs and still verifies each retained question, label type, source answer, media
reference, and annotation. Unknown IDs and modified source data fail. Each run
writes `membership.json` and `excluded-ids.txt`, rather than requiring exact equality
with all 31,327 source IDs. MELD is not included in this training set.

## Native length preflight

Use the pinned processor snapshot, original dataset, and a new output directory:

```bash
python run_pipeline.py train \
  --runtime ./runtime --dataset-root /path/to/dataset \
  --snapshot /path/to/pinned-qwen-snapshot \
  --pairs "qwen3.8omniflash annotation" \
  --output ./native-length-check --preflight-only --preflight-workers 8
```

Preflight loads no model weights and starts no GPU training. It runs the same
native encoder for every selected row, measures the complete audiovisual prompt
plus each answer and EOS, and saves maxima, longest IDs, capacity violations, and
encoding errors in `length-check.json`. Parallel workers only distribute CPU work;
they do not change the media representation. The configured limit is 4,096 tokens.
Overflowing rows are reported with their full lengths, never silently truncated
or excluded. `--preflight-only` needs neither an adapter nor preference-validation
pairs. Actual training still requires an explicit adapter, beta, and eval pairs;
this code does not invent a validation split.

Batch defaults are `--batch-size 8 --gradient-accumulation-steps 8`, with gradient
checkpointing enabled. In the official DPO implementation, batch size already
includes accumulation: this means eight preference pairs per optimizer update,
one pair per microbatch, and 3,907 updates per full training epoch. The chosen and
rejected sequences are both evaluated for each pair. This is not a claim that
eight simultaneous audiovisual pairs fit GPU memory; CPU preflight does not
measure forward/backward VRAM.

## Text audit

All 31,256 top-level rejected answers currently contain no Chinese. One unchanged
source chosen answer (`samplenew3_00063912`) still contains a Chinese emotion label;
it must be reviewed before claiming wholly English target supervision. API
reasoning/retry fields are provenance only and are not serialized into model text.
`PairData` takes `prompt`, media, `chosen`, and `rejected`; it does not train on the
whole annotation JSON. Language and style checks do not prove semantic preference
quality; human review remains necessary.

The [teacher-review reports](validation/teacher-review/README.md) include the
complete per-ID length scan, excluded IDs, full-field language counts, style
analysis and 61 samples for human review. The description-only style classifier
achieved 66.14% balanced accuracy using length/script/punctuation features alone;
this warrants review and is not proof of incorrect labels. Human review is still
pending; no additional rows have been excluded based on these heuristics.

Run CPU contract checks with `python -m unittest test_dpo_native_input test_annotation_recovery`.
Real-data preflight results and GPU memory checks are separate evidence.
