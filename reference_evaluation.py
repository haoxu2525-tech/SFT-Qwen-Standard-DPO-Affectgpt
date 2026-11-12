"""AffectGPT sampled-media evaluation for both SFT and DPO checkpoints.

The original samplers own which waveforms/pixels are observed. Omni owns its
native feature encoding. Eight separate audio clips and eight ordered images
do not assert a continuous/synchronized timeline. Training is unaffected.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SFT_INTERFACE = ROOT / "vendor/sft/Experiment/SFT Base Model Training"
SOURCE = ROOT / "vendor/affectgpt/AffectGPT"
PROTOCOL = "affectgpt-sampled-audio-face-omni-v1"
RESULT_NAME = "emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz"


def read_reference_media(source, load_audio, load_face):
    """Call the pinned original functions; never resample their outputs."""
    if not isinstance(source, dict) or set(source) != {"audio", "face"}:
        raise ValueError("Evaluation requires original audio and face paths")
    for name, path in source.items():
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"Missing original evaluation {name}: {path}")
    audio = load_audio([source["audio"]], "cpu", clips_per_video=8)[0]
    faces = load_face(face_npy=source["face"], n_frms=8, height=224,
                      width=224, sampling="uniform", return_msg=False)
    if tuple(audio.shape) != (8, 1, 32000) or tuple(faces.shape) != (3, 8, 224, 224):
        raise ValueError(f"Unexpected original media shapes: {audio.shape}, {faces.shape}")
    # Layout changes only: keep original clip/frame order, values and channels.
    return [clip[0].numpy() for clip in audio], list(faces.permute(1, 2, 3, 0).numpy())


def encode_reference_media(processor, prompt, source, load_audio, load_face):
    from omni_components import native_messages
    audio, images = read_reference_media(source, load_audio, load_face)
    # Reuse the existing system prompt, role conversion and question/subtitle.
    # The placeholder video entry is removed before any processor/media call.
    messages = native_messages(prompt, "unused-template-placeholder")
    text_part = messages[1]["content"][-1]
    messages[1]["content"] = ([{"type": "audio", "audio": x} for x in audio]
                              + [{"type": "image", "image": x} for x in images]
                              + [text_part])
    inputs = dict(processor(
        text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        images=images, audio=audio, return_tensors="pt", add_special_tokens=False,
        padding=False, truncation=False, do_resize=False, use_audio_in_video=False,
        sampling_rate=16000,
    ))
    inputs["use_audio_in_video"] = False
    return inputs


def install_reference_media(omni):
    """Replace only the evaluation media/model-call interface after SFT install."""
    from my_affectgpt.datasets.datasets import base_dataset

    def read_media(dataset, video_path=None, face_npy=None, audio_path=None, image_path=None):
        if not audio_path or not face_npy:
            raise ValueError("Reference evaluation needs original WAV and face NPY; no fallback")
        result = dict.fromkeys(("frame", "face", "audio", "image", "raw_face", "raw_audio", "raw_image"))
        # Existing NativeChat.postprocess_frame forwards this opaque media handle.
        result["raw_frame"] = {"audio": str(audio_path), "face": str(face_npy)}
        return result

    class ReferenceInferenceInterface(omni.OmniInferenceInterface):
        def generate(self, inputs_embeds, attention_mask, **generation_kwargs):
            prompt = omni.native_prompt_from_ids(self.chat.tokenizer, self.input_ids)
            inputs = encode_reference_media(self.chat.processor, prompt, self.source,
                                           base_dataset.load_audio, base_dataset.load_face)
            prefix_length = inputs["input_ids"].shape[1]
            output = self.chat.model.generate(
                **omni.thinker_inputs(self.chat.model, inputs), **generation_kwargs)
            return output[:, prefix_length:]

    base_dataset.BaseDataset.read_frame_face_audio_text = read_media
    omni.OmniInferenceInterface = ReferenceInferenceInterface
    # Chat.answer_sample, token selection, stopping and parsing stay inherited.


def checkpoint_alias(runtime, checkpoints):
    """Give the original scorer its expected result name without moving weights."""
    checkpoints = checkpoints.resolve()
    if not checkpoints.is_dir():
        raise FileNotFoundError(checkpoints)
    alias = runtime / "evaluation-checkpoints" / RESULT_NAME
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.exists() or alias.is_symlink():
        if not alias.is_symlink() or alias.resolve() != checkpoints:
            raise ValueError("Use a separate evaluation runtime for each checkpoint source")
    else:
        alias.symlink_to(checkpoints, target_is_directory=True)
    return alias


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["infer", "evaluate"])
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--checkpoint-epoch", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "infer" and (args.checkpoints is None or args.checkpoint_epoch < 1):
        parser.error("Inference requires --checkpoints and a positive --checkpoint-epoch")
    if os.environ.get("OMNI_INPUT_CACHE"):
        raise ValueError("Unset OMNI_INPUT_CACHE: continuous-video caches are incompatible with sampled evaluation")
    sys.path.insert(0, str(SFT_INTERFACE))
    import run_affectgpt_omni as runner
    # The unchanged preparer rejects an old continuous-AV runtime/its results.
    runner.WORKFLOW = PROTOCOL
    runtime = args.runtime.resolve()
    if args.stage == "evaluate":
        receipt = runtime / "evaluation-protocol.json"
        if not receipt.is_file() or json.loads(receipt.read_text())["protocol"] != PROTOCOL:
            raise ValueError("Scoring requires this protocol's inference runtime")
    runner.prepare_execution(SOURCE, runtime, args.dataset_root)
    original_args = []
    if args.stage == "infer":
        alias = checkpoint_alias(runtime, args.checkpoints)
        record = dict(protocol=PROTOCOL, source_commit=runner.SOURCE_COMMIT,
                      checkpoint_source=str(args.checkpoints.resolve()),
                      checkpoint_epoch=args.checkpoint_epoch,
                      audio="original load_audio: 8 x 2 seconds, 16 kHz",
                      vision="original load_face: 8 ordered face crops, 224 x 224",
                      serialization="8 native audio items + 8 native image items; no AV synchronization",
                      scoring="unchanged AffectGPT evaluation.py")
        receipt = runtime / "evaluation-protocol.json"
        if receipt.exists() and json.loads(receipt.read_text()) != record:
            raise ValueError("Use a new runtime for a different evaluation/checkpoint")
        receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        import omni_components as omni
        original_install = omni.install

        def install(snapshot, inference_io=False):
            original_install(snapshot, inference_io=False)
            install_reference_media(omni)

        omni.install = install
        original_args = ["--", "--options", f"inference.test_epoch={args.checkpoint_epoch}",
                         f"inference.ckpt_root={alias}"]
    sys.argv = [str(Path(runner.__file__)), args.stage, "--source", str(SOURCE),
                "--runtime", str(runtime), "--dataset-root", str(args.dataset_root.resolve()), *original_args]
    runner.main()


if __name__ == "__main__":
    main()
