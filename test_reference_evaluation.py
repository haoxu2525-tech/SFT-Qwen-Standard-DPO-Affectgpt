"""CPU tests with actual upstream sampling functions and the pinned Omni processor."""
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import random
import runpy
import sys
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import torchaudio
import zipfile

import reference_evaluation as evaluation

sys.path.insert(0, str(evaluation.SFT_INTERFACE))
import omni_components as omni


@pytest.fixture(autouse=True)
def original_generation_defaults(monkeypatch):
    from transformers import GenerationConfig
    # Values from the original Qwen2.5-7B-Instruct generation_config.json.
    config = GenerationConfig(bos_token_id=151643, pad_token_id=151643,
                              eos_token_id=[151645, 151643], do_sample=True,
                              repetition_penalty=1.05, temperature=.7, top_p=.8, top_k=20)
    monkeypatch.setattr(evaluation, "load_reference_generation_config", lambda: config)
    return config


def original_nodes(relative, names, namespace):
    path = evaluation.SOURCE / relative
    nodes = [n for n in ast.parse(path.read_text(encoding="utf-8")).body
             if getattr(n, "name", None) in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def samplers():
    # Execute untouched published functions without importing unrelated training backends.
    sampler = runpy.run_path(str(Path(torchaudio.__file__).parent.parent /
                                "pytorchvideo/data/clip_sampling.py"))["ConstantClipsPerVideoSampler"]
    scope = dict(torch=torch, torchaudio=torchaudio, np=np, cv2=cv2, rnd=random,
                 ConstantClipsPerVideoSampler=sampler, MAX_INT=sys.maxsize)
    original_nodes("my_affectgpt/models/ImageBind/data.py", {"load_audio", "get_clip_timepoints"}, scope)
    original_nodes("my_affectgpt/processors/video_processor.py", {"load_face"}, scope)
    return scope


@pytest.fixture(scope="module")
def processor():
    torch.set_num_threads(1)
    return omni.load_processor(os.environ["OMNI_PROCESSOR_SNAPSHOT"])


@pytest.fixture
def media(tmp_path):
    def make(seconds=5, frames=19, sample_rate=16000):
        wave = torch.stack([torch.linspace(-.5, .5, int(seconds * sample_rate)),
                            torch.linspace(.25, -.25, int(seconds * sample_rate))])
        audio = tmp_path / "audio.wav"
        torchaudio.save(str(audio), wave, sample_rate)
        face = tmp_path / "face.npy"
        # Frame- and channel-distinct pixels expose reordered/dropped/converted input.
        pixels = np.zeros((frames, 31, 27, 3), dtype=np.uint8)
        for i in range(frames):
            pixels[i] = [i * 3 % 255, i * 5 % 255, i * 7 % 255]
        np.save(face, pixels)
        return {"audio": str(audio), "face": str(face)}
    return make


@pytest.mark.parametrize("seconds,frames,rate", [(5, 19, 16000), (.5, 3, 16000), (3, 8, 22050)])
def test_same_waveforms_and_faces_as_original_dataset(media, samplers, seconds, frames, rate):
    source = media(seconds, frames, rate)
    scope = dict(samplers)
    original_nodes("my_affectgpt/datasets/datasets/base_dataset.py", {"BaseDataset"}, scope)
    dataset = scope["BaseDataset"].__new__(scope["BaseDataset"])
    dataset.needed_data = ["face", "audio"]
    dataset.n_frms = 8
    dataset.vis_processor = SimpleNamespace(transform=lambda x: x)
    # Legacy feature conversion is not fed to Omni; raw media must be identical.
    scope["transform_audio"] = lambda x, _: x
    expected = dataset.read_frame_face_audio_text(None, source["face"], source["audio"], None)
    audio, faces = evaluation.read_reference_media(source, samplers["load_audio"], samplers["load_face"])
    assert np.array_equal(np.stack(audio), expected["raw_audio"][:, 0].numpy())
    assert np.array_equal(np.stack(faces), expected["raw_face"].permute(1, 2, 3, 0).numpy())


def test_processor_uses_only_sampled_media_and_tiny_thinker_accepts_it(media, samplers, processor):
    source = media()
    prompt = "###Human: What is the emotion? ###Assistant: "
    inputs = evaluation.encode_reference_media(processor, prompt, source,
                                               samplers["load_audio"], samplers["load_face"])
    assert inputs["use_audio_in_video"] is False
    assert "pixel_values_videos" not in inputs and "video_second_per_grid" not in inputs
    assert inputs["image_grid_thw"].tolist() == [[1, 16, 16]] * 8
    assert inputs["feature_attention_mask"].sum(-1).tolist() == [200] * 8
    ids = inputs["input_ids"][0]
    assert (ids == processor.tokenizer.convert_tokens_to_ids(processor.audio_token)).sum() == 400
    assert (ids == processor.tokenizer.convert_tokens_to_ids(processor.image_token)).sum() == 512
    audio, images = evaluation.read_reference_media(source, samplers["load_audio"], samplers["load_face"])
    expected_audio = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt",
                                               padding="max_length", truncation=False, return_attention_mask=True)
    assert torch.equal(inputs["input_features"], expected_audio["input_features"])
    expected_images = processor.image_processor(images=images, do_resize=False, return_tensors="pt")
    assert torch.equal(inputs["pixel_values"], expected_images["pixel_values"])
    from test_omni_components import tiny_native
    model = tiny_native(processor).eval()
    with torch.no_grad():
        assert torch.isfinite(model(**inputs).logits).all()
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    assert out.shape[1] == inputs["input_ids"].shape[1] + 1


def test_missing_media_fails_without_full_video_fallback(tmp_path, samplers):
    with pytest.raises(FileNotFoundError):
        evaluation.read_reference_media({"audio": str(tmp_path / "none.wav"), "face": "none.npy"},
                                         samplers["load_audio"], samplers["load_face"])
    with pytest.raises(ValueError):
        evaluation.read_reference_media("video.mp4", samplers["load_audio"], samplers["load_face"])


@pytest.mark.skipif(not os.environ.get("MER_UNIBENCH_ARCHIVE"), reason="Original archive not supplied")
@pytest.mark.parametrize("name", ["test_dia132_utt17", "test_dia0_utt2", "test_dia100_utt14"])
def test_real_meld_media_from_original_archive(tmp_path, samplers, processor, name):
    source = {}
    with zipfile.ZipFile(os.environ["MER_UNIBENCH_ARCHIVE"]) as archive:
        for key, folder, ext in [("audio", "subaudio", "wav"), ("face", "openface_face", "npy")]:
            destination = tmp_path / f"{name}.{ext}"
            destination.write_bytes(archive.read(f"meld-process/{folder}/{name}.{ext}"))
            source[key] = str(destination)
    scope = dict(samplers, transform_audio=lambda x, _: x)
    original_nodes("my_affectgpt/datasets/datasets/base_dataset.py", {"BaseDataset"}, scope)
    ds = scope["BaseDataset"].__new__(scope["BaseDataset"])
    ds.needed_data, ds.n_frms = ["face", "audio"], 8
    ds.vis_processor = SimpleNamespace(transform=lambda x: x)
    reference = ds.read_frame_face_audio_text(None, source["face"], source["audio"], None)
    actual_audio, actual_faces = evaluation.read_reference_media(source, samplers["load_audio"], samplers["load_face"])
    assert np.array_equal(np.stack(actual_audio), reference["raw_audio"][:, 0].numpy())
    assert np.array_equal(np.stack(actual_faces), reference["raw_face"].permute(1, 2, 3, 0).numpy())
    encoded = evaluation.encode_reference_media(processor, "###Human: What is the emotion? ###Assistant: ",
                                                source, samplers["load_audio"], samplers["load_face"])
    assert encoded["feature_attention_mask"].sum(-1).tolist() == [200] * 8
    assert encoded["image_grid_thw"].tolist() == [[1, 16, 16]] * 8
    assert encoded["use_audio_in_video"] is False and "pixel_values_videos" not in encoded


def test_install_keeps_original_chat_and_generation_contract(monkeypatch, processor, media, samplers):
    from transformers import StoppingCriteria, StoppingCriteriaList
    def module(name, **attrs):
        obj = ModuleType(name)
        obj.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, obj)
        if "." in name:
            parent, leaf = name.rsplit(".", 1)
            if parent not in sys.modules:
                module(parent)
            monkeypatch.setattr(sys.modules[parent], leaf, obj, raising=False)
        return obj
    cfg = SimpleNamespace(PATH_TO_LLM={"Qwen25": "unused"})
    for kind in ("IMAGE", "AUDIO", "FRAME", "FACE", "MULTI"):
        setattr(cfg, f"DEFAULT_{kind}_PATCH_TOKEN", f"<{kind.title()}Here>")
    scope = dict(torch=torch, copy=copy, config=cfg, StoppingCriteria=StoppingCriteria,
                 StoppingCriteriaList=StoppingCriteriaList)
    original_nodes("my_affectgpt/conversation/conversation_video.py", {"Chat", "StoppingCriteriaSub"}, scope)
    Chat = scope["Chat"]
    ds = original_nodes("my_affectgpt/datasets/datasets/base_dataset.py", {"BaseDataset"}, {})["BaseDataset"]
    module("config", PATH_TO_LLM=cfg.PATH_TO_LLM)
    module("my_affectgpt.common.registry", registry=SimpleNamespace(mapping={"model_name_mapping": {}}))
    module("my_affectgpt.datasets.datasets.base_dataset", BaseDataset=ds,
           load_audio=samplers["load_audio"], load_face=samplers["load_face"])
    conv = module("my_affectgpt.conversation.conversation_video", Chat=Chat)
    tokenizer = copy.deepcopy(processor.tokenizer)
    tokenizer.bos_token = "<|im_start|>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(list(omni.MEDIA_MARKERS), special_tokens=True)
    module("my_affectgpt.models.tokenizer", load_tokenizer_from_LLM=lambda _: tokenizer)
    monkeypatch.setattr(omni, "OmniInferenceInterface", omni.OmniInferenceInterface)
    native_training_encoder = omni.encode_prompt
    omni.install(os.environ["OMNI_PROCESSOR_SNAPSHOT"])
    evaluation.install_reference_media(omni)
    assert omni.encode_prompt is native_training_encoder
    assert conv.Chat.answer_sample is Chat.answer_sample
    assert conv.Chat.to_token_ids is Chat.to_token_ids
    assert conv.Chat.replace_token_for_multimodal is Chat.replace_token_for_multimodal
    source = media()
    sample = ds.__new__(ds).read_frame_face_audio_text("never-read.mp4", source["face"], source["audio"])
    assert sample["raw_frame"] == source
    model = omni.AffectGPTThinker(torch.nn.Linear(1, 1))
    captured = {}
    def generate(**kwargs):
        captured.update(kwargs)
        answer = tokenizer("happy", return_tensors="pt", add_special_tokens=False)["input_ids"]
        return torch.cat([kwargs["input_ids"], answer], dim=1)
    monkeypatch.setattr(model, "generate", generate)
    # Use a minimal real embedding, without loading the 7B checkpoint.
    embedding = torch.nn.Embedding(len(tokenizer), 2)
    monkeypatch.setattr(model.thinker, "get_base_model", lambda: SimpleNamespace(get_input_embeddings=lambda: embedding), raising=False)
    monkeypatch.setattr(type(model), "device", property(lambda _: torch.device("cpu")))
    chat = conv.Chat(model, SimpleNamespace(llama_model="Qwen25", num_video_query_token=1,
                    num_audio_query_token=1, num_multi_query_token=1, num_image_query_token=1), device="cpu")
    chat.postprocess_frame(sample)
    result = chat.answer_sample("###Human: What is the emotion? ###Assistant: ",
                               dict.fromkeys(("frame", "face", "audio", "image", "multi")),
                               num_beams=1, temperature=1, do_sample=True, top_p=.9,
                               max_new_tokens=1200, max_length=2000)
    assert result == "happy"
    assert captured["use_audio_in_video"] is False
    assert captured["generation_config"].eos_token_id == [151645, 151643]
    assert captured["generation_config"].top_k == 20
    for key, value in dict(num_beams=1, temperature=1, do_sample=True, top_p=.9, max_new_tokens=1200).items():
        assert captured[key] == value


@pytest.mark.parametrize("ending", [151643, 151645])
def test_original_eos_stops_native_generation(ending, original_generation_defaults):
    from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessor, LogitsProcessorList
    # A forced sequence proves both EOS tokens end generate before continuation.
    class ForceEndThenContinuation(LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores.fill_(-float("inf"))
            scores[:, ending if input_ids.shape[1] == 1 else 7] = 0
            return scores
    model = GPT2LMHeadModel(GPT2Config(vocab_size=151646, n_positions=8,
                                      n_embd=8, n_layer=1, n_head=1)).eval()
    result = model.generate(torch.tensor([[3]]), attention_mask=torch.ones(1, 1, dtype=torch.long),
                            generation_config=copy.deepcopy(original_generation_defaults),
                            do_sample=False, max_new_tokens=3,
                            logits_processor=LogitsProcessorList([ForceEndThenContinuation()]))
    assert result.tolist() == [[3, ending]]


@pytest.mark.parametrize("stage,epoch", [("infer", 1), ("infer", 30), ("evaluate", 1)])
def test_launcher_dispatches_original_scripts_and_correct_checkpoint(monkeypatch, tmp_path, stage, epoch):
    import run_affectgpt_omni as runner
    import huggingface_hub
    monkeypatch.setattr(runner, "WORKFLOW", runner.WORKFLOW)
    monkeypatch.setattr(runner, "prepare_execution", lambda *args: evaluation.SOURCE)
    monkeypatch.setattr(evaluation, "checkpoint_alias", lambda *args: tmp_path / evaluation.RESULT_NAME)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: "test-snapshot")
    monkeypatch.setattr(omni, "install", lambda *a, **k: None)
    installs = []
    monkeypatch.setattr(evaluation, "install_reference_media", lambda _: installs.append(True))
    monkeypatch.delenv("OMNI_INPUT_CACHE", raising=False)
    for key in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME"):
        monkeypatch.setenv(key, "test")
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(runner.runpy, "run_path", lambda path, **kwargs: calls.append((path, list(sys.argv))))
    (tmp_path / "evaluation-protocol.json").write_text(json.dumps({"protocol": evaluation.PROTOCOL})) if stage == "evaluate" else None
    argv = ["reference_evaluation", stage, "--runtime", str(tmp_path), "--dataset-root", str(tmp_path)]
    if stage == "infer":
        argv += ["--checkpoints", str(tmp_path), "--checkpoint-epoch", str(epoch)]
    monkeypatch.setattr(sys, "argv", argv)
    evaluation.main()
    path, argv = calls[0]
    assert Path(path) == evaluation.SOURCE / runner.SCRIPTS[stage]
    if stage == "evaluate":
        assert argv == ["evaluation.py"] and installs == []
    else:
        assert installs == [True]
        # Use the actual original argparse to check later options take precedence.
        tree = ast.parse((evaluation.SOURCE / "inference_hybird.py").read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.If) and "__name__" in ast.unparse(n.test))
        nodes = []
        for node in main.body:
            nodes.append(node)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "args" for t in node.targets):
                break
        scope = dict(argparse=argparse)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "original-cli", "exec"), scope)
        assert scope["args"].dataset == "inferenceData" and scope["args"].zeroshot
        assert scope["args"].options == [f"inference.test_epoch={epoch}", f"inference.ckpt_root={tmp_path / evaluation.RESULT_NAME}"]
