# SFT-Qwen-Standard-DPO-Affectgpt
We experimented the effect of DPO for our sft qwen model in affectgpt framework.

DPO inputs use AffectGPT's original `load_audio` and `get_clip_timepoints`
functions from the unchanged pinned source: eight two-second clips from the
original dataset WAV, including its original resampling, mono conversion,
padding and clip selection. The interface executes these source functions
directly; it does not reimplement sampling. Preference rows must include the
original dataset-relative `audio` path, obtained from `_get_audio_path`.

Omni receives one video and eight separate audio inputs with
`use_audio_in_video=False`. This is an explicit change from continuous,
synchronized audio/video input. The native audio feature extractor and Thinker
remain Qwen's. The published DPO loss, trainer and all vendor files are unchanged.
The existing benchmark inference/scoring route is unchanged by this DPO-input
change. Qwen's official Decord reader avoids decoding every video frame into RAM.

Two real long MELD clips were tested on an A40 with epoch30 weights. Input
lengths fell from 7978/6225 to 768/762 tokens. With diagnostic 300-token answers,
both DPO forward/backward/update checks passed at approximately 39 GiB peak
allocated memory. These are interface/memory checks, not preference-quality or
benchmark results. The training length limit remains 4096.
