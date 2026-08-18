# Qwen3-Omni -> MiniMax-H3 audiovisual captions

`omni_h3_caption.py` is the no-client-frame-splitting video-caption path for H3 LoRA datasets.

For every input video it writes a UTF-8 sidecar with the same basename:

```text
clip001.mp4
clip001.txt
clip002.webm
clip002.txt
```

The caption is requested in MiniMax-H3 T2VA form:

```text
integrated_multimodal_description: [Shot 1] ...
overall_soundscape: ...
non_diegetic_music: ...
```

The prompt follows the H3 base-prompt conventions: chronological shot/action description, timestamps on later genuine shots when useful, stable speaker IDs and `<d>[Language] ...</d>` for confidently intelligible dialogue, soundscape separated from dialogue/diegetic music, and a separate non-diegetic music field.

## Why this path exists

Normal captionHelper video mode deliberately samples a small number of OpenCV frames and submits them as images. That is useful for generic vision models, but it throws away most video timing and cannot supply the soundtrack.

This tool instead sends the complete source file to a recent llama.cpp server using the OpenAI-compatible `input_video` content type. It also extracts the video's embedded soundtrack to a temporary WAV and sends it as `input_audio`, so Qwen3-Omni receives both modalities in the same request. There is no Python/OpenCV frame selection in this path.

Important limitation: llama.cpp/libmtmd's current video helper is still model-agnostic. It decodes video internally through ffmpeg and currently defaults to an effective 4 fps stream with periodic timestamp chunks. Therefore this is substantially better than captionHelper's arbitrary sparse N-frame splitting, but it is not bit-for-bit the same preprocessing as Qwen's official Transformers `process_mm_info(..., use_audio_in_video=True)` path. A future backend can add that official processor path without changing the caption format or sidecar convention.

## Requirements

- A recent llama.cpp build with libmtmd video support (`MTMD_VIDEO`) and Qwen3-Omni support.
- `ffmpeg` and `ffprobe` on `PATH`.
- A running llama-server exposing `/v1/chat/completions`.
- `requests` (already a captionHelper dependency).

The companion branch in `BlipOnNobodysRadar/llamacpp-server-launcher` contains a preset named:

```text
Qwen3 Omni 30B - Native AV H3 Captioning
```

Its model alias is:

```text
qwen3-omni-h3-caption
```

## Usage

From the captionHelper checkout:

```bash
uv run python omni_h3_caption.py /path/to/video/folder
```

Overwrite existing sidecars:

```bash
uv run python omni_h3_caption.py /path/to/video/folder --overwrite
```

Process nested folders:

```bash
uv run python omni_h3_caption.py /path/to/dataset --recursive
```

Specific files are also accepted:

```bash
uv run python omni_h3_caption.py clip1.mp4 clip2.mp4
```

Use another server/model:

```bash
uv run python omni_h3_caption.py /path/to/clips \
  --base-url http://127.0.0.1:8080/v1 \
  --model qwen3-omni-h3-caption
```

A JSON batch report can be written without changing the sidecar files:

```bash
uv run python omni_h3_caption.py /path/to/clips \
  --report /path/to/clips/omni_caption_report.json
```

Preview what would be processed without model calls:

```bash
uv run python omni_h3_caption.py /path/to/clips --dry-run
```

## Audio behavior

If the video has an audio stream, it is extracted only into a temporary 16 kHz mono PCM WAV for transport to llama.cpp. The original video is never modified. The temporary file is deleted after the request.

If there is no audio stream, the model is told explicitly not to invent speech, ambience, sound effects, or music. Use `--no-audio` to intentionally caption only the video stream.

## Output discipline

The tool validates that all three H3 fields are present and ordered, but intentionally does not rewrite model content after generation. A malformed result is still saved with a warning so it can be inspected rather than silently discarded.

Existing `<basename>.txt` files are skipped by default. Use `--overwrite` when regenerating captions.
