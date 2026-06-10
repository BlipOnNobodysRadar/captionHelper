# Caption Helper

A small local web UI for captioning images and short video clips with an LM Studio-compatible `/v1/chat/completions` endpoint.

It is meant for dataset prep: point it at a folder of images or clips, choose a vision model loaded in LM Studio, and write matching `.txt` caption files next to the source files.

## Features

- Chat captioning for one uploaded image or video.
- Batch captioning for a target folder.
- Image mode for still images; video mode samples frames from clips.
- Optional existing-caption grounding from matching `.txt` files.
- Assistant response prefill.
- Batch progress, cancel, active-file display, elapsed time, ETA, captions/minute, and per-item duration.
- Parallel batch workers for LM Studio's concurrent prediction slots.
- Max image side downscaling to reduce multimodal context pressure.
- Max output token cap for shorter, safer caption generations.
- Better LM Studio/API error messages, retries, and a fail-fast guard so a bad concurrency/context setting does not chew through an entire folder.

## Requirements

- Python 3.10 or newer.
- [uv](https://docs.astral.sh/uv/) for Python environment and dependency management.
- LM Studio running a local server with a vision-capable model loaded.

The default LM Studio endpoint is:

```text
http://localhost:1234/v1
```

## Install uv

macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal after installing if the `uv` command is not found.

## Setup

Clone the repo, then enter it:

```bash
git clone https://github.com/BlipOnNobodysRadar/captionHelper
cd caption-helper
```

Create the project virtual environment and install dependencies:

```bash
uv sync
```

This creates a local `.venv/` folder for the project. Do not install dependencies globally.

## Run

Start the app through uv:

```bash
uv run python app.py
```

Then open:

```text
http://localhost:5057/
```

You do not need to manually activate the venv when using `uv run`.

If you prefer to activate it anyway:

macOS/Linux:

```bash
source .venv/bin/activate
python app.py
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python app.py
```

## LM Studio setup

1. Open LM Studio.
2. Load a vision-capable model.
3. Start the local server.
4. In the app, set the model name to match the loaded model.

For parallel batch captioning, LM Studio's **Max Concurrent Predictions** should usually be equal to or greater than the app's **Parallel batch workers** setting.

## Batch usage

1. Choose image mode or video mode.
2. Set the target folder path.
3. Choose whether to overwrite existing captions.
4. Choose parallel workers.
5. Start the batch.

The app writes captions as `.txt` files beside the original media:

```text
image001.jpg  -> image001.txt
clip001.mp4   -> clip001.txt
```

By default, generated `.txt` files are ignored by git so local datasets do not get committed accidentally.

## Existing-caption grounding

If **Use existing caption** is enabled during batch mode, the app looks for a matching `.txt` file next to each image/video and adds that text to the prompt as grounding.

This is useful for converting booru tags, rough captions, or older captions into more detailed natural-language captions.

## Parallel/context guidance

The UI's **Parallel batch workers** should usually be less than or equal to LM Studio's **Max Concurrent Predictions**.

For GGUF/llama.cpp-style serving, parallel vision requests can put heavy pressure on the available context/KV cache. A useful rough mental model is:

```text
per-slot context ~= LM Studio Context Length / Max Concurrent Predictions
```

So `20000 / 4` gives roughly 5000 tokens per slot, while `20000 / 12` gives roughly 1667 tokens per slot. Large images or video frame batches can exceed that quickly because multimodal image patches count against context too.

For image captioning, start at 4 parallel workers. Test 6, 8, or 12 by comparing captions/minute. If you see `Context size has been exceeded`, do one or more of these:

- lower **Parallel batch workers** and LM Studio **Max Concurrent Predictions**;
- raise LM Studio **Context Length**, if VRAM allows;
- lower **Max image side** from 1024 to 768 or 640;
- lower **Max output tokens** from 512 to 256.

Avoid setting **Max image side** to `0` during high-parallel runs unless the source images are already small.

## Environment variables

You can override defaults with environment variables.

```bash
LMSTUDIO_BASE_URL="http://localhost:1234/v1" uv run python app.py
```

Available variables:

| Variable | Default | Meaning |
|---|---:|---|
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio-compatible API base URL. |
| `LMSTUDIO_MODEL` | `qwen2.5-vl-32b-instruct` | Default model name sent to the API. |
| `LMSTUDIO_BATCH_CONCURRENCY` | `4` | Default parallel batch workers shown in the UI. |
| `LMSTUDIO_MAX_BATCH_CONCURRENCY` | `16` | Hard cap for parallel workers accepted by the backend. |
| `LMSTUDIO_REQUEST_RETRIES` | `2` | Number of retries after API/request failures. |
| `LMSTUDIO_RETRY_BACKOFF_SEC` | `2` | Base retry backoff in seconds. |
| `LMSTUDIO_ABORT_AFTER_SERVER_ERRORS` | `3` | Abort batch after this many API-level errors. Set `0` to disable. |
| `LMSTUDIO_MAX_IMAGE_SIDE` | `1024` | Default max image dimension before sending to LM Studio. Set `0` to disable resizing. |
| `LMSTUDIO_MAX_OUTPUT_TOKENS` | `512` | Default generation limit for captions. |

## Fallback setup without uv

uv is recommended. If you cannot use it, use a normal venv instead:

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```
