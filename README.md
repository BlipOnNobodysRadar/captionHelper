# Caption Helper

A small local web UI for captioning images and short video clips with a local OpenAI-compatible `/v1/chat/completions` endpoint. It defaults to `llama.cpp`/`llama-server`, while keeping LM Studio compatibility through environment variables.

It is meant for dataset prep: point it at a folder of images or clips, choose a vision model loaded in llama.cpp/llama-server, LM Studio, or another compatible backend, and write matching `.txt` caption files next to the source files.

## Features

- Chat captioning for one uploaded image or video.
- Batch captioning for a target folder.
- Image mode for still images; video mode samples frames from clips.
- Optional existing-caption grounding from matching `.txt` files.
- Assistant response prefill.
- Batch progress, cancel, active-file display, elapsed time, ETA, captions/minute, and per-item duration.
- Parallel batch workers for llama.cpp slots, LM Studio concurrent prediction slots, or other backend concurrency.
- Max image side downscaling to reduce multimodal context pressure.
- Max output token cap for shorter, safer caption generations.
- Better backend/API error messages, retries, and a fail-fast guard so a bad concurrency/context setting does not chew through an entire folder.

## Requirements

- Python 3.10 or newer.
- [uv](https://docs.astral.sh/uv/) for Python environment and dependency management.
- `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp) running with a vision-capable model/projector, or another OpenAI-compatible vision backend such as LM Studio.

The default backend is llama.cpp at:

```text
http://localhost:8080/v1
```

LM Studio is still supported by setting `CAPTION_BACKEND=lmstudio` or `LMSTUDIO_BASE_URL=http://localhost:1234/v1`.

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

## llama.cpp setup

1. Start a llama.cpp `llama-server` with a vision-capable GGUF model and the matching multimodal projector/options required by that model.
2. Keep the default llama.cpp HTTP server port (`8080`) or set `CAPTION_API_BASE_URL` / `LLAMA_CPP_BASE_URL` to your custom `/v1` base URL.
3. In the app, set the model name to the name your server accepts. llama.cpp generally accepts any non-empty model string for OpenAI-compatible requests, but using a descriptive name is helpful for logs.

Example run command shape (adjust model/projector paths for your model):

```bash
llama-server -m /path/to/vision-model.gguf --mmproj /path/to/mmproj.gguf --host 127.0.0.1 --port 8080
```

llama.cpp's server exposes the OpenAI-compatible chat completions endpoint at `/v1/chat/completions`; captionHelper uses the base URL `http://localhost:8080/v1` by default.

For parallel batch captioning, the app's **Parallel batch workers** should usually be equal to or lower than the number of llama.cpp slots/parallel requests you configured.

## LM Studio setup

1. Open LM Studio.
2. Load a vision-capable model.
3. Start the local server.
4. Start captionHelper with `CAPTION_BACKEND=lmstudio` or `CAPTION_API_BASE_URL=http://localhost:1234/v1`.
5. In the app, set the model name to match the loaded model.

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

## Presets and prompt templates

captionHelper includes a **Preset** selector for common caption-conversion workflows:

- basic image captioning;
- detailed image captioning;
- grounded image/tag conversion;
- basic video captioning;
- grounded video caption conversion;
- Ideogram 4 image-to-JSON captioning.

Selecting a preset fills both editable prompt fields:

- **System prompt** controls the static model behavior.
- **User message template** controls the per-request message sent with the image or video frames. Use `[image]` where visual inputs should be inserted.

After editing a preset, use **Save as user preset** to store it in `user_presets.json`. This file is ignored by git so local prompt experiments, private templates, and dataset-specific presets are not committed. Set `CAPTION_USER_PRESETS_PATH=/path/to/presets.json` if you want to keep the file elsewhere. User presets can be selected like built-in presets and deleted from the UI.

The template can use these placeholders, and lines containing only empty placeholders are omitted automatically:

```text
{existing_caption}
{source_tags}
{character_tags}
{copyright_tags}
{artist_tags}
{general_tags}
{rating_tags}
{quality_tags}
{media_kind}
{input_count}
```

The optional tag-group fields are shared by chat and batch requests. For batch jobs, per-file `.txt` captions can also contain grouped lines such as `CHARACTER: ...`, `COPYRIGHT: ...`, `ARTIST: ...`, `GENERAL: ...`, `RATING: ...`, and `QUALITY: ...`; captionHelper maps those into matching template placeholders.

## Existing-caption grounding

If **Use existing caption** is enabled during batch mode, the app looks for a matching `.txt` file next to each image/video and passes that text through the active user message template as grounding.

This is useful for converting booru tags, rough captions, or older captions into more detailed natural-language captions or structured preset outputs.

## Parallel/context guidance

The UI's **Parallel batch workers** should usually be less than or equal to llama.cpp slots/parallel requests or LM Studio's **Max Concurrent Predictions**.

For GGUF/llama.cpp-style serving, parallel vision requests can put heavy pressure on the available context/KV cache. A useful rough mental model is:

```text
per-slot context ~= backend context length / backend parallel slots
```

So `20000 / 4` gives roughly 5000 tokens per slot, while `20000 / 12` gives roughly 1667 tokens per slot. Large images or video frame batches can exceed that quickly because multimodal image patches count against context too.

For image captioning, start at 4 parallel workers. Test 6, 8, or 12 by comparing captions/minute. If you see `Context size has been exceeded`, do one or more of these:

- lower **Parallel batch workers** and your backend slot/concurrency setting;
- raise the backend context length/KV cache setting, if VRAM allows;
- lower **Max image side** from 1024 to 768 or 640;
- lower **Max output tokens** from 512 to 256.

Avoid setting **Max image side** to `0` during high-parallel runs unless the source images are already small.

## Environment variables

You can override defaults with environment variables.

```bash
CAPTION_API_BASE_URL="http://localhost:8080/v1" uv run python app.py
```

Available variables:

| Variable | Default | Meaning |
|---|---:|---|
| `CAPTION_BACKEND` | `llamacpp` | Backend preset. Use `llamacpp`, `lmstudio`, or `openai`. Common aliases such as `llama.cpp` and `lm-studio` also work. |
| `CAPTION_API_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible API base URL. For LM Studio use `http://localhost:1234/v1`. |
| `CAPTION_MODEL` | `qwen2.5-vl-32b-instruct` | Default model name sent to the API. |
| `CAPTION_BATCH_CONCURRENCY` | `4` | Default parallel batch workers shown in the UI. |
| `CAPTION_MAX_BATCH_CONCURRENCY` | `16` | Hard cap for parallel workers accepted by the backend. |
| `CAPTION_REQUEST_RETRIES` | `2` | Number of retries after API/request failures. |
| `CAPTION_RETRY_BACKOFF_SEC` | `2` | Base retry backoff in seconds. |
| `CAPTION_ABORT_AFTER_SERVER_ERRORS` | `3` | Abort batch after this many API-level errors. Set `0` to disable. |
| `CAPTION_MAX_IMAGE_SIDE` | `1024` | Default max image dimension before sending to the backend. Set `0` to disable resizing. |
| `CAPTION_MAX_OUTPUT_TOKENS` | `512` | Default generation limit for captions. |
| `CAPTION_USER_PRESETS_PATH` | `user_presets.json` | Local JSON file for UI-saved user presets. Relative paths are resolved from the app directory; the default file is git-ignored. |

Backwards-compatible `LMSTUDIO_*` variables still work. New `LLAMA_CPP_*` aliases are also accepted, for example `LLAMA_CPP_BASE_URL` and `LLAMA_CPP_MODEL`. `LAMMA_CPP_*` is accepted as a typo-tolerant alias.

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
