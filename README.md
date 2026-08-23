# Caption Helper

A small local web UI for captioning images and short video clips with a local OpenAI-compatible `/v1/chat/completions` endpoint. It defaults to `llama.cpp`/`llama-server`, while keeping LM Studio compatibility through environment variables.

It is meant for dataset prep: point it at a folder of images or clips, choose a vision model loaded in llama.cpp/llama-server, LM Studio, or another compatible backend, and write matching `.txt` caption files next to the source files.

## Features

- Chat captioning for one uploaded image or video.
- Batch captioning for a target folder.
- Image mode for still images; video mode either samples frames or sends a complete native audiovisual clip.
- Optional existing-caption grounding from matching `.txt` files.
- Assistant response prefill.
- Batch progress, cancel, active-file display, elapsed time, ETA, captions/minute, and per-item duration.
- Parallel batch workers for llama.cpp slots, LM Studio concurrent prediction slots, or other backend concurrency.
- Max image side downscaling to reduce multimodal context pressure.
- Max output token cap for shorter, safer caption generations.
- Better backend/API error messages, retries, and a fail-fast guard so a bad concurrency/context setting does not chew through an entire folder.
- Resume support for batches halted by backend/API errors, reusing successfully written captions from the prior run and retrying only unfinished/errored items.

## Requirements

- Python 3.10 or newer.
- [uv](https://docs.astral.sh/uv/) for Python environment and dependency management.
- `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp) running with a vision-capable model/projector, or another OpenAI-compatible vision backend such as LM Studio.
- FFmpeg (`ffmpeg` and `ffprobe` on `PATH`) to automatically include sound in native audiovisual mode.

## Native audiovisual captioning

Choose **MiniMax H3 T2VA - Qwen3 Omni Native AV** in either chat or Batch processing. The browser selects native video input, enables audio, chooses the `qwen3-omni-h3-caption` alias, and uses one worker by default. MP4, WebM, MOV, AVI, MKV, and M4V inputs are supported. CaptionHelper sends the complete file as `input_video`, extracts a temporary mono 16 kHz PCM WAV as `input_audio`, and removes the WAV after success or failure. Videos without an audio stream continue as explicitly visual-only inputs.

This mode is intended primarily for a recent local llama.cpp server because raw base64 media can make requests large. "Native" means CaptionHelper does not arbitrarily choose OpenCV frames: llama.cpp/libmtmd performs its own model-agnostic decoding. It is not bit-identical to Qwen's official Transformers preprocessing, which would use `process_mm_info(..., use_audio_in_video=True)` with Qwen's processor; that remains a possible future backend.

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

By default captionHelper binds only to `127.0.0.1` and runs with Flask debug mode off. Keep it local unless you fully trust the network: the app can read target-folder filenames, read matching caption text, upload images/video frames to the configured vision backend, and write caption files. If you intentionally expose it beyond localhost, set `CAPTION_HOST`, `CAPTION_ALLOWED_HOSTS`, and any firewall/reverse-proxy authentication deliberately.

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

The app writes captions as `.txt` files beside the original media by default:

```text
image001.jpg  -> image001.txt
clip001.mp4   -> clip001.txt
```

Batch output options can also copy the media and caption outputs with custom filename text prepended or appended. Include any desired separator in the custom text:

```text
image001.jpg + prefix "ideogramConversion_" -> ideogramConversion_image001.jpg
                                                ideogramConversion_image001.txt
```

Enable **Copy results to target subdirectory** to write the copied media and generated captions inside a subdirectory of the target folder instead of beside the originals.

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

After editing a preset, use **Save as user preset** to store it in `user_presets.json`. User presets can include local model paths, dataset paths, output settings, checkbox states, and prompt metadata, so `user_presets*.json` files are ignored by git to keep local prompt experiments, private templates, and dataset-specific paths out of commits. Set `CAPTION_USER_PRESETS_PATH=/path/to/presets.json` if you want to keep the file elsewhere; if that path lives inside a git repository, add it to that repository's `.gitignore` before saving presets there. User presets can be selected like built-in presets and deleted from the UI.

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

## Resuming halted batches

If a batch stops after reaching the configured backend/API error limit, click **Resume halted batch** to start a new job from the failed run. captionHelper records local job state in `.caption_jobs/` by default, which is ignored by git and intended to stay on your machine.

On resume, successful or skipped items from the halted run are treated as already complete, so their existing non-errored caption files are left in place. Only files that errored or never started are queued again. Set `CAPTION_USER_JOBS_PATH=/path/to/jobs` if you want to keep these local resume records somewhere else.

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

## Security and privacy notes

captionHelper is a local dataset-preparation tool, not a multi-user web service. The web UI accepts uploaded media and target folder paths, and batch jobs expose active/result filenames in progress responses so the UI can show useful status. To reduce accidental leaks:

- The development server defaults to `127.0.0.1` instead of all network interfaces.
- Flask debug mode is disabled unless `CAPTION_DEBUG=true` is set.
- Host, Origin, and Referer checks default to localhost hosts to reduce DNS-rebinding and cross-site request risks.
- `/static` only serves the browser assets required by the UI, instead of exposing the whole repository root.
- Upload size is capped by `CAPTION_MAX_UPLOAD_BYTES` to reduce accidental denial-of-service from very large chat uploads.

Do not point `CAPTION_API_BASE_URL` at an untrusted remote service unless you are comfortable sending uploaded images/video frames, prompts, existing captions, and tag metadata to that service.

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
| `CAPTION_HOST` | `127.0.0.1` | Host/interface for the Flask development server. Use `0.0.0.0` only on a trusted network with additional access controls. |
| `CAPTION_PORT` | `5057` | Port for the Flask development server. |
| `CAPTION_DEBUG` | `false` | Enables Flask debug mode when set to `true`, `1`, `yes`, or `on`. Do not enable on shared networks. |
| `CAPTION_ALLOWED_HOSTS` | `localhost,127.0.0.1,::1` | Comma-separated hostnames accepted in Host/Origin/Referer checks. Set deliberately if exposing the app under another hostname; `*` disables these checks. |
| `CAPTION_MAX_UPLOAD_BYTES` | `536870912` | Maximum chat-upload request size in bytes (default 512 MiB). |

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

## Optional region-proposal preprocessing for Ideogram 4 JSON captions

CaptionHelper can run a first-pass image preprocessor before the existing llama.cpp/Gemma caption request. Enable **Image mode** and **Add object/OCR region candidates** in the UI, or pass the matching batch API fields. The preprocessor returns JSON region hints only; the existing OpenAI-compatible llama.cpp backend still writes the final Ideogram 4 JSON caption.

The lightweight checked-in `vision_preprocess.py` provides the orchestration contract, coordinate conversion, OCR hook, filtering, and JSON output format. Heavy model integrations are optional local dependencies:

- `--detector groundingdino` / `groundingdino1.5` is the intended object-box source.
- `--segmenter sam2` is reserved for box-prompted mask refinement; SAM2 is not used as an object discoverer by itself.
- `--ocr paddleocr` is reserved for OCR, but is temporarily disabled at runtime.
- `--detector florence2` is exposed as an alternative prototype path.

Temporarily, CaptionHelper exposes only GroundingDINO in the UI. SAM2 and PaddleOCR hooks remain in the script for future work, but are disabled at runtime because local testing showed unstable failures in those integrations.

GroundingDINO selections run through the `transformers` zero-shot object detection pipeline when the model and runtime load successfully. If GroundingDINO runs but returns zero candidates, lower the detector box/text thresholds in the UI and/or add simpler object prompts/tags.

If GroundingDINO fails with CUDA out-of-memory while Gemma is loaded in a separate classic `llama-server -m ...` process, either run llama.cpp in router mode and enable the router unload/reload checkbox, or set **Preprocessing device** to CPU. CPU preprocessing is slower but avoids competing for VRAM with the caption model.

For selected detector models, the preprocessor resolves assets in this order:

1. A user-supplied model path or directory, when provided in the UI or via CLI.
2. The expected local repo path under `~/.cache/captionhelper/vision_models`.
3. An automatic Hugging Face download into that local repo path when auto-download is enabled and `huggingface_hub` is installed.

When **Load selected preprocessing models** is enabled, CaptionHelper also asks the preprocessor to warm-load the selected detector runtime after resolving assets. This surfaces missing Python packages or incompatible checkpoints before the Gemma caption request. Disable auto-download for fully offline/private runs, or point the override field at an already-downloaded model directory/checkpoint.

If the UI reports that preprocessing was skipped because packages such as `huggingface_hub`, `transformers`, or `torch` are missing, sync the current app dependencies in the same environment running CaptionHelper:

```bash
uv sync
```

`huggingface-hub` is needed for the **Auto-download missing preprocessing models** checkbox. `transformers` and `torch` are needed to warm-load and run GroundingDINO/Florence-style detectors.

Example CLI:

```bash
python vision_preprocess.py \
  --image /path/to/image.jpg \
  --tags /path/to/image.txt \
  --out /tmp/regions.json \
  --detector groundingdino \
  --segmenter none \
  --ocr none \
  --model-root ~/.cache/captionhelper/vision_models \
  --detector-model-path /optional/path/to/groundingdino \
  --load-models \
  --max-regions 12
```

Detector pixel boxes in `[x1, y1, x2, y2]` order are normalized to Ideogram `[y_min, x_min, y_max, x_max]` coordinates on a 1000x1000 grid, clamped to 0-1000, and rejected unless both axes have positive area. Region candidates are injected into the prompt as a `REGION_CANDIDATES` block so Gemma can prefer supplied coordinates, merge duplicates, omit bad candidates, and add missing important elements only when necessary.

When region preprocessing is enabled, CaptionHelper also validates the final response as Ideogram 4 JSON and retries once with validator errors. Batch outputs get a sibling `.caption_meta.json` sidecar containing the image hash, model/backend settings, region candidates, validation result, and provenance. These local metadata files are gitignored so private dataset metadata is not committed by default.

### llama.cpp VRAM handoff during preprocessing

The preprocessor can download/resolve detector weights, but loading those models is separate from unloading Gemma in llama.cpp. A classic `llama-server -m gemma.gguf` process does not expose a general unload/reload endpoint for its single bound model. To free VRAM while GroundingDINO runs, start a recent llama.cpp server in **router mode** and enable CaptionHelper's router handoff from the UI or environment:

```bash
CAPTION_LLAMA_CPP_MODEL_MANAGEMENT=router \
CAPTION_LLAMA_CPP_UNLOAD_DURING_PREPROCESS=true \
python app.py
```

In router mode CaptionHelper calls `/models/unload` for the configured caption model immediately before preprocessing, then `/models/load` for the same model before sending the final Gemma caption request. If the router endpoints are unavailable or disabled, the handoff is skipped/non-fatal and the normal caption flow continues.

Relevant environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CAPTION_REGION_PREPROCESS_SCRIPT` | `vision_preprocess.py` | Preprocessor script path. |
| `CAPTION_REGION_DETECTOR` | `groundingdino` | Default detector selection. |
| `CAPTION_REGION_SEGMENTER` | `none` | Default segmenter selection. |
| `CAPTION_REGION_OCR` | `none` | OCR is temporarily disabled; keep this as `none`. |
| `CAPTION_REGION_MAX_REGIONS` | `12` | Maximum retained candidates. |
| `CAPTION_REGION_OCR_THRESHOLD` | `0.55` | Minimum OCR confidence. |
| `CAPTION_REGION_MODEL_ROOT` | `~/.cache/captionhelper/vision_models` | Expected local repo path for selected preprocessing models. |
| `CAPTION_REGION_AUTO_DOWNLOAD` | `true` | Whether missing selected detector assets may be downloaded automatically. |
| `CAPTION_REGION_LOAD_MODELS` | `true` | Warm-load selected preprocessing model runtimes before running detection. |
| `CAPTION_REGION_DETECTOR_MODEL_PATH` | empty | Optional detector checkpoint or directory override. |
| `CAPTION_REGION_SEGMENTER_MODEL_PATH` | empty | Optional SAM2 checkpoint or directory override. |
| `CAPTION_REGION_OCR_MODEL_PATH` | empty | Optional OCR model directory override. |
| `CAPTION_LLAMA_CPP_MODEL_MANAGEMENT` | `off` | Set to `router` to use llama.cpp router `/models/load` and `/models/unload`. |
| `CAPTION_LLAMA_CPP_UNLOAD_DURING_PREPROCESS` | `false` | Unload the configured llama.cpp caption model before preprocessing and reload it before Gemma captioning. Requires router mode. |
| `CAPTION_LLAMA_CPP_MODEL_MANAGEMENT_BASE_URL` | derived from API URL | Optional router management base URL, for example `http://localhost:8080`. |
