# LM Studio Video/Image Captioner

Small Flask UI for captioning images or short video clips through an LM Studio-compatible `/v1/chat/completions` endpoint.

## Features

- Chat captioning for one uploaded image/video.
- Batch captioning for a target folder.
- Image mode for still images; video mode samples frames.
- Optional existing-caption grounding from matching `.txt` files.
- Assistant response prefill.
- Batch progress, cancel, active-file display, elapsed time, ETA, captions/minute, and per-item duration.
- Parallel batch workers for LM Studio's concurrent prediction slots.
- Better LM Studio/API error messages, retries, and a fail-fast guard so a bad concurrency setting does not chew through an entire folder.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://localhost:5057/
```

## Environment variables

- `LMSTUDIO_BASE_URL` default: `http://localhost:1234/v1`
- `LMSTUDIO_MODEL` default: `qwen2.5-vl-32b-instruct`
- `LMSTUDIO_BATCH_CONCURRENCY` default: `4`
- `LMSTUDIO_MAX_BATCH_CONCURRENCY` default: `16`
- `LMSTUDIO_REQUEST_RETRIES` default: `2`
- `LMSTUDIO_RETRY_BACKOFF_SEC` default: `2`
- `LMSTUDIO_ABORT_AFTER_SERVER_ERRORS` default: `3`

## Parallel guidance

The UI's **Parallel batch workers** should usually be less than or equal to LM Studio's **Max Concurrent Predictions**.

For image captioning, start at 4. Test 6 or 8 by comparing captions/minute. If you see lots of quick `LM Studio HTTP 400` errors, the concurrency/context/KV-cache setup is overloaded; lower the workers, lower LM Studio context length, or both.

The **Abort after LM Studio/API errors** setting stops the batch after repeated API-level failures. Set it to `0` only if you deliberately want the batch to continue through errors.
