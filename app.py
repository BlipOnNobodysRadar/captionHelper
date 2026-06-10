import os
import base64
import io
import json
import time
import uuid
import threading
import queue
import random
from flask import Flask, render_template, request, jsonify
import requests
import cv2
from PIL import Image

# ------------------ Config ------------------
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
DEFAULT_MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen2.5-vl-32b-instruct")
DEFAULT_BATCH_CONCURRENCY = int(os.environ.get("LMSTUDIO_BATCH_CONCURRENCY", "4"))
MAX_BATCH_CONCURRENCY = int(os.environ.get("LMSTUDIO_MAX_BATCH_CONCURRENCY", "16"))
LMSTUDIO_REQUEST_RETRIES = int(os.environ.get("LMSTUDIO_REQUEST_RETRIES", "2"))
LMSTUDIO_RETRY_BACKOFF_SEC = float(os.environ.get("LMSTUDIO_RETRY_BACKOFF_SEC", "2"))
LMSTUDIO_ABORT_AFTER_SERVER_ERRORS = int(os.environ.get("LMSTUDIO_ABORT_AFTER_SERVER_ERRORS", "3"))
DEFAULT_MAX_IMAGE_SIDE = int(os.environ.get("LMSTUDIO_MAX_IMAGE_SIDE", "1024"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get("LMSTUDIO_MAX_OUTPUT_TOKENS", "512"))

ALLOWED_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
ALLOWED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Default prompts (the UI may send its own, but when modes switch we ensure sane defaults)
DEFAULT_PROMPT_VIDEO = (
    "You caption short videos for dataset creation. Only return the caption (no prelude, no quotes). "
    "Be concrete, specific, and neutral. If multiple actions occur, summarize succinctly. Include descriptions of motions implied between frames. "
    "Include information about watermarks and text if visible, and the quality/resolution if notable. "
)

DEFAULT_PROMPT_IMAGE = (
    "You caption single still images for dataset creation. Only return the caption (no prelude, no quotes). "
    "Be concrete, specific, and neutral. Focus on visible subjects, actions/poses, setting, composition, "
    "and salient attributes. Include information about watermarks and text if visible, and the quality/resolution if notable."
)

# Serve everything from project root for simplicity
app = Flask(__name__, template_folder=".", static_folder=".", static_url_path="/static")

# ------------------ Helpers ------------------
def allowed_video(path:str)->bool:
    return os.path.splitext(path)[1].lower() in ALLOWED_EXTS

def allowed_image(path:str)->bool:
    return os.path.splitext(path)[1].lower() in ALLOWED_IMG_EXTS

def frame_indices(total_frames:int, num_frames:int, sampling:str):
    if total_frames <= 0:
        return []
    n = max(1, int(num_frames))
    if sampling == "head":
        return list(range(min(n, total_frames)))
    if n == 1:
        return [0]
    step = (total_frames - 1) / (n - 1)
    return [int(round(i * step)) for i in range(n)]

def _resize_image_for_vision(im:Image.Image, max_side:int=DEFAULT_MAX_IMAGE_SIDE)->Image.Image:
    """Downscale large images before sending them to LM Studio.

    Multimodal GGUF servers count image patches/tokens against the context
    budget. Huge source images can blow up per-slot context when parallelism
    is increased. A max_side of 768-1024 is usually plenty for captioning.
    Use 0 or a negative value to disable resizing.
    """
    try:
        max_side = int(max_side)
    except (TypeError, ValueError):
        max_side = DEFAULT_MAX_IMAGE_SIDE
    if max_side <= 0:
        return im
    w, h = im.size
    longest = max(w, h)
    if longest <= max_side:
        return im
    scale = max_side / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return im.resize(new_size, Image.Resampling.LANCZOS)


def extract_frames(video_path:str, num_frames:int, sampling:str, max_image_side:int=DEFAULT_MAX_IMAGE_SIDE):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = frame_indices(total, num_frames, sampling)
    images = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        pil = _resize_image_for_vision(pil, max_image_side)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        images.append(f"data:image/jpeg;base64,{b64}")
    cap.release()
    return images

def image_to_data_url(image_path:str, max_image_side:int=DEFAULT_MAX_IMAGE_SIDE)->str:
    """Load an image from disk, optionally downscale it, and return a JPEG data URL."""
    with Image.open(image_path) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        if im.mode == "RGBA":
            im = im.convert("RGB")
        im = _resize_image_for_vision(im, max_image_side)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

def augment_system_with_existing(system_prompt:str, existing_caption:str, media_kind:str)->str:
    existing_caption = (existing_caption or "").strip()
    if not existing_caption:
        return system_prompt
    addon = (
        f"\n\nYour task is to use the existing caption below, including any tag strings if present, "
        f"and convert it into a more accurate natural-language caption of the {media_kind}. "
        "Prefer readable prose; include concrete, visible details; avoid speculation. Use specific explicit language from the booru captions if such tags are included."
        "\n\nExisting caption:\n---\n" + existing_caption + "\n---\n"
    )
    return (system_prompt or "").rstrip() + addon

class LMStudioRequestError(RuntimeError):
    def __init__(self, status_code=None, message="", body=""):
        self.status_code = status_code
        self.body = body
        super().__init__(message)


def _shorten(text:str, limit:int=700)->str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …"


def _lmstudio_error_detail(response):
    body_text = ""
    try:
        data = response.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                body_text = err.get("message") or err.get("type") or json.dumps(err, ensure_ascii=False)
            elif err:
                body_text = str(err)
            else:
                body_text = json.dumps(data, ensure_ascii=False)
        else:
            body_text = str(data)
    except Exception:
        body_text = response.text or ""
    return _shorten(body_text) or response.reason or "No response body"


def _is_retryable_lmstudio_status(status_code:int)->bool:
    # 400 is normally not retryable, but LM Studio/local llama.cpp servers can emit it
    # during overloaded multimodal/parallel batches. Retrying once keeps a brief
    # overload hiccup from ruining a whole batch, while still surfacing real bad payloads.
    return status_code in {400, 408, 409, 425, 429, 500, 502, 503, 504}


def call_lmstudio_vision(images_data_urls, system_prompt:str, model:str, prefill:str="", media_kind:str="clip", max_output_tokens:int=DEFAULT_MAX_OUTPUT_TOKENS):
    if media_kind == "image":
        lead_text = "You are given a single still image. Write a descriptive caption."
    else:
        lead_text = "You are given a few frames sampled from a short video clip. Write a descriptive caption for the clip as a whole."

    user_content = [{"type":"text","text": lead_text}]
    for url in images_data_urls:
        user_content.append({"type":"image_url","image_url":{"url": url}})

    messages = [
        {"role":"system","content": (system_prompt or "").strip()},
        {"role":"user","content": user_content}
    ]

    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "add_generation_prompt": True
    }

    try:
        max_output_tokens = int(max_output_tokens)
    except (TypeError, ValueError):
        max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
    if max_output_tokens > 0:
        payload["max_tokens"] = max_output_tokens

    if prefill and prefill.strip():
        messages.append({"role":"assistant","content": prefill})
        payload["add_generation_prompt"] = False

    url = f"{LMSTUDIO_BASE_URL}/chat/completions"
    attempts = max(1, LMSTUDIO_REQUEST_RETRIES)
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            r = requests.post(url, json=payload, timeout=300)
        except requests.RequestException as e:
            last_error = LMStudioRequestError(None, f"LM Studio request failed: {e}", str(e))
            retryable = True
        else:
            if r.ok:
                data = r.json()
                return data["choices"][0]["message"]["content"].strip()

            detail = _lmstudio_error_detail(r)
            msg = f"LM Studio HTTP {r.status_code}: {detail}"
            last_error = LMStudioRequestError(r.status_code, msg, detail)
            detail_lower = detail.lower()
            if "context size" in detail_lower or "context length" in detail_lower or "context window" in detail_lower:
                retryable = False
            else:
                retryable = _is_retryable_lmstudio_status(r.status_code)

        if attempt < attempts and retryable:
            # Jitter avoids every worker retrying at the exact same moment.
            delay = LMSTUDIO_RETRY_BACKOFF_SEC * attempt + random.uniform(0.0, 1.5)
            time.sleep(delay)
            continue
        raise last_error

    raise last_error or LMStudioRequestError(None, "LM Studio request failed for an unknown reason")

# ------------------ Simple chat route ------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat-caption", methods=["POST"])
def chat_caption():
    f = request.files.get("file")
    if not f:
        return jsonify({"error":"No file uploaded"}), 400

    image_mode = request.form.get("image_mode","false").lower() in ("1","true","yes","on")
    system_prompt_in = request.form.get("system_prompt","").strip()
    if not system_prompt_in:
        system_prompt_in = DEFAULT_PROMPT_IMAGE if image_mode else DEFAULT_PROMPT_VIDEO
    elif image_mode and system_prompt_in == DEFAULT_PROMPT_VIDEO:
        system_prompt_in = DEFAULT_PROMPT_IMAGE

    model = request.form.get("model", DEFAULT_MODEL)
    prefill = request.form.get("prefill","")
    max_image_side = _clamp_int(request.form.get("max_image_side", DEFAULT_MAX_IMAGE_SIDE), DEFAULT_MAX_IMAGE_SIDE, 0, 8192)
    max_output_tokens = _clamp_int(request.form.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS), DEFAULT_MAX_OUTPUT_TOKENS, 0, 8192)
    use_existing = request.form.get("use_existing_caption","false").lower() in ("1","true","yes","on")
    existing_caption_text = request.form.get("existing_caption","")

    media_kind = "image" if image_mode else "clip"
    if use_existing and existing_caption_text.strip():
        system_prompt_in = augment_system_with_existing(system_prompt_in, existing_caption_text, media_kind)

    tmpdir = "tmp_uploads"
    os.makedirs(tmpdir, exist_ok=True)
    original_name = f.filename or "upload"
    _, ext = os.path.splitext(original_name)
    upload_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}{ext}")
    f.save(upload_path)

    try:
        if image_mode:
            if not allowed_image(upload_path):
                raise RuntimeError("Unsupported image file extension")
            imgs = [image_to_data_url(upload_path, max_image_side=max_image_side)]
        else:
            if not allowed_video(upload_path):
                raise RuntimeError("Unsupported video file extension")
            try:
                num_frames = int(request.form.get("num_frames","5"))
            except:
                num_frames = 5
            sampling = request.form.get("sampling_type","uniform")
            imgs = extract_frames(upload_path, num_frames, sampling, max_image_side=max_image_side)

        if not imgs:
            raise RuntimeError("No visual inputs found for captioning")

        caption = call_lmstudio_vision(imgs, system_prompt_in, model, prefill=prefill, media_kind=media_kind, max_output_tokens=max_output_tokens)
        return jsonify({"caption": caption, "frames_used": len(imgs)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(upload_path)
        except:
            pass

# ------------------ Batch job system ------------------
JOBS = {}
JOBS_LOCK = threading.Lock()

def _select_targets(folder:str, image_mode:bool):
    if image_mode:
        selected = [p for p in os.listdir(folder) if allowed_image(os.path.join(folder, p))]
    else:
        selected = [p for p in os.listdir(folder) if allowed_video(os.path.join(folder, p))]
    selected.sort()
    return selected

def _clamp_int(value, default:int, minimum:int=1, maximum:int=16)->int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = default
    return max(minimum, min(maximum, n))


def _process_one_target(fn:str, params:dict):
    folder = params["target_folder"]
    image_mode = params["image_mode"]
    system_prompt_in = params["system_prompt"]
    model = params["model"]
    prefill = params["prefill"]
    num_frames = params["num_frames"]
    sampling = params["sampling_type"]
    overwrite = params["overwrite"]
    prepend_existing = params["prepend_existing"]
    use_existing = params["use_existing_caption"]
    max_image_side = params.get("max_image_side", DEFAULT_MAX_IMAGE_SIDE)
    max_output_tokens = params.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)

    media_kind = "image" if image_mode else "clip"
    in_path = os.path.join(folder, fn)
    base, _ = os.path.splitext(in_path)
    out_txt = base + ".txt"

    # Skip logic. Existing caption grounding implies we usually want to overwrite/prepend,
    # otherwise the safest behavior is still to leave existing caption files alone.
    if os.path.exists(out_txt) and (not overwrite) and (not prepend_existing):
        return {"file": fn, "skipped": True, "reason": "caption exists"}

    try:
        # Optional augmentation with existing caption file.
        sys_for_this = system_prompt_in
        if use_existing and os.path.exists(out_txt):
            try:
                with open(out_txt, "r", encoding="utf-8") as fh:
                    old_text = fh.read().strip()
            except Exception:
                old_text = ""
            if old_text:
                sys_for_this = augment_system_with_existing(system_prompt_in, old_text, media_kind)

        # Prepare inputs. This happens inside the worker, so multiple files can be prepared
        # and sent to LM Studio at once.
        if image_mode:
            imgs = [image_to_data_url(in_path, max_image_side=max_image_side)]
        else:
            imgs = extract_frames(in_path, num_frames, sampling, max_image_side=max_image_side)

        if not imgs:
            raise RuntimeError("No visual inputs found for captioning")

        caption = call_lmstudio_vision(imgs, sys_for_this, model, prefill=prefill, media_kind=media_kind, max_output_tokens=max_output_tokens)

        if os.path.exists(out_txt) and prepend_existing:
            try:
                with open(out_txt, "r", encoding="utf-8") as fh:
                    old = fh.read()
            except Exception:
                old = ""
            new_text = caption.strip() + ("\n\n" + old if old else "")
            with open(out_txt, "w", encoding="utf-8") as fh:
                fh.write(new_text)
        else:
            with open(out_txt, "w", encoding="utf-8") as fh:
                fh.write(caption.strip())

        return {"file": fn, "ok": True, "out": os.path.basename(out_txt)}
    except LMStudioRequestError as e:
        out = {"file": fn, "ok": False, "error": str(e), "server_error": True}
        if e.status_code is not None:
            out["status_code"] = e.status_code
        return out
    except requests.RequestException as e:
        return {"file": fn, "ok": False, "error": f"LM Studio request failed: {e}", "server_error": True}
    except Exception as e:
        return {"file": fn, "ok": False, "error": str(e), "server_error": False}


def _run_batch(job_id:str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        params = dict(job["params"])

    folder = params["target_folder"]
    image_mode = params["image_mode"]
    max_concurrent = params["max_concurrent"]
    abort_after_server_errors = params.get("abort_after_server_errors", LMSTUDIO_ABORT_AFTER_SERVER_ERRORS)
    targets = _select_targets(folder, image_mode)

    work_q = queue.Queue()
    for fn in targets:
        work_q.put(fn)

    batch_started_at = time.time()
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["total"] = len(targets)
        job["status"] = "running"
        job["started_at"] = batch_started_at
        job["finished_at"] = None
        job["active"] = {}
        job["current"] = None
        job["server_error_count"] = 0
        job["abort_reason"] = None

    def worker(worker_id:int):
        while True:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job or job.get("cancel", False):
                    return
            try:
                fn = work_q.get_nowait()
            except queue.Empty:
                return

            item_started_at = time.time()
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job or job.get("cancel", False):
                    work_q.task_done()
                    return
                active = job.setdefault("active", {})
                active[fn] = item_started_at
                job["current"] = ", ".join(active.keys()) or None

            res = _process_one_target(fn, params)
            item_finished_at = time.time()
            res["duration_sec"] = round(item_finished_at - item_started_at, 3)

            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    work_q.task_done()
                    return
                active = job.setdefault("active", {})
                active.pop(fn, None)
                job["current"] = ", ".join(active.keys()) or None
                job["results"].append(res)
                job["completed"] += 1
                job["last_result_at"] = item_finished_at

                if res.get("server_error"):
                    job["server_error_count"] = int(job.get("server_error_count", 0)) + 1
                    if abort_after_server_errors and job["server_error_count"] >= abort_after_server_errors:
                        job["cancel"] = True
                        job["status"] = "failing"
                        context_hint = ""
                        if "context size" in str(res.get("error", "")).lower():
                            context_hint = " Context was exceeded; use fewer parallel slots, raise LM Studio Context Length, or lower Max image side / Max output tokens."
                        job["abort_reason"] = (
                            f"Stopped after {job['server_error_count']} LM Studio/API errors."
                            f"{context_hint} "
                            "Already-written captions are left alone."
                        )
            work_q.task_done()
            time.sleep(0)

    worker_count = min(max_concurrent, len(targets)) if targets else 0
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(worker_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["finished_at"] = time.time()
        if job.get("abort_reason"):
            job["status"] = "failed"
        elif job.get("cancel", False):
            job["status"] = "cancelled"
        else:
            job["status"] = "done"
        job["active"] = {}
        job["current"] = None

@app.route("/api/batch-start", methods=["POST"])
def batch_start():
    data = request.get_json(force=True)
    folder = (data.get("target_folder") or "").strip()
    if not folder or not os.path.isdir(folder):
        return jsonify({"error":"Invalid target folder"}), 400

    image_mode = bool(data.get("image_mode", False))
    system_prompt_in = (data.get("system_prompt","") or "").strip()
    if not system_prompt_in:
        system_prompt_in = DEFAULT_PROMPT_IMAGE if image_mode else DEFAULT_PROMPT_VIDEO
    model = data.get("model", DEFAULT_MODEL)
    prefill = data.get("prefill","")
    num_frames = int(data.get("num_frames", 5))
    sampling = data.get("sampling_type", "uniform")
    overwrite = bool(data.get("overwrite", False))
    prepend_existing = bool(data.get("prepend_existing", False))
    use_existing = bool(data.get("use_existing_caption", False))
    max_image_side = _clamp_int(data.get("max_image_side", DEFAULT_MAX_IMAGE_SIDE), DEFAULT_MAX_IMAGE_SIDE, 0, 8192)
    max_output_tokens = _clamp_int(data.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS), DEFAULT_MAX_OUTPUT_TOKENS, 0, 8192)
    max_concurrent = _clamp_int(data.get("max_concurrent", DEFAULT_BATCH_CONCURRENCY), DEFAULT_BATCH_CONCURRENCY, 1, MAX_BATCH_CONCURRENCY)
    abort_after_server_errors = _clamp_int(data.get("abort_after_server_errors", LMSTUDIO_ABORT_AFTER_SERVER_ERRORS), LMSTUDIO_ABORT_AFTER_SERVER_ERRORS, 0, 999)

    job_id = uuid.uuid4().hex
    params = {
        "target_folder": folder,
        "image_mode": image_mode,
        "system_prompt": system_prompt_in,
        "model": model,
        "prefill": prefill,
        "num_frames": num_frames,
        "sampling_type": sampling,
        "overwrite": overwrite,
        "prepend_existing": prepend_existing,
        "use_existing_caption": use_existing,
        "max_image_side": max_image_side,
        "max_output_tokens": max_output_tokens,
        "max_concurrent": max_concurrent,
        "abort_after_server_errors": abort_after_server_errors
    }

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "created_at": time.time(),
            "params": params,
            "total": 0,
            "completed": 0,
            "current": None,
            "active": {},
            "results": [],
            "cancel": False,
            "started_at": None,
            "finished_at": None,
            "last_result_at": None,
            "server_error_count": 0,
            "abort_reason": None,
        }

    t = threading.Thread(target=_run_batch, args=(job_id,), daemon=True)
    t.start()

    total_guess = len(_select_targets(folder, image_mode))
    return jsonify({"job_id": job_id, "total": total_guess, "max_concurrent": max_concurrent, "max_image_side": max_image_side, "max_output_tokens": max_output_tokens})

@app.route("/api/batch-progress", methods=["GET"])
def batch_progress():
    job_id = request.args.get("job_id","").strip()
    if not job_id:
        return jsonify({"error":"Missing job_id"}), 400
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error":"Unknown job_id"}), 404
        now = time.time()
        status = job["status"]
        started_at = job.get("started_at") or job.get("created_at") or now
        finished_at = job.get("finished_at")
        elapsed_until = finished_at if (status in ("done", "cancelled") and finished_at) else now
        elapsed_sec = max(0.0, elapsed_until - started_at)
        completed = job["completed"]
        total = job["total"]
        remaining = max(0, total - completed)
        throughput_per_sec = (completed / elapsed_sec) if completed and elapsed_sec > 0 else 0.0
        eta_sec = (remaining / throughput_per_sec) if remaining and throughput_per_sec > 0 else None

        durations = [r.get("duration_sec") for r in job["results"] if isinstance(r.get("duration_sec"), (int, float))]
        avg_item_sec = (sum(durations) / len(durations)) if durations else None

        active_raw = job.get("active", {})
        if isinstance(active_raw, dict):
            active = [
                {"file": fn, "elapsed_sec": round(max(0.0, now - float(started)), 1)}
                for fn, started in active_raw.items()
            ]
        else:
            active = [{"file": fn, "elapsed_sec": None} for fn in active_raw]

        out = {
            "id": job["id"],
            "status": status,
            "total": total,
            "completed": completed,
            "current": job["current"],
            "results": list(job["results"]),
            "active": active,
            "active_count": len(active),
            "max_concurrent": job["params"].get("max_concurrent", DEFAULT_BATCH_CONCURRENCY),
            "elapsed_sec": round(elapsed_sec, 3),
            "eta_sec": round(eta_sec, 3) if eta_sec is not None else None,
            "throughput_per_min": round(throughput_per_sec * 60.0, 3),
            "avg_item_sec": round(avg_item_sec, 3) if avg_item_sec is not None else None,
            "server_error_count": job.get("server_error_count", 0),
            "abort_reason": job.get("abort_reason"),
        }
    return jsonify(out)

@app.route("/api/batch-cancel", methods=["POST"])
def batch_cancel():
    data = request.get_json(force=True)
    job_id = (data.get("job_id") or "").strip()
    if not job_id:
        return jsonify({"error":"Missing job_id"}), 400
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error":"Unknown job_id"}), 404
        job["cancel"] = True
    return jsonify({"ok": True})

# Back-compat one-shot
@app.route("/api/batch-caption", methods=["POST"])
def batch_caption_oneshot():
    # Start job using the same request body as /api/batch-start.
    resp = batch_start()
    if getattr(resp, "status_code", 200) != 200:
        return resp

    payload = resp.get_json(force=True)
    if not payload or "job_id" not in payload:
        return resp

    job_id = payload["job_id"]
    # Busy wait until done/cancelled. This route is kept for older callers;
    # the UI should use start/progress/cancel instead.
    while True:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                break
            if job["status"] in ("done", "cancelled"):
                active_raw = job.get("active", {})
                active = list(active_raw.keys()) if isinstance(active_raw, dict) else list(active_raw)
                results = {
                    "count": len(job["results"]),
                    "results": list(job["results"]),
                    "active": active,
                    "max_concurrent": job["params"].get("max_concurrent", DEFAULT_BATCH_CONCURRENCY),
                    "status": job["status"],
                    "total": job["total"],
                    "completed": job["completed"],
                    "elapsed_sec": round(((job.get("finished_at") or time.time()) - (job.get("started_at") or job.get("created_at") or time.time())), 3),
                    "abort_reason": job.get("abort_reason"),
                    "server_error_count": job.get("server_error_count", 0),
                }
                return jsonify(results)
        time.sleep(0.5)
    return jsonify({"error":"Job disappeared"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5057, debug=True)
