import os
import base64
import io
import json
import time
import uuid
import threading
import queue
import random
import shutil
import re
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
import requests
from werkzeug.exceptions import RequestEntityTooLarge
from PIL import Image
from presets import CAPTION_PRESETS, DEFAULT_IMAGE_SYSTEM_PROMPT, DEFAULT_VIDEO_SYSTEM_PROMPT

# ------------------ Config ------------------
def _env_first(*names, default=None):
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip() != "":
            return value
    return default


def _normalize_backend(value:str)->str:
    normalized = (value or "llamacpp").strip().lower().replace("_", "-")
    aliases = {
        "llama": "llamacpp",
        "llama-cpp": "llamacpp",
        "llama.cpp": "llamacpp",
        "llamacpp": "llamacpp",
        "lamma": "llamacpp",
        "lamma-cpp": "llamacpp",
        "lamma.cpp": "llamacpp",
        "lm-studio": "lmstudio",
        "lmstudio": "lmstudio",
        "openai-compatible": "openai",
        "openai": "openai",
    }
    return aliases.get(normalized, normalized)


def _default_backend()->str:
    configured = _env_first("CAPTION_BACKEND", "LLM_BACKEND", "CAPTIONHELPER_BACKEND")
    if configured:
        return configured
    if _env_first("LMSTUDIO_BASE_URL", "LMSTUDIO_MODEL"):
        return "lmstudio"
    if _env_first("LLAMA_CPP_BASE_URL", "LLAMA_CPP_MODEL", "LAMMA_CPP_BASE_URL", "LAMMA_CPP_MODEL"):
        return "llamacpp"
    return "llamacpp"


BACKEND = _normalize_backend(_default_backend())
_BACKEND_DEFAULT_BASE_URLS = {
    "llamacpp": "http://localhost:8080/v1",
    "lmstudio": "http://localhost:1234/v1",
    "openai": "http://localhost:8080/v1",
}
BACKEND_DISPLAY_NAMES = {
    "llamacpp": "llama.cpp",
    "lmstudio": "LM Studio",
    "openai": "OpenAI-compatible",
}
API_BASE_URL = _env_first(
    "CAPTION_API_BASE_URL",
    "LLAMA_CPP_BASE_URL",
    "LAMMA_CPP_BASE_URL",
    "LMSTUDIO_BASE_URL",
    default=_BACKEND_DEFAULT_BASE_URLS.get(BACKEND, _BACKEND_DEFAULT_BASE_URLS["openai"]),
).rstrip("/")
DEFAULT_MODEL = _env_first(
    "CAPTION_MODEL",
    "LLAMA_CPP_MODEL",
    "LAMMA_CPP_MODEL",
    "LMSTUDIO_MODEL",
    default="qwen2.5-vl-32b-instruct",
)
DEFAULT_BATCH_CONCURRENCY = int(_env_first("CAPTION_BATCH_CONCURRENCY", "LLAMA_CPP_BATCH_CONCURRENCY", "LAMMA_CPP_BATCH_CONCURRENCY", "LMSTUDIO_BATCH_CONCURRENCY", default="4"))
MAX_BATCH_CONCURRENCY = int(_env_first("CAPTION_MAX_BATCH_CONCURRENCY", "LLAMA_CPP_MAX_BATCH_CONCURRENCY", "LAMMA_CPP_MAX_BATCH_CONCURRENCY", "LMSTUDIO_MAX_BATCH_CONCURRENCY", default="16"))
API_REQUEST_RETRIES = int(_env_first("CAPTION_REQUEST_RETRIES", "LLAMA_CPP_REQUEST_RETRIES", "LAMMA_CPP_REQUEST_RETRIES", "LMSTUDIO_REQUEST_RETRIES", default="2"))
API_RETRY_BACKOFF_SEC = float(_env_first("CAPTION_RETRY_BACKOFF_SEC", "LLAMA_CPP_RETRY_BACKOFF_SEC", "LAMMA_CPP_RETRY_BACKOFF_SEC", "LMSTUDIO_RETRY_BACKOFF_SEC", default="2"))
API_ABORT_AFTER_SERVER_ERRORS = int(_env_first("CAPTION_ABORT_AFTER_SERVER_ERRORS", "LLAMA_CPP_ABORT_AFTER_SERVER_ERRORS", "LAMMA_CPP_ABORT_AFTER_SERVER_ERRORS", "LMSTUDIO_ABORT_AFTER_SERVER_ERRORS", default="3"))
DEFAULT_MAX_IMAGE_SIDE = int(_env_first("CAPTION_MAX_IMAGE_SIDE", "LLAMA_CPP_MAX_IMAGE_SIDE", "LAMMA_CPP_MAX_IMAGE_SIDE", "LMSTUDIO_MAX_IMAGE_SIDE", default="1024"))
DEFAULT_MAX_OUTPUT_TOKENS = int(_env_first("CAPTION_MAX_OUTPUT_TOKENS", "LLAMA_CPP_MAX_OUTPUT_TOKENS", "LAMMA_CPP_MAX_OUTPUT_TOKENS", "LMSTUDIO_MAX_OUTPUT_TOKENS", default="512"))
USER_PRESETS_PATH = _env_first("CAPTION_USER_PRESETS_PATH", default="user_presets.json")
APP_HOST = _env_first("CAPTION_HOST", default="127.0.0.1")
APP_PORT = int(_env_first("CAPTION_PORT", default="5057"))
APP_DEBUG = str(_env_first("CAPTION_DEBUG", default="false")).strip().lower() in {"1", "true", "yes", "on"}
MAX_UPLOAD_BYTES = int(_env_first("CAPTION_MAX_UPLOAD_BYTES", default=str(512 * 1024 * 1024)))
CAPTION_ALLOWED_HOSTS = _env_first("CAPTION_ALLOWED_HOSTS", default="localhost,127.0.0.1,::1")
BACKEND_DISPLAY_NAME = BACKEND_DISPLAY_NAMES.get(BACKEND, BACKEND_DISPLAY_NAMES["openai"])

# Backwards-compatible names used by older code paths and third-party snippets.
LMSTUDIO_BASE_URL = API_BASE_URL
LMSTUDIO_REQUEST_RETRIES = API_REQUEST_RETRIES
LMSTUDIO_RETRY_BACKOFF_SEC = API_RETRY_BACKOFF_SEC
LMSTUDIO_ABORT_AFTER_SERVER_ERRORS = API_ABORT_AFTER_SERVER_ERRORS

ALLOWED_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
ALLOWED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Default prompts (the UI may send its own, but when modes switch we ensure sane defaults)
DEFAULT_PROMPT_VIDEO = DEFAULT_VIDEO_SYSTEM_PROMPT
DEFAULT_PROMPT_IMAGE = DEFAULT_IMAGE_SYSTEM_PROMPT

# Serve the HTML template from the project root, but expose only the
# browser assets explicitly whitelisted below. Mapping /static to the entire
# project root would leak source files and local user_presets.json contents.
app = Flask(__name__, template_folder=".", static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_ALLOWED_STATIC_FILES = {"script.js", "style.css"}


def _configured_allowed_hosts()->set[str]:
    raw = str(CAPTION_ALLOWED_HOSTS or "").strip()
    if raw == "*":
        return {"*"}
    hosts = {item.strip().lower() for item in raw.split(",") if item.strip()}
    return hosts or {"localhost", "127.0.0.1", "::1"}


def _hostname_from_host_header(host_header:str|None)->str:
    host = (host_header or "").strip().lower()
    if host.startswith("[") and "]" in host:
        return host[1:host.index("]")]
    return host.split(":", 1)[0].strip("[]")


def _request_host_allowed(hostname:str|None)->bool:
    allowed = _configured_allowed_hosts()
    if "*" in allowed:
        return True
    host = _hostname_from_host_header(hostname)
    return host in allowed


def _same_origin_allowed(url:str|None)->bool:
    if not url:
        return True
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if not parsed.hostname:
        return True
    return _request_host_allowed(parsed.hostname)


@app.before_request
def enforce_local_request_boundaries():
    if not _request_host_allowed(request.host):
        abort(403)
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        if not _same_origin_allowed(request.headers.get("Origin")):
            abort(403)
        referer = request.headers.get("Referer")
        if referer and not _same_origin_allowed(referer):
            abort(403)


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'")
    return response


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_error):
    max_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
    return jsonify({"error": f"Uploaded file is too large. Limit is {max_mb:.0f} MB."}), 413

# ------------------ Helpers ------------------



def _copy_builtin_presets():
    presets = []
    for preset in CAPTION_PRESETS:
        item = dict(preset)
        item["source"] = "builtin"
        item["readonly"] = True
        presets.append(item)
    return presets


def _user_presets_file_path()->str:
    path = os.path.expanduser(USER_PRESETS_PATH)
    if not os.path.isabs(path):
        path = os.path.join(app.root_path, path)
    return path


def _slugify_preset_id(name:str)->str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "preset").strip().lower()).strip("-")
    return slug or "preset"


def _coerce_user_preset(raw:dict)->dict:
    if not isinstance(raw, dict):
        raise ValueError("Preset must be an object")
    name = str(raw.get("name") or "").strip()
    if not name:
        raise ValueError("Preset name is required")
    system_prompt = str(raw.get("system_prompt") or "").strip()
    user_template = str(raw.get("user_template") or "").strip()
    if not system_prompt:
        raise ValueError("System prompt is required")
    if not user_template:
        raise ValueError("User message template is required")

    requested_id = str(raw.get("id") or "").strip()
    if requested_id.startswith("user:"):
        base_id = requested_id.split(":", 1)[1]
    else:
        base_id = requested_id or _slugify_preset_id(name)
    base_id = _slugify_preset_id(base_id)
    preset_id = f"user:{base_id}"


    media = str(raw.get("media") or "image").strip().lower()
    if media not in {"image", "video"}:
        media = "image"

    try:
        max_output_tokens = int(raw.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS))
    except (TypeError, ValueError):
        max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
    max_output_tokens = max(0, min(8192, max_output_tokens))

    return {
        "id": preset_id,
        "name": name,
        "description": str(raw.get("description") or "User-saved preset.").strip(),
        "media": media,
        "system_prompt": system_prompt,
        "user_template": user_template,
        "prefill": str(raw.get("prefill") or ""),
        "max_output_tokens": max_output_tokens,
        "source": "user",
        "readonly": False,
    }


def load_user_presets()->list:
    path = _user_presets_file_path()
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("User presets file must contain a JSON list")
    presets = []
    seen = {p.get("id") for p in _copy_builtin_presets()}
    for raw in data:
        preset = _coerce_user_preset(raw)
        if preset["id"] in seen:
            preset["id"] = f"user:{_slugify_preset_id(preset['name'])}-{uuid.uuid4().hex[:8]}"
        seen.add(preset["id"])
        presets.append(preset)
    return presets


def save_user_presets(presets:list)->None:
    path = _user_presets_file_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cleaned = [_coerce_user_preset(preset) for preset in presets]
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(cleaned, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp_path, path)


def all_caption_presets()->list:
    return _copy_builtin_presets() + load_user_presets()

def _preset_by_id(preset_id:str):
    preset_id = (preset_id or "").strip()
    for preset in CAPTION_PRESETS:
        if preset.get("id") == preset_id:
            return preset
    return None


def _default_user_template(image_mode:bool, use_existing:bool=False)->str:
    preferred = "image_grounded_tags" if image_mode and use_existing else "image_basic" if image_mode else "video_grounded" if use_existing else "video_basic"
    preset = _preset_by_id(preferred)
    if preset:
        return preset.get("user_template", "")
    return "[image]\n\nCaption this image." if image_mode else "[image]\n\nCaption this clip."


def _parse_grouped_metadata(text:str)->dict:
    groups = {
        "character_tags": "",
        "copyright_tags": "",
        "artist_tags": "",
        "general_tags": "",
        "rating_tags": "",
        "quality_tags": "",
    }
    labels = {
        "CHARACTER": "character_tags",
        "CHARACTERS": "character_tags",
        "COPYRIGHT": "copyright_tags",
        "SERIES": "copyright_tags",
        "ARTIST": "artist_tags",
        "STYLE": "artist_tags",
        "GENERAL": "general_tags",
        "TAGS": "general_tags",
        "RATING": "rating_tags",
        "QUALITY": "quality_tags",
    }
    leftovers = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        label, sep, value = stripped.partition(":")
        key = labels.get(label.strip().upper()) if sep else None
        if key:
            groups[key] = (groups[key] + ", " + value.strip()).strip(", ") if groups[key] else value.strip()
        else:
            leftovers.append(line)
    groups["ungrouped_caption"] = "\n".join(leftovers).strip()
    return groups


def _context_from_request_values(values:dict, existing_caption:str="", image_mode:bool=False, input_count:int=1)->dict:
    parsed = _parse_grouped_metadata(existing_caption)
    context = {
        "media_kind": "image" if image_mode else "clip",
        "input_count": str(input_count),
        "visual_input_description": "one still image" if image_mode else f"{input_count} sampled video frame{'s' if input_count != 1 else ''}",
        "existing_caption": parsed.get("ungrouped_caption") or (existing_caption or "").strip(),
        "source_tags": (values.get("source_tags") or "").strip(),
        "character_tags": (values.get("character_tags") or parsed.get("character_tags") or "").strip(),
        "copyright_tags": (values.get("copyright_tags") or parsed.get("copyright_tags") or "").strip(),
        "artist_tags": (values.get("artist_tags") or parsed.get("artist_tags") or "").strip(),
        "general_tags": (values.get("general_tags") or parsed.get("general_tags") or "").strip(),
        "rating_tags": (values.get("rating_tags") or parsed.get("rating_tags") or "").strip(),
        "quality_tags": (values.get("quality_tags") or parsed.get("quality_tags") or "").strip(),
    }
    if not context["source_tags"]:
        source_parts = [context[k] for k in ("character_tags", "copyright_tags", "artist_tags", "general_tags", "rating_tags", "quality_tags") if context[k]]
        context["source_tags"] = ", ".join(source_parts)
    return context


def _render_user_template(template:str, context:dict)->str:
    template = (template or "").strip() or "[image]\n\nCaption this visual input."
    safe_context = {k: str(v or "") for k, v in context.items()}
    rendered_lines = []
    for line in template.splitlines():
        placeholders = [name for name in safe_context if "{" + name + "}" in line]
        try:
            rendered = line.format(**safe_context)
        except KeyError:
            rendered = line
        # Optional metadata lines disappear when all placeholders on that line are empty.
        if placeholders and not any(safe_context[name] for name in placeholders):
            continue
        rendered_lines.append(rendered.rstrip())
    rendered = "\n".join(rendered_lines).strip()
    while "\n\n\n" in rendered:
        rendered = rendered.replace("\n\n\n", "\n\n")
    return rendered


def _build_user_content(user_prompt:str, images_data_urls):
    parts = []
    prompt = (user_prompt or "").strip()
    marker = "[image]"
    if marker in prompt:
        before, after = prompt.split(marker, 1)
        if before.strip():
            parts.append({"type":"text", "text": before.strip()})
        for url in images_data_urls:
            parts.append({"type":"image_url", "image_url":{"url": url}})
        if after.strip():
            parts.append({"type":"text", "text": after.strip()})
    else:
        if prompt:
            parts.append({"type":"text", "text": prompt})
        for url in images_data_urls:
            parts.append({"type":"image_url", "image_url":{"url": url}})
    return parts

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
    """Downscale large images before sending them to a local vision API.

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
    import cv2

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

class VisionAPIRequestError(RuntimeError):
    def __init__(self, status_code=None, message="", body=""):
        self.status_code = status_code
        self.body = body
        super().__init__(message)


def _shorten(text:str, limit:int=700)->str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …"


def _api_error_detail(response):
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


def _is_retryable_api_status(status_code:int)->bool:
    # 400 is normally not retryable, but LM Studio/local llama.cpp servers can emit it
    # during overloaded multimodal/parallel batches. Retrying once keeps a brief
    # overload hiccup from ruining a whole batch, while still surfacing real bad payloads.
    return status_code in {400, 408, 409, 425, 429, 500, 502, 503, 504}


def call_vision_api(images_data_urls, system_prompt:str, model:str, prefill:str="", media_kind:str="clip", max_output_tokens:int=DEFAULT_MAX_OUTPUT_TOKENS, user_prompt:str=""):
    if not user_prompt:
        if media_kind == "image":
            user_prompt = "You are given a single still image. Write a descriptive caption.\n\n[image]"
        else:
            user_prompt = "You are given a few frames sampled from a short video clip. Write a descriptive caption for the clip as a whole.\n\n[image]"

    messages = [
        {"role":"system","content": (system_prompt or "").strip()},
        {"role":"user","content": _build_user_content(user_prompt, images_data_urls)}
    ]

    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }
    if BACKEND == "lmstudio":
        payload["add_generation_prompt"] = True

    try:
        max_output_tokens = int(max_output_tokens)
    except (TypeError, ValueError):
        max_output_tokens = DEFAULT_MAX_OUTPUT_TOKENS
    if max_output_tokens > 0:
        payload["max_tokens"] = max_output_tokens

    if prefill and prefill.strip():
        messages.append({"role":"assistant","content": prefill})
        if BACKEND == "lmstudio":
            payload["add_generation_prompt"] = False

    url = f"{API_BASE_URL}/chat/completions"
    attempts = max(1, API_REQUEST_RETRIES)
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            r = requests.post(url, json=payload, timeout=300)
        except requests.RequestException as e:
            last_error = VisionAPIRequestError(None, f"Vision API request failed: {e}", str(e))
            retryable = True
        else:
            if r.ok:
                data = r.json()
                content = data["choices"][0]["message"].get("content")
                caption = (content or "").strip()
                if caption:
                    return caption

                body = json.dumps(data, ensure_ascii=False)
                last_error = VisionAPIRequestError(
                    None,
                    "Backend returned an empty caption; refusing to write blank .txt",
                    body,
                )
                retryable = True
            else:
                detail = _api_error_detail(r)
                msg = f"{BACKEND_DISPLAY_NAME} HTTP {r.status_code}: {detail}"
                last_error = VisionAPIRequestError(r.status_code, msg, detail)
                detail_lower = detail.lower()
                if "context size" in detail_lower or "context length" in detail_lower or "context window" in detail_lower:
                    retryable = False
                else:
                    retryable = _is_retryable_api_status(r.status_code)

        if attempt < attempts and retryable:
            # Jitter avoids every worker retrying at the exact same moment.
            delay = API_RETRY_BACKOFF_SEC * attempt + random.uniform(0.0, 1.5)
            time.sleep(delay)
            continue
        raise last_error

    raise last_error or VisionAPIRequestError(None, "Vision API request failed for an unknown reason")

# ------------------ Simple chat route ------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/static/<path:filename>")
def static_assets(filename):
    if filename not in _ALLOWED_STATIC_FILES:
        abort(404)
    return send_from_directory(app.root_path, filename)


@app.route("/api/config", methods=["GET"])
def api_config():
    return jsonify({
        "backend": BACKEND,
        "backend_display_name": BACKEND_DISPLAY_NAME,
        "api_base_url": API_BASE_URL,
        "default_model": DEFAULT_MODEL,
        "default_batch_concurrency": DEFAULT_BATCH_CONCURRENCY,
        "max_batch_concurrency": MAX_BATCH_CONCURRENCY,
        "abort_after_server_errors": API_ABORT_AFTER_SERVER_ERRORS,
        "max_image_side": DEFAULT_MAX_IMAGE_SIDE,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "caption_presets": all_caption_presets(),
    })


@app.route("/api/user-presets", methods=["POST"])
def api_save_user_preset():
    data = request.get_json(force=True) or {}
    user_presets = load_user_presets()
    existing_user_ids = {preset["id"] for preset in user_presets}
    builtin_ids = {preset["id"] for preset in _copy_builtin_presets()}
    requested_id = str(data.get("id") or "").strip()

    if requested_id and requested_id in builtin_ids:
        return jsonify({"error":"Built-in presets cannot be overwritten. Save as a new user preset instead."}), 400

    try:
        preset = _coerce_user_preset(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if preset["id"] in builtin_ids:
        preset["id"] = f"user:{_slugify_preset_id(preset['name'])}"

    if requested_id and preset["id"] in existing_user_ids:
        user_presets = [preset if existing["id"] == preset["id"] else existing for existing in user_presets]
    else:
        all_ids = builtin_ids | existing_user_ids
        if preset["id"] in all_ids:
            base_id = preset["id"]
            n = 2
            while preset["id"] in all_ids:
                preset["id"] = f"{base_id}-{n}"
                n += 1
        user_presets.append(preset)

    save_user_presets(user_presets)
    return jsonify({"ok": True, "preset": preset, "caption_presets": all_caption_presets()})


@app.route("/api/user-presets/<path:preset_id>", methods=["DELETE"])
def api_delete_user_preset(preset_id):
    preset_id = (preset_id or "").strip()
    if not preset_id.startswith("user:"):
        return jsonify({"error":"Only user presets can be deleted."}), 400
    user_presets = load_user_presets()
    kept = [preset for preset in user_presets if preset.get("id") != preset_id]
    if len(kept) == len(user_presets):
        return jsonify({"error":"Unknown user preset."}), 404
    save_user_presets(kept)
    return jsonify({"ok": True, "caption_presets": all_caption_presets()})

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

    user_template = request.form.get("user_template", "").strip() or _default_user_template(image_mode)
    metadata_values = {
        "source_tags": request.form.get("source_tags", ""),
        "character_tags": request.form.get("character_tags", ""),
        "copyright_tags": request.form.get("copyright_tags", ""),
        "artist_tags": request.form.get("artist_tags", ""),
        "general_tags": request.form.get("general_tags", ""),
        "rating_tags": request.form.get("rating_tags", ""),
        "quality_tags": request.form.get("quality_tags", ""),
    }
    model = request.form.get("model", DEFAULT_MODEL)
    prefill = request.form.get("prefill","")
    max_image_side = _clamp_int(request.form.get("max_image_side", DEFAULT_MAX_IMAGE_SIDE), DEFAULT_MAX_IMAGE_SIDE, 0, 8192)
    max_output_tokens = _clamp_int(request.form.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS), DEFAULT_MAX_OUTPUT_TOKENS, 0, 8192)
    use_existing = request.form.get("use_existing_caption","false").lower() in ("1","true","yes","on")
    existing_caption_text = request.form.get("existing_caption","")

    media_kind = "image" if image_mode else "clip"

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

        existing_for_prompt = existing_caption_text if use_existing else ""
        context = _context_from_request_values(metadata_values, existing_for_prompt, image_mode=image_mode, input_count=len(imgs))
        user_prompt = _render_user_template(user_template, context)
        caption = call_vision_api(imgs, system_prompt_in, model, prefill=prefill, media_kind=media_kind, max_output_tokens=max_output_tokens, user_prompt=user_prompt)
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


def _safe_filename_fragment(value:str)->str:
    fragment = str(value or "").strip()
    if not fragment:
        return ""
    # Keep custom naming text in a single filename segment on every platform.
    return re.sub(r'[\\/:*?"<>|]+', "_", fragment)


def _safe_subdir_name(value:str)->str:
    name = _safe_filename_fragment(value) or "captionHelper_results"
    name = name.strip(". ") or "captionHelper_results"
    return name


def _build_output_paths(in_path:str, params:dict):
    folder = params["target_folder"]
    root, ext = os.path.splitext(os.path.basename(in_path))
    affix_text = _safe_filename_fragment(params.get("filename_affix_text", ""))
    affix_position = str(params.get("filename_affix_position") or "prefix").lower()

    if affix_text:
        out_root = f"{root}{affix_text}" if affix_position == "suffix" else f"{affix_text}{root}"
    else:
        out_root = root

    if params.get("output_to_subdir"):
        out_dir = os.path.join(folder, _safe_subdir_name(params.get("output_subdir_name", "")))
    else:
        out_dir = folder

    return {
        "dir": out_dir,
        "media": os.path.join(out_dir, out_root + ext),
        "caption": os.path.join(out_dir, out_root + ".txt"),
        "needs_media_copy": out_dir != folder or out_root != root,
    }


def _copy_media_if_needed(source_path:str, output_paths:dict, overwrite:bool):
    if not output_paths.get("needs_media_copy"):
        return
    dest_path = output_paths["media"]
    if os.path.abspath(source_path) == os.path.abspath(dest_path):
        return
    if os.path.exists(dest_path) and not overwrite:
        return
    shutil.copy2(source_path, dest_path)

def _write_caption_error_debug_file(output_paths:dict, error:VisionAPIRequestError):
    raw_body = (error.body or "").strip()
    if not raw_body:
        return None

    out_txt = output_paths["caption"]
    debug_path = os.path.splitext(out_txt)[0] + ".caption_error.json"
    os.makedirs(os.path.dirname(debug_path) or ".", exist_ok=True)

    debug_payload = {
        "error": str(error),
        "status_code": error.status_code,
    }
    try:
        debug_payload["raw_response"] = json.loads(raw_body)
    except json.JSONDecodeError:
        debug_payload["raw_response"] = raw_body

    with open(debug_path, "w", encoding="utf-8") as fh:
        json.dump(debug_payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return debug_path


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
    user_template = params.get("user_template") or _default_user_template(image_mode, params.get("use_existing_caption"))
    metadata_values = params.get("metadata_values", {})
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
    source_base, _ = os.path.splitext(in_path)
    source_txt = source_base + ".txt"
    output_paths = _build_output_paths(in_path, params)
    out_txt = output_paths["caption"]

    # Skip logic. Existing caption grounding implies we usually want to overwrite/prepend,
    # otherwise the safest behavior is still to leave existing caption files alone.
    if os.path.exists(out_txt) and (not overwrite) and (not prepend_existing):
        return {"file": fn, "skipped": True, "reason": "caption exists"}

    try:
        old_text = ""
        if use_existing:
            grounding_txt = source_txt if os.path.exists(source_txt) else out_txt
            if os.path.exists(grounding_txt):
                try:
                    with open(grounding_txt, "r", encoding="utf-8") as fh:
                        old_text = fh.read().strip()
                except Exception:
                    old_text = ""

        # Prepare inputs. This happens inside the worker, so multiple files can be prepared
        # and sent to the local vision API at once.
        if image_mode:
            imgs = [image_to_data_url(in_path, max_image_side=max_image_side)]
        else:
            imgs = extract_frames(in_path, num_frames, sampling, max_image_side=max_image_side)

        if not imgs:
            raise RuntimeError("No visual inputs found for captioning")

        context = _context_from_request_values(metadata_values, old_text, image_mode=image_mode, input_count=len(imgs))
        user_prompt = _render_user_template(user_template, context)
        caption = call_vision_api(imgs, system_prompt_in, model, prefill=prefill, media_kind=media_kind, max_output_tokens=max_output_tokens, user_prompt=user_prompt)

        os.makedirs(output_paths["dir"], exist_ok=True)
        _copy_media_if_needed(in_path, output_paths, overwrite)

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

        return {
            "file": fn,
            "ok": True,
            "out": os.path.relpath(out_txt, folder),
            "media_out": os.path.relpath(output_paths["media"], folder) if output_paths.get("needs_media_copy") else None,
        }
    except VisionAPIRequestError as e:
        out = {"file": fn, "ok": False, "error": str(e), "server_error": True}
        debug_path = _write_caption_error_debug_file(output_paths, e)
        if debug_path:
            out["debug_out"] = os.path.relpath(debug_path, folder)
        if e.status_code is not None:
            out["status_code"] = e.status_code
        return out
    except requests.RequestException as e:
        return {"file": fn, "ok": False, "error": f"Vision API request failed: {e}", "server_error": True}
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
    abort_after_server_errors = params.get("abort_after_server_errors", API_ABORT_AFTER_SERVER_ERRORS)
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
                            context_hint = " Context was exceeded; use fewer parallel slots, raise backend Context Length, or lower Max image side / Max output tokens."
                        job["abort_reason"] = (
                            f"Stopped after {job['server_error_count']} vision API errors."
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
    user_template = (data.get("user_template", "") or "").strip() or _default_user_template(image_mode, bool(data.get("use_existing_caption", False)))
    metadata_values = {
        "source_tags": data.get("source_tags", ""),
        "character_tags": data.get("character_tags", ""),
        "copyright_tags": data.get("copyright_tags", ""),
        "artist_tags": data.get("artist_tags", ""),
        "general_tags": data.get("general_tags", ""),
        "rating_tags": data.get("rating_tags", ""),
        "quality_tags": data.get("quality_tags", ""),
    }
    model = data.get("model", DEFAULT_MODEL)
    prefill = data.get("prefill","")
    num_frames = int(data.get("num_frames", 5))
    sampling = data.get("sampling_type", "uniform")
    overwrite = bool(data.get("overwrite", False))
    prepend_existing = bool(data.get("prepend_existing", False))
    use_existing = bool(data.get("use_existing_caption", False))
    filename_affix_text = _safe_filename_fragment(data.get("filename_affix_text", ""))
    filename_affix_position = str(data.get("filename_affix_position") or "prefix").lower()
    if filename_affix_position not in ("prefix", "suffix"):
        filename_affix_position = "prefix"
    output_to_subdir = bool(data.get("output_to_subdir", False))
    output_subdir_name = _safe_subdir_name(data.get("output_subdir_name", ""))
    max_image_side = _clamp_int(data.get("max_image_side", DEFAULT_MAX_IMAGE_SIDE), DEFAULT_MAX_IMAGE_SIDE, 0, 8192)
    max_output_tokens = _clamp_int(data.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS), DEFAULT_MAX_OUTPUT_TOKENS, 0, 8192)
    max_concurrent = _clamp_int(data.get("max_concurrent", DEFAULT_BATCH_CONCURRENCY), DEFAULT_BATCH_CONCURRENCY, 1, MAX_BATCH_CONCURRENCY)
    abort_after_server_errors = _clamp_int(data.get("abort_after_server_errors", API_ABORT_AFTER_SERVER_ERRORS), API_ABORT_AFTER_SERVER_ERRORS, 0, 999)

    job_id = uuid.uuid4().hex
    params = {
        "target_folder": folder,
        "image_mode": image_mode,
        "system_prompt": system_prompt_in,
        "user_template": user_template,
        "metadata_values": metadata_values,
        "model": model,
        "prefill": prefill,
        "num_frames": num_frames,
        "sampling_type": sampling,
        "overwrite": overwrite,
        "prepend_existing": prepend_existing,
        "use_existing_caption": use_existing,
        "filename_affix_text": filename_affix_text,
        "filename_affix_position": filename_affix_position,
        "output_to_subdir": output_to_subdir,
        "output_subdir_name": output_subdir_name,
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
    return jsonify({
        "job_id": job_id,
        "total": total_guess,
        "max_concurrent": max_concurrent,
        "max_image_side": max_image_side,
        "max_output_tokens": max_output_tokens,
        "output_to_subdir": output_to_subdir,
        "output_subdir_name": output_subdir_name,
        "filename_affix_text": filename_affix_text,
        "filename_affix_position": filename_affix_position,
    })

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
    app.run(host=APP_HOST, port=APP_PORT, debug=APP_DEBUG)
