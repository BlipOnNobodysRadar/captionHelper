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
import subprocess
import tempfile
import hashlib
import sys
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
REGION_PREPROCESS_SCRIPT = _env_first("CAPTION_REGION_PREPROCESS_SCRIPT", default=os.path.join(os.path.dirname(__file__), "vision_preprocess.py"))
REGION_PREPROCESS_DETECTOR = _env_first("CAPTION_REGION_DETECTOR", default="groundingdino")
REGION_PREPROCESS_SEGMENTER = _env_first("CAPTION_REGION_SEGMENTER", default="none")
REGION_PREPROCESS_OCR = _env_first("CAPTION_REGION_OCR", default="paddleocr")
REGION_PREPROCESS_MAX_REGIONS = int(_env_first("CAPTION_REGION_MAX_REGIONS", default="12"))
REGION_PREPROCESS_OCR_THRESHOLD = float(_env_first("CAPTION_REGION_OCR_THRESHOLD", default="0.55"))
REGION_PREPROCESS_MODEL_ROOT = _env_first("CAPTION_REGION_MODEL_ROOT", default=os.path.join(os.path.expanduser("~"), ".cache", "captionhelper", "vision_models"))
REGION_PREPROCESS_AUTO_DOWNLOAD = str(_env_first("CAPTION_REGION_AUTO_DOWNLOAD", default="true")).strip().lower() in {"1", "true", "yes", "on"}
REGION_PREPROCESS_LOAD_MODELS = str(_env_first("CAPTION_REGION_LOAD_MODELS", default="true")).strip().lower() in {"1", "true", "yes", "on"}
REGION_PREPROCESS_DETECTOR_MODEL_PATH = _env_first("CAPTION_REGION_DETECTOR_MODEL_PATH", default="")
REGION_PREPROCESS_SEGMENTER_MODEL_PATH = _env_first("CAPTION_REGION_SEGMENTER_MODEL_PATH", default="")
REGION_PREPROCESS_OCR_MODEL_PATH = _env_first("CAPTION_REGION_OCR_MODEL_PATH", default="")
LLAMA_CPP_MODEL_MANAGEMENT = _env_first("CAPTION_LLAMA_CPP_MODEL_MANAGEMENT", default="off").strip().lower()
LLAMA_CPP_UNLOAD_DURING_PREPROCESS = str(_env_first("CAPTION_LLAMA_CPP_UNLOAD_DURING_PREPROCESS", default="false")).strip().lower() in {"1", "true", "yes", "on"}
LLAMA_CPP_MODEL_MANAGEMENT_BASE_URL = _env_first("CAPTION_LLAMA_CPP_MODEL_MANAGEMENT_BASE_URL", default="")
USER_PRESETS_PATH = _env_first("CAPTION_USER_PRESETS_PATH", default="user_presets.json")
USER_JOBS_PATH = _env_first("CAPTION_USER_JOBS_PATH", default=".caption_jobs")
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
    return _local_user_file_path(USER_PRESETS_PATH)


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


def _local_user_file_path(configured_path:str)->str:
    path = os.path.expanduser(configured_path)
    if not os.path.isabs(path):
        path = os.path.join(app.root_path, path)
    return path


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


def _sha256_file(path:str)->str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _region_candidates_for_prompt(region_payload:dict|None)->list[dict]:
    if not isinstance(region_payload, dict):
        return []
    candidates = []
    for item in list(region_payload.get("regions") or []) + list(region_payload.get("ocr") or []):
        if not isinstance(item, dict):
            continue
        normalized = item.get("bbox_ideogram_yxyx") or item.get("bbox")
        if not _valid_ideogram_bbox(normalized):
            continue
        candidate = {
            "id": item.get("id"),
            "type_hint": item.get("type_hint") or ("text" if item.get("text") else "obj"),
            "bbox": normalized,
            "confidence": item.get("confidence"),
            "source": item.get("source"),
        }
        if item.get("label"):
            candidate["label"] = item.get("label")
        if item.get("text"):
            candidate["text"] = item.get("text")
        candidates.append({k: v for k, v in candidate.items() if v not in (None, "")})
    return candidates


def _augment_prompt_with_region_candidates(user_prompt:str, region_payload:dict|None)->str:
    candidates = _region_candidates_for_prompt(region_payload)
    if not candidates:
        return user_prompt
    block = json.dumps(candidates, ensure_ascii=False, indent=2)
    instruction = (
        "REGION_CANDIDATES:\n"
        f"{block}\n\n"
        "Use REGION_CANDIDATES as provisional spatial hints. Prefer these boxes over inventing "
        "coordinates when they match visible elements. Merge duplicates, rename labels naturally, "
        "omit wrong or low-value candidates, and add missing important elements only when necessary. "
        "All final boxes must use Ideogram [y_min, x_min, y_max, x_max] format.\n\n"
    )
    return instruction + (user_prompt or "")


def _detector_tags_from_context(context:dict)->str:
    """Keep likely visual tags for detection while excluding artist/style/quality/rating metadata."""
    parts = [
        context.get("character_tags", ""),
        context.get("copyright_tags", ""),
        context.get("general_tags", ""),
        context.get("existing_caption", ""),
    ]
    return ", ".join(str(part).strip() for part in parts if str(part or "").strip())


def _run_region_preprocess(image_path:str, tags_text:str="", source_caption_path:str|None=None, params:dict|None=None)->dict|None:
    params = params or {}
    if not params.get("enable_region_preprocess"):
        return None
    detector = str(params.get("region_detector") or REGION_PREPROCESS_DETECTOR)
    segmenter = str(params.get("region_segmenter") or REGION_PREPROCESS_SEGMENTER)
    ocr = str(params.get("region_ocr") or REGION_PREPROCESS_OCR)
    max_regions = _clamp_int(params.get("region_max_regions", REGION_PREPROCESS_MAX_REGIONS), REGION_PREPROCESS_MAX_REGIONS, 0, 64)
    ocr_threshold = float(params.get("region_ocr_threshold", REGION_PREPROCESS_OCR_THRESHOLD))
    model_root = str(params.get("region_model_root") or REGION_PREPROCESS_MODEL_ROOT)
    auto_download = bool(params.get("region_auto_download", REGION_PREPROCESS_AUTO_DOWNLOAD))
    load_models = bool(params.get("region_load_models", REGION_PREPROCESS_LOAD_MODELS))
    detector_model_path = str(params.get("region_detector_model_path") or REGION_PREPROCESS_DETECTOR_MODEL_PATH)
    segmenter_model_path = str(params.get("region_segmenter_model_path") or REGION_PREPROCESS_SEGMENTER_MODEL_PATH)
    ocr_model_path = str(params.get("region_ocr_model_path") or REGION_PREPROCESS_OCR_MODEL_PATH)
    with tempfile.NamedTemporaryFile(prefix="caption_regions_", suffix=".json", delete=False) as fh:
        out_path = fh.name
    cmd = [
        sys.executable, REGION_PREPROCESS_SCRIPT,
        "--image", image_path,
        "--out", out_path,
        "--detector", detector,
        "--segmenter", segmenter,
        "--ocr", ocr,
        "--model-root", model_root,
        "--detector-model-path", detector_model_path,
        "--segmenter-model-path", segmenter_model_path,
        "--ocr-model-path", ocr_model_path,
        "--max-regions", str(max_regions),
        "--ocr-threshold", str(ocr_threshold),
    ]
    if not auto_download:
        cmd.append("--no-auto-download")
    if not load_models:
        cmd.append("--no-load-models")
    if tags_text:
        cmd.extend(["--tags-text", tags_text])
    if source_caption_path:
        cmd.extend(["--tags", source_caption_path])
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=600)
        if completed.returncode != 0:
            return {
                "regions": [],
                "ocr": [],
                "error": (completed.stderr or completed.stdout or "region preprocessor failed").strip(),
                "command": cmd,
            }
        with open(out_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        payload["command"] = cmd
        return payload
    finally:
        try:
            os.remove(out_path)
        except Exception:
            pass


def _valid_ideogram_bbox(value)->bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(v, int) and 0 <= v <= 1000 for v in value)
        and value[0] < value[2]
        and value[1] < value[3]
    )


def _region_preprocess_warnings(region_payload:dict|None)->list[str]:
    if not isinstance(region_payload, dict):
        return []
    warnings = [str(item) for item in (region_payload.get("warnings") or []) if str(item).strip()]
    if region_payload.get("error"):
        warnings.insert(0, str(region_payload["error"]))
    return warnings


def _region_preprocess_summary(region_payload:dict|None)->dict:
    warnings = _region_preprocess_warnings(region_payload)
    if not isinstance(region_payload, dict):
        return {"enabled": False, "warnings": warnings, "skipped": False}
    candidates = len(region_payload.get("regions") or []) + len(region_payload.get("ocr") or [])
    selected = [
        region_payload.get("detector"),
        region_payload.get("segmenter"),
        region_payload.get("ocr_engine"),
    ]
    selected = [item for item in selected if item and item != "none"]
    model_load_status = region_payload.get("model_load_status") or {}
    failed_loads = [
        name
        for name, status in model_load_status.items()
        if isinstance(status, dict) and status.get("loaded") is False
    ]
    skipped = bool(selected) and candidates == 0 and bool(warnings or failed_loads)
    return {
        "enabled": True,
        "skipped": skipped,
        "candidate_count": candidates,
        "warnings": warnings,
        "failed_model_loads": failed_loads,
    }


def _llamacpp_management_root_url()->str:
    if LLAMA_CPP_MODEL_MANAGEMENT_BASE_URL:
        return LLAMA_CPP_MODEL_MANAGEMENT_BASE_URL.rstrip("/")
    if API_BASE_URL.endswith("/v1"):
        return API_BASE_URL[:-3].rstrip("/")
    return API_BASE_URL.rstrip("/")


def _llamacpp_router_model_action(action:str, model:str)->dict:
    if BACKEND != "llamacpp":
        return {"ok": False, "skipped": True, "reason": "backend is not llama.cpp"}
    if LLAMA_CPP_MODEL_MANAGEMENT not in {"router", "llamacpp-router"}:
        return {"ok": False, "skipped": True, "reason": "llama.cpp router model management is disabled"}
    if not model:
        return {"ok": False, "skipped": True, "reason": "no model configured"}
    url = f"{_llamacpp_management_root_url()}/models/{action}"
    try:
        response = requests.post(url, json={"model": model}, timeout=120)
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc), "url": url, "model": model, "action": action}
    if not response.ok:
        return {
            "ok": False,
            "status_code": response.status_code,
            "error": _api_error_detail(response),
            "url": url,
            "model": model,
            "action": action,
        }
    payload = None
    try:
        payload = response.json()
    except Exception:
        payload = response.text
    return {"ok": True, "url": url, "model": model, "action": action, "response": payload}


def _maybe_unload_llamacpp_for_preprocess(model:str, enabled:bool)->dict|None:
    if not enabled or not LLAMA_CPP_UNLOAD_DURING_PREPROCESS:
        return None
    return _llamacpp_router_model_action("unload", model or DEFAULT_MODEL)


def _maybe_reload_llamacpp_after_preprocess(model:str, unload_result:dict|None)->dict|None:
    if not unload_result:
        return None
    if unload_result.get("skipped"):
        return {"ok": False, "skipped": True, "reason": unload_result.get("reason")}
    # Reload even if unload failed: router mode will no-op/load as needed, and this
    # gives Gemma the best chance of being resident before the caption request.
    return _llamacpp_router_model_action("load", model or DEFAULT_MODEL)

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


_HEX_COLOR_RE = re.compile(r"^#[0-9A-F]{6}$")


def _walk_json(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def validate_ideogram4_json_caption(caption:str)->dict:
    errors = []
    raw = (caption or "").strip()
    if not (raw.startswith("{") and raw.endswith("}")):
        errors.append("Output must contain raw JSON only, starting with { and ending with }.")
    if re.search(r"```|\\bHere is\\b|\\bJSON\\b", raw):
        errors.append("Output contains trailing commentary or Markdown markers.")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"valid": False, "errors": errors + [f"JSON parse error: {exc}"], "data": None}
    if not isinstance(data, dict):
        errors.append("Top-level value must be a JSON object.")
        return {"valid": False, "errors": errors, "data": data}
    expected_top = ["high_level_description", "style_description", "compositional_deconstruction"]
    if list(data.keys()) != expected_top:
        errors.append(f"Top-level keys must be exactly in this order: {expected_top}.")
    style = data.get("style_description")
    if not isinstance(style, dict):
        errors.append("style_description must be an object.")
    else:
        has_photo = "photo" in style
        has_art_style = "art_style" in style
        if has_photo == has_art_style:
            errors.append("style_description must include exactly one of photo or art_style.")
        medium = style.get("medium")
        allowed_mediums = {"photograph", "illustration", "painting", "3d_render", "graphic_design", "mixed_media", "screenshot"}
        if not isinstance(medium, str) or not medium or (medium not in allowed_mediums and len(medium) > 40):
            errors.append("style_description.medium is missing or invalid.")
        expected_style = ["aesthetics", "lighting", "photo", "medium", "color_palette"] if has_photo else ["aesthetics", "lighting", "medium", "art_style", "color_palette"]
        if list(style.keys()) != expected_style:
            errors.append(f"style_description keys must be in this order: {expected_style}.")
        palette = style.get("color_palette")
        if not isinstance(palette, list) or not all(isinstance(c, str) and _HEX_COLOR_RE.match(c) for c in palette):
            errors.append("color_palette must contain uppercase #RRGGBB hex colors.")
    if re.search(r"\\bscore_\\d+\\b", raw, re.I):
        errors.append("Raw score_x tags are not allowed.")
    for obj in _walk_json(data):
        bbox = obj.get("bbox")
        if bbox is not None and not _valid_ideogram_bbox(bbox):
            errors.append(f"Invalid bbox: {bbox}")
        if obj.get("type") == "text" or obj.get("type_hint") == "text":
            if "text" not in obj or not str(obj.get("text") or "").strip():
                errors.append("Text elements must include non-empty text.")
        for key, value in obj.items():
            if key.endswith("color") and isinstance(value, str) and value.startswith("#") and not _HEX_COLOR_RE.match(value):
                errors.append(f"Invalid hex color for {key}: {value}")
    return {"valid": not errors, "errors": errors, "data": data}


def _caption_with_validation(imgs, system_prompt, model, prefill, media_kind, max_output_tokens, user_prompt, validate_json=False):
    caption = call_vision_api(imgs, system_prompt, model, prefill=prefill, media_kind=media_kind, max_output_tokens=max_output_tokens, user_prompt=user_prompt)
    validation = validate_ideogram4_json_caption(caption) if validate_json else {"valid": True, "errors": []}
    retried = False
    if validate_json and not validation["valid"]:
        retry_prompt = (
            user_prompt.rstrip()
            + "\n\nThe previous response failed validation:\n"
            + "\n".join(f"- {err}" for err in validation["errors"])
            + "\n\nReturn corrected Ideogram 4 JSON only. Do not include commentary."
        )
        retried = True
        caption = call_vision_api(imgs, system_prompt, model, prefill=prefill, media_kind=media_kind, max_output_tokens=max_output_tokens, user_prompt=retry_prompt)
        validation = validate_ideogram4_json_caption(caption)
    return caption, {"validation": validation, "retried": retried}

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
        "llama_cpp_model_management": {
            "mode": LLAMA_CPP_MODEL_MANAGEMENT,
            "unload_during_preprocess": LLAMA_CPP_UNLOAD_DURING_PREPROCESS,
            "base_url": _llamacpp_management_root_url(),
        },
        "region_preprocess": {
            "detector": REGION_PREPROCESS_DETECTOR,
            "segmenter": REGION_PREPROCESS_SEGMENTER,
            "ocr": REGION_PREPROCESS_OCR,
            "max_regions": REGION_PREPROCESS_MAX_REGIONS,
            "ocr_threshold": REGION_PREPROCESS_OCR_THRESHOLD,
            "model_root": REGION_PREPROCESS_MODEL_ROOT,
            "auto_download": REGION_PREPROCESS_AUTO_DOWNLOAD,
            "load_models": REGION_PREPROCESS_LOAD_MODELS,
            "detector_model_path": REGION_PREPROCESS_DETECTOR_MODEL_PATH,
            "segmenter_model_path": REGION_PREPROCESS_SEGMENTER_MODEL_PATH,
            "ocr_model_path": REGION_PREPROCESS_OCR_MODEL_PATH,
        },
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
    enable_region_preprocess = request.form.get("enable_region_preprocess","false").lower() in ("1","true","yes","on")
    validate_ideogram_json = request.form.get("validate_ideogram_json","false").lower() in ("1","true","yes","on") or enable_region_preprocess

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
        region_payload = None
        model_management = None
        if image_mode and enable_region_preprocess:
            unload_result = _maybe_unload_llamacpp_for_preprocess(model, enabled=True)
            region_payload = _run_region_preprocess(
                upload_path,
                tags_text=_detector_tags_from_context(context),
                params={
                    "enable_region_preprocess": True,
                    "region_detector": request.form.get("region_detector") or REGION_PREPROCESS_DETECTOR,
                    "region_segmenter": request.form.get("region_segmenter") or REGION_PREPROCESS_SEGMENTER,
                    "region_ocr": request.form.get("region_ocr") or REGION_PREPROCESS_OCR,
                    "region_max_regions": request.form.get("region_max_regions") or REGION_PREPROCESS_MAX_REGIONS,
                    "region_ocr_threshold": request.form.get("region_ocr_threshold") or REGION_PREPROCESS_OCR_THRESHOLD,
                    "region_model_root": request.form.get("region_model_root") or REGION_PREPROCESS_MODEL_ROOT,
                    "region_auto_download": request.form.get("region_auto_download","true").lower() in ("1","true","yes","on"),
                    "region_load_models": request.form.get("region_load_models","true").lower() in ("1","true","yes","on"),
                    "region_detector_model_path": request.form.get("region_detector_model_path") or REGION_PREPROCESS_DETECTOR_MODEL_PATH,
                    "region_segmenter_model_path": request.form.get("region_segmenter_model_path") or REGION_PREPROCESS_SEGMENTER_MODEL_PATH,
                    "region_ocr_model_path": request.form.get("region_ocr_model_path") or REGION_PREPROCESS_OCR_MODEL_PATH,
                },
            )
            reload_result = _maybe_reload_llamacpp_after_preprocess(model, unload_result)
            model_management = {"unload_before_preprocess": unload_result, "reload_after_preprocess": reload_result}
            user_prompt = _augment_prompt_with_region_candidates(user_prompt, region_payload)
        caption, caption_meta = _caption_with_validation(imgs, system_prompt_in, model, prefill, media_kind, max_output_tokens, user_prompt, validate_json=validate_ideogram_json)
        preprocess_summary = _region_preprocess_summary(region_payload) if enable_region_preprocess else {"enabled": False, "warnings": [], "skipped": False}
        return jsonify({"caption": caption, "frames_used": len(imgs), "region_preprocess": region_payload, "region_preprocess_summary": preprocess_summary, "model_management": model_management, **caption_meta})
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


def _user_jobs_dir_path()->str:
    return _local_user_file_path(USER_JOBS_PATH)


def _job_record_path(job_id:str)->str:
    return os.path.join(_user_jobs_dir_path(), f"{job_id}.json")


def _public_job_record(job:dict)->dict:
    """Return the local, user-private job state safe to persist for resuming."""
    return {
        "id": job.get("id"),
        "status": job.get("status"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "params": job.get("params", {}),
        "total": job.get("total", 0),
        "completed": job.get("completed", 0),
        "results": job.get("results", []),
        "server_error_count": job.get("server_error_count", 0),
        "abort_reason": job.get("abort_reason"),
        "selected_targets": job.get("selected_targets", []),
        "resume_of": job.get("resume_of"),
    }


def _persist_job(job:dict)->None:
    """Persist job state in a local ignored folder so failed batches can resume."""
    os.makedirs(_user_jobs_dir_path(), exist_ok=True)
    record = _public_job_record(job)
    path = _job_record_path(str(record["id"]))
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp_path, path)


def _load_job_record(job_id:str)->dict|None:
    path = _job_record_path(job_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        record = json.load(fh)
    return record if isinstance(record, dict) else None


def _resume_targets_from_record(record:dict)->list[str]:
    """Only retry files that did not produce a successful or skipped result."""
    selected = list(record.get("selected_targets") or [])
    if not selected:
        params = record.get("params") or {}
        folder = params.get("target_folder")
        if folder and os.path.isdir(folder):
            selected = _select_targets(folder, bool(params.get("image_mode")))

    terminal = {
        r.get("file")
        for r in record.get("results", [])
        if isinstance(r, dict) and (r.get("ok") or r.get("skipped"))
    }
    return [fn for fn in selected if fn not in terminal]


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


def _metadata_path_for_caption(caption_path:str)->str:
    return os.path.splitext(caption_path)[0] + ".caption_meta.json"


def _write_caption_metadata(path:str, payload:dict)->None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


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
    enable_region_preprocess = bool(params.get("enable_region_preprocess", False))
    validate_ideogram_json = bool(params.get("validate_ideogram_json", False)) or enable_region_preprocess

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
        region_payload = None
        model_management = None
        if image_mode and enable_region_preprocess:
            unload_result = _maybe_unload_llamacpp_for_preprocess(model, enabled=True)
            region_payload = _run_region_preprocess(in_path, tags_text=_detector_tags_from_context(context), source_caption_path=None, params=params)
            reload_result = _maybe_reload_llamacpp_after_preprocess(model, unload_result)
            model_management = {"unload_before_preprocess": unload_result, "reload_after_preprocess": reload_result}
            user_prompt = _augment_prompt_with_region_candidates(user_prompt, region_payload)

        caption, caption_meta = _caption_with_validation(imgs, system_prompt_in, model, prefill, media_kind, max_output_tokens, user_prompt, validate_json=validate_ideogram_json)
        validation = caption_meta.get("validation") or {}
        preprocess_summary = _region_preprocess_summary(region_payload) if enable_region_preprocess else {"enabled": False, "warnings": [], "skipped": False}

        os.makedirs(output_paths["dir"], exist_ok=True)
        _copy_media_if_needed(in_path, output_paths, overwrite)

        metadata_payload = {
            "image_path": in_path,
            "image_hash": _sha256_file(in_path) if image_mode else None,
            "source_caption_path": source_txt if os.path.exists(source_txt) else None,
            "final_caption_path": out_txt,
            "prompt_version": "captionhelper-region-proposal-v1",
            "model_backend": BACKEND_DISPLAY_NAME,
            "model": model or DEFAULT_MODEL,
            "llama_cpp_settings": {
                "api_base_url": API_BASE_URL,
                "max_image_side": max_image_side,
                "max_output_tokens": max_output_tokens,
                "temperature": 0.2,
            },
            "region_proposal": region_payload,
            "region_preprocess_summary": preprocess_summary,
            "llama_cpp_model_management": model_management,
            "region_candidates_used_in_prompt": _region_candidates_for_prompt(region_payload),
            "final_validation_result": validation,
            "validation_retried": bool(caption_meta.get("retried")),
            "manual_reviewed": False,
        }
        metadata_payload = {k: v for k, v in metadata_payload.items() if v is not None}
        meta_path = _metadata_path_for_caption(out_txt)
        if validate_ideogram_json and not validation.get("valid"):
            _write_caption_metadata(meta_path, metadata_payload)
            return {"file": fn, "ok": False, "error": "Caption failed Ideogram JSON validation after retry", "validation_errors": validation.get("errors", []), "metadata_out": os.path.relpath(meta_path, folder), "region_preprocess_summary": preprocess_summary}

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
        _write_caption_metadata(meta_path, metadata_payload)

        return {
            "file": fn,
            "ok": True,
            "out": os.path.relpath(out_txt, folder),
            "media_out": os.path.relpath(output_paths["media"], folder) if output_paths.get("needs_media_copy") else None,
            "metadata_out": os.path.relpath(meta_path, folder),
            "region_preprocess_summary": preprocess_summary,
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
    targets = list(params["resume_targets"] if "resume_targets" in params else _select_targets(folder, image_mode))

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
        job["selected_targets"] = targets
        _persist_job(job)

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
                _persist_job(job)
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
        _persist_job(job)

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
    enable_region_preprocess = bool(data.get("enable_region_preprocess", False))
    validate_ideogram_json = bool(data.get("validate_ideogram_json", False)) or enable_region_preprocess
    region_auto_download = bool(data.get("region_auto_download", REGION_PREPROCESS_AUTO_DOWNLOAD))
    region_load_models = bool(data.get("region_load_models", REGION_PREPROCESS_LOAD_MODELS))

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
        "abort_after_server_errors": abort_after_server_errors,
        "enable_region_preprocess": enable_region_preprocess,
        "validate_ideogram_json": validate_ideogram_json,
        "region_detector": data.get("region_detector", REGION_PREPROCESS_DETECTOR),
        "region_segmenter": data.get("region_segmenter", REGION_PREPROCESS_SEGMENTER),
        "region_ocr": data.get("region_ocr", REGION_PREPROCESS_OCR),
        "region_max_regions": _clamp_int(data.get("region_max_regions", REGION_PREPROCESS_MAX_REGIONS), REGION_PREPROCESS_MAX_REGIONS, 0, 64),
        "region_ocr_threshold": float(data.get("region_ocr_threshold", REGION_PREPROCESS_OCR_THRESHOLD)),
        "region_model_root": data.get("region_model_root") or REGION_PREPROCESS_MODEL_ROOT,
        "region_auto_download": region_auto_download,
        "region_load_models": region_load_models,
        "region_detector_model_path": data.get("region_detector_model_path") or REGION_PREPROCESS_DETECTOR_MODEL_PATH,
        "region_segmenter_model_path": data.get("region_segmenter_model_path") or REGION_PREPROCESS_SEGMENTER_MODEL_PATH,
        "region_ocr_model_path": data.get("region_ocr_model_path") or REGION_PREPROCESS_OCR_MODEL_PATH,
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
            "selected_targets": [],
        }
        _persist_job(JOBS[job_id])

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


@app.route("/api/batch-resume", methods=["POST"])
def batch_resume():
    data = request.get_json(force=True) or {}
    source_job_id = (data.get("job_id") or "").strip()
    if not source_job_id:
        return jsonify({"error":"Missing job_id"}), 400

    with JOBS_LOCK:
        source_job = JOBS.get(source_job_id)
        source_record = _public_job_record(source_job) if source_job else None
    if source_record is None:
        source_record = _load_job_record(source_job_id)
    if not source_record:
        return jsonify({"error":"Unknown job_id"}), 404
    if source_record.get("status") not in {"failed", "cancelled", "done"}:
        return jsonify({"error":"Only completed, cancelled, or failed batches can be resumed."}), 400

    params = dict(source_record.get("params") or {})
    folder = (params.get("target_folder") or "").strip()
    if not folder or not os.path.isdir(folder):
        return jsonify({"error":"Original target folder is no longer available."}), 400

    resume_targets = _resume_targets_from_record(source_record)
    params["resume_targets"] = resume_targets
    # A resume should not fail immediately because the previous run already hit errors.
    params["abort_after_server_errors"] = _clamp_int(
        params.get("abort_after_server_errors", API_ABORT_AFTER_SERVER_ERRORS),
        API_ABORT_AFTER_SERVER_ERRORS,
        0,
        999,
    )

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "created_at": time.time(),
            "params": params,
            "total": len(resume_targets),
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
            "selected_targets": resume_targets,
            "resume_of": source_job_id,
        }
        _persist_job(JOBS[job_id])

    t = threading.Thread(target=_run_batch, args=(job_id,), daemon=True)
    t.start()

    return jsonify({
        "job_id": job_id,
        "resumed_from": source_job_id,
        "total": len(resume_targets),
        "max_concurrent": params.get("max_concurrent", DEFAULT_BATCH_CONCURRENCY),
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
        elapsed_until = finished_at if (status in ("done", "cancelled", "failed") and finished_at) else now
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
            if job["status"] in ("done", "cancelled", "failed"):
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
