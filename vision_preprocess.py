#!/usr/bin/env python3
"""Optional region-proposal preprocessor for CaptionHelper.

The script intentionally keeps heavy vision dependencies optional. When a selected
backend is not installed it returns a valid JSON payload with a warning instead
of failing the whole captioning run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any

from PIL import Image

FALLBACK_VOCABULARY = [
    "person", "woman", "man", "face", "hair", "hand", "eye", "animal", "bird",
    "cat", "dog", "weapon", "sword", "gun", "bow", "arrow", "book", "flower",
    "rose", "dress", "armor", "hat", "crown", "jewelry", "chair", "table",
    "vehicle", "building", "window", "text", "logo", "signature", "watermark",
]
META_TAG_RE = re.compile(r"^(score_\d+|rating[:_].+|best_quality|highres|absurdres|masterpiece|safe|questionable|explicit)$", re.I)
LOW_VALUE_DETECTOR_TAG_RE = re.compile(r"^(source |style |artist |game model|looking at |solo$|1girl$|1boy$|2girls$|2boys$)", re.I)


def write_progress(path: str, *, stage: str, message: str, percent: float | None = None) -> None:
    if not path:
        return
    payload = {"stage": stage, "message": message, "percent": percent, "updated_at": time.time()}
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        pass
DEFAULT_MODEL_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "captionhelper", "vision_models")
MODEL_SPECS = {
    "detector": {
        "groundingdino": {
            "repo_id": "IDEA-Research/grounding-dino-tiny",
            "local_name": "groundingdino",
            "description": "GroundingDINO object detector",
        },
        "groundingdino1.5": {
            "repo_id": "IDEA-Research/grounding-dino-base",
            "local_name": "groundingdino1.5",
            "description": "GroundingDINO 1.5/base object detector",
        },
        "florence2": {
            "repo_id": "microsoft/Florence-2-large",
            "local_name": "florence2-large",
            "description": "Florence-2 large prototype detector/region captioner",
        },
    },
    "segmenter": {
        "sam2": {
            "repo_id": "facebook/sam2.1-hiera-large",
            "local_name": "sam2.1-hiera-large",
            "description": "SAM2 box-prompted segmentation/refinement model",
        },
    },
}


@dataclass
class RegionCandidate:
    id: str
    type_hint: str
    bbox_pixel_xyxy: list[int]
    bbox_ideogram_yxyx: list[int]
    confidence: float
    source: str
    label: str | None = None
    text: str | None = None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


def clamp(value: int, minimum: int = 0, maximum: int = 1000) -> int:
    return max(minimum, min(maximum, value))


def xyxy_to_ideogram(box: list[float] | tuple[float, float, float, float], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = [float(v) for v in box]
    y_min = clamp(round((y1 / height) * 1000))
    x_min = clamp(round((x1 / width) * 1000))
    y_max = clamp(round((y2 / height) * 1000))
    x_max = clamp(round((x2 / width) * 1000))
    if y_min >= y_max or x_min >= x_max:
        return None
    return [y_min, x_min, y_max, x_max]


def sanitize_xyxy(box: list[float] | tuple[float, float, float, float], width: int, height: int) -> list[int] | None:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(width, x1)); x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1)); y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1), round(y1), round(x2), round(y2)]


def iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1, area_a + area_b - inter)


def image_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_tags(text: str) -> list[str]:
    chunks = re.split(r"[,\n]", text or "")
    tags = []
    for chunk in chunks:
        tag = chunk.strip().strip("#")
        if not tag or META_TAG_RE.match(tag):
            continue
        tag = tag.replace("_", " ")
        tag = re.sub(r"\([^)]*\)", "", tag).strip()
        if LOW_VALUE_DETECTOR_TAG_RE.match(tag):
            continue
        if len(tag) <= 40:
            tags.append(tag)
    return tags


def build_detector_prompts(tags_text: str, max_prompts: int = 48) -> list[str]:
    prompts = []
    for tag in split_tags(tags_text):
        # Keep likely visible nouns; skip artist/copyright-ish names by relying on length and meta filtering.
        if tag.lower() not in {p.lower() for p in prompts}:
            prompts.append(tag)
    for word in FALLBACK_VOCABULARY:
        if word.lower() not in {p.lower() for p in prompts}:
            prompts.append(word)
    return prompts[:max_prompts]


def _expand_path(path: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def _model_expected_path(model_root: str, component: str, selection: str) -> str:
    spec = MODEL_SPECS.get(component, {}).get(selection, {})
    local_name = spec.get("local_name") or selection
    return os.path.join(_expand_path(model_root), component, local_name)


def resolve_model_asset(
    *,
    component: str,
    selection: str,
    provided_path: str = "",
    model_root: str = DEFAULT_MODEL_ROOT,
    auto_download: bool = True,
    progress_out: str = "",
) -> tuple[dict[str, Any] | None, list[str]]:
    """Resolve or optionally download model assets for a selected preprocessor.

    This keeps model acquisition separate from inference. The current lightweight
    implementation records the resolved path/repo for downstream integrations,
    while installed libraries can use the same path to load local weights.
    """
    warnings: list[str] = []
    if not selection or selection == "none":
        return None, warnings

    if component == "ocr" and selection == "paddleocr":
        if provided_path:
            path = _expand_path(provided_path)
            if not os.path.exists(path):
                warnings.append(f"Provided PaddleOCR model path does not exist: {path}")
            return {
                "component": component,
                "selection": selection,
                "source": "user_path",
                "path": path,
                "available": os.path.exists(path),
                "auto_downloaded": False,
            }, warnings
        return {
            "component": component,
            "selection": selection,
            "source": "paddleocr_default_cache",
            "path": None,
            "available": True,
            "auto_downloaded": False,
            "note": "PaddleOCR manages its own model cache/downloads when the package is installed.",
        }, warnings

    spec = MODEL_SPECS.get(component, {}).get(selection)
    if not spec:
        warnings.append(f"No model manifest entry for {component} selection: {selection}")
        return None, warnings

    if provided_path:
        path = _expand_path(provided_path)
        available = os.path.exists(path)
        if not available:
            warnings.append(f"Provided {component} model path does not exist: {path}")
        return {
            "component": component,
            "selection": selection,
            "repo_id": spec["repo_id"],
            "source": "user_path",
            "path": path,
            "available": available,
            "auto_downloaded": False,
        }, warnings

    expected_path = _model_expected_path(model_root, component, selection)
    if os.path.exists(expected_path):
        return {
            "component": component,
            "selection": selection,
            "repo_id": spec["repo_id"],
            "source": "local_cache",
            "path": expected_path,
            "available": True,
            "auto_downloaded": False,
        }, warnings

    if not auto_download:
        warnings.append(f"{spec['description']} not found at expected path {expected_path}; enable auto-download or provide a model path.")
        return {
            "component": component,
            "selection": selection,
            "repo_id": spec["repo_id"],
            "source": "missing",
            "path": expected_path,
            "available": False,
            "auto_downloaded": False,
        }, warnings

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        warnings.append(
            f"Cannot auto-download {spec['description']} because huggingface_hub is unavailable: {exc}. "
            "Install/update app dependencies with `uv sync` or provide a local model path."
        )
        return {
            "component": component,
            "selection": selection,
            "repo_id": spec["repo_id"],
            "source": "missing",
            "path": expected_path,
            "available": False,
            "auto_downloaded": False,
        }, warnings

    os.makedirs(expected_path, exist_ok=True)
    try:
        write_progress(
            progress_out,
            stage="downloading_model",
            message=f"Downloading {spec['description']} from {spec['repo_id']}...",
            percent=None,
        )
        downloaded_path = snapshot_download(
            repo_id=spec["repo_id"],
            local_dir=expected_path,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        warnings.append(f"Auto-download failed for {spec['description']} ({spec['repo_id']}): {exc}")
        return {
            "component": component,
            "selection": selection,
            "repo_id": spec["repo_id"],
            "source": "download_failed",
            "path": expected_path,
            "available": False,
            "auto_downloaded": False,
        }, warnings

    write_progress(
        progress_out,
        stage="model_downloaded",
        message=f"Downloaded {spec['description']} to {downloaded_path}.",
        percent=100,
    )
    return {
        "component": component,
        "selection": selection,
        "repo_id": spec["repo_id"],
        "source": "auto_download",
        "path": downloaded_path,
        "available": True,
        "auto_downloaded": True,
    }, warnings


def _model_load_reference(asset: dict[str, Any]) -> str | None:
    path = asset.get("path")
    if path and asset.get("available"):
        return str(path)
    return asset.get("repo_id")


def load_selected_model_assets(model_assets: dict[str, Any], *, load_models: bool, auto_download: bool, progress_out: str = "") -> tuple[dict[str, Any], list[str]]:
    """Best-effort warm loading for selected preprocessing models.

    Loading is intentionally optional because these dependencies are large. When
    enabled, the function imports the relevant runtime and instantiates the model
    from the resolved local path, or from the repo id when auto-download is
    allowed. Loaded objects are immediately released; the goal is to make sure
    missing weights/dependencies are surfaced before inference work begins and
    to populate local caches for a smoother first run.
    """
    status: dict[str, Any] = {}
    warnings: list[str] = []
    if not load_models:
        return status, warnings

    detector = model_assets.get("detector")
    if detector:
        selection = detector.get("selection")
        write_progress(progress_out, stage="loading_model", message=f"Loading {selection} preprocessing model...", percent=None)
        ref = _model_load_reference(detector) if auto_download or detector.get("available") else None
        item = {"selection": selection, "loaded": False, "reference": ref}
        if not ref:
            item["error"] = "No available detector path and auto-download is disabled."
        elif selection in {"groundingdino", "groundingdino1.5"}:
            try:
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # type: ignore

                processor = AutoProcessor.from_pretrained(ref, local_files_only=not auto_download)
                model = AutoModelForZeroShotObjectDetection.from_pretrained(ref, local_files_only=not auto_download)
                item.update({
                    "loaded": True,
                    "runtime": "transformers",
                    "processor_class": processor.__class__.__name__,
                    "model_class": model.__class__.__name__,
                })
                del processor, model
            except Exception as exc:
                item["error"] = str(exc)
                warnings.append(
                    f"Could not load {selection} detector model: {exc}. "
                    "Install optional preprocessing dependencies with `uv sync --extra preprocess`, "
                    "or disable 'Load selected preprocessing models'."
                )
        elif selection == "florence2":
            try:
                from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore

                processor = AutoProcessor.from_pretrained(ref, trust_remote_code=True, local_files_only=not auto_download)
                model = AutoModelForCausalLM.from_pretrained(ref, trust_remote_code=True, local_files_only=not auto_download)
                item.update({
                    "loaded": True,
                    "runtime": "transformers",
                    "processor_class": processor.__class__.__name__,
                    "model_class": model.__class__.__name__,
                })
                del processor, model
            except Exception as exc:
                item["error"] = str(exc)
                warnings.append(
                    f"Could not load Florence-2 model: {exc}. "
                    "Install optional preprocessing dependencies with `uv sync --extra preprocess`, "
                    "or disable 'Load selected preprocessing models'."
                )
        status["detector"] = item

    segmenter = model_assets.get("segmenter")
    if segmenter and segmenter.get("selection") == "sam2":
        write_progress(progress_out, stage="loading_model", message="Loading SAM2 preprocessing model...", percent=None)
        ref = _model_load_reference(segmenter) if auto_download or segmenter.get("available") else None
        item = {"selection": "sam2", "loaded": False, "reference": ref}
        if not ref:
            item["error"] = "No available SAM2 path and auto-download is disabled."
        else:
            try:
                import sam2  # type: ignore

                item.update({"loaded": True, "runtime": "sam2", "module": getattr(sam2, "__name__", "sam2")})
            except Exception as exc:
                item["error"] = str(exc)
                warnings.append(
                    f"Could not import/load SAM2 runtime for model at {ref}: {exc}. "
                    "Install SAM2 in the app environment or disable SAM2 refinement."
                )
        status["segmenter"] = item

    ocr = model_assets.get("ocr")
    if ocr and ocr.get("selection") == "paddleocr":
        write_progress(progress_out, stage="loading_model", message="Loading PaddleOCR preprocessing model...", percent=None)
        item = {"selection": "paddleocr", "loaded": False, "reference": ocr.get("path")}
        try:
            from paddleocr import PaddleOCR  # type: ignore

            paddle_kwargs = {"use_textline_orientation": True, "lang": "en"}
            if ocr.get("path"):
                # PaddleOCR accepts model directory overrides in newer releases;
                # exact per-stage dirs can still be configured by advanced users
                # through PaddleOCR's own environment/config if needed.
                paddle_kwargs["text_detection_model_dir"] = ocr.get("path")
                paddle_kwargs["text_recognition_model_dir"] = ocr.get("path")
            engine = PaddleOCR(**paddle_kwargs)
            item.update({"loaded": True, "runtime": "paddleocr", "engine_class": engine.__class__.__name__})
            del engine
        except Exception as exc:
            item["error"] = str(exc)
            warnings.append(
                f"Could not load PaddleOCR runtime: {exc}. "
                "Install optional preprocessing dependencies with `uv sync --extra preprocess`, "
                "or set OCR to 'none'."
            )
        status["ocr"] = item

    return status, warnings


def run_groundingdino_detector(
    path: str,
    prompts: list[str],
    asset: dict[str, Any],
    width: int,
    height: int,
    *,
    auto_download: bool,
    box_threshold: float,
    text_threshold: float,
    progress_out: str = "",
) -> tuple[list[RegionCandidate], dict[str, Any], list[str]]:
    """Run GroundingDINO through transformers and return object candidates."""
    diagnostics: dict[str, Any] = {
        "selection": asset.get("selection"),
        "runtime": "transformers",
        "attempted": True,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "prompt_count": len(prompts),
        "raw_detection_count": 0,
        "kept_detection_count": 0,
    }
    warnings: list[str] = []
    candidates: list[RegionCandidate] = []
    ref = _model_load_reference(asset) if auto_download or asset.get("available") else None
    if not ref:
        diagnostics["error"] = "No available detector path and auto-download is disabled."
        warnings.append("GroundingDINO detection skipped because no detector model path is available.")
        return candidates, diagnostics, warnings

    try:
        import torch  # type: ignore
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # type: ignore
    except Exception as exc:
        diagnostics["error"] = str(exc)
        warnings.append(
            f"GroundingDINO detection skipped because required runtime packages are unavailable: {exc}. "
            "Run `uv sync` after pulling the latest dependency changes."
        )
        return candidates, diagnostics, warnings

    try:
        write_progress(progress_out, stage="detecting_objects", message="Running GroundingDINO object detection...", percent=None)
        with Image.open(path) as im:
            image = im.convert("RGB")
        processor = AutoProcessor.from_pretrained(ref, local_files_only=not auto_download)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(ref, local_files_only=not auto_download)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        labels = [p.strip().lower() for p in prompts if p.strip()][:48]
        text_prompt = ". ".join(labels)
        if text_prompt and not text_prompt.endswith("."):
            text_prompt += "."
        diagnostics["device"] = device
        diagnostics["text_prompt"] = text_prompt
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[height, width]], device=device)
        try:
            # Newer Transformers uses `threshold`; older examples used
            # `box_threshold`. Try the current API first.
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
                text_labels=[labels],
            )
            diagnostics["post_process_api"] = "threshold"
        except TypeError:
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes,
            )
            diagnostics["post_process_api"] = "box_threshold"
        result = results[0] if results else {}
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        text_labels = result.get("text_labels") or result.get("labels", [])
        diagnostics["raw_detection_count"] = len(boxes)
        for box, score, label in zip(boxes, scores, text_labels):
            if hasattr(box, "detach"):
                box = box.detach().cpu().tolist()
            if hasattr(score, "detach"):
                score = float(score.detach().cpu().item())
            label = str(label)
            add_candidate(
                candidates,
                type_hint="obj",
                bbox=[float(v) for v in box],
                width=width,
                height=height,
                confidence=float(score),
                source="groundingdino",
                label=label,
            )
        diagnostics["kept_detection_count"] = len(candidates)
        if not candidates:
            warnings.append(
                "GroundingDINO ran but produced zero object candidates. "
                f"Try lowering thresholds (box={box_threshold}, text={text_threshold}) or adding simpler object prompts."
            )
    except Exception as exc:
        diagnostics["error"] = str(exc)
        warnings.append(f"GroundingDINO detection failed: {exc}")
    return candidates, diagnostics, warnings


def add_candidate(out: list[RegionCandidate], *, type_hint: str, bbox: list[float], width: int, height: int, confidence: float, source: str, label: str | None = None, text: str | None = None) -> None:
    pixel = sanitize_xyxy(bbox, width, height)
    if not pixel:
        return
    normalized = xyxy_to_ideogram(pixel, width, height)
    if not normalized:
        return
    prefix = "t" if type_hint == "text" else "r"
    out.append(RegionCandidate(f"{prefix}{len(out) + 1}", type_hint, pixel, normalized, round(float(confidence), 4), source, label, text))


def run_paddleocr(path: str, width: int, height: int, threshold: float) -> tuple[list[RegionCandidate], list[str]]:
    warnings: list[str] = []
    regions: list[RegionCandidate] = []
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as exc:
        return [], [f"PaddleOCR unavailable: {exc}"]
    ocr = PaddleOCR(use_textline_orientation=True, lang="en")
    result = ocr.predict(path) if hasattr(ocr, "predict") else ocr.ocr(path, cls=True)
    items = result if isinstance(result, list) else []
    for page in items:
        if isinstance(page, dict):
            polys = page.get("dt_polys") or page.get("rec_polys") or []
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            for poly, text, score in zip(polys, texts, scores):
                if float(score) < threshold or not str(text).strip():
                    continue
                xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                add_candidate(regions, type_hint="text", bbox=[min(xs), min(ys), max(xs), max(ys)], width=width, height=height, confidence=float(score), source="paddleocr", text=str(text).strip())
        elif isinstance(page, list):
            for item in page:
                try:
                    poly, (text, score) = item
                    if float(score) < threshold or not str(text).strip():
                        continue
                    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                    add_candidate(regions, type_hint="text", bbox=[min(xs), min(ys), max(xs), max(ys)], width=width, height=height, confidence=float(score), source="paddleocr", text=str(text).strip())
                except Exception:
                    continue
    return regions, warnings


def filter_candidates(candidates: list[RegionCandidate], width: int, height: int, max_regions: int, iou_threshold: float) -> list[RegionCandidate]:
    image_area = width * height
    kept: list[RegionCandidate] = []
    for cand in sorted(candidates, key=lambda c: c.confidence, reverse=True):
        x1, y1, x2, y2 = cand.bbox_pixel_xyxy
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(1, image_area)
        if area_ratio < 0.0005:
            continue
        if area_ratio > 0.92 and (not cand.label or cand.label.lower() in {"image", "background", "scene", "object"}):
            continue
        if any(iou(cand.bbox_pixel_xyxy, other.bbox_pixel_xyxy) >= iou_threshold for other in kept):
            continue
        kept.append(cand)
        if len(kept) >= max_regions:
            break
    for idx, cand in enumerate(kept, start=1):
        cand.id = ("t" if cand.type_hint == "text" else "r") + str(idx)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate optional region/OCR candidates for CaptionHelper.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--tags", default="")
    parser.add_argument("--tags-text", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress-out", default="")
    parser.add_argument("--detector", default="none", choices=["none", "groundingdino", "groundingdino1.5", "florence2"])
    parser.add_argument("--segmenter", default="none", choices=["none", "sam2"])
    parser.add_argument("--ocr", default="none", choices=["none", "paddleocr"])
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--detector-model-path", default="")
    parser.add_argument("--segmenter-model-path", default="")
    parser.add_argument("--ocr-model-path", default="")
    parser.add_argument("--auto-download", dest="auto_download", action="store_true", default=True)
    parser.add_argument("--no-auto-download", dest="auto_download", action="store_false")
    parser.add_argument("--load-models", dest="load_models", action="store_true", default=True)
    parser.add_argument("--no-load-models", dest="load_models", action="store_false")
    parser.add_argument("--max-regions", type=int, default=12)
    parser.add_argument("--ocr-threshold", type=float, default=0.55)
    parser.add_argument("--iou-threshold", type=float, default=0.65)
    parser.add_argument("--detector-box-threshold", type=float, default=0.30)
    parser.add_argument("--detector-text-threshold", type=float, default=0.25)
    args = parser.parse_args()

    with Image.open(args.image) as im:
        width, height = im.size
    write_progress(args.progress_out, stage="starting", message="Preparing region preprocessing...", percent=0)
    tags_text = args.tags_text
    if args.tags and os.path.exists(args.tags):
        with open(args.tags, "r", encoding="utf-8", errors="replace") as fh:
            tags_text = (tags_text + "\n" + fh.read()).strip()

    warnings: list[str] = []
    regions: list[RegionCandidate] = []
    detector_diagnostics: dict[str, Any] = {}
    prompts = build_detector_prompts(tags_text)
    model_assets: dict[str, Any] = {}

    for component, selection, provided_path in (
        ("detector", args.detector, args.detector_model_path),
        ("segmenter", args.segmenter, args.segmenter_model_path),
        ("ocr", args.ocr, args.ocr_model_path),
    ):
        asset, asset_warnings = resolve_model_asset(
            component=component,
            selection=selection,
            provided_path=provided_path,
            model_root=args.model_root,
            auto_download=args.auto_download,
            progress_out=args.progress_out,
        )
        if asset:
            model_assets[component] = asset
        warnings.extend(asset_warnings)
    model_load_status, load_warnings = load_selected_model_assets(
        model_assets,
        load_models=args.load_models,
        auto_download=args.auto_download,
        progress_out=args.progress_out,
    )
    warnings.extend(load_warnings)

    if args.detector != "none":
        detector_asset = model_assets.get("detector") or {}
        asset_note = f" Resolved model path: {detector_asset.get('path')}." if detector_asset.get("path") else ""
        detector_loaded = (model_load_status.get("detector") or {}).get("loaded")
        if args.detector in {"groundingdino", "groundingdino1.5"} and (detector_loaded or detector_asset.get("available") or args.auto_download):
            detected_regions, detector_diagnostics, detector_warnings = run_groundingdino_detector(
                args.image,
                prompts,
                detector_asset,
                width,
                height,
                auto_download=args.auto_download,
                box_threshold=args.detector_box_threshold,
                text_threshold=args.detector_text_threshold,
                progress_out=args.progress_out,
            )
            regions.extend(detected_regions)
            warnings.extend(detector_warnings)
        elif not detector_loaded:
            warnings.append(f"{args.detector} preprocessing was skipped because the detector runtime/model did not load.{asset_note}")
    if args.segmenter != "none":
        segmenter_asset = model_assets.get("segmenter") or {}
        asset_note = f" Resolved model path: {segmenter_asset.get('path')}." if segmenter_asset.get("path") else ""
        warnings.append(f"SAM2 refinement requires detector boxes and a local SAM2 integration; no masks were generated.{asset_note}")
    if args.ocr == "paddleocr":
        write_progress(args.progress_out, stage="ocr", message="Running OCR preprocessing...", percent=None)
        ocr_regions, ocr_warnings = run_paddleocr(args.image, width, height, args.ocr_threshold)
        regions.extend(ocr_regions)
        warnings.extend(ocr_warnings)

    kept = filter_candidates(regions, width, height, max(0, args.max_regions), args.iou_threshold)
    payload = {
        "image_path": args.image,
        "image_hash": image_hash(args.image),
        "image_width": width,
        "image_height": height,
        "detector": args.detector,
        "segmenter": args.segmenter,
        "ocr_engine": args.ocr,
        "model_root": _expand_path(args.model_root),
        "auto_download": args.auto_download,
        "load_models": args.load_models,
        "model_assets": model_assets,
        "model_load_status": model_load_status,
        "detector_prompts": prompts,
        "detector_diagnostics": detector_diagnostics,
        "regions": [c.to_json() for c in kept if c.type_hint == "obj"],
        "ocr": [c.to_json() for c in kept if c.type_hint == "text"],
        "warnings": warnings,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    write_progress(args.progress_out, stage="done", message="Region preprocessing complete.", percent=100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
