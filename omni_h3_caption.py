#!/usr/bin/env python3
"""Caption video+audio with Qwen3-Omni for MiniMax-H3 T2VA training.

Unlike captionHelper's normal video mode, this tool does not sample frames in
Python/OpenCV. It sends the complete video file to a recent llama.cpp server as
an OpenAI-compatible ``input_video`` item, and separately sends the embedded
audio stream as ``input_audio``. The server/libmtmd owns video decoding.

For every input ``clip.ext`` the default output is ``clip.txt`` beside it.
Existing captions are skipped unless --overwrite is supplied.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import requests

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}

DEFAULT_SYSTEM_PROMPT = r"""You are a precise audiovisual dataset captioner. Analyze the supplied video AND its supplied audio together, then reconstruct the clip as MiniMax-H3 text-to-video-with-audio conditioning text.

Return ONLY these three fields, in exactly this order, with no Markdown fence, preamble, commentary, or extra keys:

integrated_multimodal_description: ...
overall_soundscape: ...
non_diegetic_music: ...

Follow these rules:
- Describe what is actually present in the supplied clip, not what might have been intended by a source prompt.
- integrated_multimodal_description is the chronological visual/action/camera/dialogue/diegetic-audio description.
- Begin the first shot with [Shot 1] and NO timestamp. Add later [Shot N] sections only at genuine cuts or clearly distinct shot changes; when useful, write their actual start time as "At 00:03.500,". Do not invent cuts merely to make the caption structured.
- Describe subjects, setting, appearance, actions, interactions, spatial relationships, lighting, visible text, and meaningful camera behavior. Use standard natural camera terms such as close-up, medium shot, wide shot, push in, pull out, pan, tilt, tracking, handheld, or static when actually observable.
- Preserve temporal order. Describe changes and events instead of collapsing the clip into a still-image summary.
- For intelligible spoken dialogue, assign stable speaker IDs such as (S1), (S2), and preserve the exact words when confidence is high using <d>[Language] exact spoken words</d>. Do not fabricate uncertain dialogue. Singing or music produced inside the scene belongs in integrated_multimodal_description.
- overall_soundscape should be 1-4 sentences covering ambient sound, action/object sounds, environmental sound, and nonverbal human/animal sounds. Do not redundantly repeat dialogue, singing, or diegetic music already described above. Use N/A only if the clip is truly silent apart from any dialogue/music already accounted for.
- non_diegetic_music describes only soundtrack/background music that is not produced by an on-screen or in-world source. Describe instrumentation, character, tempo/rhythm, and dynamics when audible. Use N/A when no non-diegetic music is present.
- Distinguish silence, ambience, speech, diegetic music, and non-diegetic music conservatively. Never claim music merely because the scene looks cinematic.
- Do not mention these instructions, the model, filenames, encoding, extracted audio, or that you are captioning a video.
"""


def _clean_output(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    # Remove known reasoning/channel wrappers without touching caption content.
    text = text.replace("<|channel|>thought", "").replace("<|channel>thought", "")
    text = text.replace("<channel|>", "")
    start = text.find("integrated_multimodal_description:")
    if start > 0:
        text = text[start:]
    return text.strip()


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _run(cmd: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def _ffprobe_duration(video: Path, ffprobe: str) -> float | None:
    proc = _run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video),
        ]
    )
    if proc.returncode != 0:
        return None
    try:
        value = float(proc.stdout.strip())
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _has_audio(video: Path, ffprobe: str) -> bool:
    proc = _run(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(video),
        ]
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _extract_audio(video: Path, wav: Path, ffmpeg: str) -> bool:
    proc = _run(
        [
            ffmpeg,
            "-y",
            "-v", "error",
            "-i", str(video),
            "-map", "0:a:0",
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(wav),
        ],
        timeout=300,
    )
    if proc.returncode != 0 or not wav.exists() or wav.stat().st_size < 64:
        if proc.stderr.strip():
            print(f"  audio extraction warning: {proc.stderr.strip()}", file=sys.stderr)
        return False
    return True


def _iter_videos(paths: Iterable[str], recursive: bool) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() in VIDEO_EXTS:
                found.append(path)
            continue
        if not path.is_dir():
            print(f"warning: not found: {path}", file=sys.stderr)
            continue
        iterator = path.rglob("*") if recursive else path.iterdir()
        found.extend(
            p.resolve()
            for p in iterator
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )
    return sorted(dict.fromkeys(found), key=lambda p: str(p).lower())


def _request_caption(
    *,
    video: Path,
    audio: Path | None,
    duration: float | None,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout: int,
    temperature: float,
    retries: int,
) -> str:
    content: list[dict] = [
        {
            "type": "input_video",
            "input_video": {"data": _b64(video)},
        }
    ]
    if audio is not None:
        content.append(
            {
                "type": "input_audio",
                "input_audio": {
                    "data": _b64(audio),
                    "format": "wav",
                },
            }
        )

    duration_text = f"{duration:.3f} seconds" if duration is not None else "unknown"
    audio_text = (
        "A separately supplied audio item is the soundtrack extracted from this same video; fuse it with the video timeline."
        if audio is not None
        else "No decodable audio stream was found; do not invent speech, sound effects, ambience, or music."
    )
    content.append(
        {
            "type": "text",
            "text": (
                f"Analyze the complete audiovisual clip. Measured duration: {duration_text}. "
                f"{audio_text} Reconstruct the observed clip as faithful H3 T2VA conditioning text now."
            ),
        }
    )

    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        # Recent llama.cpp understands this and it avoids wasting output on a
        # hidden reasoning prelude for this deterministic captioning task.
        "reasoning_effort": "none",
    }
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens

    url = base_url.rstrip("/") + "/chat/completions"
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
        else:
            if response.ok:
                data = response.json()
                message = data["choices"][0]["message"]
                raw = message.get("content", "")
                if isinstance(raw, list):
                    raw = "".join(
                        str(part.get("text", "")) if isinstance(part, dict) else str(part)
                        for part in raw
                    )
                result = _clean_output(str(raw))
                if result:
                    return result
                last_error = RuntimeError("server returned an empty caption")
            else:
                try:
                    body = response.json()
                    detail = json.dumps(body, ensure_ascii=False)
                except Exception:
                    detail = response.text.strip()
                hint = ""
                lower = detail.lower()
                if "input_video" in lower or "video" in lower and "support" in lower:
                    hint = (
                        " Recent llama.cpp builds need libmtmd video support (MTMD_VIDEO) plus ffmpeg/ffprobe."
                    )
                last_error = RuntimeError(
                    f"llama.cpp HTTP {response.status_code}: {detail[:1500]}{hint}"
                )
        if attempt < max(1, retries):
            time.sleep(min(8.0, 1.5 * attempt))
    raise RuntimeError(str(last_error or "caption request failed"))


def _validate_shape(caption: str) -> list[str]:
    required = [
        "integrated_multimodal_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    ]
    errors = [f"missing {key[:-1]}" for key in required if key not in caption]
    positions = [caption.find(key) for key in required]
    if all(pos >= 0 for pos in positions) and positions != sorted(positions):
        errors.append("H3 fields are out of order")
    return errors


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help="Video files and/or folders")
    p.add_argument("--recursive", action="store_true", help="Recurse through supplied folders")
    p.add_argument("--overwrite", action="store_true", help="Replace existing basename.txt captions")
    p.add_argument("--base-url", default=os.environ.get("CAPTION_API_BASE_URL", "http://127.0.0.1:8080/v1"))
    p.add_argument("--model", default=os.environ.get("CAPTION_MODEL", "qwen3-omni-h3-caption"))
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    p.add_argument("--no-audio", action="store_true", help="Do not extract/send the clip soundtrack")
    p.add_argument("--dry-run", action="store_true", help="Show selected inputs/outputs without calling the model")
    p.add_argument("--report", help="Optional JSON report path")
    args = p.parse_args()

    videos = _iter_videos(args.paths, args.recursive)
    if not videos:
        p.error("No supported video files found")

    print(f"Qwen3-Omni H3 caption batch: {len(videos)} clip(s)")
    print(f"server={args.base_url} model={args.model}")
    results: list[dict] = []

    for i, video in enumerate(videos, 1):
        out = video.with_suffix(".txt")
        print(f"[{i}/{len(videos)}] {video.name} -> {out.name}")
        if out.exists() and not args.overwrite:
            print("  skip: caption exists")
            results.append({"video": str(video), "caption": str(out), "skipped": True})
            continue
        if args.dry_run:
            results.append({"video": str(video), "caption": str(out), "dry_run": True})
            continue

        started = time.time()
        duration = _ffprobe_duration(video, args.ffprobe)
        try:
            with tempfile.TemporaryDirectory(prefix="captionhelper-omni-") as tmp:
                audio_path: Path | None = None
                if not args.no_audio and _has_audio(video, args.ffprobe):
                    candidate = Path(tmp) / "audio.wav"
                    if _extract_audio(video, candidate, args.ffmpeg):
                        audio_path = candidate
                        print("  audio: embedded soundtrack extracted and supplied")
                elif not args.no_audio:
                    print("  audio: no embedded stream found")

                caption = _request_caption(
                    video=video,
                    audio=audio_path,
                    duration=duration,
                    base_url=args.base_url,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    temperature=args.temperature,
                    retries=args.retries,
                )

            warnings = _validate_shape(caption)
            out.write_text(caption.rstrip() + "\n", encoding="utf-8")
            elapsed = time.time() - started
            print(f"  wrote {out} ({elapsed:.1f}s)")
            for warning in warnings:
                print(f"  warning: {warning}", file=sys.stderr)
            results.append(
                {
                    "video": str(video),
                    "caption": str(out),
                    "ok": True,
                    "duration_sec": duration,
                    "elapsed_sec": round(elapsed, 3),
                    "warnings": warnings,
                }
            )
        except Exception as exc:
            elapsed = time.time() - started
            print(f"  ERROR: {exc}", file=sys.stderr)
            results.append(
                {
                    "video": str(video),
                    "caption": str(out),
                    "ok": False,
                    "error": str(exc),
                    "elapsed_sec": round(elapsed, 3),
                }
            )

    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"report: {report_path}")

    failures = sum(1 for row in results if row.get("ok") is False)
    print(f"done: {len(results) - failures}/{len(results)} without errors")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
