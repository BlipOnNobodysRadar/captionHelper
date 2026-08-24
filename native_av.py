"""Shared preparation helpers for llama.cpp native audiovisual requests."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager


class NativeAVError(RuntimeError):
    """A user-actionable native media preparation error."""


def _require_executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise NativeAVError(
            f"{name} is required for native video audio handling but was not found on PATH. "
            f"Install {name} (normally provided by FFmpeg) or turn off Include video audio."
        )
    return path


def probe_video(video_path: str) -> dict:
    """Return duration and whether ffprobe reports an audio stream."""
    ffprobe = _require_executable("ffprobe")
    command = [ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", video_path]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise NativeAVError(f"ffprobe could not inspect the video: {(result.stderr or 'unknown error').strip()}")
    try:
        payload = json.loads(result.stdout or "{}")
        duration = float((payload.get("format") or {}).get("duration") or 0)
        has_audio = any(stream.get("codec_type") == "audio" for stream in payload.get("streams", []))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise NativeAVError(f"ffprobe returned invalid video metadata: {exc}") from exc
    return {"duration_sec": duration, "audio_stream_found": has_audio}


def extract_audio_temp(video_path: str) -> str:
    """Extract conservative mono 16 kHz PCM audio to a temporary WAV."""
    ffmpeg = _require_executable("ffmpeg")
    fd, wav_path = tempfile.mkstemp(prefix="captionhelper-audio-", suffix=".wav")
    os.close(fd)
    command = [ffmpeg, "-v", "error", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", wav_path]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        try:
            os.remove(wav_path)
        except OSError:
            pass
        raise NativeAVError(f"ffmpeg audio extraction failed: {(result.stderr or 'unknown error').strip()}")
    return wav_path


def _raw_base64_with_sha256(path: str) -> tuple[str, str]:
    with open(path, "rb") as handle:
        raw = handle.read()
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest()


def _raw_base64(path: str) -> str:
    return _raw_base64_with_sha256(path)[0]


def build_native_av_content(video_path: str, text: str, audio_path: str | None = None, video_data: str | None = None) -> list[dict]:
    """Build recent llama.cpp OpenAI-compatible input_video/input_audio content."""
    content = [{"type": "input_video", "input_video": {"data": video_data or _raw_base64(video_path)}}]
    if audio_path:
        content.append({"type": "input_audio", "input_audio": {"data": _raw_base64(audio_path), "format": "wav"}})
    content.append({"type": "text", "text": text})
    return content


@contextmanager
def prepare_native_av(video_path: str, text: str, include_audio: bool = True, allow_visual_fallback: bool = True):
    """Prepare content and always remove extracted audio after request consumption."""
    audio_path = None
    video_data, video_sha256 = _raw_base64_with_sha256(video_path)
    metadata = {
        "media_input_mode": "native_av", "video_path": video_path,
        "video_size_bytes": os.path.getsize(video_path),
        "video_sha256": video_sha256,
        "audio_requested": bool(include_audio), "audio_stream_found": False, "audio_supplied": False,
        "audio_extraction_settings": "PCM s16le, mono, 16 kHz" if include_audio else None,
        "warnings": [],
    }
    try:
        if include_audio:
            info = probe_video(video_path)
            metadata["video_duration_sec"] = info["duration_sec"]
            metadata["audio_stream_found"] = info["audio_stream_found"]
            if info["audio_stream_found"]:
                try:
                    audio_path = extract_audio_temp(video_path)
                    metadata["audio_supplied"] = True
                except NativeAVError as exc:
                    if not allow_visual_fallback:
                        raise
                    metadata["warnings"].append(str(exc) + " Continuing with visual-only input.")
        audio_note = (
            "Fuse the supplied complete video and soundtrack as one temporally related audiovisual clip."
            if metadata["audio_supplied"] else
            "No audio was supplied. Analyze visuals only; do not invent speech, soundtrack, or other sounds."
        )
        isolation_note = "This request contains exactly one clip. Describe only this supplied clip; ignore any prior or cached media."
        yield build_native_av_content(video_path, f"{text}\n\n{audio_note}\n{isolation_note}", audio_path, video_data), metadata
    finally:
        if audio_path:
            try:
                os.remove(audio_path)
            except OSError:
                pass
