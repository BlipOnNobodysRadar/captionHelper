import os

import pytest

import native_av
from presets import CAPTION_PRESETS


def test_h3_preset_defaults():
    preset = next(item for item in CAPTION_PRESETS if item["id"] == "h3_qwen3_omni_native_av")
    assert preset["video_input_mode"] == "native_av"
    assert preset["include_audio"] is True
    assert preset["model"] == "qwen3-omni-h3-caption"
    assert preset["max_concurrent"] == 1


def test_native_content_video_only(tmp_path):
    video = tmp_path / "clip.webm"
    video.write_bytes(b"video")
    content = native_av.build_native_av_content(str(video), "caption")
    assert [part["type"] for part in content] == ["input_video", "text"]


def test_native_content_with_audio(tmp_path):
    video, audio = tmp_path / "clip.mp4", tmp_path / "audio.wav"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    content = native_av.build_native_av_content(str(video), "caption", str(audio))
    assert [part["type"] for part in content] == ["input_video", "input_audio", "text"]


def test_audio_cleanup_after_context_failure(tmp_path, monkeypatch):
    video, audio = tmp_path / "clip.mp4", tmp_path / "temporary.wav"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    monkeypatch.setattr(native_av, "probe_video", lambda _path: {"duration_sec": 1.5, "audio_stream_found": True})
    monkeypatch.setattr(native_av, "extract_audio_temp", lambda _path: str(audio))
    with pytest.raises(RuntimeError):
        with native_av.prepare_native_av(str(video), "caption") as (content, metadata):
            assert metadata["audio_supplied"]
            assert any(part["type"] == "input_audio" for part in content)
            raise RuntimeError("backend failed")
    assert not os.path.exists(audio)
