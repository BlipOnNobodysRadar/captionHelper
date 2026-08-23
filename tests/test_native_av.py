import os
from contextlib import contextmanager

import pytest

import native_av
import app
from presets import CAPTION_PRESETS


def test_h3_preset_defaults():
    preset = next(item for item in CAPTION_PRESETS if item["id"] == "h3_qwen3_omni_native_av")
    assert preset["video_input_mode"] == "native_av"
    assert preset["include_audio"] is True
    assert preset["validate_h3_output"] is False
    assert preset["model"] == "qwen3-omni-h3-caption"
    assert preset["max_output_tokens"] == 0
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


def _video_batch_params(folder, mode):
    return {
        "target_folder": str(folder), "image_mode": False,
        "system_prompt": "system", "user_template": "Caption this clip.",
        "metadata_values": {}, "model": "model", "prefill": "",
        "num_frames": 3, "sampling_type": "uniform", "overwrite": True,
        "prepend_existing": False, "use_existing_caption": False,
        "filename_affix_text": "", "filename_affix_position": "prefix",
        "output_to_subdir": False, "output_subdir_name": "",
        "max_image_side": 0, "max_output_tokens": 256,
        "video_input_mode": mode, "include_audio": True,
    }


def test_batch_native_av_does_not_sample_frames_and_writes_sidecar(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    @contextmanager
    def fake_prepare(path, text, include_audio):
        assert path == str(video)
        assert include_audio is True
        yield [{"type": "input_video", "input_video": {"data": "encoded"}}], {
            "media_input_mode": "native_av", "audio_supplied": True,
        }

    monkeypatch.setattr(app, "prepare_native_av", fake_prepare)
    monkeypatch.setattr(app, "extract_frames", lambda *_args, **_kwargs: pytest.fail("native AV sampled frames"))
    monkeypatch.setattr(
        app, "_call_native_av_content",
        lambda *_args, **_kwargs: "cut-off but non-empty caption",
    )

    result = app._process_one_target("clip.mp4", _video_batch_params(tmp_path, "native_av"))

    assert result["ok"] is True
    assert result["out"] == "clip.txt"
    assert (tmp_path / "clip.txt").is_file()
    assert (tmp_path / "clip.txt").read_text() == "cut-off but non-empty caption"


def test_batch_sampled_video_still_extracts_frames(tmp_path, monkeypatch):
    video = tmp_path / "clip.webm"
    video.write_bytes(b"video")
    sampled = []

    def fake_extract(*args, **kwargs):
        sampled.append((args, kwargs))
        return ["data:image/jpeg;base64,frame"]

    monkeypatch.setattr(app, "extract_frames", fake_extract)
    monkeypatch.setattr(app, "call_vision_api", lambda *_args, **_kwargs: "sampled caption")

    result = app._process_one_target("clip.webm", _video_batch_params(tmp_path, "sampled_frames"))

    assert result["ok"] is True
    assert len(sampled) == 1


def test_native_h3_retries_once_with_same_media_when_format_is_malformed(monkeypatch):
    calls = []

    def fake_call(content, *_args):
        calls.append(content)
        if len(calls) == 1:
            return "Here is your caption without the requested envelope."
        return (
            "integrated_multimodal_description: [Shot 1] A clip.\n"
            "overall_soundscape: Quiet.\nnon_diegetic_music: N/A"
        )

    monkeypatch.setattr(app, "_call_native_av_content", fake_call)
    media = {"type": "input_video", "input_video": {"data": "encoded-once"}}
    caption, metadata = app._caption_native_with_validation(
        [media, {"type": "text", "text": "original prompt"}], "system", "model", "", 256, True,
    )

    assert metadata["retried"] is True
    assert metadata["validation"]["valid"] is True
    assert caption.startswith("integrated_multimodal_description:")
    assert calls[0][0] is media
    assert calls[1][0] is media
    assert "encoded-once" in calls[1][0]["input_video"]["data"]


def test_native_h3_retry_failure_remains_flagged(monkeypatch):
    monkeypatch.setattr(app, "_call_native_av_content", lambda *_args: "malformed")

    caption, metadata = app._caption_native_with_validation(
        [{"type": "input_video", "input_video": {"data": "encoded"}}],
        "system", "model", "", 256, True,
    )

    assert caption == "malformed"
    assert metadata["retried"] is True
    assert metadata["validation"]["valid"] is False


def test_native_h3_validation_is_off_by_default_and_does_not_retry(monkeypatch):
    calls = []

    def fake_call(*_args):
        calls.append(True)
        return "cut-off but non-empty caption"

    monkeypatch.setattr(app, "_call_native_av_content", fake_call)
    caption, metadata = app._caption_native_with_validation(
        [{"type": "input_video", "input_video": {"data": "encoded"}}],
        "system", "model", "", 256,
    )

    assert caption == "cut-off but non-empty caption"
    assert metadata["validation"] == {"enabled": False, "valid": True, "errors": []}
    assert metadata["retried"] is False
    assert len(calls) == 1


def test_native_request_omits_zero_token_cap_and_warns_for_thinking_only(monkeypatch, caplog):
    posted = {}

    class Response:
        ok = True

        @staticmethod
        def json():
            return {
                "choices": [{"message": {"content": "<think>\n\n</think>"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 3},
            }

    def fake_post(_url, **kwargs):
        posted.update(kwargs["json"])
        return Response()

    monkeypatch.setattr(app.requests, "post", fake_post)
    with caplog.at_level("WARNING"):
        caption = app._call_native_av_content([], "system", "model", max_output_tokens=0)

    assert caption == "<think>\n\n</think>"
    assert "max_tokens" not in posted
    assert "backend generation/chat-template issue" in caplog.text
    assert "completion_tokens': 3" in caplog.text
