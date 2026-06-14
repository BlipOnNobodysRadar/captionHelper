import os
import tempfile
import unittest
from unittest import mock

import app


def _params(folder, **overrides):
    params = {
        "target_folder": folder,
        "image_mode": True,
        "system_prompt": "system",
        "user_template": "Existing: {existing_caption}\n[image]",
        "metadata_values": {},
        "model": "model",
        "prefill": "",
        "num_frames": 1,
        "sampling_type": "uniform",
        "overwrite": True,
        "prepend_existing": False,
        "use_existing_caption": True,
        "filename_affix_text": "",
        "filename_affix_position": "prefix",
        "output_to_subdir": True,
        "output_subdir_name": "converted",
        "max_image_side": 0,
        "max_output_tokens": 32,
    }
    params.update(overrides)
    return params


class BatchCopyOutputTests(unittest.TestCase):
    def test_copy_subdir_overwrite_preserves_original_caption_and_writes_copy_only(self):
        with tempfile.TemporaryDirectory() as folder:
            image_path = os.path.join(folder, "sample.jpg")
            source_txt = os.path.join(folder, "sample.txt")
            os.makedirs(os.path.join(folder, "converted"), exist_ok=True)
            copied_txt = os.path.join(folder, "converted", "sample.txt")

            with open(image_path, "wb") as fh:
                fh.write(b"original image bytes")
            with open(source_txt, "w", encoding="utf-8") as fh:
                fh.write("original caption")
            with open(copied_txt, "w", encoding="utf-8") as fh:
                fh.write("previous copied caption")

            prompts = []

            def fake_call(_imgs, _system, _model, **kwargs):
                prompts.append(kwargs["user_prompt"])
                return "new copied caption"

            with mock.patch.object(app, "image_to_data_url", return_value="data:image/jpeg;base64,abc"), \
                 mock.patch.object(app, "call_vision_api", side_effect=fake_call):
                result = app._process_one_target("sample.jpg", _params(folder))

            self.assertTrue(result["ok"])
            self.assertEqual(result["out"], os.path.join("converted", "sample.txt"))
            with open(source_txt, "r", encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "original caption")
            with open(copied_txt, "r", encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "new copied caption")
            self.assertIn("original caption", prompts[0])
            self.assertNotIn("previous copied caption", prompts[0])
            with open(os.path.join(folder, "converted", "sample.jpg"), "rb") as fh:
                self.assertEqual(fh.read(), b"original image bytes")

    def test_select_targets_ignores_directories_with_media_extensions(self):
        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "not_a_real_image.jpg"))
            with open(os.path.join(folder, "real.jpg"), "wb") as fh:
                fh.write(b"x")

            self.assertEqual(app._select_targets(folder, image_mode=True), ["real.jpg"])


if __name__ == "__main__":
    unittest.main()
