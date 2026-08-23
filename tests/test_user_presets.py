import unittest

import app


class UserPresetTests(unittest.TestCase):
    def test_user_preset_preserves_full_job_settings(self):
        preset = app._coerce_user_preset({
            "id": "user:prior-run",
            "name": "Prior run",
            "system_prompt": "system",
            "user_template": "[image]",
            "prefill": "{",
            "max_output_tokens": 256,
            "media": "image",
            "saved_settings": {
                "target_folder": "/datasets/job-a",
                "model": "/models/vision-model.gguf",
                "prefill": "{",
                "num_frames": 8,
                "sampling_type": "head",
                "video_input_mode": "native_av",
                "include_audio": True,
                "max_image_side": 1536,
                "max_output_tokens": 256,
                "max_concurrent": 2,
                "abort_after_server_errors": 0,
                "overwrite": True,
                "prepend_existing": True,
                "filename_affix_text": "converted_",
                "filename_affix_position": "suffix",
                "output_to_subdir": True,
                "output_subdir_name": "out",
                "use_existing_caption": True,
                "image_mode": True,
                "existing_caption": "old caption",
                "source_tags": "tag one",
                "character_tags": "character",
                "copyright_tags": "series",
                "artist_tags": "artist",
                "general_tags": "general",
                "rating_tags": "safe",
                "quality_tags": "best",
            },
        })

        self.assertEqual(preset["saved_settings"]["target_folder"], "/datasets/job-a")
        self.assertEqual(preset["saved_settings"]["model"], "/models/vision-model.gguf")
        self.assertEqual(preset["saved_settings"]["num_frames"], 8)
        self.assertEqual(preset["saved_settings"]["sampling_type"], "head")
        self.assertEqual(preset["saved_settings"]["video_input_mode"], "native_av")
        self.assertTrue(preset["saved_settings"]["include_audio"])
        self.assertTrue(preset["saved_settings"]["overwrite"])
        self.assertTrue(preset["saved_settings"]["prepend_existing"])
        self.assertTrue(preset["saved_settings"]["output_to_subdir"])
        self.assertTrue(preset["saved_settings"]["use_existing_caption"])
        self.assertEqual(preset["saved_settings"]["source_tags"], "tag one")


if __name__ == "__main__":
    unittest.main()
