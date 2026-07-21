import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("GEMINI_API_KEY", "test-key")

if "pytz" not in sys.modules:
    import types

    fake_pytz = types.ModuleType("pytz")
    fake_pytz.timezone = lambda name: None
    sys.modules["pytz"] = fake_pytz


class PdpaMaskingTests(unittest.TestCase):
    def _write_synthetic_label(self, path: str, include_divider: bool = True) -> None:
        image = np.full((520, 820, 3), 235, dtype=np.uint8)

        # Simulate personal data above the divider.
        cv2.putText(image, "BANYA SOOKJAI 0612899146", (70, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        cv2.putText(image, "Customer: Allergies / Address", (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

        if include_divider:
            cv2.line(image, (55, 160), (765, 160), (0, 0, 0), 4)

        # Simulate medication content below the divider.
        cv2.putText(image, "DUTROSS DM 8 MG", (70, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
        cv2.putText(image, "DEXTROMETHORPHAN", (70, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        cv2.putText(image, "After meals morning noon evening", (70, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

        cv2.imwrite(path, image)

    def _write_synthetic_label_with_moderate_glare(self, path: str) -> None:
        image = np.full((520, 820, 3), 210, dtype=np.uint8)
        cv2.putText(image, "BANYA SOOKJAI 0612899146", (70, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        cv2.line(image, (55, 160), (765, 160), (0, 0, 0), 4)
        cv2.putText(image, "DUTROSS DM 8 MG", (70, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
        cv2.rectangle(image, (600, 260), (790, 440), (255, 255, 255), -1)
        cv2.imwrite(path, image)

    def _write_label_on_background_with_internal_divider(self, path: str) -> None:
        image = np.full((720, 960, 3), (70, 55, 45), dtype=np.uint8)
        cv2.rectangle(image, (210, 130), (760, 600), (228, 228, 218), -1)
        cv2.rectangle(image, (210, 130), (760, 600), (170, 170, 160), 3)
        cv2.putText(image, "BANYA SOOKJAI 0612899146", (235, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 35, 35), 2)
        cv2.putText(image, "Customer: allergy history", (235, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (35, 35, 35), 2)
        cv2.line(image, (230, 255), (735, 255), (10, 10, 10), 3)
        cv2.putText(image, "PINRONE/NORCA 5 MG", (235, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        cv2.putText(image, "NORETHISTERONE", (235, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
        cv2.imwrite(path, image)

    def _write_label_on_background_without_detectable_divider(self, path: str) -> None:
        image = np.full((720, 960, 3), (70, 55, 45), dtype=np.uint8)
        cv2.rectangle(image, (210, 130), (760, 600), (228, 228, 218), -1)
        cv2.rectangle(image, (210, 130), (760, 600), (170, 170, 160), 3)
        cv2.putText(image, "BANYA SOOKJAI 0612899146", (235, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 35, 35), 2)
        cv2.putText(image, "Customer: allergy history", (235, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (35, 35, 35), 2)
        cv2.putText(image, "PINRONE/NORCA 5 MG", (235, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        cv2.putText(image, "NORETHISTERONE", (235, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
        cv2.imwrite(path, image)

    def _write_white_background_label_with_thin_divider(self, path: str) -> None:
        image = np.full((520, 820, 3), 245, dtype=np.uint8)
        cv2.putText(image, "BANYA SOOKJAI 0612899146", (70, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        cv2.putText(image, "Customer: allergy history", (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
        cv2.line(image, (70, 165), (610, 165), (120, 120, 120), 1)
        cv2.putText(image, "HEPALAC 100ML", (70, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
        cv2.putText(image, "LACTULOSE", (70, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
        cv2.imwrite(path, image)

    def _write_large_low_contrast_label(self, path: str) -> None:
        image = np.full((2400, 3600, 3), 225, dtype=np.uint8)
        cv2.putText(image, "BANYA SOOKJAI 0612899146", (560, 420), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (85, 85, 85), 5)
        cv2.putText(image, "Customer: allergy history", (560, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (90, 90, 90), 4)
        cv2.line(image, (560, 720), (2860, 720), (75, 75, 75), 3)
        cv2.putText(image, "HEPALAC 100ML", (560, 980), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (45, 45, 45), 8)
        cv2.putText(image, "LACTULOSE", (560, 1160), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (45, 45, 45), 7)
        cv2.imwrite(path, image)

    def _assert_top_masked_and_body_readable(self, safe_image) -> None:
        gray = cv2.cvtColor(safe_image, cv2.COLOR_BGR2GRAY)
        top_band_dark_ratio = np.mean(gray[:120, :] < 20)
        body_band_dark_ratio = np.mean(gray[220:, :] < 80)

        self.assertGreater(top_band_dark_ratio, 0.85)
        self.assertGreater(body_band_dark_ratio, 0.003)

    def test_create_pdpa_safe_image_masks_personal_data_above_divider(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "label.jpg")
            output_path = str(Path(temp_dir) / "safe_label.jpg")
            self._write_synthetic_label(input_path)

            ok, message = main.create_pdpa_safe_image(input_path, output_path)

            self.assertTrue(ok, message)
            safe_image = cv2.imread(output_path)
            self.assertIsNotNone(safe_image)
            self.assertEqual(safe_image.shape[:2], (520, 820))
            self._assert_top_masked_and_body_readable(safe_image)

    def test_create_pdpa_safe_image_uses_conservative_mask_when_divider_is_missing(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "label_without_divider.jpg")
            output_path = str(Path(temp_dir) / "safe_label.jpg")
            self._write_synthetic_label(input_path, include_divider=False)

            ok, message = main.create_pdpa_safe_image(input_path, output_path)

            self.assertTrue(ok, message)
            safe_image = cv2.imread(output_path)
            self.assertIsNotNone(safe_image)
            self.assertEqual(safe_image.shape[:2], (520, 820))
            self._assert_top_masked_and_body_readable(safe_image)

    def test_create_pdpa_safe_image_detects_internal_divider_inside_label_border(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "label_on_background.jpg")
            output_path = str(Path(temp_dir) / "safe_label.jpg")
            self._write_label_on_background_with_internal_divider(input_path)

            ok, message = main.create_pdpa_safe_image(input_path, output_path)

            self.assertTrue(ok, message)
            safe_image = cv2.imread(output_path)
            self.assertIsNotNone(safe_image)
            self.assertEqual(safe_image.shape[:2], (720, 960))
            self._assert_top_masked_and_body_readable(safe_image)

    def test_create_pdpa_safe_image_uses_label_bounds_fallback_when_divider_is_not_detectable(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "label_on_background_without_divider.jpg")
            output_path = str(Path(temp_dir) / "safe_label.jpg")
            self._write_label_on_background_without_detectable_divider(input_path)

            ok, message = main.create_pdpa_safe_image(input_path, output_path)

            self.assertTrue(ok, message)
            safe_image = cv2.imread(output_path)
            self.assertIsNotNone(safe_image)
            self.assertEqual(safe_image.shape[:2], (720, 960))
            self._assert_top_masked_and_body_readable(safe_image)

    def test_create_pdpa_safe_image_detects_thin_divider_on_white_background(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "thin_divider_label.jpg")
            output_path = str(Path(temp_dir) / "safe_label.jpg")
            self._write_white_background_label_with_thin_divider(input_path)

            ok, message = main.create_pdpa_safe_image(input_path, output_path)

            self.assertTrue(ok, message)
            safe_image = cv2.imread(output_path)
            self.assertIsNotNone(safe_image)
            gray = cv2.cvtColor(safe_image, cv2.COLOR_BGR2GRAY)
            masked_rows = np.where(np.mean(gray < 20, axis=1) > 0.95)[0]

            self.assertGreater(masked_rows.size, 0)
            self.assertLess(int(masked_rows[-1]), 205)
            self._assert_top_masked_and_body_readable(safe_image)

    def test_normalize_label_image_for_ai_resizes_large_image_and_preserves_color(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "large_label.jpg")
            output_path = str(Path(temp_dir) / "normalized_label.jpg")
            self._write_large_low_contrast_label(input_path)

            ok, message = main.normalize_label_image_for_ai(input_path, output_path)

            self.assertTrue(ok, message)
            normalized = cv2.imread(output_path)
            self.assertIsNotNone(normalized)
            self.assertEqual(len(normalized.shape), 3)
            self.assertLessEqual(normalized.shape[1], 1800)
            self.assertGreaterEqual(normalized.shape[1], 1000)

    def test_normalized_image_can_be_pdpa_masked_without_hiding_medicine_name(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "large_label.jpg")
            normalized_path = str(Path(temp_dir) / "normalized_label.jpg")
            safe_path = str(Path(temp_dir) / "safe_label.jpg")
            self._write_large_low_contrast_label(input_path)

            ok, message = main.normalize_label_image_for_ai(input_path, normalized_path)
            self.assertTrue(ok, message)
            ok, message = main.create_pdpa_safe_image(normalized_path, safe_path)

            self.assertTrue(ok, message)
            safe_image = cv2.imread(safe_path)
            self.assertIsNotNone(safe_image)
            gray = cv2.cvtColor(safe_image, cv2.COLOR_BGR2GRAY)
            masked_rows = np.where(np.mean(gray < 20, axis=1) > 0.95)[0]

            self.assertGreater(masked_rows.size, 0)
            self.assertLess(int(masked_rows[-1]), int(safe_image.shape[0] * 0.38))
            self._assert_top_masked_and_body_readable(safe_image)

    def test_check_image_quality_allows_moderate_glare_for_user_experience(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "label_with_glare.jpg")
            self._write_synthetic_label_with_moderate_glare(input_path)

            ok, message = main.check_image_quality(input_path)

            self.assertTrue(ok, message)


if __name__ == "__main__":
    unittest.main()
