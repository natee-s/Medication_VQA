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

    def _write_distant_label(self, path: str) -> None:
        image = np.full((900, 1200, 3), 238, dtype=np.uint8)
        label_x, label_y, label_w, label_h = 510, 360, 210, 150
        cv2.rectangle(image, (label_x, label_y), (label_x + label_w, label_y + label_h), (226, 226, 218), -1)
        cv2.rectangle(image, (label_x, label_y), (label_x + label_w, label_y + label_h), (120, 120, 115), 2)
        cv2.putText(image, "BANYA 0612899146", (label_x + 12, label_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)
        cv2.line(image, (label_x + 12, label_y + 55), (label_x + label_w - 14, label_y + 55), (20, 20, 20), 1)
        cv2.putText(image, "HEPALAC", (label_x + 12, label_y + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2)
        cv2.putText(image, "LACTULOSE", (label_x + 12, label_y + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1)
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

    def _write_low_label_with_background_line(self, path: str) -> None:
        image = np.full((1200, 900, 3), (98, 75, 52), dtype=np.uint8)
        for x in range(0, 900, 60):
            cv2.line(image, (x, 0), (x + 120, 1199), (125, 105, 78), 3)

        # Background line that used to be mistaken for the PDPA divider.
        cv2.line(image, (60, 340), (840, 340), (8, 8, 8), 3)

        label_x, label_y, label_w, label_h = 130, 520, 640, 520
        cv2.rectangle(image, (label_x, label_y), (label_x + label_w, label_y + label_h), (232, 232, 222), -1)
        cv2.rectangle(image, (label_x, label_y), (label_x + label_w, label_y + label_h), (170, 170, 160), 3)
        cv2.putText(image, "BANYA SOOKJAI 0612899146", (label_x + 30, label_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 35, 35), 2)
        cv2.putText(image, "Customer allergy history", (label_x + 30, label_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 2)
        cv2.line(image, (label_x + 25, label_y + 155), (label_x + label_w - 35, label_y + 155), (15, 15, 15), 3)
        cv2.putText(image, "PINRONE/NORCA 5 MG", (label_x + 30, label_y + 230), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
        cv2.putText(image, "NORETHISTERONE", (label_x + 30, label_y + 275), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        cv2.imwrite(path, image)

    def _write_plastic_bag_label_with_slanted_divider(self, path: str) -> None:
        image = np.full((900, 1200, 3), 185, dtype=np.uint8)
        cv2.rectangle(image, (0, 0), (1199, 360), (205, 205, 205), -1)
        for y in range(70, 360, 35):
            cv2.line(image, (0, y), (1000, y + 90), (245, 245, 245), 3)
        cv2.ellipse(image, (300, 180), (170, 45), -8, 0, 360, (245, 245, 245), -1)
        cv2.rectangle(image, (245, 220), (1085, 860), (228, 228, 222), -1)
        cv2.rectangle(image, (245, 220), (1085, 860), (155, 155, 150), 3)
        cv2.putText(image, "BANYA SOOKJAI 0612899146", (285, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 35, 35), 2)
        cv2.putText(image, "Customer: allergy history", (285, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 35), 2)
        cv2.putText(image, "No drug allergy", (620, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 35), 2)
        cv2.putText(image, "Customer type", (285, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 35), 2)
        cv2.line(image, (285, 515), (930, 477), (20, 20, 20), 3)
        cv2.putText(image, "ITRAFUNGAL 100 MG 10'S", (285, 620), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 4)
        cv2.putText(image, "ITRACONAZOLE 100 mg", (285, 685), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
        cv2.putText(image, "After meals morning evening", (285, 790), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
        cv2.imwrite(path, image)

    def _write_label_with_medicine_name_close_to_divider(self, path: str) -> None:
        image = np.full((1000, 1200, 3), 235, dtype=np.uint8)
        label_x, label_y, label_w, label_h = 210, 330, 810, 560
        cv2.rectangle(image, (label_x, label_y), (label_x + label_w, label_y + label_h), (226, 226, 218), -1)
        cv2.rectangle(image, (label_x, label_y), (label_x + label_w, label_y + label_h), (170, 170, 160), 3)
        cv2.putText(image, "BANYA SOOKJAI 0612899146", (250, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (35, 35, 35), 2)
        cv2.putText(image, "Customer: allergy history", (250, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (35, 35, 35), 2)
        cv2.putText(image, "General customer", (250, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (35, 35, 35), 2)
        cv2.line(image, (250, 530), (970, 530), (25, 25, 25), 2)
        cv2.putText(image, "DIOCTAHEDRAL SMECTITE", (250, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2)
        cv2.line(image, (250, 610), (920, 610), (25, 25, 25), 2)
        cv2.putText(image, "Treats diarrhea", (250, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 3)
        cv2.putText(image, "Take 2-3 sachets daily", (250, 745), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2)
        cv2.imwrite(path, image)

    def _write_white_background_label_with_thin_divider(self, path: str) -> None:
        image = np.full((520, 820, 3), 245, dtype=np.uint8)
        cv2.putText(image, "BANYA SOOKJAI 0612899146", (70, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        cv2.putText(image, "Customer: allergy history", (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
        cv2.line(image, (70, 165), (610, 165), (120, 120, 120), 1)
        cv2.putText(image, "HEPALAC 100ML", (70, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
        cv2.putText(image, "LACTULOSE", (70, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
        cv2.imwrite(path, image)

    def _write_label_with_false_top_line_and_no_border(self, path: str) -> None:
        image = np.full((900, 1200, 3), 230, dtype=np.uint8)
        cv2.line(image, (130, 105), (1060, 105), (35, 35, 35), 3)
        cv2.putText(image, "BANYA SOOKJAI 0612899146", (270, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 35, 35), 2)
        cv2.putText(image, "Customer: allergy history", (270, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 35), 2)
        cv2.putText(image, "HEPALAC 100ML", (270, 440), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 4)
        cv2.putText(image, "LACTULOSE", (270, 510), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
        cv2.imwrite(path, image)

    def _write_large_low_contrast_label(self, path: str) -> None:
        image = np.full((2400, 3600, 3), 225, dtype=np.uint8)
        cv2.putText(image, "BANYA SOOKJAI 0612899146", (560, 420), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (85, 85, 85), 5)
        cv2.putText(image, "Customer: allergy history", (560, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (90, 90, 90), 4)
        cv2.line(image, (560, 720), (2860, 720), (75, 75, 75), 3)
        cv2.putText(image, "HEPALAC 100ML", (560, 980), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (45, 45, 45), 8)
        cv2.putText(image, "LACTULOSE", (560, 1160), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (45, 45, 45), 7)
        cv2.imwrite(path, image)

    def _write_overmerged_header_before_medicine_name(self) -> np.ndarray:
        image = np.full((900, 1200, 3), 235, dtype=np.uint8)
        cv2.rectangle(image, (210, 140), (1030, 820), (226, 226, 218), -1)
        cv2.rectangle(image, (245, 260), (980, 470), (20, 20, 20), -1)
        cv2.putText(image, "DIOCTAHEDRAL SMECTITE", (250, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 3)
        cv2.putText(image, "Treats diarrhea", (250, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2)
        return image

    def _write_rotated_label_on_background(self, path: str, angle: float = 8.0) -> None:
        label = np.full((520, 820, 3), 235, dtype=np.uint8)
        cv2.rectangle(label, (8, 8), (812, 512), (170, 170, 160), 3)
        cv2.putText(label, "BANYA SOOKJAI 0612899146", (70, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        cv2.putText(label, "Customer: allergy history", (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
        cv2.line(label, (70, 165), (720, 165), (20, 20, 20), 3)
        cv2.putText(label, "HEPALAC 100ML", (70, 235), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
        cv2.putText(label, "LACTULOSE", (70, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)

        height, width = label.shape[:2]
        matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        rotated_width = int((height * sin) + (width * cos))
        rotated_height = int((height * cos) + (width * sin))
        matrix[0, 2] += (rotated_width / 2) - (width // 2)
        matrix[1, 2] += (rotated_height / 2) - (height // 2)
        rotated = cv2.warpAffine(
            label,
            matrix,
            (rotated_width, rotated_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(210, 210, 210),
        )

        image = np.full((900, 1200, 3), (210, 210, 210), dtype=np.uint8)
        y = (image.shape[0] - rotated.shape[0]) // 2
        x = (image.shape[1] - rotated.shape[1]) // 2
        image[y:y + rotated.shape[0], x:x + rotated.shape[1]] = rotated
        cv2.imwrite(path, image)

    def _dominant_horizontal_angle(self, image) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 55, minLineLength=180, maxLineGap=18)
        if lines is None:
            return 0.0

        candidates = []
        line_segments = np.asarray(lines).reshape(-1, 4)
        for x1, y1, x2, y2 in line_segments:
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            length = float(np.hypot(dx, dy))
            angle = float(np.degrees(np.arctan2(dy, dx)))
            if abs(angle) <= 15.0:
                candidates.append((angle, length))

        if not candidates:
            return 0.0

        angles = np.array([angle for angle, _ in candidates])
        weights = np.array([weight for _, weight in candidates])
        return float(np.average(angles, weights=weights))

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

    def test_rectify_then_mask_low_label_with_background_line_uses_standard_crop(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "low_label_with_background_line.jpg")
            rectified_path = str(Path(temp_dir) / "rectified_label.jpg")
            output_path = str(Path(temp_dir) / "safe_label.jpg")
            self._write_low_label_with_background_line(input_path)

            ok, message = main.rectify_label_image_for_ai(input_path, rectified_path)
            self.assertTrue(ok, message)
            ok, message = main.create_pdpa_safe_image(rectified_path, output_path)

            self.assertTrue(ok, message)
            safe_image = cv2.imread(output_path)
            self.assertIsNotNone(safe_image)
            self.assertEqual(safe_image.shape[:2], (main.STANDARD_LABEL_HEIGHT, main.STANDARD_LABEL_WIDTH))
            gray = cv2.cvtColor(safe_image, cv2.COLOR_BGR2GRAY)

            self.assertGreater(np.mean(gray[:main.PDPA_MASK_HEIGHT, :] < 20), 0.95)
            self.assertLess(np.mean(gray[main.PDPA_MASK_HEIGHT + 45:main.PDPA_MASK_HEIGHT + 180, :] < 20), 0.45)

    def test_rectify_then_mask_plastic_bag_label_uses_standard_crop(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "plastic_bag_slanted_divider.jpg")
            rectified_path = str(Path(temp_dir) / "rectified_label.jpg")
            output_path = str(Path(temp_dir) / "safe_label.jpg")
            self._write_plastic_bag_label_with_slanted_divider(input_path)

            ok, message = main.rectify_label_image_for_ai(input_path, rectified_path)
            self.assertTrue(ok, message)
            ok, message = main.create_pdpa_safe_image(rectified_path, output_path)

            self.assertTrue(ok, message)
            safe_image = cv2.imread(output_path)
            self.assertIsNotNone(safe_image)
            self.assertEqual(safe_image.shape[:2], (main.STANDARD_LABEL_HEIGHT, main.STANDARD_LABEL_WIDTH))
            gray = cv2.cvtColor(safe_image, cv2.COLOR_BGR2GRAY)

            self.assertGreater(np.mean(gray[:main.PDPA_MASK_HEIGHT, :] < 20), 0.95)
            self.assertLess(np.mean(gray[main.PDPA_MASK_HEIGHT + 55:main.PDPA_MASK_HEIGHT + 210, :] < 20), 0.45)

    def test_rectify_then_mask_keeps_medicine_name_when_close_to_divider(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "medicine_name_close_to_divider.jpg")
            rectified_path = str(Path(temp_dir) / "rectified_label.jpg")
            output_path = str(Path(temp_dir) / "safe_label.jpg")
            self._write_label_with_medicine_name_close_to_divider(input_path)

            ok, message = main.rectify_label_image_for_ai(input_path, rectified_path)
            self.assertTrue(ok, message)
            ok, message = main.create_pdpa_safe_image(rectified_path, output_path)

            self.assertTrue(ok, message)
            safe_image = cv2.imread(output_path)
            self.assertIsNotNone(safe_image)
            self.assertEqual(safe_image.shape[:2], (main.STANDARD_LABEL_HEIGHT, main.STANDARD_LABEL_WIDTH))
            gray = cv2.cvtColor(safe_image, cv2.COLOR_BGR2GRAY)

            self.assertGreater(np.mean(gray[:main.PDPA_MASK_HEIGHT, :] < 20), 0.99)
            self.assertLess(np.mean(gray[main.PDPA_MASK_HEIGHT + 35:main.PDPA_MASK_HEIGHT + 190, :] < 20), 0.45)

    def test_find_first_large_text_y_ignores_overmerged_header_blocks(self):
        import main

        image = self._write_overmerged_header_before_medicine_name()

        first_text_y = main.find_first_large_text_y(image, 220, 760, (210, 1030))

        self.assertIsNotNone(first_text_y)
        self.assertGreaterEqual(first_text_y, 540)

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

    def test_rectify_then_mask_ignores_false_top_line_before_body_text(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "false_top_line_label.jpg")
            rectified_path = str(Path(temp_dir) / "rectified_label.jpg")
            output_path = str(Path(temp_dir) / "safe_label.jpg")
            self._write_label_with_false_top_line_and_no_border(input_path)

            ok, message = main.rectify_label_image_for_ai(input_path, rectified_path)
            self.assertTrue(ok, message)
            ok, message = main.create_pdpa_safe_image(rectified_path, output_path)

            self.assertTrue(ok, message)
            safe_image = cv2.imread(output_path)
            self.assertIsNotNone(safe_image)
            self.assertEqual(safe_image.shape[:2], (main.STANDARD_LABEL_HEIGHT, main.STANDARD_LABEL_WIDTH))
            gray = cv2.cvtColor(safe_image, cv2.COLOR_BGR2GRAY)

            self.assertGreater(np.mean(gray[:main.PDPA_MASK_HEIGHT, :] < 20), 0.99)
            self.assertLess(np.mean(gray[main.PDPA_MASK_HEIGHT + 50:main.PDPA_MASK_HEIGHT + 230, :] < 20), 0.45)

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

    def test_rectify_label_image_for_ai_reduces_moderate_skew(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "rotated_label.jpg")
            output_path = str(Path(temp_dir) / "rectified_label.jpg")
            self._write_rotated_label_on_background(input_path, angle=8.0)

            before = cv2.imread(input_path)
            ok, message = main.rectify_label_image_for_ai(input_path, output_path, use_yolo_obb=False)

            self.assertTrue(ok, message)
            rectified = cv2.imread(output_path)
            self.assertIsNotNone(rectified)
            self.assertEqual(rectified.shape[:2], (main.STANDARD_LABEL_HEIGHT, main.STANDARD_LABEL_WIDTH))
            self.assertLess(abs(self._dominant_horizontal_angle(rectified)), abs(self._dominant_horizontal_angle(before)) - 3.0)

    def test_yolo_header_orientation_prefers_header_at_top(self):
        import main

        image = np.full((main.STANDARD_LABEL_HEIGHT, main.STANDARD_LABEL_WIDTH, 3), 240, dtype=np.uint8)
        label_quad = np.array(
            [
                [0, 0],
                [main.STANDARD_LABEL_WIDTH - 1, 0],
                [main.STANDARD_LABEL_WIDTH - 1, main.STANDARD_LABEL_HEIGHT - 1],
                [0, main.STANDARD_LABEL_HEIGHT - 1],
            ],
            dtype=np.float32,
        )
        header_quad = np.array(
            [
                [1060, 120],
                [1280, 120],
                [1280, 880],
                [1060, 880],
            ],
            dtype=np.float32,
        )

        rectified, _, transformed_header, orientation = main._warp_label_quad_to_standard_with_header(
            image,
            label_quad,
            header_quad,
        )

        self.assertIsNotNone(rectified)
        self.assertIsNotNone(transformed_header)
        self.assertNotEqual(orientation, "shift_0")
        header_width = np.max(transformed_header[:, 0]) - np.min(transformed_header[:, 0])
        header_height = np.max(transformed_header[:, 1]) - np.min(transformed_header[:, 1])
        header_center_y = (np.max(transformed_header[:, 1]) + np.min(transformed_header[:, 1])) / 2
        self.assertGreater(header_width, header_height)
        self.assertLess(header_center_y, main.STANDARD_LABEL_HEIGHT * 0.36)

    def test_create_pdpa_safe_image_masks_fixed_top_quarter_on_standard_label(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "standard_label.jpg")
            output_path = str(Path(temp_dir) / "safe_label.jpg")
            image = np.full((main.STANDARD_LABEL_HEIGHT, main.STANDARD_LABEL_WIDTH, 3), 235, dtype=np.uint8)
            cv2.putText(image, "BANYA SOOKJAI 0612899146", (70, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 3)
            cv2.putText(image, "HEPALAC 100ML", (70, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 4)
            cv2.putText(image, "LACTULOSE", (70, 410), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
            cv2.imwrite(input_path, image)

            ok, message = main.create_pdpa_safe_image(input_path, output_path)

            self.assertTrue(ok, message)
            safe_image = cv2.imread(output_path)
            self.assertIsNotNone(safe_image)
            self.assertEqual(safe_image.shape[:2], (main.STANDARD_LABEL_HEIGHT, main.STANDARD_LABEL_WIDTH))
            gray = cv2.cvtColor(safe_image, cv2.COLOR_BGR2GRAY)
            self.assertGreater(np.mean(gray[:main.PDPA_MASK_HEIGHT, :] < 20), 0.99)
            self.assertLess(np.mean(gray[main.PDPA_MASK_HEIGHT + 40:main.PDPA_MASK_HEIGHT + 180, :] < 20), 0.30)

    def test_create_pdpa_safe_image_extends_mask_when_header_tail_crosses_boundary(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "standard_label_with_header_tail.jpg")
            output_path = str(Path(temp_dir) / "safe_label.jpg")
            image = np.full((main.STANDARD_LABEL_HEIGHT, main.STANDARD_LABEL_WIDTH, 3), 235, dtype=np.uint8)
            cv2.putText(image, "Customer allergy history", (70, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25, 25, 25), 3)
            cv2.line(image, (70, 285), (1040, 285), (35, 35, 35), 2)
            cv2.putText(image, "MUCOLID 30MG. 10 S.", (70, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (20, 20, 20), 4)
            cv2.imwrite(input_path, image)

            ok, message = main.create_pdpa_safe_image(input_path, output_path)

            self.assertTrue(ok, message)
            safe_image = cv2.imread(output_path)
            self.assertIsNotNone(safe_image)
            gray = cv2.cvtColor(safe_image, cv2.COLOR_BGR2GRAY)
            self.assertGreater(np.mean(gray[260:295, :] < 20), 0.90)
            self.assertLess(np.mean(gray[350:430, :] < 20), 0.35)

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

    def test_check_image_quality_rejects_extremely_distant_label(self):
        import main

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = str(Path(temp_dir) / "distant_label.jpg")
            self._write_distant_label(input_path)

            ok, _ = main.check_image_quality(input_path)

            self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
