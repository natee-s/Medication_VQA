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


class DebugPdpaImageScriptTests(unittest.TestCase):
    def _write_label(self, path: Path) -> None:
        image = np.full((520, 820, 3), 245, dtype=np.uint8)
        cv2.putText(image, "BANYA SOOKJAI 0612899146", (70, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        cv2.putText(image, "Customer: allergy history", (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
        cv2.line(image, (70, 165), (610, 165), (120, 120, 120), 1)
        cv2.putText(image, "HEPALAC 100ML", (70, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
        cv2.putText(image, "LACTULOSE", (70, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
        cv2.imwrite(str(path), image)

    def test_process_pdpa_debug_image_creates_normalized_and_safe_outputs(self):
        from tools.debug_pdpa_image import process_pdpa_debug_image

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "sample label.jpg"
            output_dir = temp_path / "debug_pdpa"
            self._write_label(input_path)

            result = process_pdpa_debug_image(input_path, output_dir, run_qc=False)

            self.assertTrue(result.normalized_path.exists())
            self.assertTrue(result.safe_path.exists())
            self.assertEqual(result.normalized_path.name, "sample_label_normalized.jpg")
            self.assertEqual(result.safe_path.name, "sample_label_safe.jpg")

            safe_image = cv2.imread(str(result.safe_path))
            self.assertIsNotNone(safe_image)
            gray = cv2.cvtColor(safe_image, cv2.COLOR_BGR2GRAY)
            self.assertGreater(np.mean(gray[:120, :] < 20), 0.85)
            self.assertGreater(np.mean(gray[220:, :] < 80), 0.003)


if __name__ == "__main__":
    unittest.main()
