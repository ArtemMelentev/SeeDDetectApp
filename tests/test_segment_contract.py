import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from segment_seeds_scan import analyze_image


class AnalyzeImageContractTest(unittest.TestCase):
    def test_returns_structured_payload_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "sample.jpg"
            out_dir = tmp_path / "out"

            image = np.full((1400, 2000, 3), 255, dtype=np.uint8)
            cv2.circle(image, (700, 700), 130, (20, 20, 20), -1)
            cv2.ellipse(image, (1200, 760), (120, 60), 20, 0, 360, (80, 70, 55), -1)
            ok = cv2.imwrite(str(input_path), image)
            self.assertTrue(ok)

            result = analyze_image(str(input_path), str(out_dir), max_pixels=1_200_000)

            self.assertTrue(result.get("ok"))
            self.assertIn("seed_count", result)
            self.assertIn("all_seed_area_px", result)
            self.assertIn("black_seed_count", result)
            self.assertIn("black_seed_area_px", result)
            self.assertIn("processing_ms", result)

            artifacts = result.get("artifacts", {})
            self.assertTrue(Path(artifacts.get("all_mask", "")).exists())
            self.assertTrue(Path(artifacts.get("all_overlay", "")).exists())
            self.assertTrue(Path(artifacts.get("black_mask", "")).exists())
            self.assertTrue(Path(artifacts.get("black_overlay", "")).exists())

    def test_returns_not_found_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            result = analyze_image("missing_file.jpg", str(out_dir))
            self.assertFalse(result.get("ok"))
            self.assertEqual("file_not_found", result.get("error", {}).get("code"))


if __name__ == "__main__":
    unittest.main()
