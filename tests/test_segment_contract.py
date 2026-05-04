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
            self.assertIn("all_seed_ratio_pct", result)
            self.assertIn("black_seed_ratio_pct", result)
            self.assertIn("black_to_all_seed_ratio_pct", result)
            self.assertIn("processing_ms", result)

            self.assertGreaterEqual(result["all_seed_ratio_pct"], 0.0)
            self.assertLessEqual(result["all_seed_ratio_pct"], 100.0)
            self.assertGreaterEqual(result["black_seed_ratio_pct"], 0.0)
            self.assertLessEqual(result["black_seed_ratio_pct"], 100.0)
            self.assertGreaterEqual(result["black_to_all_seed_ratio_pct"], 0.0)
            self.assertLessEqual(result["black_to_all_seed_ratio_pct"], 100.0)

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

    def test_photo_with_paper_background_produces_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "photo.jpg"
            out_dir = tmp_path / "out"

            # Dark background + bright paper rectangle + a few dark seeds.
            image = np.full((1600, 2400, 3), 30, dtype=np.uint8)
            cv2.rectangle(image, (250, 180), (2150, 1420), (245, 245, 245), -1)
            cv2.ellipse(image, (900, 720), (140, 70), 25, 0, 360, (25, 25, 25), -1)
            cv2.ellipse(image, (1400, 880), (120, 55), -15, 0, 360, (55, 50, 45), -1)
            cv2.ellipse(image, (1200, 560), (80, 45), 10, 0, 360, (15, 15, 15), -1)
            ok = cv2.imwrite(str(input_path), image)
            self.assertTrue(ok)

            result = analyze_image(str(input_path), str(out_dir), max_pixels=1_200_000)
            self.assertTrue(result.get("ok"))
            self.assertGreaterEqual(int(result.get("seed_count", 0)), 1)
            self.assertGreaterEqual(int(result.get("black_seed_count", 0)), 1)

            artifacts = result.get("artifacts", {})
            self.assertTrue(Path(artifacts.get("all_mask", "")).exists())
            self.assertTrue(Path(artifacts.get("all_overlay", "")).exists())
            self.assertTrue(Path(artifacts.get("black_mask", "")).exists())
            self.assertTrue(Path(artifacts.get("black_overlay", "")).exists())

    def test_release_mode_writes_only_black_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "sample.jpg"
            out_dir = tmp_path / "out"

            # Simple synthetic: white paper background + one dark seed.
            image = np.full((1400, 2000, 3), 255, dtype=np.uint8)
            cv2.circle(image, (900, 700), 140, (20, 20, 20), -1)
            ok = cv2.imwrite(str(input_path), image)
            self.assertTrue(ok)

            result = analyze_image(str(input_path), str(out_dir), max_pixels=1_200_000, release_mode=True)
            self.assertTrue(result.get("ok"))

            artifacts = result.get("artifacts", {})
            self.assertEqual({"black_overlay"}, set(artifacts.keys()))
            self.assertTrue(Path(artifacts.get("black_overlay", "")).exists())

            stem = input_path.stem
            self.assertFalse((out_dir / f"{stem}_all_mask.png").exists())
            self.assertFalse((out_dir / f"{stem}_all_overlay.jpg").exists())
            self.assertFalse((out_dir / f"{stem}_black_mask.png").exists())


if __name__ == "__main__":
    unittest.main()
