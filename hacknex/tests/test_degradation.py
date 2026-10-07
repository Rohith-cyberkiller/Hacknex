"""Unit tests for Phase 5 degradation hardening and robustness evaluation."""

import unittest
import numpy as np

from src.degradation import (
    DEGRADATION_PROFILES,
    apply_all_profiles,
    apply_degradation_profile,
    compute_psnr,
    downscale_upscale,
    gaussian_blur,
    jpeg_compress,
)
from visual_detector import VisualDetector, evaluate_robustness


class TestDegradationHardening(unittest.TestCase):
    """Test suite for social-media degradation simulations."""

    def setUp(self):
        # Create a synthetic color pattern
        self.test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        self.test_img[:100, :100] = [255, 0, 0]
        self.test_img[:100, 100:] = [0, 255, 0]
        self.test_img[100:, :100] = [0, 0, 255]
        self.test_img[100:, 100:] = [255, 255, 255]

    def test_jpeg_compression(self):
        """Test JPEG compression effects on image array and PSNR."""
        q90 = jpeg_compress(self.test_img, quality=90)
        q30 = jpeg_compress(self.test_img, quality=30)

        self.assertEqual(q90.shape, self.test_img.shape)
        self.assertEqual(q30.shape, self.test_img.shape)

        psnr_90 = compute_psnr(self.test_img, q90)
        psnr_30 = compute_psnr(self.test_img, q30)

        # Higher quality should have higher PSNR (less distortion)
        self.assertGreater(psnr_90, psnr_30)

    def test_gaussian_blur(self):
        """Test Gaussian blur maintains shape and reduces sharpness."""
        blurred = gaussian_blur(self.test_img, kernel_size=5)
        self.assertEqual(blurred.shape, self.test_img.shape)
        psnr = compute_psnr(self.test_img, blurred)
        self.assertGreater(psnr, 0.0)

    def test_downscale_upscale(self):
        """Test downscale and upscale cycle restores original resolution."""
        res = downscale_upscale(self.test_img, scale_factor=0.5)
        self.assertEqual(res.shape, self.test_img.shape)
        psnr = compute_psnr(self.test_img, res)
        self.assertGreater(psnr, 0.0)

    def test_psnr_identical(self):
        """Identical images must yield infinite PSNR."""
        self.assertEqual(compute_psnr(self.test_img, self.test_img), float("inf"))

    def test_apply_all_profiles(self):
        """Test that all degradation profiles execute cleanly."""
        results = apply_all_profiles(self.test_img)
        self.assertEqual(len(results), len(DEGRADATION_PROFILES))
        for name, deg in results.items():
            self.assertEqual(deg.degraded_image.shape, self.test_img.shape)
            self.assertIsNotNone(deg.psnr)

    def test_evaluate_robustness_on_blank_frame(self):
        """Test detector robustness evaluation on black frame."""
        detector = VisualDetector(verbose=False, device="cpu")
        report = detector.evaluate_robustness(
            self.test_img,
            profiles=["clean", "jpeg_q70", "blur_k3"],
        )
        self.assertIn("clean", report)
        self.assertIn("jpeg_q70", report)
        self.assertIn("blur_k3", report)
        for p_name, metrics in report.items():
            self.assertIn("score", metrics)
            self.assertIn("score_delta", metrics)
            self.assertIn("label", metrics)


if __name__ == "__main__":
    unittest.main()
