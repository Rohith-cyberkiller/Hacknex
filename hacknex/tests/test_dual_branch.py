"""Unit and contract tests for Phase 3 Dual-Branch Detector and Score Fusion."""

import unittest
import numpy as np
import torch

from src.rgb_branch import RGBBranch, RGBClassifier
from src.frequency_branch import (
    FrequencyBranch,
    compute_fft_magnitude,
    apply_srm_filters,
    prepare_frequency_features,
)
from src.fusion import ScoreFusion


class TestRGBBranch(unittest.TestCase):
    """Test suite for RGB spatial branch (EfficientNet-B0)."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = RGBClassifier(pretrained_backbone=False)
        cls.rgb_branch = RGBBranch(device="cpu")

    def test_rgb_classifier_forward(self):
        """Test forward pass of RGBClassifier with dummy tensor."""
        dummy_tensor = torch.randn(2, 3, 256, 256)
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(dummy_tensor)
        self.assertEqual(logits.shape, (2, 2))

    def test_rgb_feature_map_extraction(self):
        """Test feature map extraction for Grad-CAM."""
        dummy_tensor = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            feat = self.classifier.get_feature_map(dummy_tensor)
        # EfficientNet-B0 feature map has 1280 channels and 8x8 spatial resolution
        self.assertEqual(feat.shape, (1, 1280, 8, 8))

    def test_rgb_branch_score_faces(self):
        """Test score_faces wrapper on synthetic tensors."""
        tensors = [torch.randn(3, 256, 256) for _ in range(3)]
        scores = self.rgb_branch.score_faces(tensors)
        self.assertEqual(len(scores), 3)
        for s in scores:
            self.assertIsInstance(s, float)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_rgb_branch_empty_faces(self):
        """Test score_faces handles empty input gracefully."""
        scores = self.rgb_branch.score_faces([])
        self.assertEqual(scores, [])


class TestFrequencyBranch(unittest.TestCase):
    """Test suite for Frequency domain branch (FFT + SRM)."""

    @classmethod
    def setUpClass(cls):
        cls.freq_branch = FrequencyBranch(device="cpu")

    def test_fft_magnitude_computation(self):
        """Test 2D FFT magnitude spectrum output dimensions and range."""
        gray = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        mag = compute_fft_magnitude(gray)
        self.assertEqual(mag.shape, (256, 256))
        self.assertGreaterEqual(mag.min(), 0.0)
        self.assertLessEqual(mag.max(), 1.0)

    def test_srm_residual_extraction(self):
        """Test SRM 3-channel residual filter output shape."""
        gray = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        srm = apply_srm_filters(gray)
        self.assertEqual(srm.shape, (3, 256, 256))
        self.assertEqual(srm.dtype, np.float32)

    def test_prepare_frequency_features(self):
        """Test prepare_frequency_features dictionary extraction."""
        crop = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        feats = prepare_frequency_features(crop)
        self.assertIn("fft_tensor", feats)
        self.assertIn("srm_tensor", feats)
        self.assertEqual(feats["fft_tensor"].shape, (1, 256, 256))
        self.assertEqual(feats["srm_tensor"].shape, (3, 256, 256))


    def test_frequency_branch_score_faces(self):
        """Test frequency branch scoring on RGB crops."""
        crops = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(2)]
        scores = self.freq_branch.score_faces(crops)
        self.assertEqual(len(scores), 2)
        for s in scores:
            self.assertIsInstance(s, float)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_frequency_branch_empty(self):
        """Test frequency branch on empty input."""
        scores = self.freq_branch.score_faces([])
        self.assertEqual(scores, [])


class TestScoreFusion(unittest.TestCase):
    """Test suite for score fusion and multi-face aggregation."""

    def test_weight_validation(self):
        """Ensure invalid weights raise AssertionError."""
        with self.assertRaises(AssertionError):
            ScoreFusion(rgb_weight=0.7, freq_weight=0.7)

    def test_per_face_fusion_calculation(self):
        """Verify weighted average calculation."""
        fusion = ScoreFusion(rgb_weight=0.6, freq_weight=0.4)
        rgb_scores = [0.8, 0.4]
        freq_scores = [0.6, 0.2]
        fused = fusion.fuse_per_face(rgb_scores, freq_scores)
        # Face 1: 0.6 * 0.8 + 0.4 * 0.6 = 0.48 + 0.24 = 0.72
        # Face 2: 0.6 * 0.4 + 0.4 * 0.2 = 0.24 + 0.08 = 0.32
        self.assertAlmostEqual(fused[0], 0.72, places=4)
        self.assertAlmostEqual(fused[1], 0.32, places=4)

    def test_aggregation_min(self):
        """Verify min aggregation returns most suspicious score."""
        fusion = ScoreFusion(aggregation="min")
        self.assertEqual(fusion.aggregate_image_score([0.9, 0.3, 0.7]), 0.3)

    def test_aggregation_mean(self):
        """Verify mean aggregation."""
        fusion = ScoreFusion(aggregation="mean")
        self.assertAlmostEqual(fusion.aggregate_image_score([0.8, 0.4]), 0.6)

    def test_no_face_aggregation(self):
        """Verify empty faces return 1.0 (authentic / no face manipulation)."""
        fusion = ScoreFusion()
        self.assertEqual(fusion.aggregate_image_score([]), 1.0)


if __name__ == "__main__":
    unittest.main()
