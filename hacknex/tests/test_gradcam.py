"""Unit tests for Phase 4 Grad-CAM explainability and heatmap compositing."""

import unittest
import numpy as np
import torch

from src.rgb_branch import RGBClassifier
from src.gradcam import (
    GradCAM,
    composite_heatmap_multi_face,
    generate_gradcam_for_faces,
    map_heatmap_to_frame,
)


class TestGradCAM(unittest.TestCase):
    """Test suite for Grad-CAM explainability pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.classifier = RGBClassifier(pretrained_backbone=False)

    def test_gradcam_generation_shape_and_range(self):
        """Test GradCAM output dimensions and [0.0, 1.0] value normalization."""
        gradcam = GradCAM(self.classifier)
        tensor = torch.randn(1, 3, 256, 256)
        heatmap = gradcam.generate(tensor, target_class=1, target_size=(256, 256))
        gradcam.cleanup()

        self.assertEqual(heatmap.shape, (256, 256))
        self.assertEqual(heatmap.dtype, np.float32)
        self.assertGreaterEqual(heatmap.min(), 0.0)
        self.assertLessEqual(heatmap.max(), 1.0)

    def test_generate_gradcam_for_multiple_faces(self):
        """Test batch wrapper generate_gradcam_for_faces."""
        tensors = [torch.randn(3, 256, 256) for _ in range(2)]
        heatmaps = generate_gradcam_for_faces(self.classifier, tensors, crop_size=256)

        self.assertEqual(len(heatmaps), 2)
        for hmap in heatmaps:
            self.assertEqual(hmap.shape, (256, 256))
            self.assertEqual(hmap.dtype, np.float32)

    def test_map_heatmap_to_frame(self):
        """Test mapping a face crop heatmap back to full image coordinate space."""
        crop_hmap = np.ones((256, 256), dtype=np.float32) * 0.8
        box = (50, 100, 150, 200)  # x1, y1, x2, y2 -> w=100, h=100
        frame_shape = (500, 500)

        mapped = map_heatmap_to_frame(crop_hmap, box, frame_shape)
        self.assertEqual(mapped.shape, (500, 500))

        # Check inside box
        region = mapped[100:200, 50:150]
        self.assertAlmostEqual(float(region.mean()), 0.8, places=3)

        # Check outside box (should be zeros)
        outside_val = mapped[0:50, 0:50]
        self.assertEqual(float(outside_val.max()), 0.0)

    def test_composite_heatmap_multi_face(self):
        """Test max-pooling compositing across multiple face bounding boxes."""
        hmap1 = np.ones((256, 256), dtype=np.float32) * 0.4
        hmap2 = np.ones((256, 256), dtype=np.float32) * 0.9

        box1 = (50, 50, 150, 150)
        box2 = (200, 200, 300, 300)

        composite = composite_heatmap_multi_face(
            [hmap1, hmap2],
            [box1, box2],
            (500, 500),
        )

        self.assertEqual(composite.shape, (500, 500))
        self.assertAlmostEqual(float(composite[60, 60]), 0.4, places=3)
        self.assertAlmostEqual(float(composite[210, 210]), 0.9, places=3)
        self.assertEqual(float(composite[0, 0]), 0.0)

    def test_out_of_bounds_face_box_graceful(self):
        """Ensure invalid or negative box coordinates do not throw crashes."""
        crop_hmap = np.ones((256, 256), dtype=np.float32)
        box = (-50, -50, -10, -10)
        mapped = map_heatmap_to_frame(crop_hmap, box, (400, 400))
        self.assertEqual(mapped.shape, (400, 400))
        self.assertEqual(float(mapped.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
