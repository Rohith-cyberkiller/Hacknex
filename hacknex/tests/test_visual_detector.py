"""Unit and contract tests for Visual Forensics Deepfake Detector (Phase 2)."""

import os
from pathlib import Path
import unittest
import numpy as np
import torch

from visual_detector import VisualDetector, detect_visual_authenticity
from src.face_extraction import FaceDetector, align_face_similarity_transform
from src.types import AlignedFace, DetectionResult, FacialLandmarks
from src.utils import (
    create_overlay_image,
    create_synthetic_sample_image,
    load_image,
    save_aligned_face_crops,
)

ROOT_DIR = Path(__file__).resolve().parent.parent


class TestVisualDetectorPhase2(unittest.TestCase):
    """Test suite verifying Phase 2 real face extraction, landmark alignment, and edge cases."""

    @classmethod
    def setUpClass(cls):
        cls.detector = VisualDetector(verbose=False, device="cpu", target_size=256)
        cls.portrait_path = ROOT_DIR / "samples" / "portrait_face.jpg"
        cls.multi_faces_path = ROOT_DIR / "samples" / "multi_faces.jpg"
        cls.no_face_path = ROOT_DIR / "samples" / "sample.jpg"

    def test_single_face_detection_and_alignment(self):
        """Verify real detection, 5 landmarks, and 256x256 crop on single portrait."""
        self.assertTrue(self.portrait_path.exists(), "Test asset portrait_face.jpg missing.")
        result = self.detector.detect(str(self.portrait_path))

        self.assertIsInstance(result, DetectionResult)
        self.assertTrue(result.face_detected)
        self.assertEqual(len(result.bounding_boxes), 1)
        self.assertEqual(len(result.faces), 1)

        face: AlignedFace = result.faces[0]
        # Check box coordinates
        x1, y1, x2, y2 = face.box.as_tuple()
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)
        self.assertGreaterEqual(face.confidence, 0.70)

        # Check landmarks
        self.assertIsNotNone(face.landmarks)
        self.assertIsInstance(face.landmarks, FacialLandmarks)
        lms_arr = face.landmarks.as_array()
        self.assertEqual(lms_arr.shape, (5, 2))

        # Check aligned crops
        self.assertEqual(face.crop_bgr.shape, (256, 256, 3))
        self.assertEqual(face.crop_rgb.shape, (256, 256, 3))
        self.assertEqual(face.crop_bgr.dtype, np.uint8)

        # Check normalized PyTorch tensor
        self.assertIsNotNone(face.normalized_tensor)
        self.assertIsInstance(face.normalized_tensor, torch.Tensor)
        self.assertEqual(list(face.normalized_tensor.shape), [3, 256, 256])
        self.assertEqual(face.normalized_tensor.dtype, torch.float32)

    def test_multi_face_detection(self):
        """Verify multi-face detection parses both faces independently."""
        self.assertTrue(self.multi_faces_path.exists(), "Test asset multi_faces.jpg missing.")
        result = self.detector.detect(str(self.multi_faces_path))

        self.assertIsInstance(result, DetectionResult)
        self.assertTrue(result.face_detected)
        self.assertEqual(len(result.bounding_boxes), 2)
        self.assertEqual(len(result.faces), 2)

        # Each face should have distinct bounding boxes
        box1 = result.faces[0].box.as_tuple()
        box2 = result.faces[1].box.as_tuple()
        self.assertNotEqual(box1, box2)

    def test_no_face_handling_graceful(self):
        """Verify no-face image returns clear status and does not crash."""
        self.assertTrue(self.no_face_path.exists(), "Test asset sample.jpg missing.")
        result = self.detector.detect(str(self.no_face_path))

        self.assertIsInstance(result, DetectionResult)
        self.assertFalse(result.face_detected)
        self.assertEqual(len(result.bounding_boxes), 0)
        self.assertEqual(len(result.faces), 0)
        self.assertEqual(result.predicted_label, "NO_FACE_DETECTED")
        self.assertEqual(result.visual_authenticity_score, 1.0)

    def test_blank_array_no_face(self):
        """Verify completely black image array gracefully returns 0 faces."""
        black_img = np.zeros((300, 300, 3), dtype=np.uint8)
        result = self.detector.detect(black_img)

        self.assertIsInstance(result, DetectionResult)
        self.assertFalse(result.face_detected)
        self.assertEqual(len(result.bounding_boxes), 0)

    def test_backward_compatible_three_field_contract(self):
        """Ensure Phase 1 public contract is strictly preserved."""
        result = self.detector.detect(str(self.portrait_path))

        # 1. visual_authenticity_score float [0.0, 1.0]
        self.assertIsInstance(result.visual_authenticity_score, float)
        self.assertGreaterEqual(result.visual_authenticity_score, 0.0)
        self.assertLessEqual(result.visual_authenticity_score, 1.0)

        # 2. bounding_boxes list of 4-tuples
        self.assertIsInstance(result.bounding_boxes, list)
        for box in result.bounding_boxes:
            self.assertEqual(len(box), 4)

        # 3. gradcam_heatmap_array 2D numpy array
        self.assertIsInstance(result.gradcam_heatmap_array, np.ndarray)
        self.assertEqual(result.gradcam_heatmap_array.ndim, 2)

    def test_save_aligned_crops_utility(self):
        """Verify saving aligned crops to an output directory."""
        result = self.detector.detect(str(self.portrait_path))
        temp_crops_dir = ROOT_DIR / "outputs" / "temp_test_crops"
        saved = save_aligned_face_crops(result.faces, temp_crops_dir)

        self.assertEqual(len(saved), 1)
        self.assertTrue(saved[0].exists())

        # Cleanup
        if saved[0].exists():
            saved[0].unlink()
        if temp_crops_dir.exists():
            temp_crops_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
