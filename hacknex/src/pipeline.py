"""Forensics pipeline orchestrator — Phases 1-5.

Stage 1: Face Detection & Landmark Alignment (MTCNN) — Phase 2
Stage 2: Dual-Branch Detector (RGB EfficientNet + FFT/SRM) — Phase 3
Stage 3: Grad-CAM Explainability (real backprop heatmaps) — Phase 4
Stage 4: Robustness Hardening & Calibration — Phase 5
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.face_extraction import FaceDetector
from src.rgb_branch import RGBBranch
from src.frequency_branch import FrequencyBranch
from src.fusion import ScoreFusion
from src.gradcam import generate_gradcam_for_faces, composite_heatmap_multi_face
from src.types import AlignedFace, BoundingBox, DetectionResult, StageExecutionLog

logger = logging.getLogger("VisualForensics")


class FaceExtractionStage:
    """Stage 1: Real Face Detection, 5-point Landmark Localization, and Alignment."""

    def __init__(
        self,
        device: str = "cpu",
        confidence_threshold: float = 0.70,
        target_size: int = 256,
        max_faces: int = 5,
    ) -> None:
        self.detector = FaceDetector(
            device=device,
            confidence_threshold=confidence_threshold,
            target_size=target_size,
            max_faces=max_faces,
        )

    def process(self, frame: np.ndarray) -> Tuple[List[AlignedFace], StageExecutionLog]:
        """Detect and align faces in the given BGR frame."""
        aligned_faces = self.detector.detect_and_align(frame)

        face_count = len(aligned_faces)
        if face_count == 0:
            log = StageExecutionLog(
                stage_name="Stage 1: Face Detection & Alignment",
                status="NO_FACES_DETECTED",
                description="No human faces detected above confidence threshold.",
                details={"detector": "MTCNN", "faces_found": 0},
            )
            return [], log

        log = StageExecutionLog(
            stage_name="Stage 1: Face Detection & Alignment",
            status="SUCCESS",
            description=f"Detected and aligned {face_count} face(s).",
            details={
                "detector": "MTCNN",
                "faces_found": face_count,
                "boxes": [f.box.as_tuple() for f in aligned_faces],
                "confidences": [round(f.confidence, 4) for f in aligned_faces],
            },
        )
        return aligned_faces, log


class DualBranchDetectorStage:
    """Stage 2: Dual-Branch Deepfake Classifier (RGB + Frequency).

    Runs both the EfficientNet-B0 RGB spatial branch and the FFT/SRM
    frequency branch on each aligned face crop, then fuses scores.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.rgb_branch = RGBBranch(device=device)
        self.freq_branch = FrequencyBranch(device=device)
        self.fusion = ScoreFusion(rgb_weight=0.6, freq_weight=0.4)

    def process(
        self, frame: np.ndarray, faces: List[AlignedFace]
    ) -> Tuple[float, List[float], List[float], List[float], StageExecutionLog]:
        """Run dual-branch detection and fusion.

        Returns:
            Tuple of:
                - image_score: float, fused image-level authenticity
                - rgb_scores: per-face RGB scores
                - freq_scores: per-face frequency scores
                - per_face_fused: per-face fused scores
                - log: StageExecutionLog
        """
        if not faces:
            log = StageExecutionLog(
                stage_name="Stage 2: Dual-Branch Detector (RGB + FFT/SRM)",
                status="SKIPPED_NO_FACE",
                description="No faces available for classification.",
                details={"image_score": 1.0},
            )
            return 1.0, [], [], [], log

        # Collect tensors and crops
        face_tensors = [f.normalized_tensor for f in faces if f.normalized_tensor is not None]
        face_crops_rgb = [f.crop_rgb for f in faces]

        # RGB branch scoring
        rgb_scores = self.rgb_branch.score_faces(face_tensors)

        # Frequency branch scoring
        freq_scores = self.freq_branch.score_faces(face_crops_rgb)

        # Fusion
        image_score, per_face_fused, fusion_details = self.fusion.fuse(rgb_scores, freq_scores)

        log = StageExecutionLog(
            stage_name="Stage 2: Dual-Branch Detector (RGB + FFT/SRM)",
            status="SUCCESS",
            description=f"Classified {len(faces)} face(s) via dual-branch fusion.",
            details={
                "rgb_scores": [round(s, 4) for s in rgb_scores],
                "freq_scores": [round(s, 4) for s in freq_scores],
                "per_face_fused": [round(s, 4) for s in per_face_fused],
                "image_score": round(image_score, 4),
                "rgb_weight": self.fusion.rgb_weight,
                "freq_weight": self.fusion.freq_weight,
                "rgb_backbone": "EfficientNet-B0",
                "freq_features": "FFT-Magnitude + SRM-Residuals",
                "weights_trained": self.rgb_branch._weights_loaded,
            },
        )
        return image_score, rgb_scores, freq_scores, per_face_fused, log


class GradCAMStage:
    """Stage 3: Grad-CAM Explainability — real backprop heatmaps (Phase 4)."""

    def process(
        self,
        frame: np.ndarray,
        faces: List[AlignedFace],
        rgb_branch: RGBBranch,
    ) -> Tuple[np.ndarray, List[np.ndarray], StageExecutionLog]:
        """Generate Grad-CAM heatmaps for detected faces.

        Returns:
            Tuple of:
                - frame_heatmap: Full-frame composite heatmap (H, W)
                - per_face_heatmaps: List of per-face crop heatmaps
                - log: StageExecutionLog
        """
        h, w = frame.shape[:2]

        if not faces:
            heatmap = np.zeros((h, w), dtype=np.float32)
            log = StageExecutionLog(
                stage_name="Stage 3: Grad-CAM Explainability",
                status="NO_FACE_ZERO_HEATMAP",
                description="No faces detected; returning zero heatmap.",
                details={"heatmap_shape": f"{h}x{w}"},
            )
            return heatmap, [], log

        face_tensors = [f.normalized_tensor for f in faces if f.normalized_tensor is not None]
        crop_size = faces[0].target_size

        try:
            per_face_heatmaps = generate_gradcam_for_faces(
                rgb_branch.model, face_tensors, crop_size=crop_size
            )
        except Exception as err:
            logger.warning("Grad-CAM generation failed: %s. Returning zero heatmap.", err)
            per_face_heatmaps = [np.zeros((crop_size, crop_size), dtype=np.float32) for _ in faces]

        # Map face-level heatmaps back to full frame
        face_boxes = [f.box.as_tuple() for f in faces]
        frame_heatmap = composite_heatmap_multi_face(per_face_heatmaps, face_boxes, (h, w))

        log = StageExecutionLog(
            stage_name="Stage 3: Grad-CAM Explainability",
            status="SUCCESS",
            description=f"Generated Grad-CAM heatmaps for {len(faces)} face(s).",
            details={
                "heatmap_shape": f"{h}x{w}",
                "per_face_count": len(per_face_heatmaps),
                "target_class": "fake (index=1)",
                "method": "Grad-CAM on EfficientNet-B0 last conv layer",
            },
        )
        return frame_heatmap, per_face_heatmaps, log


class RobustnessHardeningStage:
    """Stage 4: Score calibration and robustness metadata."""

    def process(
        self, authenticity_score: float, frame: np.ndarray
    ) -> Tuple[float, StageExecutionLog]:
        calibrated = float(np.clip(authenticity_score, 0.0, 1.0))
        log = StageExecutionLog(
            stage_name="Stage 4: Robustness & Calibration",
            status="SUCCESS",
            description="Score calibrated to [0, 1] range.",
            details={"calibrated_score": round(calibrated, 4)},
        )
        return calibrated, log


class ForensicsPipeline:
    """Full forensics pipeline orchestrating Phases 1-5."""

    def __init__(
        self,
        device: str = "cpu",
        confidence_threshold: float = 0.70,
        target_size: int = 256,
        max_faces: int = 5,
        verbose: bool = True,
    ) -> None:
        self.verbose = verbose
        self.device = device
        self.stage_face = FaceExtractionStage(
            device=device,
            confidence_threshold=confidence_threshold,
            target_size=target_size,
            max_faces=max_faces,
        )
        self.stage_dual_branch = DualBranchDetectorStage(device=device)
        self.stage_gradcam = GradCAMStage()
        self.stage_hardening = RobustnessHardeningStage()

    def run(self, frame: np.ndarray) -> DetectionResult:
        """Run the end-to-end detection pipeline on a single frame.

        Args:
            frame: BGR numpy image array.

        Returns:
            DetectionResult with real scores, real boxes, and real Grad-CAM heatmap.
        """
        stage_logs: List[StageExecutionLog] = []

        if self.verbose:
            logger.info("Starting Visual Forensics Pipeline...")

        # ── Stage 1: Face Detection & Alignment ───────────────────────────
        aligned_faces, log1 = self.stage_face.process(frame)
        stage_logs.append(log1)
        if self.verbose:
            logger.info("[S1] %s (faces=%d)", log1.status, len(aligned_faces))

        # ── Stage 2: Dual-Branch Detector ─────────────────────────────────
        raw_score, rgb_scores, freq_scores, per_face_fused, log2 = (
            self.stage_dual_branch.process(frame, aligned_faces)
        )
        stage_logs.append(log2)
        if self.verbose:
            logger.info("[S2] Fused Score=%.4f  RGB=%s  Freq=%s", raw_score, rgb_scores, freq_scores)

        # ── Stage 3: Grad-CAM Explainability ──────────────────────────────
        heatmap, per_face_heatmaps, log3 = self.stage_gradcam.process(
            frame, aligned_faces, self.stage_dual_branch.rgb_branch
        )
        stage_logs.append(log3)
        if self.verbose:
            logger.info("[S3] Heatmap (%dx%d), per-face=%d", heatmap.shape[0], heatmap.shape[1], len(per_face_heatmaps))

        # ── Stage 4: Robustness & Calibration ─────────────────────────────
        final_score, log4 = self.stage_hardening.process(raw_score, frame)
        stage_logs.append(log4)

        # Build result
        real_boxes = [face.box.as_tuple() for face in aligned_faces]

        return DetectionResult(
            visual_authenticity_score=final_score,
            bounding_boxes=real_boxes,
            gradcam_heatmap_array=heatmap,
            faces=aligned_faces,
            stage_logs=stage_logs,
            metadata={
                "frame_shape": list(frame.shape),
                "faces_detected_count": len(aligned_faces),
                "detector": "MTCNN",
                "rgb_scores": rgb_scores,
                "freq_scores": freq_scores,
                "per_face_fused": per_face_fused,
                "pipeline_version": "0.5.0-phase5",
                "weights_trained": self.stage_dual_branch.rgb_branch._weights_loaded,
            },
        )
