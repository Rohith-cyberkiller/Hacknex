"""Video frame extraction, primary face tracking, and mouth region cropping."""
import os
import logging
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np

from core.video.face_mesh import MediaPipeFaceMeshTracker

logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    """Raised when video frames cannot be read or processed."""
    pass


class NoFaceDetectedError(VideoProcessingError):
    """Raised when no human face is detected across the video."""
    pass


class ShortVideoError(VideoProcessingError):
    """Raised when the video has too few frames for forensic analysis."""
    pass


class VideoFrameExtractor:
    """Extracts frames, tracks primary faces, and crops mouth regions.
    
    Combines MediaPipe Face Mesh for sub-pixel lip geometry with OpenCV
    chrominance and anatomical fallbacks.
    """

    def __init__(self, target_fps: int = 20):
        self.target_fps = target_fps
        self.face_cascade = None
        self.mesh_tracker = MediaPipeFaceMeshTracker()

        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(cascade_path):
                try:
                    cascade = cv2.CascadeClassifier(cascade_path)
                    if not cascade.empty():
                        self.face_cascade = cascade
                except Exception as e:
                    logger.debug(f"Could not initialize CascadeClassifier: {e}")

    def _detect_face_box(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Locate primary face bounding box (x, y, w, h) in frame using CV techniques."""
        h, w = frame_bgr.shape[:2]

        # 1. Try Haar Cascade if available
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=(int(w * 0.15), int(h * 0.15)),
            )
            if len(faces) > 0:
                return max(faces, key=lambda b: b[2] * b[3])

        # 2. Chrominance skin-color segmentation (YCrCb color space)
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 175, 127]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > (h * w * 0.03):
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = bh / max(bw, 1)
                if 0.7 <= aspect <= 2.5:
                    candidates.append((bx, by, bw, bh, area))

        if candidates:
            best = max(candidates, key=lambda x: x[4])
            return best[0], best[1], best[2], best[3]

        # 3. Canonical central talking-head framing fallback
        cx = int(w * 0.20)
        cy = int(h * 0.15)
        cw = int(w * 0.60)
        ch = int(h * 0.65)
        return cx, cy, cw, ch

    def extract(
        self, video_path: str
    ) -> Tuple[List[np.ndarray], np.ndarray, float, float, Dict[str, Any]]:
        """Extract frames, mouth patches, timestamps, and visual quality metric.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            mouth_crops: List of cropped grayscale mouth region frames (resized to 96x96).
            timestamps: 1D numpy array of timestamps (seconds).
            actual_fps: Effective sample rate.
            visual_quality: Estimated visual quality metric (0.0 to 1.0).
            metadata: Diagnostic stats.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoProcessingError(f"OpenCV failed to open video file: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_orig_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if orig_fps <= 0 or np.isnan(orig_fps):
            orig_fps = 25.0

        step = max(1, int(round(orig_fps / self.target_fps)))
        effective_fps = orig_fps / step

        mouth_crops: List[np.ndarray] = []
        timestamps_list: List[float] = []
        blur_scores: List[float] = []
        lar_series: List[float] = []

        frame_idx = 0
        face_detected_count = 0
        mesh_tracked_count = 0
        last_known_mouth_box: Optional[Tuple[int, int, int, int]] = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % step == 0:
                    t_sec = frame_idx / orig_fps
                    h, w = frame.shape[:2]
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Prioritize MediaPipe Face Mesh for lip tracking
                    mesh_info = self.mesh_tracker.process_frame(frame)
                    mouth_box = None

                    if mesh_info is not None:
                        mesh_tracked_count += 1
                        face_detected_count += 1
                        mouth_box = mesh_info["mouth_box"]
                        lar_series.append(mesh_info["lip_aspect_ratio"])
                        last_known_mouth_box = mouth_box
                    else:
                        # Fallback to CV face detector + anatomical box
                        face_box = self._detect_face_box(frame)
                        if face_box is not None:
                            face_detected_count += 1
                            fx, fy, fw, fh = face_box
                            mx = max(0, fx + int(0.18 * fw))
                            my = max(0, fy + int(0.62 * fh))
                            mw = min(w - mx, int(0.64 * fw))
                            mh = min(h - my, int(0.36 * fh))
                            mouth_box = (mx, my, mw, mh)
                            last_known_mouth_box = mouth_box
                        elif last_known_mouth_box is not None:
                            mouth_box = last_known_mouth_box
                        else:
                            mouth_box = (int(w * 0.3), int(h * 0.6), int(w * 0.4), int(h * 0.25))

                    mx, my, mw, mh = mouth_box
                    if mw > 10 and mh > 10:
                        mouth_roi = gray[my : my + mh, mx : mx + mw]
                        mouth_resized = cv2.resize(mouth_roi, (96, 96), interpolation=cv2.INTER_AREA)
                    else:
                        mouth_resized = np.zeros((96, 96), dtype=np.uint8)

                    mouth_crops.append(mouth_resized)
                    timestamps_list.append(t_sec)

                    # Visual blur metric: variance of the Laplacian
                    lap_var = float(cv2.Laplacian(mouth_resized, cv2.CV_64F).var())
                    blur_scores.append(lap_var)

                frame_idx += 1
        finally:
            cap.release()
            self.mesh_tracker.close()

        n_sampled = len(mouth_crops)
        if n_sampled < 8:
            raise ShortVideoError(
                f"Video is too short for cross-modal analysis (only {n_sampled} frames sampled, minimum 8 required)."
            )

        face_tracking_ratio = face_detected_count / n_sampled

        # Visual Quality Metric (0.0 to 1.0)
        mean_blur = float(np.mean(blur_scores)) if blur_scores else 50.0
        sharpness_score = np.clip((mean_blur - 20.0) / 150.0, 0.2, 1.0)
        tracking_score = np.clip(face_tracking_ratio, 0.3, 1.0)
        visual_quality = round(float(0.5 * sharpness_score + 0.5 * tracking_score), 2)

        metadata = {
            "total_orig_frames": total_orig_frames,
            "sampled_frames": n_sampled,
            "face_detected_ratio": round(face_tracking_ratio, 2),
            "mesh_tracked_ratio": round(mesh_tracked_count / n_sampled, 2),
            "effective_fps": round(effective_fps, 2),
            "mediapipe_active": mesh_tracked_count > 0,
        }

        return (
            mouth_crops,
            np.array(timestamps_list, dtype=np.float64),
            effective_fps,
            visual_quality,
            metadata,
        )
