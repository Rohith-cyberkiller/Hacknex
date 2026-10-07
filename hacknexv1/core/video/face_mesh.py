"""MediaPipe Face Mesh integration for high-precision lip and face tracking."""
import logging
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except (ImportError, Exception):
    mp = None
    MEDIAPIPE_AVAILABLE = False


# Canonical MediaPipe FaceMesh Landmark Indices for Lips
LIP_INNER_UPPER = 13
LIP_INNER_LOWER = 14
LIP_CORNER_LEFT = 78
LIP_CORNER_RIGHT = 308

LIP_LANDMARKS_ALL = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    95, 88, 178, 87, 14, 317, 402, 318, 324,
]


class MediaPipeFaceMeshTracker:
    """Tracks facial landmarks and extracts articulatory lip metrics using MediaPipe Face Mesh."""

    def __init__(self, refine_landmarks: bool = True):
        self.is_available = MEDIAPIPE_AVAILABLE
        self._mesh = None
        if self.is_available:
            try:
                self._mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=refine_landmarks,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                logger.info("MediaPipe Face Mesh tracker initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize MediaPipe Face Mesh: {e}")
                self.is_available = False

    def process_frame(
        self, frame_bgr: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Extract face bounding box, mouth crop, and lip geometry from a BGR video frame.
        
        Returns:
            Dict containing:
                'face_box': (fx, fy, fw, fh)
                'mouth_box': (mx, my, mw, mh)
                'lip_aspect_ratio': float
                'lip_opening': float
                'lip_landmarks': np.ndarray of shape (K, 2)
            or None if face mesh is unavailable or not detected.
        """
        if not self.is_available or self._mesh is None:
            return None

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        try:
            results = self._mesh.process(frame_rgb)
        except Exception as e:
            logger.debug(f"MediaPipe Face Mesh inference error: {e}")
            return None

        if not results or not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        # Convert normalized coordinates to pixel space
        pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)

        # 1. Overall Face Bounding Box
        fx_min, fy_min = np.min(pts, axis=0)
        fx_max, fy_max = np.max(pts, axis=0)
        fx = max(0, int(fx_min))
        fy = max(0, int(fy_min))
        fw = min(w - fx, int(fx_max - fx_min))
        fh = min(h - fy, int(fy_max - fy_min))

        # 2. Precise Lip Region Bounding Box
        lip_pts = pts[LIP_LANDMARKS_ALL]
        lx_min, ly_min = np.min(lip_pts, axis=0)
        lx_max, ly_max = np.max(lip_pts, axis=0)
        
        # Add 20% margin around lips for optical flow context
        pad_x = int((lx_max - lx_min) * 0.20)
        pad_y = int((ly_max - ly_min) * 0.25)
        
        mx = max(0, int(lx_min - pad_x))
        my = max(0, int(ly_min - pad_y))
        mw = min(w - mx, int(lx_max - lx_min + 2 * pad_x))
        mh = min(h - my, int(ly_max - ly_min + 2 * pad_y))

        # 3. Geometric Articulation Metrics
        # Vertical lip opening: inner upper 13 to inner lower 14
        p13 = pts[LIP_INNER_UPPER]
        p14 = pts[LIP_INNER_LOWER]
        lip_opening = float(np.linalg.norm(p13 - p14))

        # Horizontal mouth width: corner 78 to corner 308
        p78 = pts[LIP_CORNER_LEFT]
        p308 = pts[LIP_CORNER_RIGHT]
        lip_width = float(np.linalg.norm(p78 - p308))

        # Lip Aspect Ratio (LAR)
        lar = lip_opening / (lip_width + 1e-6)

        return {
            "face_box": (fx, fy, fw, fh),
            "mouth_box": (mx, my, mw, mh),
            "lip_opening": lip_opening,
            "lip_aspect_ratio": float(lar),
            "lip_landmarks": lip_pts,
        }

    def close(self):
        if self._mesh is not None:
            try:
                self._mesh.close()
            except Exception:
                pass
