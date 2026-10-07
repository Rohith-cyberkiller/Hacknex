"""Client SDK for Member 1 and Member 2 integration with Cross-Modal Deepfake Forensics."""
import os
import requests
from typing import Optional, Dict, Any
from pathlib import Path

from core.models.schemas import VideoAnalysisResponse, HealthResponse


class CrossModalForensicsClient:
    """Python Client SDK for communicating with the Cross-Modal Forensics backend.
    
    Enables Member 1 (Visual Lead) and Member 2 (Audio Lead) to send local detector
    predictions and receive unified, calibrated forensic authenticity reports.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> HealthResponse:
        """Query backend system health and dependency availability."""
        url = f"{self.base_url}/health"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return HealthResponse.model_validate(resp.json())

    def analyze_video(
        self,
        video_path: str,
        visual_fake_probability: Optional[float] = None,
        visual_quality: Optional[float] = None,
        audio_fake_probability: Optional[float] = None,
        audio_quality: Optional[float] = None,
        sample_fps: int = 20,
        model_type: str = "heuristic",
    ) -> VideoAnalysisResponse:
        """Submit a video and upstream forensic scores for multimodal fusion.
        
        Args:
            video_path: Local path to the talking-head video file.
            visual_fake_probability: Member 1 visual fake probability (0-100 or 0-1).
            visual_quality: Member 1 visual quality index (0.0 to 1.0).
            audio_fake_probability: Member 2 audio fake probability (0-100 or 0-1).
            audio_quality: Member 2 audio quality index (0.0 to 1.0).
            sample_fps: Frame sampling rate (default: 20).
            model_type: 'heuristic' (default) or 'pretrained'.
            
        Returns:
            VideoAnalysisResponse with calibrated scores, AV offset, and evidence.
        """
        p = Path(video_path)
        if not p.exists():
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        url = f"{self.base_url}/analyze/video"
        data: Dict[str, Any] = {
            "sample_fps": sample_fps,
            "model_type": model_type,
        }
        if visual_fake_probability is not None:
            data["visual_fake_probability"] = visual_fake_probability
        if visual_quality is not None:
            data["visual_quality"] = visual_quality
        if audio_fake_probability is not None:
            data["audio_fake_probability"] = audio_fake_probability
        if audio_quality is not None:
            data["audio_quality"] = audio_quality

        with open(p, "rb") as f:
            files = {"video": (p.name, f, "video/mp4")}
            resp = requests.post(url, data=data, files=files, timeout=self.timeout)

        resp.raise_for_status()
        return VideoAnalysisResponse.model_validate(resp.json())

    def get_report(self, analysis_id: str) -> VideoAnalysisResponse:
        """Retrieve a stored forensic report by its UUID."""
        url = f"{self.base_url}/report/{analysis_id}"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return VideoAnalysisResponse.model_validate(resp.json())
