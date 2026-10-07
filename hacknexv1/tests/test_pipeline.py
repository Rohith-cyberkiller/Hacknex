"""End-to-end forensic pipeline test with synthetic audiovisual media."""
import os
import cv2
import pytest
import numpy as np
from scipy.io import wavfile
import imageio_ffmpeg
import subprocess

from backend.service import ForensicAnalysisService
from core.models.schemas import VideoAnalysisResponse


def create_synthetic_test_video(tmp_path) -> str:
    """Create a 3-second test MP4 containing a talking head with audio."""
    fps = 20
    duration = 3
    sr = 16000
    n_frames = fps * duration

    v_path = str(tmp_path / "raw_video.avi")
    a_path = str(tmp_path / "raw_audio.wav")
    out_mp4 = str(tmp_path / "test_talking_head.mp4")

    # 1. Video stream
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(v_path, fourcc, fps, (240, 240))
    for i in range(n_frames):
        frame = np.ones((240, 240, 3), dtype=np.uint8) * 40
        # Face ellipse
        cv2.ellipse(frame, (120, 120), (60, 80), 0, 0, 360, (180, 200, 220), -1)
        # Eyes
        cv2.circle(frame, (95, 90), 8, (30, 30, 40), -1)
        cv2.circle(frame, (145, 90), 8, (30, 30, 40), -1)
        # Mouth
        mouth_h = int(8 + 6 * np.sin(2 * np.pi * 1.5 * (i / fps)))
        cv2.ellipse(frame, (120, 160), (18, mouth_h), 0, 0, 360, (60, 70, 180), -1)
        vw.write(frame)
    vw.release()

    # 2. Audio stream
    t = np.linspace(0, duration, sr * duration, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 200 * t) * (0.5 + 0.4 * np.sin(2 * np.pi * 1.5 * t))
    audio_int16 = (sig * 30000).astype(np.int16)
    wavfile.write(a_path, sr, audio_int16)

    # 3. Combine with FFmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", v_path,
        "-i", a_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        out_mp4,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return out_mp4


def test_end_to_end_analysis_and_cleanup(tmp_path):
    """Verify complete pipeline execution, output JSON schema, and cleanup."""
    video_file = create_synthetic_test_video(tmp_path)
    with open(video_file, "rb") as f:
        video_bytes = f.read()

    service = ForensicAnalysisService()
    response = service.analyze_video_file(
        video_bytes=video_bytes,
        filename="test_sample.mp4",
        visual_fake_probability=35.0,
        audio_fake_probability=25.0,
        visual_quality=0.85,
        audio_quality=0.90,
    )

    # Check response instance
    assert isinstance(response, VideoAnalysisResponse)
    data = response.model_dump()

    # Verify exact required output JSON shape
    assert "analysis_id" in data
    assert data["classification"] in ["REAL", "SUSPICIOUS", "FAKE"]
    assert 0.0 <= data["authenticity_score"] <= 100.0
    assert 0.0 <= data["final_fake_probability"] <= 100.0
    assert data["confidence"] in ["low", "medium", "high"]

    # Lip sync block
    assert "lip_sync" in data
    assert 0.0 <= data["lip_sync"]["score"] <= 100.0
    assert isinstance(data["lip_sync"]["estimated_offset_ms"], int)
    assert -1.0 <= data["lip_sync"]["correlation"] <= 1.0
    assert data["lip_sync"]["analysis_mode"] == "heuristic"
    assert isinstance(data["lip_sync"]["suspicious_segments"], list)

    # Fusion block
    assert "fusion" in data
    assert "visual_contribution" in data["fusion"]
    assert "audio_contribution" in data["fusion"]
    assert "lip_sync_contribution" in data["fusion"]
    assert "visual_quality" in data["fusion"]
    assert "audio_quality" in data["fusion"]

    # Evidence and limitations
    assert len(data["evidence"]) >= 1
    assert len(data["limitations"]) >= 1

    # Verify report was cached
    retrieved = service.get_report(response.analysis_id)
    assert retrieved is not None
    assert retrieved.analysis_id == response.analysis_id

    # Verify temp_media directory contains no leftover files for this analysis_id
    from backend.config import settings
    leftovers = list(settings.temp_dir.glob(f"*{response.analysis_id}*"))
    assert len(leftovers) == 0
