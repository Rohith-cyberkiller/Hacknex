"""Unit tests for MediaPipe face tracking, Kaggle downloader, and audio features."""
import os
import pytest
import numpy as np

from core.video.face_mesh import MediaPipeFaceMeshTracker
from core.datasets.downloader import BENCHMARK_DATASETS, check_kaggle_credentials
from core.audio.processor import load_audio_waveform, compute_speech_energy_profile


def test_face_mesh_instantiation_and_graceful_handling():
    """Verify MediaPipe Face Mesh tracker initializes and safely handles blank frames."""
    tracker = MediaPipeFaceMeshTracker()
    blank_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    
    # Blank black frame should yield None rather than crashing
    result = tracker.process_frame(blank_frame)
    assert result is None or isinstance(result, dict)
    
    # Close method must be idempotent and safe
    tracker.close()


def test_kaggle_dataset_definitions():
    """Verify Kaggle benchmark datasets configuration."""
    assert "fakeavceleb" in BENCHMARK_DATASETS
    assert "dfdc" in BENCHMARK_DATASETS
    assert "voxceleb" in BENCHMARK_DATASETS
    assert "slug" in BENCHMARK_DATASETS["fakeavceleb"]

    # Credential check must return boolean without throwing
    creds_exist = check_kaggle_credentials()
    assert isinstance(creds_exist, bool)


def test_audio_energy_with_synthetic_speech():
    """Verify speech energy profiling computes reasonable bounds."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    # 2 Hz amplitude-modulated tone simulating words
    waveform = np.sin(2 * np.pi * 220 * t) * (0.5 + 0.4 * np.sin(2 * np.pi * 2.0 * t))
    waveform = waveform.astype(np.float32)

    fps = 20.0
    timestamps = np.linspace(0, duration, int(fps * duration))

    energy, quality = compute_speech_energy_profile(waveform, sr, timestamps, fps)
    assert len(energy) == len(timestamps)
    assert 0.0 <= np.min(energy) <= np.max(energy) <= 1.0
    assert 0.0 <= quality <= 1.0
