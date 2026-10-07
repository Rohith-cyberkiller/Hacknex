"""Integration tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify health endpoint response structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "ffmpeg_available" in data
    assert data["default_model"] == "heuristic"


def test_get_nonexistent_report():
    """Requesting an unknown report ID should return 404."""
    response = client.get("/report/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_analyze_video_empty_file():
    """Uploading an empty file should return 400 error."""
    files = {"video": ("empty.mp4", b"", "video/mp4")}
    response = client.post("/analyze/video", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_analyze_video_invalid_format():
    """Uploading an unallowed file format should return 400 error."""
    files = {"video": ("document.pdf", b"%PDF-1.4...", "application/pdf")}
    response = client.post("/analyze/video", files=files)
    assert response.status_code == 400
    assert "unsupported video format" in response.json()["detail"].lower()
