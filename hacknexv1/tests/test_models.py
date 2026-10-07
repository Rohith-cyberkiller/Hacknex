"""Unit tests for LipSyncModel adapters."""
import pytest
import numpy as np

from core.models.heuristic import HeuristicLipSyncModel
from core.models.syncnet import PretrainedSyncNetAdapter


def test_heuristic_model_properties_and_execution():
    """Verify HeuristicLipSyncModel identifies as heuristic and returns valid LipSyncResult."""
    model = HeuristicLipSyncModel()
    assert model.name == "Heuristic Audiovisual Correlation Model"
    assert model.analysis_mode == "heuristic"
    assert model.is_available() is True

    n = 60
    t = np.linspace(0, 3.0, n)
    mouth = 0.5 + 0.3 * np.sin(2 * np.pi * 1.0 * t)
    speech = mouth.copy()

    result = model.analyze(mouth, speech, t, fps=20.0)
    assert result.analysis_mode == "heuristic"
    assert 0.0 <= result.score <= 100.0
    assert -1.0 <= result.correlation <= 1.0
    assert isinstance(result.estimated_offset_ms, int)
    assert len(result.timeline_series) == n


def test_pretrained_syncnet_fallback_never_deceives():
    """When weights are absent, PretrainedSyncNetAdapter must report 'heuristic' mode."""
    adapter = PretrainedSyncNetAdapter(checkpoint_path="/non/existent/path.pth")
    assert adapter.is_available() is False
    assert adapter.analysis_mode == "heuristic"

    n = 40
    t = np.linspace(0, 2.0, n)
    mouth = np.random.rand(n).astype(np.float32)
    speech = np.random.rand(n).astype(np.float32)

    result = adapter.analyze(mouth, speech, t, fps=20.0)
    # Critical requirement: Never present heuristic results as pretrained
    assert result.analysis_mode == "heuristic"
