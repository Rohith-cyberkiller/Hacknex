"""Unit tests for calibration training and benchmark verification harness."""
import os
import pytest
import numpy as np

from core.calibration.trainer import CalibrationTrainer, compute_expected_calibration_error
from core.models.syncnet_architecture import SyncNet, TORCH_AVAILABLE


def test_ece_computation():
    """Verify Expected Calibration Error calculation on known predictions."""
    y_true = np.array([1, 1, 0, 0], dtype=int)
    # Perfect calibration
    p_perfect = np.array([1.0, 1.0, 0.0, 0.0])
    ece_zero = compute_expected_calibration_error(p_perfect, y_true, n_bins=5)
    assert abs(ece_zero) < 1e-4

    # Imperfect / overconfident calibration (predicting 0.95 when true labels are 0)
    p_imperfect = np.array([0.95, 0.95, 0.95, 0.95])
    ece_imperfect = compute_expected_calibration_error(p_imperfect, np.array([0, 0, 0, 0]), n_bins=5)
    assert ece_imperfect > 0.5


def test_calibration_trainer_synthetic_fit():
    """Verify CalibrationTrainer optimizes temperature and reduces NLL."""
    trainer = CalibrationTrainer(n_bins=5)
    raw_probs, labels = trainer.generate_synthetic_benchmark_dataset(n_samples=200, random_seed=42)

    calibrator, metrics = trainer.fit(raw_probs, labels, fit_bias=False)
    assert calibrator.temperature > 0.1
    assert "uncalibrated_nll" in metrics
    assert "calibrated_nll" in metrics
    assert "calibrated_ece" in metrics
    # Temperature scaling should reduce or match NLL on overconfident scores
    assert metrics["calibrated_nll"] <= metrics["uncalibrated_nll"] + 0.05


def test_syncnet_architecture_if_torch_available():
    """Verify SyncNet module can instantiate when PyTorch is present."""
    if TORCH_AVAILABLE:
        model = SyncNet()
        assert model is not None
        assert hasattr(model, "audio_encoder")
        assert hasattr(model, "visual_encoder")
    else:
        pytest.skip("PyTorch not installed in environment.")
