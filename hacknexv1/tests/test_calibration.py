"""Unit tests for temperature scaling probability calibration."""
import os
import json
import pytest
import numpy as np

from core.calibration.calibrator import TemperatureCalibrator


def test_temperature_scaling_overconfidence_smoothing():
    """T > 1 should smooth extreme overconfident scores toward 50%."""
    calibrator = TemperatureCalibrator(temperature=1.5, bias=0.0)

    # 95% fake raw score
    calibrated_high = calibrator.calibrate(95.0)
    assert calibrated_high < 95.0
    assert calibrated_high > 50.0

    # 5% fake raw score
    calibrated_low = calibrator.calibrate(5.0)
    assert calibrated_low > 5.0
    assert calibrated_low < 50.0


def test_temperature_neutral_preservation():
    """T = 1.0, bias = 0.0 should preserve original probability."""
    calibrator = TemperatureCalibrator(temperature=1.0, bias=0.0)
    for p in [10.0, 30.0, 50.0, 70.0, 90.0]:
        assert abs(calibrator.calibrate(p) - p) < 0.5


def test_boundary_safety():
    """Extreme boundaries 0.0 and 100.0 must not produce NaNs or infinities."""
    calibrator = TemperatureCalibrator(temperature=1.2)
    assert 0.0 <= calibrator.calibrate(0.0) <= 10.0
    assert 90.0 <= calibrator.calibrate(100.0) <= 100.0
    assert not np.isnan(calibrator.calibrate(0.0))
    assert not np.isnan(calibrator.calibrate(100.0))


def test_save_and_load_persistence(tmp_path):
    """Artifact JSON persistence and re-loading."""
    model_file = str(tmp_path / "test_calibrator.json")
    calibrator = TemperatureCalibrator(temperature=1.45, bias=0.1)
    calibrator.save(model_file)

    assert os.path.exists(model_file)

    loaded = TemperatureCalibrator(model_path=model_file)
    assert loaded.is_custom_loaded is True
    assert abs(loaded.temperature - 1.45) < 1e-4
    assert abs(loaded.bias - 0.1) < 1e-4


def test_fallback_on_corrupted_file(tmp_path):
    """Corrupted model file should safely fall back to defaults."""
    bad_file = str(tmp_path / "corrupt.json")
    with open(bad_file, "w") as f:
        f.write("{invalid json")

    calibrator = TemperatureCalibrator(model_path=bad_file)
    assert calibrator.is_custom_loaded is False
    assert calibrator.temperature == 1.15
