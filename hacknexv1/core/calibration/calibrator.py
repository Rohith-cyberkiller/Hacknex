"""Probability calibration using temperature scaling with persistence and safe fallback."""
import os
import json
import logging
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class TemperatureCalibrator:
    """Calibrates uncalibrated forensic fake probabilities using Temperature Scaling.
    
    Temperature scaling applies a scalar parameter T > 0 to logits:
        logit = ln(p / (1 - p))
        scaled_logit = logit / T + bias
        calibrated_p = sigmoid(scaled_logit)
        
    T > 1 alleviates overconfidence (common in neural and heuristic detectors).
    T < 1 sharpens underconfident scores.
    """

    def __init__(
        self,
        temperature: float = 1.15,
        bias: float = 0.0,
        model_path: Optional[str] = None,
    ):
        self.temperature = float(temperature)
        self.bias = float(bias)
        self.model_path = model_path
        self.is_custom_loaded = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            logger.info(
                f"Using safe default TemperatureCalibrator (T={self.temperature:.2f}, bias={self.bias:.2f})."
            )

    def calibrate(self, uncalibrated_prob: float) -> float:
        """Calibrate a single probability value in range [0.0, 100.0] or [0.0, 1.0].
        
        Returns calibrated probability on scale [0.0, 100.0].
        """
        # Normalize input to [0.0, 1.0]
        if uncalibrated_prob > 1.0:
            p = uncalibrated_prob / 100.0
        else:
            p = uncalibrated_prob

        # Numerical safety clamping
        p = np.clip(p, 0.001, 0.999)

        # Compute logit
        logit = np.log(p / (1.0 - p))

        # Apply temperature and bias
        t = max(self.temperature, 0.05)
        scaled_logit = (logit / t) + self.bias

        # Inverse sigmoid
        calibrated_01 = 1.0 / (1.0 + np.exp(-scaled_logit))
        calibrated_100 = float(calibrated_01 * 100.0)

        return round(float(np.clip(calibrated_100, 0.0, 100.0)), 1)

    def calibrate_array(self, probs: np.ndarray) -> np.ndarray:
        """Calibrate an array of probabilities in [0.0, 1.0]."""
        p = np.clip(probs, 0.001, 0.999)
        logits = np.log(p / (1.0 - p))
        t = max(self.temperature, 0.05)
        scaled_logits = (logits / t) + self.bias
        return 1.0 / (1.0 + np.exp(-scaled_logits))

    def save(self, filepath: str) -> None:
        """Persist calibration parameters to JSON artifact."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "temperature": self.temperature,
            "bias": self.bias,
            "calibrator_type": "temperature_scaling",
            "version": "1.0",
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved calibrator artifact to {filepath}")

    def load(self, filepath: str) -> None:
        """Load calibration parameters from JSON artifact."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.temperature = float(data.get("temperature", 1.15))
            self.bias = float(data.get("bias", 0.0))
            self.is_custom_loaded = True
            logger.info(
                f"Successfully loaded calibrator from {filepath} (T={self.temperature:.2f}, bias={self.bias:.2f})"
            )
        except Exception as e:
            logger.warning(f"Failed to load calibrator from {filepath}: {e}. Retaining safe defaults.")
            self.temperature = 1.15
            self.bias = 0.0
            self.is_custom_loaded = False
