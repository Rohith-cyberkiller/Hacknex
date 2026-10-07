"""Temperature Scaling Probability Calibration Trainer modeled after SCLBD/DeepfakeBench.

References:
- SCLBD/DeepfakeBench (NeurIPS 2023): Unified Deepfake Detection Benchmark
  (https://github.com/SCLBD/DeepfakeBench)
- Guo et al. (ICML 2017): On Calibration of Modern Neural Networks
"""
import os
import json
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, brier_score_loss

from core.calibration.calibrator import TemperatureCalibrator

logger = logging.getLogger(__name__)


def compute_expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) across M confidence bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(labels[in_bin])
            bin_conf = np.mean(probs[in_bin])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


class CalibrationTrainer:
    """Fits temperature scaling parameters on forensic validation predictions."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def fit(
        self,
        uncalibrated_probs: np.ndarray,
        ground_truth_labels: np.ndarray,
        fit_bias: bool = False,
    ) -> Tuple[TemperatureCalibrator, Dict[str, Any]]:
        """Find optimal temperature T > 0 minimizing Negative Log-Likelihood (NLL).
        
        Args:
            uncalibrated_probs: Array of raw model probabilities in [0.0, 1.0].
            ground_truth_labels: Binary ground truth (1 = fake, 0 = real).
            fit_bias: If True, also fits a shift bias b.
            
        Returns:
            (calibrator, diagnostic_metrics)
        """
        p = np.clip(uncalibrated_probs, 1e-4, 1.0 - 1e-4)
        y = ground_truth_labels.astype(float)
        logits = np.log(p / (1.0 - p))

        # Objective function: Negative Log-Likelihood
        def nll_objective(params):
            if fit_bias:
                t, b = params
            else:
                t = params[0]
                b = 0.0

            t = max(t, 0.05)
            scaled = (logits / t) + b
            prob_cal = 1.0 / (1.0 + np.exp(-scaled))
            prob_cal = np.clip(prob_cal, 1e-6, 1.0 - 1e-6)
            loss = -np.mean(y * np.log(prob_cal) + (1.0 - y) * np.log(1.0 - prob_cal))
            return loss

        init_params = [1.15, 0.0] if fit_bias else [1.15]
        bounds = [(0.1, 5.0), (-2.0, 2.0)] if fit_bias else [(0.1, 5.0)]

        res = minimize(nll_objective, init_params, bounds=bounds, method="L-BFGS-B")
        best_t = float(res.x[0])
        best_b = float(res.x[1]) if fit_bias else 0.0

        calibrator = TemperatureCalibrator(temperature=best_t, bias=best_b)

        # Evaluate metrics before and after calibration
        calibrated_p = calibrator.calibrate_array(p)

        metrics = {
            "temperature": round(best_t, 3),
            "bias": round(best_b, 3),
            "uncalibrated_nll": round(float(log_loss(y, p)), 4),
            "calibrated_nll": round(float(log_loss(y, calibrated_p)), 4),
            "uncalibrated_ece": round(compute_expected_calibration_error(p, y, self.n_bins), 4),
            "calibrated_ece": round(compute_expected_calibration_error(calibrated_p, y, self.n_bins), 4),
            "uncalibrated_brier": round(float(brier_score_loss(y, p)), 4),
            "calibrated_brier": round(float(brier_score_loss(y, calibrated_p)), 4),
        }

        logger.info(f"Calibration finished: ECE improved {metrics['uncalibrated_ece']} -> {metrics['calibrated_ece']}")
        return calibrator, metrics

    def generate_synthetic_benchmark_dataset(
        self, n_samples: int = 500, random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic uncalibrated validation scores with overconfidence bias."""
        rng = np.random.RandomState(random_seed)
        labels = rng.binomial(1, 0.5, size=n_samples)

        # Overconfident predictions (beta distribution skewed toward 0 or 1)
        raw_probs = np.zeros(n_samples)
        for i in range(n_samples):
            if labels[i] == 1:
                # Fakes peaked near 0.85 - 0.98
                raw_probs[i] = rng.beta(6, 1.5)
            else:
                # Real peaked near 0.02 - 0.15
                raw_probs[i] = rng.beta(1.5, 6)

        return raw_probs, labels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit Temperature Calibrator on Deepfake benchmark predictions.")
    parser.add_argument("--save-path", type=str, default="core/calibration/artifacts/calibrator.json")
    parser.add_argument("--fit-bias", action="store_true")
    args = parser.parse_args()

    trainer = CalibrationTrainer()
    raw_p, y_true = trainer.generate_synthetic_benchmark_dataset(n_samples=1000)

    calib, stats = trainer.fit(raw_p, y_true, fit_bias=args.fit_bias)
    calib.save(args.save_path)

    print("\n--- DeepfakeBench Calibration Results ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"Calibrator artifact saved to: {args.save_path}\n")
