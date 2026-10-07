"""FaceForensics++ integration and forensic verification harness.

References:
- Rossler et al. (ICCV 2019): FaceForensics++: Learning to Detect Manipulated Facial Images
  (https://github.com/ondyari/FaceForensics)
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

from backend.service import ForensicAnalysisService
from core.models.schemas import VideoAnalysisResponse

logger = logging.getLogger(__name__)

# FaceForensics++ canonical manipulation categories
FF_METHODS = ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
FF_COMPRESSIONS = ["raw", "c23", "c40"]


class FaceForensicsVerifier:
    """Evaluates the cross-modal forensic pipeline against FaceForensics++ video samples."""

    def __init__(self, service: Optional[ForensicAnalysisService] = None):
        self.service = service or ForensicAnalysisService()

    def run_verification(
        self,
        dataset_dir: str,
        max_samples_per_class: int = 10,
        output_report_path: str = "data/verification_report.json",
    ) -> Dict[str, Any]:
        """Verify the forensic pipeline on pristine vs manipulated video subsets.
        
        Args:
            dataset_dir: Path to FaceForensics directory containing 'original' and manipulated folders.
            max_samples_per_class: Max video files to evaluate per method.
            output_report_path: Target path for the verification benchmark report.
            
        Returns:
            Dictionary containing AUC, classification accuracy, and per-class stats.
        """
        data_path = Path(dataset_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"FaceForensics directory not found at: {dataset_dir}")

        y_true = []
        y_pred_probs = []
        y_pred_classes = []
        class_stats: Dict[str, Dict[str, Any]] = {}

        for method in FF_METHODS:
            method_dir = data_path / method
            if not method_dir.exists():
                logger.debug(f"Subdirectory {method_dir} not found; skipping.")
                continue

            video_files = list(method_dir.glob("*.mp4")) + list(method_dir.glob("*.avi"))
            video_files = video_files[:max_samples_per_class]

            if not video_files:
                continue

            is_fake = 0 if method == "original" else 1
            method_scores = []
            method_offsets = []

            for v in video_files:
                try:
                    with open(v, "rb") as f:
                        v_bytes = f.read()

                    report = self.service.analyze_video_file(
                        video_bytes=v_bytes,
                        filename=v.name,
                    )

                    y_true.append(is_fake)
                    y_pred_probs.append(report.final_fake_probability / 100.0)
                    y_pred_classes.append(1 if report.classification in ["FAKE", "SUSPICIOUS"] else 0)

                    method_scores.append(report.authenticity_score)
                    method_offsets.append(report.lip_sync.estimated_offset_ms)
                except Exception as e:
                    logger.warning(f"Error evaluating sample {v.name}: {e}")

            if method_scores:
                class_stats[method] = {
                    "sample_count": len(method_scores),
                    "mean_authenticity": round(float(np.mean(method_scores)), 2),
                    "mean_offset_ms": round(float(np.mean(method_offsets)), 1),
                }

        if len(set(y_true)) < 2:
            auc = 0.5
        else:
            auc = float(roc_auc_score(y_true, y_pred_probs))

        acc = float(accuracy_score(y_true, y_pred_classes))
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, average="binary", zero_division=0)

        results = {
            "dataset": "FaceForensics++",
            "evaluated_samples": len(y_true),
            "auc_score": round(auc, 4),
            "accuracy": round(acc, 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "categories": class_stats,
            "forensic_mode": "quality_aware_multimodal_fusion",
        }

        # Save verification report
        out_file = Path(output_report_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"FaceForensics verification complete: AUC={results['auc_score']}, Acc={results['accuracy']}")
        return results


def print_faceforensics_download_instructions():
    """Print official FaceForensics++ academic download protocol instructions."""
    print(
        "\n=======================================================\n"
        " FaceForensics++ Benchmark Access Instructions\n"
        "=======================================================\n"
        "Reference: https://github.com/ondyari/FaceForensics\n\n"
        "1. Submit the academic access request form at:\n"
        "   https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5Xd1ToSgah83ULeNO9Jf2N1-IX_sJqqWb_DnJlWn-A/viewform\n"
        "2. You will receive an official download script: 'download-FaceForensics.py'\n"
        "3. Download talking-head video samples using:\n"
        "   python download-FaceForensics.py data/faceforensics -d Deepfakes -c c23 -t videos\n"
        "   python download-FaceForensics.py data/faceforensics -d original_sequences/youtube -c c23 -t videos\n"
        "4. Run verification benchmark:\n"
        "   python -m core.datasets.faceforensics --dataset-dir data/faceforensics\n"
        "=======================================================\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FaceForensics++ verification harness.")
    parser.add_argument("--dataset-dir", type=str, default="data/faceforensics")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--instructions", action="store_true")
    args = parser.parse_args()

    if args.instructions or not os.path.exists(args.dataset_dir):
        print_faceforensics_download_instructions()
    else:
        verifier = FaceForensicsVerifier()
        rep = verifier.run_verification(args.dataset_dir, max_samples_per_class=args.max_samples)
        print("\n--- Verification Report ---")
        print(json.dumps(rep, indent=2))
