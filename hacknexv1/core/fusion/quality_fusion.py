"""Quality-aware attention / weighted multimodal fusion engine."""
import logging
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from core.models.schemas import FusionWeights, LipSyncResult
from core.calibration.calibrator import TemperatureCalibrator

logger = logging.getLogger(__name__)


class QualityAwareMultimodalFusion:
    """Combines visual, acoustic, and cross-modal lip-sync predictions with quality weighting.
    
    Dynamically adjusts modal contributions based on respective stream fidelity.
    When audio quality degrades, audio weighting is attenuated; when visual quality
    degrades, visual influence is attenuated. Cross-modal lip-sync requires joint fidelity.
    """

    def __init__(
        self,
        base_w_visual: float = 0.35,
        base_w_audio: float = 0.25,
        base_w_lipsync: float = 0.40,
        calibrator: Optional[TemperatureCalibrator] = None,
    ):
        self.base_w_visual = base_w_visual
        self.base_w_audio = base_w_audio
        self.base_w_lipsync = base_w_lipsync
        self.calibrator = calibrator or TemperatureCalibrator()

    def fuse(
        self,
        lip_sync_result: LipSyncResult,
        visual_fake_prob: Optional[float] = None,
        audio_fake_prob: Optional[float] = None,
        visual_quality: float = 1.0,
        audio_quality: float = 1.0,
    ) -> Tuple[float, float, str, str, FusionWeights, List[str], List[str]]:
        """Perform quality-aware multimodal fusion.
        
        Args:
            lip_sync_result: Result from LipSyncModel.
            visual_fake_prob: Fake probability from Member 1 (0 to 100 or 0 to 1).
            audio_fake_prob: Fake probability from Member 2 (0 to 100 or 0 to 1).
            visual_quality: Visual quality score (0.0 to 1.0).
            audio_quality: Audio quality score (0.0 to 1.0).
            
        Returns:
            (final_fake_prob, authenticity_score, classification, confidence,
             fusion_weights, evidence_list, limitations_list)
        """
        # 1. Normalize quality scores to [0.05, 1.0]
        q_v = float(np.clip(visual_quality, 0.05, 1.0))
        q_a = float(np.clip(audio_quality, 0.05, 1.0))

        # Lip-sync fake probability (inverse of lip-sync consistency score)
        ls_fake_prob = float(np.clip(100.0 - lip_sync_result.score, 0.0, 100.0))

        # Normalize upstream scores to [0.0, 100.0] if given in [0.0, 1.0]
        norm_v_fake: Optional[float] = None
        if visual_fake_prob is not None:
            norm_v_fake = visual_fake_prob * 100.0 if visual_fake_prob <= 1.0 and visual_fake_prob > 0 else visual_fake_prob
            norm_v_fake = float(np.clip(norm_v_fake, 0.0, 100.0))

        norm_a_fake: Optional[float] = None
        if audio_fake_prob is not None:
            norm_a_fake = audio_fake_prob * 100.0 if audio_fake_prob <= 1.0 and audio_fake_prob > 0 else audio_fake_prob
            norm_a_fake = float(np.clip(norm_a_fake, 0.0, 100.0))

        # 2. Quality-weighted attention computation
        # Visual weight scales with visual quality
        w_v = (self.base_w_visual * (q_v ** 1.2)) if norm_v_fake is not None else 0.0

        # Audio weight scales with audio quality
        w_a = (self.base_w_audio * (q_a ** 1.2)) if norm_a_fake is not None else 0.0

        # Cross-modal weight depends on joint audiovisual quality
        q_cross = np.sqrt(q_v * q_a)
        w_ls = self.base_w_lipsync * (q_cross ** 1.0)

        # Fallback if all weights are zero
        total_w = w_v + w_a + w_ls
        if total_w < 1e-6:
            w_ls = 1.0
            total_w = 1.0

        # Normalized contributions (summing to 1.0)
        c_v = w_v / total_w
        c_a = w_a / total_w
        c_ls = w_ls / total_w

        # Format contributions nicely to 2 decimals, ensuring exact 1.0 sum
        c_v_round = round(float(c_v), 2)
        c_a_round = round(float(c_a), 2)
        c_ls_round = round(1.0 - (c_v_round + c_a_round), 2)

        # 3. Compute raw fused fake probability
        raw_fake_prob = 0.0
        if norm_v_fake is not None:
            raw_fake_prob += c_v * norm_v_fake
        if norm_a_fake is not None:
            raw_fake_prob += c_a * norm_a_fake
        raw_fake_prob += c_ls * ls_fake_prob

        # 4. Temperature calibration
        calibrated_fake_prob = self.calibrator.calibrate(raw_fake_prob)
        authenticity_score = round(float(np.clip(100.0 - calibrated_fake_prob, 0.0, 100.0)), 1)

        # 5. Forensic classification
        if calibrated_fake_prob > 65.0:
            classification = "FAKE"
        elif calibrated_fake_prob >= 35.0:
            classification = "SUSPICIOUS"
        else:
            classification = "REAL"

        # 6. Confidence rating
        # High confidence requires good quality (mean quality > 0.70) and at least 2 contributing modalities
        modalities_count = 1 + (1 if norm_v_fake is not None else 0) + (1 if norm_a_fake is not None else 0)
        mean_quality = 0.5 * (q_v + q_a)

        if mean_quality >= 0.75 and modalities_count >= 2:
            confidence = "high"
        elif mean_quality >= 0.50:
            confidence = "medium"
        else:
            confidence = "low"

        # 7. Generate evidence list
        evidence: List[str] = []
        if abs(lip_sync_result.estimated_offset_ms) > 60:
            evidence.append(
                f"Detected an estimated {lip_sync_result.estimated_offset_ms} ms audio-video offset."
            )
        else:
            evidence.append(
                f"Audio-video stream timing is closely aligned ({lip_sync_result.estimated_offset_ms} ms offset)."
            )

        if lip_sync_result.suspicious_segments:
            for seg in lip_sync_result.suspicious_segments[:2]:
                evidence.append(
                    f"Lip movement was weakly correlated with spoken-audio energy between "
                    f"{seg.start_sec:.1f} and {seg.end_sec:.1f} seconds."
                )
        else:
            evidence.append(
                f"No persistent temporal drift windows observed across the evaluated speech segments."
            )

        if norm_v_fake is not None:
            if norm_v_fake > 60.0:
                evidence.append(f"Visual forensics reported high manipulation risk ({norm_v_fake:.1f}%).")
            else:
                evidence.append(f"Visual forensics reported low manipulation indicators ({norm_v_fake:.1f}%).")

        if norm_a_fake is not None:
            if norm_a_fake > 60.0:
                evidence.append(f"Audio forensics flagged potential synthetic voice synthesis ({norm_a_fake:.1f}%).")
            else:
                evidence.append(f"Audio forensics reported natural acoustic properties ({norm_a_fake:.1f}%).")

        if q_a < 0.60:
            evidence.append(
                f"Audio quality was degraded (quality index {q_a:.2f}); audio detector influence was dynamically attenuated."
            )
        if q_v < 0.60:
            evidence.append(
                f"Visual quality was degraded (quality index {q_v:.2f}); visual detector influence was dynamically attenuated."
            )

        # 8. Forensic limitations
        limitations: List[str] = [
            "This score is probabilistic forensic evidence, not proof of manipulation.",
            "Cross-modal synchronisation can be influenced by broadcast compression, encoding lag, or non-frontal head poses.",
            f"Quality ratings applied: visual quality {q_v:.2f}, audio quality {q_a:.2f}.",
        ]
        if lip_sync_result.analysis_mode == "heuristic":
            limitations.append(
                "Cross-modal synchrony evaluated in Heuristic analysis mode (optical flow & acoustic envelopes)."
            )

        fusion_weights = FusionWeights(
            visual_contribution=c_v_round,
            audio_contribution=c_a_round,
            lip_sync_contribution=c_ls_round,
            visual_quality=round(q_v, 2),
            audio_quality=round(q_a, 2),
        )

        return (
            calibrated_fake_prob,
            authenticity_score,
            classification,
            confidence,
            fusion_weights,
            evidence,
            limitations,
        )
