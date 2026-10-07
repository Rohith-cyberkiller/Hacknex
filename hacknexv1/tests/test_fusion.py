"""Unit tests for quality-aware multimodal fusion engine."""
import pytest
from core.models.schemas import LipSyncResult, SuspiciousSegment
from core.fusion.quality_fusion import QualityAwareMultimodalFusion
from core.calibration.calibrator import TemperatureCalibrator


@pytest.fixture
def dummy_lipsync_result():
    return LipSyncResult(
        score=75.0,
        estimated_offset_ms=20,
        correlation=0.65,
        analysis_mode="heuristic",
        suspicious_segments=[],
    )


@pytest.fixture
def fusion_engine():
    calibrator = TemperatureCalibrator(temperature=1.0)  # Neutral T for deterministic testing
    return QualityAwareMultimodalFusion(calibrator=calibrator)


def test_fusion_with_all_modalities_equal_quality(fusion_engine, dummy_lipsync_result):
    """Test fusion when all detectors provide scores at standard quality."""
    fake_prob, auth, verdict, conf, weights, evidence, limits = fusion_engine.fuse(
        lip_sync_result=dummy_lipsync_result,
        visual_fake_prob=20.0,
        audio_fake_prob=30.0,
        visual_quality=1.0,
        audio_quality=1.0,
    )

    # Weights must sum to 1.0
    total_weights = weights.visual_contribution + weights.audio_contribution + weights.lip_sync_contribution
    assert abs(total_weights - 1.0) < 0.02
    assert weights.visual_contribution > 0.0
    assert weights.audio_contribution > 0.0
    assert weights.lip_sync_contribution > 0.0
    assert verdict in ["REAL", "SUSPICIOUS", "FAKE"]
    assert len(evidence) > 0


def test_quality_attenuation_audio_degraded(fusion_engine, dummy_lipsync_result):
    """Degrading audio quality must attenuate audio detector contribution."""
    _, _, _, _, weights_high, _, _ = fusion_engine.fuse(
        lip_sync_result=dummy_lipsync_result,
        visual_fake_prob=20.0,
        audio_fake_prob=80.0,
        visual_quality=1.0,
        audio_quality=1.0,
    )

    _, _, _, _, weights_low_audio, _, _ = fusion_engine.fuse(
        lip_sync_result=dummy_lipsync_result,
        visual_fake_prob=20.0,
        audio_fake_prob=80.0,
        visual_quality=1.0,
        audio_quality=0.2,
    )

    # Audio contribution should drop significantly when audio quality is 0.2
    assert weights_low_audio.audio_contribution < weights_high.audio_contribution


def test_quality_attenuation_visual_degraded(fusion_engine, dummy_lipsync_result):
    """Degrading visual quality must attenuate visual detector contribution."""
    _, _, _, _, weights_high, _, _ = fusion_engine.fuse(
        lip_sync_result=dummy_lipsync_result,
        visual_fake_prob=80.0,
        audio_fake_prob=20.0,
        visual_quality=1.0,
        audio_quality=1.0,
    )

    _, _, _, _, weights_low_vis, _, _ = fusion_engine.fuse(
        lip_sync_result=dummy_lipsync_result,
        visual_fake_prob=80.0,
        audio_fake_prob=20.0,
        visual_quality=0.2,
        audio_quality=1.0,
    )

    # Visual contribution should drop when visual quality is 0.2
    assert weights_low_vis.visual_contribution < weights_high.visual_contribution


def test_missing_upstream_scores(fusion_engine, dummy_lipsync_result):
    """When Member 1 and Member 2 scores are absent, lip-sync contributes 100%."""
    fake_prob, auth, verdict, conf, weights, evidence, limits = fusion_engine.fuse(
        lip_sync_result=dummy_lipsync_result,
        visual_fake_prob=None,
        audio_fake_prob=None,
        visual_quality=0.9,
        audio_quality=0.9,
    )

    assert weights.visual_contribution == 0.0
    assert weights.audio_contribution == 0.0
    assert weights.lip_sync_contribution == 1.0


def test_classification_thresholds(fusion_engine):
    """Verify REAL, SUSPICIOUS, and FAKE threshold transitions."""
    # Highly authentic
    real_ls = LipSyncResult(
        score=95.0, estimated_offset_ms=10, correlation=0.8,
        analysis_mode="heuristic", suspicious_segments=[]
    )
    fake_p, auth, verdict_real, _, _, _, _ = fusion_engine.fuse(
        lip_sync_result=real_ls, visual_fake_prob=10.0, audio_fake_prob=10.0
    )
    assert verdict_real == "REAL"
    assert auth > 65.0

    # Highly fake
    fake_ls = LipSyncResult(
        score=10.0, estimated_offset_ms=300, correlation=-0.2,
        analysis_mode="heuristic", suspicious_segments=[
            SuspiciousSegment(start_sec=1.0, end_sec=4.0, severity="high", reason="Desynced")
        ]
    )
    fake_p, auth, verdict_fake, _, _, _, _ = fusion_engine.fuse(
        lip_sync_result=fake_ls, visual_fake_prob=90.0, audio_fake_prob=85.0
    )
    assert verdict_fake == "FAKE"
    assert fake_p > 65.0
