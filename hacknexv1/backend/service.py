"""Forensic analysis orchestration service."""
import os
import uuid
import logging
from typing import Optional, Dict
from pathlib import Path

from core.models.schemas import (
    VideoAnalysisResponse,
    LipSyncResult,
)
from core.models.heuristic import HeuristicLipSyncModel
from core.models.syncnet import PretrainedSyncNetAdapter
from core.video.extractor import (
    VideoFrameExtractor,
    VideoProcessingError,
    NoFaceDetectedError,
    ShortVideoError,
)
from core.video.motion import compute_mouth_motion_energy
from core.audio.extractor import (
    extract_audio_from_video,
    AudioExtractionError,
    NoAudioFoundError,
)
from core.audio.processor import (
    load_audio_waveform,
    compute_speech_energy_profile,
)
from core.fusion.quality_fusion import QualityAwareMultimodalFusion
from core.calibration.calibrator import TemperatureCalibrator
from backend.config import settings

logger = logging.getLogger(__name__)

# In-memory storage for analysis reports
REPORTS_CACHE: Dict[str, VideoAnalysisResponse] = {}


class ForensicAnalysisService:
    """End-to-end coordinator for multimodal deepfake forensic examination."""

    def __init__(self):
        self.calibrator = TemperatureCalibrator(
            temperature=settings.temperature,
            model_path=str(settings.calibration_path) if settings.calibration_path.exists() else None,
        )
        self.fusion_engine = QualityAwareMultimodalFusion(calibrator=self.calibrator)
        self.heuristic_model = HeuristicLipSyncModel()
        self.syncnet_adapter = PretrainedSyncNetAdapter()

    def analyze_video_file(
        self,
        video_bytes: bytes,
        filename: str,
        visual_fake_probability: Optional[float] = None,
        audio_fake_probability: Optional[float] = None,
        visual_quality: Optional[float] = None,
        audio_quality: Optional[float] = None,
        sample_fps: Optional[int] = None,
        model_type: Optional[str] = "heuristic",
    ) -> VideoAnalysisResponse:
        """Run complete forensic pipeline and clean up all temporary files."""
        analysis_id = str(uuid.uuid4())
        fps = sample_fps or settings.default_fps
        ext = Path(filename).suffix or ".mp4"
        
        temp_video_path = str(settings.temp_dir / f"input_{analysis_id}{ext}")
        temp_wav_path = str(settings.temp_dir / f"audio_{analysis_id}.wav")

        try:
            # 1. Write uploaded bytes to disk temporarily
            with open(temp_video_path, "wb") as f:
                f.write(video_bytes)

            logger.info(f"[{analysis_id}] Commencing analysis for {filename} ({len(video_bytes)} bytes)")

            # 2. Extract visual stream, detect primary face, crop mouth
            extractor = VideoFrameExtractor(target_fps=fps)
            (
                mouth_crops,
                timestamps,
                effective_fps,
                computed_visual_quality,
                meta,
            ) = extractor.extract(temp_video_path)

            v_qual = visual_quality if visual_quality is not None else computed_visual_quality

            # 3. Extract audio via FFmpeg
            extract_audio_from_video(temp_video_path, temp_wav_path)
            waveform, sr = load_audio_waveform(temp_wav_path)

            # 4. Compute acoustic speech envelope and audio quality
            speech_energy, computed_audio_quality = compute_speech_energy_profile(
                waveform=waveform,
                sr=sr,
                timestamps=timestamps,
                fps=effective_fps,
            )

            a_qual = audio_quality if audio_quality is not None else computed_audio_quality

            # 5. Compute mouth motion dynamics across sampled frames
            mouth_energy = compute_mouth_motion_energy(mouth_crops)

            # 6. Select lip-sync analyzer
            if model_type == "pretrained" and self.syncnet_adapter.is_available():
                active_model = self.syncnet_adapter
            else:
                active_model = self.heuristic_model

            lip_sync_result = active_model.analyze(
                mouth_energy=mouth_energy,
                speech_energy=speech_energy,
                timestamps=timestamps,
                fps=effective_fps,
            )

            # 7. Execute quality-aware multimodal fusion
            (
                final_fake_prob,
                authenticity_score,
                classification,
                confidence,
                fusion_weights,
                evidence,
                limitations,
            ) = self.fusion_engine.fuse(
                lip_sync_result=lip_sync_result,
                visual_fake_prob=visual_fake_probability,
                audio_fake_prob=audio_fake_probability,
                visual_quality=v_qual,
                audio_quality=a_qual,
            )

            response = VideoAnalysisResponse(
                analysis_id=analysis_id,
                classification=classification,
                authenticity_score=authenticity_score,
                final_fake_probability=final_fake_prob,
                confidence=confidence,
                lip_sync=lip_sync_result,
                fusion=fusion_weights,
                evidence=evidence,
                limitations=limitations,
            )

            # Cache report for GET /report/{analysis_id}
            REPORTS_CACHE[analysis_id] = response
            logger.info(f"[{analysis_id}] Analysis finished: verdict={classification}, auth={authenticity_score}")
            return response

        finally:
            # Clean up temporary media
            for p in [temp_video_path, temp_wav_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception as err:
                        logger.warning(f"Failed to delete temporary file {p}: {err}")

    def get_report(self, analysis_id: str) -> Optional[VideoAnalysisResponse]:
        """Retrieve stored report by UUID."""
        return REPORTS_CACHE.get(analysis_id)
