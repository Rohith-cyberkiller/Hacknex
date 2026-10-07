"""Pydantic request and response schemas for Cross-Modal Forensics."""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class SuspiciousSegment(BaseModel):
    start_sec: float = Field(..., description="Start timestamp of anomaly in seconds")
    end_sec: float = Field(..., description="End timestamp of anomaly in seconds")
    severity: Literal["low", "medium", "high"] = Field(..., description="Severity of the synchronization drift")
    reason: str = Field(..., description="Forensic rationale for flagging the segment")


class TimelinePoint(BaseModel):
    timestamp_sec: float = Field(..., description="Timestamp in seconds")
    mouth_energy: float = Field(..., description="Normalized mouth motion energy (0.0 to 1.0)")
    speech_energy: float = Field(..., description="Normalized speech acoustic energy (0.0 to 1.0)")
    is_suspicious: bool = Field(False, description="Flag indicating if the point falls in a drift window")


class LipSyncResult(BaseModel):
    score: float = Field(..., description="Lip-sync consistency score from 0.0 to 100.0")
    estimated_offset_ms: int = Field(..., description="Audio-video temporal offset in milliseconds")
    correlation: float = Field(..., description="Pearson or normalized cross-correlation (-1.0 to 1.0)")
    analysis_mode: str = Field("heuristic", description="'heuristic' or 'pretrained'")
    suspicious_segments: List[SuspiciousSegment] = Field(default_factory=list, description="List of detected drift periods")
    timeline_series: Optional[List[TimelinePoint]] = Field(None, description="Detailed sampled timeline series")


class FusionWeights(BaseModel):
    visual_contribution: float = Field(..., description="Relative contribution weight of visual fake probability")
    audio_contribution: float = Field(..., description="Relative contribution weight of audio fake probability")
    lip_sync_contribution: float = Field(..., description="Relative contribution weight of cross-modal lip-sync signal")
    visual_quality: float = Field(..., description="Visual stream quality metric (0.0 to 1.0)")
    audio_quality: float = Field(..., description="Audio stream quality metric (0.0 to 1.0)")


class VideoAnalysisResponse(BaseModel):
    analysis_id: str = Field(..., description="Unique UUID for this analysis session")
    classification: Literal["REAL", "SUSPICIOUS", "FAKE"] = Field(..., description="Categorical forensic verdict")
    authenticity_score: float = Field(..., description="Calibrated authenticity rating (0.0 to 100.0, higher = authentic)")
    final_fake_probability: float = Field(..., description="Calibrated fake probability (0.0 to 100.0)")
    confidence: Literal["low", "medium", "high"] = Field(..., description="Forensic confidence rating")
    lip_sync: LipSyncResult = Field(..., description="Cross-modal synchronization metrics")
    fusion: FusionWeights = Field(..., description="Quality-aware multimodal fusion weighting")
    evidence: List[str] = Field(..., description="Human-readable forensic findings")
    limitations: List[str] = Field(..., description="Forensic boundaries and analytical caveats")


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    ffmpeg_available: bool = True
    default_model: str = "heuristic"
    torch_available: bool = False
