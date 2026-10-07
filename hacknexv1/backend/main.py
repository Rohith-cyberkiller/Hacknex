"""FastAPI server for Cross-Modal Deepfake Forensics."""
import os
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.models.schemas import (
    VideoAnalysisResponse,
    HealthResponse,
)
from core.audio.extractor import get_ffmpeg_executable, AudioExtractionError, NoAudioFoundError
from core.video.extractor import NoFaceDetectedError, ShortVideoError, VideoProcessingError
from backend.service import ForensicAnalysisService
from backend.config import settings

# Structured logging configuration
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("forensics_api")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Member 3 Forensic Engine: Cross-modal audiovisual synchronization analysis "
        "and calibrated quality-aware multimodal deepfake fusion."
    ),
)

# Enable CORS for cross-origin frontend and member integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = ForensicAnalysisService()


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check and forensic dependency verification."""
    ffmpeg_exe = get_ffmpeg_executable()
    has_ffmpeg = ffmpeg_exe is not None

    torch_avail = False
    try:
        import torch
        torch_avail = True
    except ImportError:
        pass

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        ffmpeg_available=has_ffmpeg,
        default_model=settings.default_model,
        torch_available=torch_avail,
    )


@app.post(
    "/analyze/video",
    response_model=VideoAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Forensics"],
)
async def analyze_video(
    video: UploadFile = File(..., description="Target talking-head video file (MP4, MOV, AVI, WebM)"),
    visual_fake_probability: Optional[float] = Form(None, description="Optional visual fake probability from Member 1 (0-100 or 0-1)"),
    audio_fake_probability: Optional[float] = Form(None, description="Optional audio fake probability from Member 2 (0-100 or 0-1)"),
    visual_quality: Optional[float] = Form(None, description="Optional visual stream quality metric (0.0 to 1.0)"),
    audio_quality: Optional[float] = Form(None, description="Optional audio stream quality metric (0.0 to 1.0)"),
    sample_fps: Optional[int] = Form(20, description="Sampling rate in frames per second (default 20)"),
    model_type: Optional[str] = Form("heuristic", description="'heuristic' or 'pretrained'"),
):
    """Analyze talking-head video for cross-modal desynchronization and calculate fused calibrated authenticity report."""
    # Validate extension
    allowed_exts = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
    ext = os.path.splitext(video.filename or "")[1].lower()
    if ext and ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{ext}'. Allowed formats: {', '.join(allowed_exts)}",
        )

    try:
        content = await video.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )
        if len(content) > settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Video file exceeds maximum allowed size ({settings.max_upload_size_bytes // (1024*1024)} MB).",
            )

        response = service.analyze_video_file(
            video_bytes=content,
            filename=video.filename or f"upload{ext or '.mp4'}",
            visual_fake_probability=visual_fake_probability,
            audio_fake_probability=audio_fake_probability,
            visual_quality=visual_quality,
            audio_quality=audio_quality,
            sample_fps=sample_fps,
            model_type=model_type,
        )
        return response

    except HTTPException:
        raise
    except NoFaceDetectedError as e:
        logger.warning(f"Video analysis rejected: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except NoAudioFoundError as e:
        logger.warning(f"Video analysis rejected: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except ShortVideoError as e:
        logger.warning(f"Video analysis rejected: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except AudioExtractionError as e:
        logger.error(f"FFmpeg audio extraction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except VideoProcessingError as e:
        logger.error(f"Video processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(f"Unexpected forensic pipeline error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal forensic analysis failure: {str(e)}",
        )


@app.get(
    "/report/{analysis_id}",
    response_model=VideoAnalysisResponse,
    tags=["Forensics"],
)
async def get_report(analysis_id: str):
    """Retrieve a previously generated forensic authenticity report by analysis UUID."""
    report = service.get_report(analysis_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Forensic report with ID '{analysis_id}' was not found or has expired from cache.",
        )
    return report


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=True)
