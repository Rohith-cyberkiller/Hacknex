"""Backend configuration and environment management."""
import os
from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Cross-Modal Deepfake Forensics API"
    app_version: str = "1.0.0"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    # Directory setup
    base_dir: Path = Path(__file__).resolve().parent.parent
    temp_dir: Path = base_dir / "temp_media"
    calibration_path: Path = base_dir / "core" / "calibration" / "artifacts" / "calibrator.json"

    # Default parameters
    default_fps: int = int(os.getenv("DEFAULT_SAMPLE_FPS", "20"))
    temperature: float = float(os.getenv("TEMPERATURE_SCALE", "1.15"))
    default_model: str = os.getenv("DEFAULT_ANALYSIS_MODE", "heuristic")

    max_upload_size_bytes: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100")) * 1024 * 1024


settings = Settings()
settings.temp_dir.mkdir(parents=True, exist_ok=True)
settings.calibration_path.parent.mkdir(parents=True, exist_ok=True)
