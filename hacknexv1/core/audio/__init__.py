"""Audio processing exports."""
from core.audio.extractor import (
    extract_audio_from_video,
    get_ffmpeg_executable,
    AudioExtractionError,
    NoAudioFoundError,
)
from core.audio.processor import (
    load_audio_waveform,
    compute_speech_energy_profile,
)

__all__ = [
    "extract_audio_from_video",
    "get_ffmpeg_executable",
    "AudioExtractionError",
    "NoAudioFoundError",
    "load_audio_waveform",
    "compute_speech_energy_profile",
]
