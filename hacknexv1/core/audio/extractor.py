"""Audio extraction utilities using FFmpeg with robust fallback support."""
import os
import shutil
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AudioExtractionError(Exception):
    """Raised when audio cannot be extracted from the video."""
    pass


class NoAudioFoundError(AudioExtractionError):
    """Raised when the video file contains no audio stream."""
    pass


def get_ffmpeg_executable() -> Optional[str]:
    """Find FFmpeg binary from PATH or imageio-ffmpeg."""
    # 1. System PATH
    cmd = shutil.which("ffmpeg")
    if cmd:
        return cmd
        
    # 2. Check imageio-ffmpeg package
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except ImportError:
        pass

    # 3. Common Windows paths
    common_win_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg\bin\ffmpeg.exe"),
    ]
    for p in common_win_paths:
        if os.path.exists(p):
            return p

    return None


def extract_audio_from_video(
    video_path: str,
    output_wav_path: str,
    target_sample_rate: int = 16000,
) -> str:
    """Extract mono 16kHz PCM WAV audio from a video container using FFmpeg.
    
    Args:
        video_path: Path to source video file (MP4, MOV, AVI, WebM).
        output_wav_path: Target path for the uncompressed PCM WAV.
        target_sample_rate: Audio sampling frequency (Hz).
        
    Returns:
        output_wav_path if successful.
        
    Raises:
        AudioExtractionError or NoAudioFoundError on failure.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    ffmpeg_bin = get_ffmpeg_executable()
    if not ffmpeg_bin:
        raise AudioExtractionError(
            "FFmpeg executable not found on system PATH or via imageio-ffmpeg. "
            "Please install FFmpeg or pip install imageio-ffmpeg."
        )

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(target_sample_rate),
        "-ac", "1",
        output_wav_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise AudioExtractionError("FFmpeg audio extraction timed out after 60 seconds.")
    except Exception as e:
        raise AudioExtractionError(f"Subprocess error running FFmpeg: {str(e)}")

    if result.returncode != 0:
        stderr_lower = result.stderr.lower()
        if "does not contain any stream" in stderr_lower or "matches no streams" in stderr_lower or "no audio" in stderr_lower:
            raise NoAudioFoundError("The uploaded video file contains no audio stream.")
        raise AudioExtractionError(f"FFmpeg extraction failed (exit code {result.returncode}): {result.stderr[-400:]}")

    if not os.path.exists(output_wav_path) or os.path.getsize(output_wav_path) == 0:
        raise NoAudioFoundError("Extracted audio file is empty. Video may lack an active audio track.")

    logger.info(f"Successfully extracted audio to {output_wav_path}")
    return output_wav_path
