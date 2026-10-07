"""Acoustic feature processing: speech energy envelopes and quality metrics using librosa, SoundFile, and SciPy."""
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)

# Try importing soundfile and librosa
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    sf = None
    SOUNDFILE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None
    LIBROSA_AVAILABLE = False


def load_audio_waveform(wav_path: str) -> Tuple[np.ndarray, int]:
    """Load mono audio file as normalized float32 array in [-1.0, 1.0].
    
    Uses SoundFile if available, with SciPy wavfile fallback.
    """
    if SOUNDFILE_AVAILABLE:
        try:
            data, sr = sf.read(wav_path, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)  # Average channels to mono
            return data.astype(np.float32), sr
        except Exception as e:
            logger.debug(f"SoundFile read fallback to scipy: {e}")

    # Fallback to scipy.io.wavfile
    sr, raw = wavfile.read(wav_path)
    if raw.ndim > 1:
        raw = raw[:, 0]

    if raw.dtype == np.int16:
        waveform = raw.astype(np.float32) / 32768.0
    elif raw.dtype == np.int32:
        waveform = raw.astype(np.float32) / 2147483648.0
    elif raw.dtype == np.uint8:
        waveform = (raw.astype(np.float32) - 128.0) / 128.0
    else:
        waveform = raw.astype(np.float32)

    return waveform, sr


def compute_speech_energy_profile(
    waveform: np.ndarray,
    sr: int,
    timestamps: np.ndarray,
    fps: float,
) -> Tuple[np.ndarray, float]:
    """Compute frame-synchronized speech energy envelope and audio quality metric.
    
    Supports librosa RMS extraction when available, complemented by SciPy windowing.
    
    Args:
        waveform: 1D audio samples.
        sr: Sampling rate (e.g. 16000 Hz).
        timestamps: 1D array of video frame timestamps in seconds.
        fps: Video frame rate.
        
    Returns:
        (speech_energy, audio_quality):
            speech_energy: 1D normalized array of length len(timestamps) in [0.0, 1.0].
            audio_quality: Estimated quality score in [0.0, 1.0].
    """
    n_frames = len(timestamps)
    if n_frames == 0 or len(waveform) == 0:
        return np.zeros(n_frames, dtype=np.float32), 0.1

    win_len = int(max(sr * 0.08, sr / max(fps, 1.0)))
    half_win = win_len // 2

    raw_energies = np.zeros(n_frames, dtype=np.float32)

    # Librosa feature extraction if available
    if LIBROSA_AVAILABLE and len(waveform) > 512:
        try:
            hop = int(max(1, sr / max(fps, 1.0)))
            frame_len = max(win_len, 512)
            rms = librosa.feature.rms(y=waveform, frame_length=frame_len, hop_length=hop)[0]
            # Interpolate librosa RMS to match timestamps length
            if len(rms) > 0:
                t_rms = np.linspace(0, len(waveform) / sr, len(rms))
                raw_energies = np.interp(timestamps, t_rms, rms).astype(np.float32)
        except Exception as e:
            logger.debug(f"Librosa RMS fallback to SciPy windowing: {e}")
            raw_energies = np.zeros(n_frames, dtype=np.float32)

    # Standard SciPy / NumPy windowing calculation if librosa wasn't used
    if np.all(raw_energies == 0):
        for i, t in enumerate(timestamps):
            center_sample = int(round(t * sr))
            start_idx = max(0, center_sample - half_win)
            end_idx = min(len(waveform), center_sample + half_win)

            if end_idx > start_idx:
                chunk = waveform[start_idx:end_idx]
                rms_val = np.sqrt(np.mean(chunk ** 2) + 1e-10)
                raw_energies[i] = rms_val
            else:
                raw_energies[i] = 0.0

    # Moving average smoothing over 3 frames to emulate vocal tract inertia
    kernel = np.ones(3) / 3.0
    smoothed = np.convolve(raw_energies, kernel, mode="same")

    # Min-max normalization with robust percentile clamping
    p10 = np.percentile(smoothed, 10)
    p95 = np.percentile(smoothed, 95)
    spread = p95 - p10

    if spread > 1e-5:
        norm_energy = np.clip((smoothed - p10) / spread, 0.0, 1.0)
    else:
        norm_energy = np.zeros_like(smoothed)

    # -------------------------------------------------------------------------
    # Audio Quality Estimation (0.0 to 1.0)
    # -------------------------------------------------------------------------
    overall_rms = np.sqrt(np.mean(waveform ** 2) + 1e-12)
    if overall_rms < 0.005:
        quality = 0.20
    else:
        clipped_ratio = np.mean(np.abs(waveform) > 0.98)
        clip_penalty = min(0.4, clipped_ratio * 4.0)

        dynamic_ratio = p95 / (p10 + 1e-5)
        dr_score = np.clip(np.log10(dynamic_ratio + 1.0) / 2.0, 0.3, 1.0)

        quality = float(np.clip(dr_score - clip_penalty, 0.1, 0.98))

    return norm_energy, round(quality, 2)
