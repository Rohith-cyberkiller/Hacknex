"""Streamlit Dashboard for Cross-Modal Deepfake Forensics."""
import os
import io
import json
import logging
import requests
import streamlit as st
import numpy as np
import cv2
from scipy.io import wavfile
import subprocess

from frontend.components import (
    render_summary_cards,
    render_timeline_chart,
    render_fusion_breakdown,
    render_evidence_and_limitations,
)

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Cross-Modal Deepfake Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS styles
css_path = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def check_backend_health(api_url: str):
    """Check connectivity to the FastAPI backend."""
    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, None


def generate_synthetic_demo_video(offset_anomaly: bool = True) -> bytes:
    """Generate a valid talking-head synthetic demo video with audio track in-memory."""
    import tempfile
    import imageio_ffmpeg

    fps = 20
    duration_sec = 4
    total_frames = fps * duration_sec
    sr = 16000

    temp_v = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    temp_a = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)

    temp_v.close()
    temp_a.close()
    temp_out.close()

    try:
        # 1. Synthesize frames with simulated facial oval and articulating mouth box
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(temp_v.name, fourcc, fps, (320, 320))

        for i in range(total_frames):
            frame = np.ones((320, 320, 3), dtype=np.uint8) * 30
            # Draw face oval
            cv2.ellipse(frame, (160, 160), (90, 120), 0, 0, 360, (180, 190, 210), -1)
            # Eyes
            cv2.circle(frame, (125, 120), 10, (40, 40, 50), -1)
            cv2.circle(frame, (195, 120), 10, (40, 40, 50), -1)

            # Mouth articulation rhythm: open/close at 1.5 Hz
            t = i / fps
            mouth_opening = int(12 + 10 * np.sin(2 * np.pi * 1.5 * t))
            if offset_anomaly and 2.0 <= t <= 3.5:
                # Stationary lips anomaly during active voice!
                mouth_opening = 3

            cv2.ellipse(frame, (160, 220), (25, mouth_opening), 0, 0, 360, (50, 60, 160), -1)
            out.write(frame)

        out.release()

        # 2. Synthesize audio track: harmonic voice with pitch ~180 Hz modulated at 1.5 Hz
        t_audio = np.linspace(0, duration_sec, sr * duration_sec, endpoint=False)
        envelope = 0.5 + 0.45 * np.sin(2 * np.pi * 1.5 * t_audio)
        carrier = np.sin(2 * np.pi * 180 * t_audio) + 0.3 * np.sin(2 * np.pi * 360 * t_audio)
        audio_signal = (carrier * envelope * 20000).astype(np.int16)
        wavfile.write(temp_a.name, sr, audio_signal)

        # 3. Combine into MP4 using imageio-ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", temp_v.name,
            "-i", temp_a.name,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            temp_out.name,
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        with open(temp_out.name, "rb") as f:
            content = f.read()

        return content
    finally:
        for p in [temp_v.name, temp_a.name, temp_out.name]:
            if os.path.exists(p):
                os.remove(p)


# =============================================================================
# SIDEBAR CONFIGURATION & MULTIMODAL UPSTREAM INTEGRATION
# =============================================================================
with st.sidebar:
    st.markdown("## 🛡️ Forensics Console")
    st.markdown("**Member 3 Lead:** Synchronization & Fusion")

    api_url = st.text_input("Backend API Base URL", value="http://localhost:8000")
    is_online, health_info = check_backend_health(api_url)

    if is_online:
        st.markdown(
            f'<div style="color:#34d399; font-size:0.85rem; margin-bottom:1rem;">'
            f'● Backend Online (v{health_info.get("version", "1.0")})'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="color:#f87171; font-size:0.85rem; margin-bottom:1rem;">'
            f'○ Backend Offline / Connecting...'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🤝 Upstream Team Signals")

    include_m1 = st.checkbox("Include Member 1 (Visual Detector)", value=True)
    if include_m1:
        m1_fake_prob = st.slider("M1 Visual Fake Probability (%)", 0.0, 100.0, 42.0, 1.0)
        m1_quality = st.slider("M1 Visual Quality Index", 0.0, 1.0, 0.88, 0.05)
    else:
        m1_fake_prob = None
        m1_quality = None

    st.markdown("---")
    include_m2 = st.checkbox("Include Member 2 (Audio Detector)", value=True)
    if include_m2:
        m2_fake_prob = st.slider("M2 Audio Fake Probability (%)", 0.0, 100.0, 35.0, 1.0)
        m2_quality = st.slider("M2 Audio Quality Index", 0.0, 1.0, 0.82, 0.05)
    else:
        m2_fake_prob = None
        m2_quality = None

    st.markdown("---")
    st.markdown("### ⚙️ Video & Sync Analysis")
    sample_fps = st.selectbox("Sample FPS", options=[15, 20, 25, 30], index=1)
    analysis_mode = st.selectbox(
        "Lip-Sync Model Mode",
        options=["heuristic", "pretrained"],
        format_func=lambda x: "Heuristic (Optical Correlation)" if x == "heuristic" else "Pretrained SyncNet (Adapter)",
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#64748b;'>Cross-Modal Deepfake Forensics v1.0<br>Calibrated Multimodal Architecture</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN FORENSIC DASHBOARD
# =============================================================================
st.markdown(
    """
    <div style="margin-bottom: 1.5rem;">
        <h1 style="margin: 0; font-size: 2.3rem; color: #f8fafc;">
            Cross-Modal Deepfake Forensics
        </h1>
        <p style="margin: 0.3rem 0 0 0; color: #94a3b8; font-size: 1.05rem;">
            Audiovisual Synchronization Analysis & Calibrated Multimodal Decision Fusion
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Upload and Demo Row
col_upload, col_demo = st.columns([2.5, 1.2])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload talking-head video for cross-modal forensic inspection",
        type=["mp4", "mov", "avi", "webm"],
        help="Upload a video containing a human talking head with speech audio.",
    )

with col_demo:
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    use_synthetic_sample = st.button(
        "⚡ Load Synthetic Demo Video",
        help="Generate an in-memory talking-head video with simulated desynchronization anomaly.",
        use_container_width=True,
    )

# Manage state for selected video bytes
if "video_bytes" not in st.session_state:
    st.session_state.video_bytes = None
    st.session_state.video_filename = None

if uploaded_file is not None:
    st.session_state.video_bytes = uploaded_file.getvalue()
    st.session_state.video_filename = uploaded_file.name

if use_synthetic_sample:
    with st.spinner("Synthesizing talking head test sample..."):
        try:
            demo_bytes = generate_synthetic_demo_video(offset_anomaly=True)
            st.session_state.video_bytes = demo_bytes
            st.session_state.video_filename = "synthetic_talking_head_demo.mp4"
            st.toast("Generated synthetic test video with desync anomaly!", icon="⚡")
        except Exception as e:
            st.error(f"Failed to generate synthetic sample: {e}")

# If video is loaded in session state
if st.session_state.video_bytes:
    analyze_col, preview_col = st.columns([1.5, 1.5])
    
    with preview_col:
        st.video(st.session_state.video_bytes)
        st.caption(f"Loaded: `{st.session_state.video_filename}` ({len(st.session_state.video_bytes) / 1024:.1f} KB)")

    with analyze_col:
        st.markdown("### Ready for Inspection")
        st.markdown(
            "This module will extract the audio track, sample frames, isolate the mouth region, "
            "compute optical motion dynamics, correlate with speech acoustics, and fuse with Member 1 & 2 signals."
        )
        run_analysis = st.button(
            "🚀 Execute Forensic Analysis",
            type="primary",
            use_container_width=True,
        )

    if run_analysis:
        if not is_online:
            st.error("Cannot connect to FastAPI backend at " + api_url + ". Please ensure the backend is running.")
        else:
            with st.spinner("Processing video frames and extracting cross-modal signals..."):
                try:
                    files = {
                        "video": (
                            st.session_state.video_filename,
                            st.session_state.video_bytes,
                            "video/mp4",
                        )
                    }
                    data_form = {
                        "sample_fps": sample_fps,
                        "model_type": analysis_mode,
                    }
                    if include_m1:
                        data_form["visual_fake_probability"] = m1_fake_prob
                        data_form["visual_quality"] = m1_quality
                    if include_m2:
                        data_form["audio_fake_probability"] = m2_fake_prob
                        data_form["audio_quality"] = m2_quality

                    resp = requests.post(
                        f"{api_url}/analyze/video",
                        files=files,
                        data=data_form,
                        timeout=120,
                    )

                    if resp.status_code == 200:
                        report_data = resp.json()
                        st.session_state.latest_report = report_data
                        st.success("Forensic examination completed successfully!")
                    else:
                        error_detail = resp.json().get("detail", resp.text)
                        st.error(f"Analysis failed ({resp.status_code}): {error_detail}")

                except Exception as err:
                    st.error(f"Connection error executing analysis: {err}")

# Display Analysis Report if Available
if "latest_report" in st.session_state and st.session_state.latest_report:
    report = st.session_state.latest_report
    st.markdown("---")

    # 1. Forensic Summary Metric Cards
    render_summary_cards(report)

    st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)

    # 2. Interactive Timeline Chart
    render_timeline_chart(report)

    # 3. Multimodal Fusion Breakdown
    render_fusion_breakdown(report)

    # 4. Forensic Findings & Limitations
    render_evidence_and_limitations(report)

    # 5. Export Report
    st.markdown("---")
    exp_col1, exp_col2 = st.columns([1.5, 2.5])
    with exp_col1:
        st.markdown("### 📥 Official Forensic Report")
        report_json = json.dumps(report, indent=2)
        st.download_button(
            label="Download Signed Forensic Report (.json)",
            data=report_json,
            file_name=f"forensic_report_{report.get('analysis_id', 'session')}.json",
            mime="application/json",
            use_container_width=True,
        )

    with exp_col2:
        with st.expander("View Full Raw JSON Response"):
            st.json(report)

else:
    st.info("👈 Upload a talking-head video or click **'Load Synthetic Demo Video'** to begin forensic inspection.")
