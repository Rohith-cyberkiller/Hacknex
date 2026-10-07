"""Plotly visualizations and UI components for the forensic dashboard."""
from typing import Dict, Any, List
import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def render_summary_cards(data: Dict[str, Any]):
    """Render top-level forensic verdict and metric cards."""
    classification = data.get("classification", "UNKNOWN")
    auth_score = data.get("authenticity_score", 0.0)
    fake_prob = data.get("final_fake_probability", 0.0)
    confidence = data.get("confidence", "low").upper()
    lip_sync = data.get("lip_sync", {})
    ls_score = lip_sync.get("score", 0.0)
    offset_ms = lip_sync.get("estimated_offset_ms", 0)
    mode = lip_sync.get("analysis_mode", "heuristic")

    # Status badge HTML
    if classification == "REAL":
        badge_html = f'<span class="badge-real">AUTHENTIC / REAL</span>'
    elif classification == "SUSPICIOUS":
        badge_html = f'<span class="badge-suspicious">SUSPICIOUS</span>'
    else:
        badge_html = f'<span class="badge-fake">POTENTIALLY FAKE</span>'

    mode_label = "Heuristic Analysis Mode" if mode == "heuristic" else "Pretrained Neural Mode"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="forensic-card">
                <div class="card-title">Forensic Classification</div>
                <div>{badge_html}</div>
                <div class="card-sub">Confidence: <strong>{confidence}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="forensic-card">
                <div class="card-title">Calibrated Authenticity</div>
                <div class="card-value" style="color: {'#34d399' if auth_score > 60 else ('#fbbf24' if auth_score >= 35 else '#f87171')};">
                    {auth_score:.1f}%
                </div>
                <div class="card-sub">Fake Probability: {fake_prob:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="forensic-card">
                <div class="card-title">Lip-Sync Consistency</div>
                <div class="card-value" style="color: {'#38bdf8' if ls_score > 60 else '#fbbf24'};">
                    {ls_score:.1f}%
                </div>
                <div class="card-sub">Correlation: {lip_sync.get('correlation', 0.0):.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        sign = "+" if offset_ms > 0 else ""
        offset_color = "#34d399" if abs(offset_ms) <= 80 else ("#fbbf24" if abs(offset_ms) <= 180 else "#f87171")
        st.markdown(
            f"""
            <div class="forensic-card">
                <div class="card-title">AV Timing Offset</div>
                <div class="card-value" style="color: {offset_color};">
                    {sign}{offset_ms} <span style="font-size:1.1rem; color:#94a3b8;">ms</span>
                </div>
                <div class="card-sub"><span class="badge-mode">{mode_label}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_timeline_chart(data: Dict[str, Any]):
    """Render interactive temporal audiovisual synchronization timeline with drift anomalies."""
    lip_sync = data.get("lip_sync", {})
    timeline_series = lip_sync.get("timeline_series", [])
    suspicious_segments = lip_sync.get("suspicious_segments", [])

    if not timeline_series:
        st.info("No sampled timeline series points available to render.")
        return

    df = pd.DataFrame(timeline_series)

    fig = go.Figure()

    # Mouth motion trace
    fig.add_trace(
        go.Scatter(
            x=df["timestamp_sec"],
            y=df["mouth_energy"],
            mode="lines",
            name="Mouth Motion Energy",
            line=dict(color="#38bdf8", width=2.2),
            hovertemplate="Time: %{x:.2f}s<br>Mouth Energy: %{y:.3f}<extra></extra>",
        )
    )

    # Speech acoustic energy trace
    fig.add_trace(
        go.Scatter(
            x=df["timestamp_sec"],
            y=df["speech_energy"],
            mode="lines",
            name="Speech Acoustic Energy",
            line=dict(color="#fbbf24", width=2.0, dash="solid"),
            hovertemplate="Time: %{x:.2f}s<br>Speech Energy: %{y:.3f}<extra></extra>",
        )
    )

    # Highlight suspicious drift segments
    shapes = []
    annotations = []

    for idx, seg in enumerate(suspicious_segments):
        start = seg.get("start_sec", 0.0)
        end = seg.get("end_sec", 0.0)
        sev = seg.get("severity", "medium")
        reason = seg.get("reason", "Sync mismatch")

        color = "rgba(239, 68, 68, 0.25)" if sev == "high" else "rgba(245, 158, 11, 0.20)"
        line_color = "rgba(239, 68, 68, 0.7)" if sev == "high" else "rgba(245, 158, 11, 0.7)"

        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=start,
                x1=end,
                y0=0,
                y1=1,
                fillcolor=color,
                line=dict(color=line_color, width=1, dash="dot"),
            )
        )

        annotations.append(
            dict(
                x=(start + end) / 2.0,
                y=1.02,
                xref="x",
                yref="paper",
                text=f"⚠️ {sev.upper()}: {reason}",
                showarrow=False,
                font=dict(size=10, color="#f87171" if sev == "high" else "#fbbf24"),
                bgcolor="rgba(15, 23, 42, 0.8)",
                bordercolor=line_color,
                borderwidth=1,
                borderpad=3,
            )
        )

    fig.update_layout(
        title=dict(
            text="Audiovisual Alignment & Temporal Drift Timeline",
            font=dict(size=16, color="#f1f5f9"),
        ),
        xaxis=dict(
            title="Timeline (Seconds)",
            gridcolor="#1e293b",
            showgrid=True,
            zeroline=False,
            color="#94a3b8",
        ),
        yaxis=dict(
            title="Normalized Signal Amplitude",
            gridcolor="#1e293b",
            showgrid=True,
            zeroline=False,
            range=[-0.05, 1.15],
            color="#94a3b8",
        ),
        shapes=shapes,
        annotations=annotations,
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="right",
            x=1,
            font=dict(color="#cbd5e1"),
        ),
        margin=dict(l=45, r=25, t=55, b=45),
        height=380,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_fusion_breakdown(data: Dict[str, Any]):
    """Render multimodal fusion contribution breakdown chart and quality indicators."""
    fusion = data.get("fusion", {})
    c_v = fusion.get("visual_contribution", 0.0)
    c_a = fusion.get("audio_contribution", 0.0)
    c_ls = fusion.get("lip_sync_contribution", 0.0)
    q_v = fusion.get("visual_quality", 1.0)
    q_a = fusion.get("audio_quality", 1.0)

    col1, col2 = st.columns([1.5, 1.0])

    with col1:
        labels = ["Visual Detector (M1)", "Audio Detector (M2)", "Lip-Sync Fusion (M3)"]
        values = [c_v * 100, c_a * 100, c_ls * 100]
        colors = ["#818cf8", "#38bdf8", "#34d399"]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.55,
                    marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
                    textinfo="label+percent",
                    textposition="inside",
                    hovertemplate="%{label}: <b>%{value:.1f}%</b> contribution<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=dict(
                text="Quality-Aware Multimodal Weighting",
                font=dict(size=14, color="#f1f5f9"),
            ),
            showlegend=False,
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            margin=dict(l=20, r=20, t=40, b=20),
            height=260,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Quality Metrics & Modulation")
        st.markdown(
            f"""
            <div style="background:#151d2c; padding:1rem; border-radius:8px; border:1px solid #283548;">
                <div style="margin-bottom:0.75rem;">
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#94a3b8;">
                        <span>Visual Stream Quality</span>
                        <strong style="color:#e2e8f0;">{q_v:.2f} / 1.00</strong>
                    </div>
                    <div style="background:#1e293b; height:6px; border-radius:3px; margin-top:4px;">
                        <div style="background:#818cf8; width:{q_v*100}%; height:6px; border-radius:3px;"></div>
                    </div>
                </div>
                <div>
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#94a3b8;">
                        <span>Audio Stream Quality</span>
                        <strong style="color:#e2e8f0;">{q_a:.2f} / 1.00</strong>
                    </div>
                    <div style="background:#1e293b; height:6px; border-radius:3px; margin-top:4px;">
                        <div style="background:#38bdf8; width:{q_a*100}%; height:6px; border-radius:3px;"></div>
                    </div>
                </div>
                <div style="font-size:0.78rem; color:#64748b; margin-top:0.85rem; line-height:1.4;">
                    ℹ️ Quality degradation automatically down-weights unreliable modalities and prioritizes cross-modal synchronization.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_evidence_and_limitations(data: Dict[str, Any]):
    """Render human-readable forensic evidence list and limitations panel."""
    evidence = data.get("evidence", [])
    limitations = data.get("limitations", [])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔍 Forensic Findings & Evidence")
        if evidence:
            for item in evidence:
                st.markdown(f'<div class="evidence-item">{item}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="evidence-item">No anomalous indicators flagged.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### ⚠️ Forensic Boundaries & Limitations")
        if limitations:
            for item in limitations:
                st.markdown(f'<div class="limitation-item">{item}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="limitation-item">This score is probabilistic forensic evidence, not proof of manipulation.</div>',
                unsafe_allow_html=True,
            )
