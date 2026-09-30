"""
LibraryGuard - Streamlit dashboard.

A privacy-first, local-first dashboard for prohibited-activity detection in
CCTY-style footage:

* upload a video (or use a bundled sample)
* select behaviors and tune thresholds/durations
* run the full ML pipeline with live progress
* review annotated output + event table with confidence labels
* download event reports (CSV / JSON)

Run with::

    streamlit run app/streamlit_app.py

No footage ever leaves the machine: everything runs locally.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

# Make src/ importable regardless of CWD.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("libraryguard.app")

BEHAVIOR_META = {
    "drinking": {"emoji": "\U0001F964", "label": "Drinking", "color": "#4FC3F7"},
    "sleeping": {"emoji": "\U0001F634", "label": "Sleeping", "color": "#F48FB1"},
    "phone_usage": {"emoji": "\U0001F4F1", "label": "Phone Usage", "color": "#FFB74D"},
}

st.set_page_config(
    page_title="LibraryGuard - AI Behavior Monitoring",
    page_icon="\U0001F4E1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; letter-spacing: 0.06em; margin-bottom: 0; }
    .subtitle { color: #9e9e9e; font-size: 1.05rem; margin-top: 0.25rem; }
    .event-card {
        border-radius: 10px; padding: 0.85rem 1.1rem; margin: 0.45rem 0;
        background: #1e1e2e; border-left: 5px solid #4FC3F7; color: #eceff1;
    }
    .stProgress > div > div > div > div { background-color: #4FC3F7; }
    .metric-pill {
        display: inline-block; padding: 0.15rem 0.65rem; border-radius: 999px;
        font-size: 0.8rem; margin-right: 0.4rem; background: #2b2b3d; color: #cfd8dc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def init_state() -> None:
    defaults = {
        "processed": False,
        "result_summary": None,
        "events": [],
        "annotated_path": None,
        "report_json": None,
        "report_csv": None,
        "error": None,
        "running": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def load_config_cached(path: str, mtime: float):
    from src.utils.config import load_config

    return load_config(path)


@st.cache_resource(show_spinner=False)
def get_pipeline():
    from src.pipeline import VideoPipeline

    return VideoPipeline()


def reset_results() -> None:
    st.session_state.processed = False
    st.session_state.result_summary = None
    st.session_state.events = []
    st.session_state.annotated_path = None
    st.session_state.report_json = None
    st.session_state.report_csv = None
    st.session_state.error = None


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------


def render_header() -> None:
    left, right = st.columns([4, 1])
    with left:
        st.markdown('<div class="main-title">\U0001F4E1 LIBRARYGUARD</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle">AI BEHAVIOR MONITORING &mdash; temporal detection for CCTV-style video</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div style="text-align:right;color:#78909c;font-size:0.85rem;">'
            "\U0001F512 100% local processing<br>no facial recognition</div>",
            unsafe_allow_html=True,
        )


def render_sidebar() -> Dict:
    st.sidebar.markdown("## \u2699\ufe0f Detection Settings")

    st.sidebar.markdown("### Behaviors")
    behaviors = {}
    for key, meta in BEHAVIOR_META.items():
        behaviors[key] = st.sidebar.checkbox(
            f"{meta['emoji']} {meta['label']}", value=True, key=f"enable_{key}"
        )

    st.sidebar.markdown("### Thresholds")
    confidence = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.10, max_value=0.95, value=0.55, step=0.05,
        help="Minimum smoothed confidence before a behavior can trigger an event.",
    )
    min_duration = st.sidebar.slider(
        "Minimum event duration (s)",
        min_value=0.5, max_value=30.0, value=2.0, step=0.5,
        help="Events shorter than this are discarded as noise.",
    )
    cooldown = st.sidebar.slider(
        "Event cooldown (s)",
        min_value=0.0, max_value=60.0, value=5.0, step=1.0,
        help="Suppress repeat events for the same person and behavior within this window.",
    )
    frame_skip = st.sidebar.select_slider(
        "Frame skip (speed vs. detail)",
        options=[1, 2, 3, 5, 8, 10], value=2,
        help="Process every Nth frame. Higher = faster, slightly coarser temporal resolution.",
    )

    st.sidebar.markdown("### Privacy")
    blur_faces = st.sidebar.checkbox(
        "\U0001F576\ufe0f Blur faces in output", value=True,
        help="Strongly recommended. Faces are blurred in the annotated video.",
    )
    st.sidebar.caption(
        "LibraryGuard performs behavioral detection only. It does not recognize or "
        "identify individuals, and no footage is uploaded to external services."
    )

    return {
        "behaviors": behaviors,
        "confidence": confidence,
        "min_duration": min_duration,
        "cooldown": cooldown,
        "frame_skip": int(frame_skip),
        "blur_faces": blur_faces,
    }


def apply_settings_to_config(config, settings: Dict):
    config.detection.global_confidence_threshold = settings["confidence"]
    config.detection.frame_skip = settings["frame_skip"]
    config.privacy.blur_faces = settings["blur_faces"]

    for key, enabled in settings["behaviors"].items():
        section = getattr(config.behavior, key)
        section.enabled = enabled
        section.confidence_threshold = settings["confidence"]
        section.min_duration_seconds = settings["min_duration"]
        section.cooldown_seconds = settings["cooldown"]
        # Scale min consecutive frames with duration to stay consistent.
        fps_guess = 25.0
        section.min_consecutive_frames = max(2, int(0.4 * settings["min_duration"] * fps_guess / 2))
    return config


def render_upload() -> Optional[object]:
    st.markdown("### 1\ufe0f\u20e3 Video Input")
    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv", "webm", "m4v"],
        help="CCTV-style footage works best. The file stays on this machine.",
    )
    if uploaded is not None:
        # Persist the upload to a temp file so OpenCV can read it.
        suffix = Path(uploaded.name).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="libraryguard_")
        tmp.write(uploaded.getbuffer())
        tmp.close()
        st.session_state["upload_path"] = tmp.name
        st.session_state["upload_name"] = uploaded.name

        from src.preprocessing.video import VideoCaptureError, probe_video

        try:
            info = probe_video(tmp.name)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Resolution", f"{info.width}\u00d7{info.height}")
            c2.metric("FPS", f"{info.fps:.1f}")
            c3.metric("Duration", f"{info.duration:.1f}s")
            c4.metric("Frames", f"{info.frame_count}")
        except VideoCaptureError as exc:
            st.error(f"\u274c {exc}")
            return None
        return tmp.name

    sample_path = ROOT / "data" / "sample" / "sample_video.mp4"
    if sample_path.exists():
        if st.checkbox("Use bundled sample video", value=False):
            st.session_state["upload_path"] = str(sample_path)
            st.session_state["upload_name"] = sample_path.name
            return str(sample_path)
    return None


def run_inference(video_path: str, settings: Dict) -> None:
    from src.preprocessing.video import VideoCaptureError
    from src.reporting.exporters import export_csv, export_json
    from src.reporting.events import EventLog
    from src.utils.config import load_config
    from src.pipeline import VideoPipeline

    config = load_config()
    config = apply_settings_to_config(config, settings)

    out_dir = Path(tempfile.mkdtemp(prefix="libraryguard_out_"))
    annotated_path = out_dir / "annotated.mp4"
    report_json = out_dir / "events.json"
    report_csv = out_dir / "events.csv"

    progress_bar = st.progress(0.0, text="Starting pipeline\u2026")
    status_box = st.empty()

    def progress(frame_index: int, total: int, frame_result) -> None:
        fraction = (frame_index + 1) / total if total else 0.0
        fraction = min(1.0, fraction)
        progress_bar.progress(
            fraction,
            text=f"Processing frame {frame_index + 1}" + (f"/{total}" if total else ""),
        )
        if frame_result.live_statuses:
            parts = [
                f"{BEHAVIOR_META.get(s.behavior, {}).get('emoji', '')} "
                f"{s.behavior} {s.confidence * 100:.0f}%"
                for s in frame_result.live_statuses[:3]
            ]
            status_box.markdown(
                f'<span class="metric-pill">\U0001F464 {frame_result.person_count} persons</span>'
                + "".join(f'<span class="metric-pill">{p}</span>' for p in parts),
                unsafe_allow_html=True,
            )

    try:
        pipeline = VideoPipeline(config=config)
        t0 = time.time()
        result = pipeline.process_video(
            video_path,
            output_path=annotated_path,
            progress_callback=progress,
        )
        elapsed = time.time() - t0
    except VideoCaptureError as exc:
        progress_bar.empty()
        st.session_state.error = str(exc)
        return
    except MemoryError:
        progress_bar.empty()
        st.session_state.error = (
            "Ran out of memory while processing. Try a shorter video or lower resolution."
        )
        return
    except Exception as exc:  # pragma: no cover - defensive UI layer
        progress_bar.empty()
        logger.exception("Pipeline failed")
        st.session_state.error = f"Processing failed: {exc}"
        return
    finally:
        progress_bar.empty()

    log = EventLog(source=result.source, config_snapshot=config.to_dict())
    log.extend(result.events)
    report = log.build_report(
        video_duration=result.video_duration,
        video_fps=result.video_fps,
        video_frames=result.video_frames,
        frames_processed=result.frames_processed,
        processing_fps=result.processing_fps,
    )
    export_json(report, report_json)
    export_csv(result.events, report_csv)

    st.session_state.processed = True
    st.session_state.events = [e.to_dict() for e in result.events]
    st.session_state.result_summary = {
        "frames_processed": result.frames_processed,
        "processing_fps": result.processing_fps,
        "video_duration": result.video_duration,
        "elapsed": elapsed,
        "counts": report.counts_by_type(),
    }
    st.session_state.annotated_path = str(annotated_path) if annotated_path.exists() else None
    st.session_state.report_json = str(report_json)
    st.session_state.report_csv = str(report_csv)


def render_events_table() -> None:
    events = st.session_state.events
    st.markdown("### \U0001F4CB Events")
    if not events:
        st.info(
            "No events detected with the current settings. Try lowering the confidence "
            "threshold or minimum duration, or verify the footage contains the behaviors."
        )
        return

    import pandas as pd

    rows = []
    for event in events:
        meta = BEHAVIOR_META.get(event["event"], {"emoji": "\u2753", "label": event["event"]})
        rows.append(
            {
                "": meta["emoji"],
                "Time": event["timestamp"],
                "Event": meta["label"],
                "Person": f"#{event['person_id']}",
                "Confidence": f"{event['confidence'] * 100:.0f}%",
                "Label": event.get("label", ""),
                "Duration": f"{event['duration']:.1f}s",
                "Frames": f"{event['frame_start']}\u2013{event['frame_end']}",
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    for event in events:
        meta = BEHAVIOR_META.get(event["event"], {"emoji": "\u2753", "label": event["event"]})
        conf = event["confidence"]
        st.markdown(
            f'<div class="event-card" style="border-left-color:{meta["color"]}">'
            f'<strong>{meta["emoji"]} {meta["label"]}</strong> '
            f'<span class="metric-pill">{event["timestamp"]}</span> '
            f'<span class="metric-pill">Person #{event["person_id"]}</span> '
            f'<span class="metric-pill">Confidence: {conf * 100:.0f}%</span> '
            f'<span class="metric-pill">{event.get("label", "")}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_downloads() -> None:
    st.markdown("### \U0001F4E5 Export")
    cols = st.columns(3)
    if st.session_state.report_json and Path(st.session_state.report_json).exists():
        with open(st.session_state.report_json, "rb") as handle:
            cols[0].download_button(
                "\U0001F4C4 Download JSON report",
                data=handle.read(),
                file_name="libraryguard_events.json",
                mime="application/json",
            )
    if st.session_state.report_csv and Path(st.session_state.report_csv).exists():
        with open(st.session_state.report_csv, "rb") as handle:
            cols[1].download_button(
                "\U0001F4CA Download CSV report",
                data=handle.read(),
                file_name="libraryguard_events.csv",
                mime="text/csv",
            )
    if st.session_state.annotated_path and Path(st.session_state.annotated_path).exists():
        with open(st.session_state.annotated_path, "rb") as handle:
            cols[2].download_button(
                "\U0001F3AC Download annotated video",
                data=handle.read(),
                file_name="libraryguard_annotated.mp4",
                mime="video/mp4",
            )


def render_results() -> None:
    summary = st.session_state.result_summary
    if not summary:
        return

    st.markdown("### \U0001F4CA Processing Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames processed", f"{summary['frames_processed']}")
    c2.metric("Processing speed", f"{summary['processing_fps']:.1f} FPS")
    c3.metric("Video duration", f"{summary['video_duration']:.1f}s")
    c4.metric("Events found", f"{sum(summary['counts'].values())}")

    if st.session_state.annotated_path and Path(st.session_state.annotated_path).exists():
        st.markdown("### \U0001F3AC Annotated Video")
        st.video(st.session_state.annotated_path)

    render_events_table()
    render_downloads()

    with st.expander("\u2139\ufe0f About these predictions"):
        st.markdown(
            """
            Computer-vision predictions are **probabilistic and can be wrong**.

            * **HIGH CONFIDENCE** &mdash; strong, temporally persistent evidence
            * **POSSIBLE** &mdash; moderate evidence; treat as a hint, not a verdict
            * **DETECTED** &mdash; weak/early evidence only

            Behavior labels are anonymous (`Person #N`) and are reset per video.
            No facial recognition or identity matching is performed.
            """
        )


def render_privacy_footer() -> None:
    st.markdown("---")
    st.markdown(
        """
        <div style="color:#78909c;font-size:0.85rem;">
        \U0001F512 <strong>Privacy:</strong> all processing happens on this machine.
        Raw footage is not stored by the pipeline; the annotated output is generated
        for review only. LibraryGuard performs <em>behavioral</em> detection and never
        attempts to identify individuals. Use responsibly and in accordance with
        local law and institutional policy.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    init_state()
    render_header()
    settings = render_sidebar()

    video_path = render_upload()

    st.markdown("### 2\ufe0f\u20e3 Run Detection")
    run_clicked = st.button(
        "\u25b6\ufe0f Run detection",
        type="primary",
        disabled=video_path is None,
        use_container_width=True,
    )

    if video_path is None:
        st.caption("Upload a video to enable detection.")

    if run_clicked and video_path is not None:
        reset_results()
        with st.spinner("Running detection pipeline\u2026"):
            run_inference(video_path, settings)
        st.rerun()

    if st.session_state.error:
        st.error(f"\u274c {st.session_state.error}")

    if st.session_state.processed:
        render_results()

    render_privacy_footer()


if __name__ == "__main__":
    main()
