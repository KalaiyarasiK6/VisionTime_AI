"""
app.py
------
VisionTime AI - AI-Powered Human Activity Timeline Generator
Professional CCTV Analytics Dashboard (Streamlit)

Pipeline:
    Video -> PersonDetector (YOLOv8) -> PersonTracker (ByteTrack)
          -> EventEngine (generic zone + dwell + speed logic)
          -> Timeline (DataFrame/CSV) -> Summary (NL text + alerts)

Works on ANY CCTV environment - airport, hospital, college, office, mall,
warehouse, public area, government building, factory, parking area - because
zones are a generic spatial grid and events are derived purely from
tracking geometry, never appearance-based activity labels.
"""

import os
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from activity import EventEngine, ZoneGrid
from detector import PersonDetector
from summary import generate_alerts, generate_summary
from timeline import compute_stats, events_to_dataframe, export_csv, important_events
from tracker import PersonTracker

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VisionTime AI | CCTV Analytics",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
#MainMenu, footer {visibility: hidden;}

.stApp {
    background: radial-gradient(circle at 10% 0%, #0f1729 0%, #0a0e1a 45%, #060810 100%);
    color: #e6ebf5;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1424 0%, #0a0f1c 100%);
    border-right: 1px solid rgba(0, 209, 178, 0.15);
}

h1, h2, h3, h4 { color: #f0f4fa !important; font-weight: 700 !important; }

.vt-hero {
    padding: 22px 28px;
    border-radius: 16px;
    background: linear-gradient(120deg, rgba(0,209,178,0.14), rgba(59,130,246,0.10));
    border: 1px solid rgba(0,209,178,0.25);
    margin-bottom: 18px;
}
.vt-hero h1 { margin: 0; font-size: 1.9rem; letter-spacing: 0.3px; }
.vt-hero p { margin: 6px 0 0 0; color: #9fb0c8; font-size: 0.95rem; }

.vt-badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.5px;
    margin-right: 6px; text-transform: uppercase;
}
.vt-badge.live { background: rgba(0,209,178,0.18); color: #00d1b2; border: 1px solid rgba(0,209,178,0.4); }
.vt-badge.gpu { background: rgba(59,130,246,0.18); color: #3b82f6; border: 1px solid rgba(59,130,246,0.4); }
.vt-badge.cpu { background: rgba(234,179,8,0.18); color: #eab308; border: 1px solid rgba(234,179,8,0.4); }

.vt-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 16px 18px;
}

.vt-status-row { display:flex; justify-content: space-between; padding: 5px 0; font-size: 0.85rem; }
.vt-status-row span:first-child { color: #8ea0b8; }
.vt-status-row span:last-child { color: #e6ebf5; font-weight: 600; }

.vt-alert { padding: 10px 14px; border-radius: 10px; margin-bottom: 8px; font-size: 0.88rem; }
.vt-alert.info { background: rgba(59,130,246,0.10); border-left: 3px solid #3b82f6; }
.vt-alert.warning { background: rgba(234,179,8,0.10); border-left: 3px solid #eab308; }
.vt-alert.danger { background: rgba(239,68,68,0.12); border-left: 3px solid #ef4444; }

.vt-summary-box {
    background: linear-gradient(120deg, rgba(0,209,178,0.08), rgba(255,255,255,0.02));
    border: 1px solid rgba(0,209,178,0.2);
    border-radius: 14px; padding: 18px 20px; font-size: 0.95rem; line-height: 1.6; color: #d4dde8;
}

.vt-footer {
    margin-top: 30px; padding: 16px; text-align: center; color: #64748b;
    font-size: 0.8rem; border-top: 1px solid rgba(255,255,255,0.08);
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 12px 16px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_detector(model_path: str, conf: float) -> PersonDetector:
    return PersonDetector(model_path=model_path, conf_threshold=conf)


# ---------------------------------------------------------------------------
# Video processing pipeline
# ---------------------------------------------------------------------------

TRAIL_COLORS = [
    (0, 209, 178), (59, 130, 246), (234, 179, 8),
    (239, 68, 68), (168, 85, 247), (16, 185, 129),
]


def _trail_color(track_id: int):
    return TRAIL_COLORS[track_id % len(TRAIL_COLORS)]


def draw_overlays(frame, people, zone_grid: ZoneGrid, active_zones: set, trail_history: dict):
    """Draw zone grid, bounding boxes, IDs, and fading movement trails onto a frame."""
    overlay = frame.copy()

    for (x1, y1, x2, y2, label) in zone_grid.boundaries():
        color = (0, 209, 178) if label in active_zones else (70, 70, 90)
        thickness = 2 if label in active_zones else 1
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(overlay, label, (x1 + 6, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Fading trajectory trails (drawn before boxes so boxes sit on top)
    for tid, points in trail_history.items():
        if len(points) < 2:
            continue
        color = _trail_color(tid)
        n = len(points)
        for i in range(1, n):
            alpha = i / n  # older segments fade out
            pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            faded = tuple(int(c * alpha) for c in color)
            cv2.line(overlay, pt1, pt2, faded, 2, cv2.LINE_AA)

    for person in people:
        x1, y1, x2, y2 = [int(v) for v in person.bbox]
        color = _trail_color(person.track_id)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"ID {person.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 15, 25), 2, cv2.LINE_AA)
        cx, cy = int(person.centroid[0]), int(person.centroid[1])
        cv2.circle(overlay, (cx, cy), 4, (0, 255, 200), -1)

    return overlay


def process_video(video_path, detector, tracker_cfg, zone_rows, zone_cols,
                   frame_skip, progress_callback=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    tracker = PersonTracker(detector, tracker_cfg=tracker_cfg)
    tracker.reset()
    zone_grid = ZoneGrid(width, height, rows=zone_rows, cols=zone_cols)
    engine = EventEngine(zone_grid=zone_grid, fps=fps)

    out_path = os.path.join(tempfile.gettempdir(), f"visiontime_out_{int(time.time())}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Heatmap accumulator: builds up "hot" regions of the scene from every
    # person position seen, used for the Movement Heatmap panel.
    heatmap_acc = np.zeros((height, width), dtype=np.float32)
    TRAIL_LENGTH = 20  # how many past points to keep per person for the trail
    trail_history: dict = {}
    last_frame_bg = None

    frame_idx = 0
    last_people = []
    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = frame_idx / fps

        if frame_idx % frame_skip == 0:
            infer_frame = frame
            scale = 1.0
            max_w = 960
            if width > max_w:
                scale = max_w / width
                infer_frame = cv2.resize(frame, (max_w, int(height * scale)))

            people = tracker.update(infer_frame)
            if scale != 1.0:
                for p in people:
                    x1, y1, x2, y2 = p.bbox
                    p.bbox = (x1 / scale, y1 / scale, x2 / scale, y2 / scale)
                    p.centroid = (p.centroid[0] / scale, p.centroid[1] / scale)
            last_people = people
        else:
            people = last_people

        engine.update(frame_idx, timestamp, people)

        # Update trail history and heatmap accumulator
        active_ids = set()
        for p in people:
            active_ids.add(p.track_id)
            trail_history.setdefault(p.track_id, []).append(p.centroid)
            if len(trail_history[p.track_id]) > TRAIL_LENGTH:
                trail_history[p.track_id] = trail_history[p.track_id][-TRAIL_LENGTH:]
            cx, cy = int(p.centroid[0]), int(p.centroid[1])
            if 0 <= cy < height and 0 <= cx < width:
                cv2.circle(heatmap_acc, (cx, cy), 25, 1.0, -1)
        # Drop trails for people no longer visible so old paths don't linger forever
        for tid in list(trail_history.keys()):
            if tid not in active_ids and len(trail_history[tid]) > 0:
                trail_history[tid] = trail_history[tid][1:] if len(trail_history[tid]) > 1 else []

        active_zones = {zone_grid.zone_for_point(*p.centroid) for p in people}
        annotated = draw_overlays(frame, people, zone_grid, active_zones, trail_history)
        writer.write(annotated)
        last_frame_bg = frame

        frame_idx += 1
        if progress_callback and total_frames > 0:
            progress_callback(min(frame_idx / total_frames, 1.0))

    engine.finalize(frame_idx / fps if fps else frame_idx)

    cap.release()
    writer.release()

    # Build the movement heatmap image (background frame + colorized heat overlay)
    heatmap_path = None
    if last_frame_bg is not None and heatmap_acc.max() > 0:
        norm = cv2.normalize(heatmap_acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm = cv2.GaussianBlur(norm, (0, 0), sigmaX=15, sigmaY=15)
        heat_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(last_frame_bg, 0.55, heat_color, 0.45, 0)
        heatmap_path = os.path.join(tempfile.gettempdir(), f"visiontime_heatmap_{int(time.time())}.png")
        cv2.imwrite(heatmap_path, blended)

    elapsed = time.time() - start_time
    proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0

    return {
        "output_path": out_path,
        "events": engine.sorted_events(),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": frame_idx,
        "duration": frame_idx / fps if fps else 0,
        "elapsed": elapsed,
        "proc_fps": proc_fps,
        "zone_grid": zone_grid,
        "heatmap_path": heatmap_path,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🎥 VisionTime AI")
    st.caption("Intelligent Event Timeline Engine")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload CCTV Video", type=["mp4", "avi", "mov", "mkv"])

    st.markdown("#### ⚙️ Zone Configuration")
    zone_cols = st.slider("Zone Columns", min_value=2, max_value=6, value=5)
    zone_rows = st.slider("Zone Rows", min_value=1, max_value=3, value=1)

    st.markdown("#### ⚡ Performance")
    frame_skip = st.slider("Frame Skip (higher = faster, but more ID switches)",
                            min_value=1, max_value=5, value=1)
    model_choice = st.selectbox("YOLOv8 Model", ["yolov8n.pt", "yolov8s.pt"], index=0)
    conf_threshold = st.slider("Detection Confidence", 0.15, 0.8, 0.35, 0.05)

    st.markdown("---")
    st.markdown("#### 🖥️ System Status")

    try:
        detector = load_detector(model_choice, conf_threshold)
        status = detector.status()
        badge = "gpu" if status["gpu_available"] else "cpu"
        badge_text = f"GPU · {status['gpu_name']}" if status["gpu_available"] else "CPU MODE"
        st.markdown(f'<span class="vt-badge {badge}">{badge_text}</span>'
                    f'<span class="vt-badge live">MODEL READY</span>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="vt-status-row"><span>Model</span><span>{status['model_path']}</span></div>
            <div class="vt-status-row"><span>Device</span><span>{status['device']}</span></div>
            <div class="vt-status-row"><span>Precision</span><span>{'FP16' if status['half_precision'] else 'FP32'}</span></div>
            <div class="vt-status-row"><span>Confidence</span><span>{status['conf_threshold']}</span></div>
            """,
            unsafe_allow_html=True,
        )
        model_ok = True
    except Exception as e:
        st.error(f"Model load failed: {e}")
        model_ok = False

    st.markdown("---")
    process_btn = st.button("▶ Process Video", type="primary", use_container_width=True,
                             disabled=(uploaded_file is None or not model_ok))

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="vt-hero">
        <h1>VisionTime AI</h1>
        <p>AI-Powered Human Activity Timeline Generator &middot;
        Universal CCTV Intelligence for Airports, Hospitals, Offices, Warehouses &amp; More</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.input_path = None

# ---------------------------------------------------------------------------
# Handle upload preview
# ---------------------------------------------------------------------------

if uploaded_file is not None and st.session_state.input_path is None:
    tmp_in = os.path.join(tempfile.gettempdir(), f"visiontime_in_{int(time.time())}_{uploaded_file.name}")
    with open(tmp_in, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.input_path = tmp_in

col_orig, col_proc = st.columns(2)
with col_orig:
    st.markdown("#### 📼 Original Video")
    if st.session_state.input_path:
        st.video(st.session_state.input_path)
    else:
        st.info("Upload a video from the sidebar to begin.")

with col_proc:
    st.markdown("#### 🎯 Processed Video")
    proc_placeholder = st.empty()
    if st.session_state.result:
        proc_placeholder.video(st.session_state.result["output_path"])
    else:
        proc_placeholder.info("Processed output will appear here.")

# ---------------------------------------------------------------------------
# Processing trigger
# ---------------------------------------------------------------------------

if process_btn and st.session_state.input_path:
    progress_bar = st.progress(0.0, text="Initializing pipeline...")

    def _update(p):
        progress_bar.progress(p, text=f"Processing frames... {int(p * 100)}%")

    with st.spinner("Running detection, tracking, and event generation..."):
        result = process_video(
            st.session_state.input_path,
            detector,
            tracker_cfg="custom_bytetrack.yaml",
            zone_rows=zone_rows,
            zone_cols=zone_cols,
            frame_skip=frame_skip,
            progress_callback=_update,
        )
    progress_bar.progress(1.0, text="Complete!")
    st.session_state.result = result
    proc_placeholder.video(result["output_path"])
    st.rerun()

# ---------------------------------------------------------------------------
# Results dashboard
# ---------------------------------------------------------------------------

result = st.session_state.result

if result:
    df = events_to_dataframe(result["events"])
    stats = compute_stats(df)
    ai_summary = generate_summary(df, stats)
    alerts = generate_alerts(df)

    st.markdown("---")
    st.markdown("### 📊 Analytics Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Persons Detected", stats["total_persons"])
    c2.metric("Events Generated", stats["total_events"])
    c3.metric("Duration", f"{int(stats['duration_seconds'])}s")
    c4.metric("Detection Status", "Active", delta=f"{result['proc_fps']:.1f} FPS")
    risk_delta_color = "inverse" if stats["risk_level"] != "Low" else "off"
    c5.metric("Risk Score", f"{stats['risk_score']}/100", delta=stats["risk_level"],
              delta_color=risk_delta_color)
    risk_bar_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[stats["risk_level"]]
    st.progress(stats["risk_score"] / 100, text=f"{risk_bar_color} Scene Risk Level: {stats['risk_level']}")

    st.markdown("### 🕒 Intelligent Event Timeline")
    st.dataframe(df, use_container_width=True, height=320, hide_index=True)

    col_a, col_b = st.columns([1.3, 1])

    with col_a:
        st.markdown("### 🧠 AI Summary")
        st.markdown(f'<div class="vt-summary-box">{ai_summary}</div>', unsafe_allow_html=True)

        st.markdown("### 🚨 Smart Alerts")
        for severity, message in alerts:
            st.markdown(f'<div class="vt-alert {severity}">{message}</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown("### 📈 Event Distribution")
        if stats["event_type_counts"]:
            dist_df = pd.DataFrame(
                {"Event Type": list(stats["event_type_counts"].keys()),
                 "Count": list(stats["event_type_counts"].values())}
            ).set_index("Event Type")
            st.bar_chart(dist_df, use_container_width=True)
        else:
            st.info("No events to chart.")

        st.markdown("### 🔥 Movement Heatmap")
        if result.get("heatmap_path") and os.path.exists(result["heatmap_path"]):
            st.image(result["heatmap_path"], use_container_width=True,
                      caption="Cumulative occupancy - warmer areas = more time spent")
        else:
            st.info("No movement data to build a heatmap.")

    st.markdown("### ⭐ Important Events")
    imp_df = important_events(df)
    if imp_df.empty:
        st.info("No long stays, loitering, crowding, or fast movement detected.")
    else:
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

    st.markdown("### ⬇️ Downloads")
    dl1, dl2, dl3 = st.columns(3)
    csv_path = os.path.join(tempfile.gettempdir(), "visiontime_timeline.csv")
    export_csv(df, csv_path)

    with dl1:
        with open(csv_path, "rb") as f:
            st.download_button("Download Timeline CSV", f, file_name="visiontime_timeline.csv",
                                mime="text/csv", use_container_width=True)
    with dl2:
        with open(result["output_path"], "rb") as f:
            st.download_button("Download Processed Video", f, file_name="visiontime_processed.mp4",
                                mime="video/mp4", use_container_width=True)
    with dl3:
        if result.get("heatmap_path") and os.path.exists(result["heatmap_path"]):
            with open(result["heatmap_path"], "rb") as f:
                st.download_button("Download Heatmap Image", f, file_name="visiontime_heatmap.png",
                                    mime="image/png", use_container_width=True)

st.markdown(
    """
    <div class="vt-footer">
        VisionTime AI &middot; Universal CCTV Event Intelligence &middot;
        Built with YOLOv8 + ByteTrack + Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
