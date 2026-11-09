"""SnakeWatch: live frame-by-frame RT-DETRv2 snake detection dashboard.

This app is intentionally separate from the offline replay app.py. It loads the
fine-tuned checkpoint and processes a selected video while updating the Streamlit
preview during inference. FPS is measured; this is not assumed to be real-time.
"""
from __future__ import annotations

import json
import math
import tempfile
import time
from datetime import datetime, timezone
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import supervision as sv
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
import subprocess

PROJECT_DIR_DEFAULT = Path("/content/drive/MyDrive/snake_detection/Snake_detection")
CHECKPOINT_DEFAULT = PROJECT_DIR_DEFAULT / "rtdetrv2_snake_finetuned_v3_earlystop" / "checkpoint-8072"
PROCESSOR_ID = "PekingU/rtdetr_v2_r18vd"

st.set_page_config(page_title="SnakeWatch Live", page_icon="🐍", layout="wide")
st.markdown("""
<style>
.stApp{background:#0e1a14;color:#e7f0e9}
[data-testid="stSidebar"]{background:#101f17;border-right:1px solid #294334}
[data-testid="stMetric"]{background:#14251b;border:1px solid #294334;padding:12px;border-radius:12px}
[data-testid="stMetricLabel"]{color:#a9c3b1}
.hero{padding:1rem 1.3rem;border:1px solid #294334;border-radius:16px;background:linear-gradient(120deg,#14291d,#17251d);margin-bottom:1rem}
.muted{color:#a9c3b1}
.pill{display:inline-block;padding:.18rem .55rem;border-radius:999px;font-size:.78rem;font-weight:700;border:1px solid #496453;margin-right:.25rem}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading fine-tuned RT-DETRv2 model…")
def load_model(checkpoint_path: str, processor_id: str):
    ckpt = Path(checkpoint_path).expanduser()
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint folder was not found: {ckpt}\n"
            "Mount Google Drive in Colab and confirm that checkpoint-8072 exists."
        )
    processor = RTDetrImageProcessor.from_pretrained(processor_id)
    model = RTDetrV2ForObjectDetection.from_pretrained(
        str(ckpt),
        num_labels=1,
        id2label={0: "snake"},
        label2id={"snake": 0},
        ignore_mismatched_sizes=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device


def build_tracker(frame_rate: float, activation_threshold: float, lost_buffer: int, match_threshold: float):
    """Create ByteTrack and apply the compatibility fixes used in Step 5.3."""
    # These patches mirror the user's Step 5.3 implementation. Keep this in
    # the app process only; never import the original pipeline script (it runs
    # its whole offline inference loop at import time).
    try:
        from supervision.tracker.byte_tracker.single_object_track import STrack, TrackState
        from supervision.tracker.byte_tracker import matching

        def patched_activate(self, kalman_filter, frame_id):
            self.kalman_filter = kalman_filter
            self.internal_track_id = self.internal_id_counter.new_id()
            self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
            self.tracklet_len = 1
            self.state = TrackState.Tracked
            if self.tracklet_len >= self.minimum_consecutive_frames:
                self.is_activated = True
                if self.external_track_id == self.external_id_counter.NO_ID:
                    self.external_track_id = self.external_id_counter.new_id()
            self.frame_id = frame_id
            self.start_frame = frame_id

        def patched_fuse_score(cost_matrix, detections):
            return cost_matrix

        STrack.activate = patched_activate
        matching.fuse_score = patched_fuse_score
    except Exception as exc:
        # Different supervision versions may have moved private internals.
        # Continue using the installed ByteTrack implementation if patching is unavailable.
        st.warning(f"ByteTrack compatibility patch not applied ({type(exc).__name__}: {exc}). Using installed tracker behavior.")

    tracker = sv.ByteTrack(
        track_activation_threshold=float(activation_threshold),
        lost_track_buffer=int(lost_buffer),
        minimum_matching_threshold=float(match_threshold),
        frame_rate=max(1, int(round(frame_rate or 30))),
    )
    # Step 5.3 allows low-confidence candidates into ByteTrack and uses a higher
    # strong-detection threshold internally when the installed version exposes it.
    if hasattr(tracker, "det_thresh"):
        tracker.det_thresh = 0.055
    return tracker


@torch.inference_mode()
def detect(frame_bgr: np.ndarray, processor: RTDetrImageProcessor, model: RTDetrV2ForObjectDetection,
           device: torch.device, threshold: float, nms_iou: float) -> list[dict[str, Any]]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    h, w = frame_bgr.shape[:2]
    results = processor.post_process_object_detection(
        outputs,
        threshold=float(threshold),
        target_sizes=torch.tensor([[h, w]], device=device),
    )[0]
    candidates: list[dict[str, Any]] = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if int(label.item()) != 0:
            continue
        score_f = float(score.item())
        if score_f < threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        candidates.append({"bbox": [x1, y1, x2, y2], "confidence": score_f})

    # NMS equivalent to Step 5.3 to limit duplicate boxes.
    if not candidates:
        return []
    boxes_xywh = [[d["bbox"][0], d["bbox"][1], d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]] for d in candidates]
    scores = [d["confidence"] for d in candidates]
    keep = cv2.dnn.NMSBoxes(boxes_xywh, scores, float(threshold), float(nms_iou))
    if keep is None or len(keep) == 0:
        return []
    return [candidates[int(i)] for i in np.asarray(keep).reshape(-1)]


def to_sv(detections: list[dict[str, Any]]) -> sv.Detections:
    if not detections:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.asarray([d["bbox"] for d in detections], dtype=np.float32),
        confidence=np.asarray([d["confidence"] for d in detections], dtype=np.float32),
        class_id=np.zeros(len(detections), dtype=int),
    )


def draw_tracks(frame: np.ndarray, detections: sv.Detections, histories: dict[int, deque], trail_length: int, trail_enabled: bool = True) -> tuple[np.ndarray, list[dict[str, Any]]]:
    painted = frame.copy()
    records: list[dict[str, Any]] = []
    boxes = detections.xyxy if len(detections) else []
    ids = detections.tracker_id if len(detections) else None
    scores = detections.confidence if len(detections) else None
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        tid = int(ids[i]) if ids is not None and ids[i] is not None else -1
        conf = float(scores[i]) if scores is not None else float("nan")
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if tid >= 0:
            histories[tid].append((cx, cy))
            while len(histories[tid]) > trail_length:
                histories[tid].popleft()
        color = (70, 230, 130)
        cv2.rectangle(painted, (x1, y1), (x2, y2), color, 2)
        band = "HIGHER SCORE" if conf >= 0.055 else "LOWER SCORE"
        track_state = "TRACKED" if tid >= 0 else "TENTATIVE"
        label = f"{track_state} {tid if tid >= 0 else ''} | {band}"
        cv2.rectangle(painted, (x1, max(0, y1 - 27)), (min(painted.shape[1] - 1, x1 + max(160, len(label) * 8)), y1), color, -1)
        cv2.putText(painted, label, (x1 + 4, max(17, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 20, 12), 1, cv2.LINE_AA)
        if tid >= 0 and trail_enabled:
            pts = list(histories[tid])
            for p, q in zip(pts[:-1], pts[1:]):
                cv2.line(painted, p, q, (0, 220, 255), 2, cv2.LINE_AA)
        records.append({"track_id": tid if tid >= 0 else None, "confidence": conf,
                        "score_band": "Higher detector score" if conf >= 0.055 else "Lower detector score",
                        "track_state": "Tracked" if tid >= 0 else "Tentative",
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2, "center_x_px": cx, "center_y_px": cy})
    return painted, records


def open_video(uploaded_file, video_path: str):
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.getvalue())
        tmp.flush()
        tmp.close()
        return cv2.VideoCapture(tmp.name), tmp.name
    path = Path(video_path).expanduser() if video_path.strip() else None
    if path is None or not path.exists():
        raise FileNotFoundError("Choose a video using the uploader or enter a valid video path.")
    return cv2.VideoCapture(str(path)), None



# SNAKEWATCH_TWO_TAB_INTEGRATION_V1

tab_live, tab_boundary, tab_safety = st.tabs([
    "Live Detection",
    "Boundary Simulation",
    "Human Safety Monitoring"
])

with tab_live:
    st.markdown("""
    <div class="hero">
     <div style="font-size:.8rem;letter-spacing:.12em;color:#8fbd9e;font-weight:700">FIELD SAFETY · COMPUTER VISION</div>
     <h1 style="margin:.25rem 0">🐍 SnakeWatch Live</h1>
     <div class="muted">Snake detection and track continuity for research and field-safety prototyping.</div>
     <div class="muted">Detection score bands are not probabilities. Any alert shown here is provisional, not a verified intrusion alert.</div>
     <div class="muted">This app displays frames while they are being processed. Processing may be slower than the source video FPS.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("SnakeWatch controls")
        view_mode = st.radio("View", ["Operator", "Advanced / analysis"], horizontal=False)
        checkpoint = st.text_input("Fine-tuned checkpoint folder", value=str(CHECKPOINT_DEFAULT))
        uploaded = st.file_uploader("Upload a test video", type=["mp4", "mov", "avi", "mkv", "webm"])
        video_path = st.text_input("Or use a video path (when running in Colab)", value=str(PROJECT_DIR_DEFAULT / "snake_videos" / "10231672-hd_1080_1920_30fps.mp4"))

        preset_label = st.selectbox(
            "Detection sensitivity",
            ["Debug (lower threshold)", "Balanced (validation threshold)", "Conservative candidate (may miss snakes)"],
            index=0,
            help="Debug mode uses a lower threshold to reveal weak candidates. It can increase false positives and is not a validated final setting.",
        )
        preset_thresholds = {
            "Debug (lower threshold)": 0.03,
            "Balanced (validation threshold)": 0.055,
            "Conservative candidate (may miss snakes)": 0.08,
        }
        threshold = preset_thresholds[preset_label]
        st.caption(f"Detection threshold applied: {threshold:.3f}")

        trail_enabled = st.toggle("Show movement trails", value=True)
        process_entire_video = st.checkbox("Process entire video", value=False)
        if process_entire_video:
            max_frames = 0
        else:
            max_frames = st.number_input("Frame limit (0 = all)", min_value=0, max_value=100000, value=400, step=50)
        stride = st.selectbox("Processing speed", [1, 2, 3, 5], index=0, format_func=lambda x: {1: "Every frame (best coverage)", 2: "Every 2nd frame", 3: "Every 3rd frame", 5: "Every 5th frame (faster, may miss brief appearances)"}[x])

        nms_iou = 0.50
        activation = 0.03
        lost_buffer = 5
        match_threshold = 0.80
        trail_length = 20
        if view_mode == "Advanced / analysis":
            with st.expander("Advanced parameters", expanded=True):
                nms_iou = st.slider("NMS IoU threshold", 0.20, 0.80, 0.50, 0.05)
                activation = st.slider("ByteTrack activation threshold", 0.01, 0.50, 0.03, 0.01)
                lost_buffer = st.slider("Lost-track buffer (frames)", 1, 60, 5, 1)
                match_threshold = st.slider("ByteTrack matching threshold", 0.10, 0.95, 0.80, 0.05)
                trail_length = st.slider("Track trail length", 5, 50, 20, 1)
                st.caption("Tracker values are engineering defaults and have not been fully validated. Detection score is not a calibrated probability.")
        run_clicked = st.button("▶ Analyze video", type="primary", use_container_width=True)

    # Preserve completed output paths across Streamlit reruns.
    if "snakewatch_output_video" not in st.session_state:
        st.session_state["snakewatch_output_video"] = None
    if "snakewatch_best_frame" not in st.session_state:
        st.session_state["snakewatch_best_frame"] = None


    st.caption("Research prototype · alerts are provisional · no SMS delivery or calibrated distance/risk estimate is implemented in this live view.")

    if run_clicked:
        try:
            processor, model, device = load_model(checkpoint.strip(), PROCESSOR_ID)
            cap, temp_video_path = open_video(uploaded, video_path)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError("OpenCV could not open this video. Try an MP4 encoded with H.264 or upload another clip.")
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            source_fps = source_fps if math.isfinite(source_fps) and source_fps > 0 else 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            tracker = build_tracker(source_fps, activation, lost_buffer, match_threshold)
            histories: dict[int, deque] = defaultdict(lambda: deque())
            st.info(f"Model device: **{device}** · Source: **{source_fps:.2f} FPS** · Frames reported by video: **{total_frames or 'unknown'}**")
            progress = st.progress(0.0, text="Starting inference…")
            preview = st.empty()
            stat_cols = st.columns(5)
            frame_metric = stat_cols[0].empty()
            fps_metric = stat_cols[1].empty()
            det_metric = stat_cols[2].empty()
            ids_metric = stat_cols[3].empty()
            status_metric = stat_cols[4].empty()
            status_line = st.empty()
            log_area = st.empty()
            records_all: list[dict[str, Any]] = []
            diagnostic_rows: list[dict[str, Any]] = []
            # Save an annotated MP4 from the frames actually analyzed. When frame
            # stepping is enabled, lower the output FPS proportionally so duration
            # stays close to the source duration.
            run_stamp = int(time.time())
            raw_output_video_path = str(Path(tempfile.gettempdir()) / f"snakewatch_raw_{run_stamp}.mp4")
            output_video_path = str(Path(tempfile.gettempdir()) / f"snakewatch_annotated_{run_stamp}.mp4")
            video_writer = None
            best_frame_path = str(Path(tempfile.gettempdir()) / f"snakewatch_best_frame_{int(time.time())}.jpg")
            best_score = -1.0
            best_frame_index = None
            best_frame_time = None
            frames_processed = 0
            frames_with_snake = 0
            frame_detection_flags = []
            unique_ids: set[int] = set()
            start_t = time.perf_counter()
            frame_index = 0
            last_preview_t = start_t
            stop_reason = "Video ended"
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                current_index = frame_index
                frame_index += 1
                if current_index % int(stride) != 0:
                    continue
                if int(max_frames) > 0 and frames_processed >= int(max_frames):
                    stop_reason = f"Stopped at configured limit ({int(max_frames)} processed frames)"
                    break
                infer_start = time.perf_counter()
                detections = detect(frame, processor, model, device, threshold, nms_iou)
                tracked = tracker.update_with_detections(to_sv(detections))
                # Keep a compact per-frame trace so we can tell whether detections
                # are being filtered before tracking or dropped by ByteTrack.
                diagnostic_rows.append({
                    "frame_index": current_index,
                    "video_time_s": round(current_index / source_fps, 3),
                    "detections_before_tracking": len(detections),
                    "detections_after_tracking": len(tracked),
                    "detector_scores": ", ".join(f"{d['confidence']:.4f}" for d in detections),
                })
                annotated, recs = draw_tracks(frame, tracked, histories, int(trail_length), trail_enabled=trail_enabled)
                if video_writer is None:
                    h_out, w_out = annotated.shape[:2]
                    output_fps = max(1.0, source_fps / max(1, int(stride)))
                    video_writer = cv2.VideoWriter(
                        raw_output_video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (w_out, h_out),
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError("Could not create the annotated MP4. Try a different input video format.")
                video_writer.write(annotated)
                frame_scores = [float(d.get("confidence", 0.0)) for d in detections]
                frame_best_score = max(frame_scores, default=-1.0)
                if frame_best_score > best_score:
                    best_score = frame_best_score
                    best_frame_index = current_index
                    best_frame_time = current_index / source_fps
                    cv2.imwrite(best_frame_path, annotated)
                elapsed = max(time.perf_counter() - start_t, 1e-6)
                processed_fps = (frames_processed + 1) / elapsed
                current_t = current_index / source_fps
                frames_processed += 1
                frame_detection_flags.append(bool(recs))
                if recs:
                    frames_with_snake += 1
                for r in recs:
                    r.update({"frame_index": current_index, "video_time_s": current_t})
                    records_all.append(r)
                    if r["track_id"] is not None:
                        unique_ids.add(int(r["track_id"]))
                # Update Streamlit periodically; inference, tracking and video
                # writing still run on every selected frame.
                if frames_processed % 5 == 0 or (
                    total_frames and frame_index >= total_frames
                ):
                    preview.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True,
                    )
                    frame_metric.metric("Frames processed", f"{frames_processed:,}")
                    fps_metric.metric("Measured inference FPS", f"{processed_fps:.2f}")
                    det_metric.metric("Current snake boxes", len(recs))
                    ids_metric.metric("Track IDs seen", len(unique_ids))
                    status_metric.metric(
                        "Provisional status",
                        "Candidate present" if recs else "No candidate"
                    )
                    status_line.caption(
                        f"Source frame {current_index + 1:,} / "
                        f"{total_frames or '?'} · Video timestamp {current_t:.2f}s · "
                        f"Current-frame inference "
                        f"{time.perf_counter() - infer_start:.2f}s"
                    )

                if frames_processed % 5 == 0:
                    if total_frames:
                        progress.progress(
                            min(1.0, frame_index / total_frames),
                            text=f"Processing source frame {frame_index:,} "
                                 f"of {total_frames:,}…"
                        )
                    else:
                        progress.progress(
                            min(
                                0.99,
                                frames_processed /
                                max(1, int(max_frames) or frames_processed + 100)
                            ),
                            text=f"Processed {frames_processed:,} frames…"
                        )
            cap.release()
            if video_writer is not None:
                video_writer.release()
                conversion = subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", raw_output_video_path,
                        "-an", "-c:v", "libx264", "-preset", "fast",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        output_video_path,
                    ],
                    capture_output=True,
                    text=True,
                )
                if conversion.returncode != 0:
                    raise RuntimeError(
                        "H.264 video conversion failed: "
                        + conversion.stderr[-2000:]
                    )
                Path(raw_output_video_path).unlink(missing_ok=True)


            st.session_state["snakewatch_output_video"] = (
                output_video_path if Path(output_video_path).is_file() else None
            )
            st.session_state["snakewatch_best_frame"] = (
                best_frame_path if Path(best_frame_path).is_file() else None
            )
            progress.progress(1.0, text="Inference complete")
            total_elapsed = max(time.perf_counter() - start_t, 1e-6)
            preview.empty()
            # SNAKEWATCH_GRADED_ALERT_V1
            # Detection rate and persistence are evidence indicators, not
            # calibrated probabilities or proof that a detection is a snake.
            detection_rate = (
                100.0 * frames_with_snake / frames_processed
                if frames_processed > 0 else 0.0
            )

            confidence_values = [
                float(r.get("confidence", 0.0))
                for r in records_all
                if r.get("confidence") is not None
            ]
            mean_confidence = (
                sum(confidence_values) / len(confidence_values)
                if confidence_values else 0.0
            )

            longest_run = 0
            current_run = 0
            for detected in frame_detection_flags:
                if detected:
                    current_run += 1
                    longest_run = max(longest_run, current_run)
                else:
                    current_run = 0

            # SNAKEWATCH_ALERT_THRESHOLDS_V2
            # Detection rate is not a calibrated probability of snake presence.
            if detection_rate < 15.0:
                alert_level = "NONE"
                snake_status = "NO CONFIRMED ALERT"
                alert_message = (
                    "No significant snake-like activity detected. "
                    "Any isolated detections may be false positives. "
                    "This does not guarantee the area is snake-free."
                )
            elif detection_rate < 30.0:
                alert_level = "MODERATE"
                snake_status = "POSSIBLE SNAKE"
                alert_message = (
                    "Possible snake-like activity detected across multiple frames. "
                    "Maintain a safe distance and review the annotated video."
                )
            elif detection_rate < 50.0:
                alert_level = "HIGH"
                snake_status = "PERSISTENT DETECTIONS"
                alert_message = (
                    "Persistent snake-like detections observed. "
                    "Do not approach the suspected object; arrange safe verification."
                )
            else:
                alert_level = "CRITICAL"
                snake_status = "SUSTAINED DETECTIONS"
                alert_message = (
                    "Sustained snake-like detections observed. "
                    "Keep away from the suspected area and seek trained assistance "
                    "if a snake is suspected."
                )

            st.session_state["sw_glance"] = {
                "snake_status": snake_status,
                "detection_rate": detection_rate,
                "confidence": mean_confidence,
                "alert_level": alert_level,
                "longest_run": longest_run,
            }

            # SNAKEWATCH_PROMINENT_ALERT_CARD_V1
            # A prominent status card replaces the default Streamlit alert banner.
            alert_styles = {
                "NONE": {
                    "bg": "#10251F", "border": "#55A878",
                    "accent": "#8FE0AE", "icon": "✓",
                    "title": "NO CONFIRMED ALERT",
                },
                "MODERATE": {
                    "bg": "#30260F", "border": "#D9A441",
                    "accent": "#FFD27A", "icon": "!",
                    "title": "MODERATE ALERT",
                },
                "HIGH": {
                    "bg": "#351B10", "border": "#F08A42",
                    "accent": "#FFAD72", "icon": "!",
                    "title": "HIGH ALERT",
                },
                "CRITICAL": {
                    "bg": "#351419", "border": "#F05C68",
                    "accent": "#FF9AA3", "icon": "!",
                    "title": "CRITICAL ALERT",
                },
            }
            style = alert_styles.get(alert_level, alert_styles["NONE"])

            st.markdown(
                f"""
                <div style="
                    background:{style['bg']};
                    border:2px solid {style['border']};
                    border-left:9px solid {style['border']};
                    border-radius:16px;
                    padding:22px 24px;
                    margin:8px 0 18px 0;
                    box-shadow:0 5px 18px rgba(0,0,0,0.20);
                ">
                  <div style="display:flex;align-items:center;gap:16px;">
                    <div style="
                        flex:0 0 54px;height:54px;border-radius:14px;
                        background:{style['border']};
                        color:#FFFFFF;font-size:32px;font-weight:800;
                        display:flex;align-items:center;justify-content:center;
                    ">{style['icon']}</div>
                    <div style="flex:1;">
                      <div style="
                          color:{style['accent']};font-size:12px;
                          font-weight:700;letter-spacing:2px;
                          margin-bottom:5px;
                      ">SNAKEWATCH · ANALYSIS RESULT</div>
                      <div style="
                          color:#FFFFFF;font-size:clamp(22px,3vw,32px);
                          font-weight:800;line-height:1.2;
                      ">{style['title']}</div>
                      <div style="
                          color:#F1F5F2;font-size:16px;
                          line-height:1.55;margin-top:9px;
                      ">{alert_message}</div>
                    </div>
                  </div>
                  <div style="
                      display:flex;flex-wrap:wrap;gap:12px;
                      border-top:1px solid rgba(255,255,255,0.20);
                      margin-top:18px;padding-top:15px;
                  ">
                    <div style="flex:1;min-width:145px;">
                      <div style="color:#C4D4CC;font-size:11px;
                                  letter-spacing:1px;font-weight:700;">
                        FRAMES WITH DETECTIONS
                      </div>
                      <div style="color:#FFFFFF;font-size:25px;
                                  font-weight:800;margin-top:4px;">
                        {detection_rate:.1f}%
                      </div>
                      <div style="color:#D2DDD7;font-size:12px;">
                        {frames_with_snake:,} of {frames_processed:,} frames
                      </div>
                    </div>
                    <div style="flex:1;min-width:145px;">
                      <div style="color:#C4D4CC;font-size:11px;
                                  letter-spacing:1px;font-weight:700;">
                        LONGEST CONSECUTIVE RUN
                      </div>
                      <div style="color:#FFFFFF;font-size:25px;
                                  font-weight:800;margin-top:4px;">
                        {longest_run} frames
                      </div>
                      <div style="color:#D2DDD7;font-size:12px;">
                        Consecutive detection evidence
                      </div>
                    </div>
                  </div>
                  <div style="
                      color:#F2DCA6;font-size:12px;line-height:1.5;
                      margin-top:15px;
                  ">
                    Safety note: detection rate is not a calibrated probability.
                    This result cannot guarantee the area is snake-free.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.success(f"{stop_reason}. Processed {frames_processed:,} frames in {total_elapsed:.1f}s ({frames_processed / total_elapsed:.2f} FPS measured).")
            st.info("Provisional detections only: this live view does not apply the full zone/risk logic from later research stages. A persistent track is not, by itself, proof of a snake intrusion.")
            run_config = {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint_path": str(Path(checkpoint).expanduser()),
                "processor_id": PROCESSOR_ID,
                "torch_version": torch.__version__,
                "supervision_version": getattr(sv, "__version__", "unknown"),
                "device": str(device),
                "video_name": uploaded.name if uploaded is not None else str(video_path),
                "preset": preset_label,
                "preset_validation_status": "0.055 identified as validation-selected; 0.03 and 0.08 are candidate settings requiring evaluation",
                "detection_threshold": float(threshold),
                "nms_iou_threshold": float(nms_iou),
                "tracker_activation_threshold": float(activation),
                "tracker_lost_buffer_frames": int(lost_buffer),
                "tracker_matching_threshold": float(match_threshold),
                "frame_step": int(stride),
                "max_processed_frames": int(max_frames),
                "trail_enabled": bool(trail_enabled),
                "trail_length": int(trail_length),
                "source_fps": float(source_fps),
                "frames_processed": int(frames_processed),
                "frames_with_candidates": int(frames_with_snake),
                "measured_inference_fps": round(frames_processed / total_elapsed, 4),
                "status_semantics": "Provisional candidates; no calibrated probability or verified zone intrusion alert",
            }
            st.download_button("Download run configuration (JSON)", data=json.dumps(run_config, indent=2), file_name="snakewatch_run_config.json", mime="application/json")
            if diagnostic_rows:
                diag_df = pd.DataFrame(diagnostic_rows)
                st.subheader("Detection pipeline diagnostics")
                st.caption("Compare detections before tracking with detections returned by ByteTrack. If the first count is positive and the second is zero, investigate tracker filtering/confirmation.")
                st.dataframe(diag_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download pipeline diagnostics (CSV)",
                    data=diag_df.to_csv(index=False).encode("utf-8"),
                    file_name="snakewatch_pipeline_diagnostics.csv",
                    mime="text/csv",
                )
            if records_all:
                df = pd.DataFrame(records_all)
                st.subheader("Current run — detection observations")
                st.dataframe(df.tail(250), use_container_width=True, hide_index=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download detection log (CSV)", data=csv, file_name="snakewatch_live_detections.csv", mime="text/csv")
            else:
                st.warning("No snake detections were returned for the processed frames. This may be a true negative or a threshold/model/video issue.")
            st.caption("Tracks, centers, and trails are in image-pixel coordinates. No pixel-to-meter calibration or SMS delivery is implemented here.")
            if temp_video_path:
                try:
                    Path(temp_video_path).unlink(missing_ok=True)
                except OSError:
                    pass
        except Exception as exc:
            st.error(f"Live inference failed: {type(exc).__name__}: {exc}")
            st.exception(exc)
    else:
        st.info("Choose a video and select **Analyze video**. Operator mode keeps technical parameters out of the way; Advanced / analysis exposes them for research tuning.")
        st.markdown("**How to read the overlay**")
        st.markdown("""- **Tentative:** a detection without an assigned persistent track ID.
    - **Tracked:** a detection assigned to a tracker ID in the current frame.
    - **Higher / lower detector score:** score band only, not a probability that the object is a snake.
    - **Provisional status:** this app does not yet issue verified zone-based intrusion alerts.""")
        st.markdown("**Demo fallback:** keep the existing offline replay dashboard (`app.py`) and its saved JSON/video outputs available for presentations if Colab GPU or the public tunnel is unavailable.")


    # Keep completed results visible after download-button reruns.
    # SNAKEWATCH_GLANCE_METRICS_V2
    # SNAKEWATCH_GLANCE_METRICS_V1
    def show_snakewatch_glance():
        metrics = st.session_state.get("sw_glance")
        if not metrics:
            return

        panels = [
            ("SNAKE STATUS", str(metrics.get("snake_status", "Awaiting analysis"))),
            ("DETECTION RATE", f"{float(metrics.get('detection_rate', 0)):.1f}%"),
            ("MEAN MODEL CONFIDENCE",
             f"{float(metrics['confidence']):.3f}"
             if metrics.get("confidence") is not None else "N/A"),
            ("ALERT LEVEL / LONGEST RUN",
             f"{metrics.get('alert_level', 'NONE')} / "
             f"{int(metrics.get('longest_run', 0))} frames"),
        ]

        cols = st.columns(4, gap="small")
        for col, (label, value) in zip(cols, panels):
            with col:
                st.markdown(
                    f"""
                    <div style="min-height:105px;padding:16px 12px;
                        border:1px solid rgba(150,170,160,.4);
                        border-radius:12px;
                        background:rgba(100,130,115,.12);
                        overflow-wrap:anywhere;">
                        <div style="font-size:11px;font-weight:700;
                            letter-spacing:1px;">{label}</div>
                        <div style="font-size:19px;font-weight:750;
                            margin-top:12px;">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )



    # SNAKEWATCH_BROWSER_BUZZER_V1
    import streamlit.components.v1 as components


    def render_snakewatch_browser_alarm(alert_level, detection_rate, longest_run, alarm_id):
        import json

        alert_json = json.dumps(str(alert_level).upper())
        rate_json = json.dumps(float(detection_rate))
        run_json = json.dumps(int(longest_run))
        id_json = json.dumps(str(alarm_id))

        html = r"""
        <div style="box-sizing:border-box;width:100%;padding:18px;
            border:1px solid #365342;border-radius:16px;background:#122219;
            color:#edf7ef;font-family:Arial,Helvetica,sans-serif;">

          <div style="font-size:20px;font-weight:750;margin-bottom:5px">
            🔊 SnakeWatch Audio Alerts
          </div>
          <div style="font-size:13px;color:#b8cdbd;margin-bottom:16px">
            Moderate: one short beep · High: three short beeps · Critical: continuous 6.5-second tone
          </div>

          <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px">
            <button id="enableSound" type="button"
              style="border:0;border-radius:9px;padding:11px 15px;
              background:#c9f0d3;color:#102519;font-weight:700;cursor:pointer">
              Enable audio & test
            </button>
            <button id="testSound" type="button"
              style="border:1px solid #547561;border-radius:9px;padding:11px 15px;
              background:#1b3225;color:#f0f7f1;font-weight:700;cursor:pointer">
              Test sound
            </button>
          </div>

          <div id="soundStatus" role="status"
            style="font-size:13px;line-height:1.5;color:#c5d8ca;margin-bottom:12px">
            Audio requires activation in this browser.
          </div>

          <div id="alertCard"
            style="padding:15px;border-radius:11px;background:#1b3023;
            border:1px solid #45634e">
            <div id="alertTitle"
              style="font-size:16px;font-weight:800;margin-bottom:6px">
              Checking alert status
            </div>
            <div id="alertDetails"
              style="font-size:13px;line-height:1.6;color:#d2e1d6">
              Run video analysis to evaluate the current alert level.
            </div>
          </div>

          <div style="font-size:11px;line-height:1.5;color:#9fb7a6;margin-top:12px">
            Keep this dashboard open and your device volume on. Alerts are
            provisional model outputs, not confirmation that a snake is present.
          </div>
        </div>

        <script>
        (() => {
          const level = __ALERT__;
          const rate = __RATE__;
          const run = __RUN__;
          const alarmId = __ID__;

          const enableButton = document.getElementById("enableSound");
          const testButton = document.getElementById("testSound");
          const soundStatus = document.getElementById("soundStatus");
          const alertCard = document.getElementById("alertCard");
          const alertTitle = document.getElementById("alertTitle");
          const alertDetails = document.getElementById("alertDetails");

          const enabledKey = "snakewatch_sound_enabled";
          const playedKey = "snakewatch_played_alert";
          let audioContext = null;
          let busy = false;

          function isEnabled() {
            try {
              return localStorage.getItem(enabledKey) === "yes";
            } catch (_) {
              return false;
            }
          }

          async function getAudioContext() {
            const AudioContextClass =
              window.AudioContext || window.webkitAudioContext;

            if (!AudioContextClass) {
              throw new Error("Web Audio is not supported by this browser.");
            }

            if (!audioContext || audioContext.state === "closed") {
              audioContext = new AudioContextClass();
            }

            if (audioContext.state !== "running") {
              await audioContext.resume();
            }

            if (audioContext.state !== "running") {
              throw new Error("Audio is blocked. Click Enable audio & test.");
            }

            return audioContext;
          }

          function tone(ctx, startTime, duration, frequency = 880) {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();

            osc.type = "sine";
            osc.frequency.setValueAtTime(frequency, startTime);
            gain.gain.setValueAtTime(0.001, startTime);
            gain.gain.linearRampToValueAtTime(0.18, startTime + 0.04);
            gain.gain.setValueAtTime(0.18, startTime + duration - 0.05);
            gain.gain.linearRampToValueAtTime(0.001, startTime + duration);

            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(startTime);
            osc.stop(startTime + duration);
          }

          async function playHigh() {
            if (busy) return;
            busy = true;
            try {
              const ctx = await getAudioContext();
              const now = ctx.currentTime + 0.05;
              // Three distinct short beeps.
              tone(ctx, now, 0.22);
              tone(ctx, now + 0.36, 0.22);
              tone(ctx, now + 0.72, 0.22);
              soundStatus.textContent = "HIGH alert sound: three short beeps.";
              setTimeout(() => { busy = false; }, 1300);
            } catch (e) {
              busy = false;
              soundStatus.textContent = e.message;
            }
          }

          async function playModerate() {
            if (busy) return;
            busy = true;
            try {
              const ctx = await getAudioContext();
              tone(ctx, ctx.currentTime + 0.05, 0.22, 880);
              soundStatus.textContent = "MODERATE alert sound: one short beep.";
              setTimeout(() => { busy = false; }, 500);
            } catch (e) {
              busy = false;
              soundStatus.textContent = e.message;
            }
          }

          async function playCritical() {
            if (busy) return;
            busy = true;
            try {
              const ctx = await getAudioContext();
              // One continuous tone lasting approximately 6.5 seconds.
              tone(ctx, ctx.currentTime + 0.05, 6.5, 740);
              soundStatus.textContent =
                "CRITICAL alert sound: continuous tone for 6.5 seconds.";
              setTimeout(() => { busy = false; }, 6800);
            } catch (e) {
              busy = false;
              soundStatus.textContent = e.message;
            }
          }

          async function testSound() {
            try {
              const ctx = await getAudioContext();
              tone(ctx, ctx.currentTime + 0.05, 0.25, 880);
              soundStatus.textContent = "Test tone played.";
            } catch (e) {
              soundStatus.textContent = e.message;
            }
          }

          enableButton.addEventListener("click", async () => {
            try {
              // Save preference only after the user's explicit interaction.
              localStorage.setItem(enabledKey, "yes");
            } catch (_) {}
            await testSound();
            enableButton.textContent = "Audio enabled ✓";
          });

          testButton.addEventListener("click", testSound);

          if (isEnabled()) {
            enableButton.textContent = "Audio enabled ✓";
            soundStatus.textContent =
              "Audio preference saved. Browser policy may still block playback.";
          }

          if (level === "CRITICAL") {
            alertCard.style.background = "#481b20";
            alertCard.style.border = "2px solid #ff626b";
            alertTitle.style.color = "#ffffff";
            alertTitle.textContent = "🚨 CRITICAL PROVISIONAL ALERT";
            alertDetails.textContent =
              "Detection rate: " + rate.toFixed(1) + "% · Longest run: " +
              run + " frames. Keep away from the suspected object and verify safely.";
          } else if (level === "HIGH") {
            alertCard.style.background = "#402914";
            alertCard.style.border = "2px solid #f5a742";
            alertTitle.style.color = "#fff2d8";
            alertTitle.textContent = "⚠ HIGH PROVISIONAL ALERT";
            alertDetails.textContent =
              "Detection rate: " + rate.toFixed(1) + "% · Longest run: " +
              run + " frames. Three short beeps are assigned to this alert level.";
          } else if (level === "MODERATE") {
            alertTitle.textContent = "MODERATE — Possible snake-like activity";
            alertDetails.textContent =
              "Detection rate: " + rate.toFixed(1) + "% · One short beep indicates moderate activity.";
          } else {
            alertTitle.textContent = "No confirmed alert";
            alertDetails.textContent =
              "Detection rate: " + rate.toFixed(1) +
              "%. This does not guarantee the area is snake-free.";
          }

          // Play at most once per distinct level + alert ID in this browser.
          // Audio only attempts automatic playback after the user has enabled it.
          if (isEnabled() && (level === "MODERATE" || level === "HIGH" || level === "CRITICAL")) {
            const thisAlertKey = level + ":" + alarmId;
            let alreadyPlayed = false;

            try {
              alreadyPlayed = localStorage.getItem(playedKey) === thisAlertKey;
            } catch (_) {}

            if (!alreadyPlayed) {
              try {
                localStorage.setItem(playedKey, thisAlertKey);
              } catch (_) {}

              if (level === "MODERATE") playModerate();
              if (level === "HIGH") playHigh();
              if (level === "CRITICAL") playCritical();
            }
          } else if (level === "MODERATE" || level === "HIGH" || level === "CRITICAL") {
            soundStatus.textContent =
              "An audible alert is available. Click Enable audio & test to activate sound.";
          }
        })();
        </script>
        """

        html = (html
            .replace("__ALERT__", alert_json)
            .replace("__RATE__", rate_json)
            .replace("__RUN__", run_json)
            .replace("__ID__", id_json))

        components.html(html, height=325, scrolling=False)


    saved_video = st.session_state.get("snakewatch_output_video")
    saved_frame = st.session_state.get("snakewatch_best_frame")

    if saved_video or saved_frame:
        st.divider()
        st.subheader("Completed SnakeWatch Results")
        show_snakewatch_glance()

        metrics = st.session_state.get("sw_glance", {})
        if metrics:
            render_snakewatch_browser_alarm(
                metrics.get("alert_level", "NONE"),
                metrics.get("detection_rate", 0.0),
                metrics.get("longest_run", 0),
                metrics.get(
                    "alarm_id",
                    f'{metrics.get("alert_level")}-'
                    f'{metrics.get("detection_rate")}-'
                    f'{metrics.get("longest_run")}'
                ),
            )


        if saved_video and Path(saved_video).is_file():
            st.subheader("Annotated Video")
            st.video(saved_video)
            st.download_button(
                "Download Annotated Video (MP4)",
                data=Path(saved_video).read_bytes(),
                file_name="snakewatch_annotated.mp4",
                mime="video/mp4",
                key="download_annotated_video",
            )

        if saved_frame and Path(saved_frame).is_file():
            st.subheader("Best Candidate Frame")
            st.image(saved_frame, use_container_width=True)
            st.download_button(
                "Download Candidate Image (JPG)",
                data=Path(saved_frame).read_bytes(),
                file_name="snakewatch_best_frame.jpg",
                mime="image/jpeg",
                key="download_candidate_image",
            )



with tab_boundary:
    st.subheader("SnakeWatch Boundary Simulation")

    # SNAKEWATCH_BROWSER_AUDIO_ALERT_CONTROLLER_V1
    audio_html = r"""
    <div style="font-family:Arial,sans-serif;background:#14251b;
                color:#e7f0e9;padding:12px 14px;border-radius:12px;
                border:1px solid #496453;">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
        <button id="enableAudio"
          style="background:#315c40;color:white;border:0;border-radius:8px;
                 padding:10px 16px;font-weight:700;cursor:pointer;">
          Enable Sound Alerts
        </button>
        <span id="audioStatus" style="font-size:14px;">
          Sound disabled — enable audio before starting analysis.
        </span>
      </div>
      <div id="alertStatus" style="font-size:13px;margin-top:9px;color:#a9c3b1;">
        Alert state: CLEAR
      </div>
    </div>

    <script>
    (() => {
      const STATE_URL = "https://8503-gpu-t4-s-ass1c1-3ar477pem2c58-c.asia-southeast1.prod.r.colab.dev/state";
      const enableButton = document.getElementById("enableAudio");
      const audioStatus = document.getElementById("audioStatus");
      const alertStatus = document.getElementById("alertStatus");

      let audioContext = null;
      let enabled = false;
      let currentMode = "CLEAR";
      let soundTimer = null;

      function tone(frequency, duration, volume=0.18) {
        if (!enabled || !audioContext) return;

        const oscillator = audioContext.createOscillator();
        const gain = audioContext.createGain();

        oscillator.type = "sine";
        oscillator.frequency.value = frequency;
        gain.gain.setValueAtTime(volume, audioContext.currentTime);
        gain.gain.exponentialRampToValueAtTime(
          0.001, audioContext.currentTime + duration
        );

        oscillator.connect(gain);
        gain.connect(audioContext.destination);
        oscillator.start();
        oscillator.stop(audioContext.currentTime + duration);
      }

      function stopSound() {
        if (soundTimer !== null) {
          clearInterval(soundTimer);
          soundTimer = null;
        }
      }

      function approachingPattern() {
        let beep = 0;
        const frequencies = [850, 850, 850];

        soundTimer = setInterval(() => {
          if (beep < 3) {
            tone(frequencies[beep], 0.18, 0.16);
            beep++;
          } else {
            beep = 0;
          }
        }, 260);
      }

      function insidePattern() {
        let phase = 0;

        soundTimer = setInterval(() => {
          tone(phase % 2 === 0 ? 1050 : 720, 0.38, 0.22);
          phase++;
        }, 450);
      }

      function applyState(state) {
        const nextMode = state.state || "CLEAR";

        alertStatus.textContent =
          "Alert state: " + nextMode +
          (state.inside_track_ids?.length
            ? " | Inside tracks: " + state.inside_track_ids.join(", ")
            : "") +
          (state.approaching_track_ids?.length
            ? " | Approaching tracks: " +
              state.approaching_track_ids.join(", ")
            : "");

        if (nextMode === currentMode) return;

        currentMode = nextMode;
        stopSound();

        if (!enabled) return;

        if (nextMode === "INSIDE") {
          insidePattern();
        } else if (nextMode === "APPROACHING") {
          approachingPattern();
        }
      }

      enableButton.addEventListener("click", async () => {
        try {
          const AudioContextClass =
            window.AudioContext || window.webkitAudioContext;

          if (!AudioContextClass) {
            audioStatus.textContent = "Web Audio is not supported.";
            return;
          }

          if (!audioContext) {
            audioContext = new AudioContextClass();
          }

          await audioContext.resume();
          enabled = true;
          enableButton.textContent = "Sound Alerts Enabled";
          enableButton.disabled = true;
          enableButton.style.opacity = "0.7";
          audioStatus.textContent = "Sound enabled — monitoring alerts.";

          if (currentMode === "INSIDE") {
            insidePattern();
          } else if (currentMode === "APPROACHING") {
            approachingPattern();
          }
        } catch (error) {
          audioStatus.textContent = "Could not enable sound: " + error.message;
        }
      });

      window.snakeWatchReceiveState = function(state) {
        applyState(state);
        audioStatus.textContent = "Sound enabled — monitoring alerts.";
      };

      function pollState() {
        const oldScript = document.getElementById("snakeWatchAudioPoll");
        if (oldScript) oldScript.remove();

        const script = document.createElement("script");
        script.id = "snakeWatchAudioPoll";
        script.src = STATE_URL.slice(0, STATE_URL.lastIndexOf("/")) + "/state.js"
          + "?callback=snakeWatchReceiveState&poll=" + Date.now();

        script.onerror = function() {
          audioStatus.textContent = "Alert service connection failed.";
          script.remove();
        };

        document.body.appendChild(script);
      }

      pollState();
      setInterval(pollState, 200);
    })();
    </script>
    """

    components.html(audio_html, height=105, scrolling=False)

    st.caption(
        "Draw boundary polygons on two video previews and save "
        "their coordinates for the movement-analysis stage."
    )

    import sys
    sys.path.insert(0, str(Path(r'/content/drive/MyDrive/snake_detection/Snake_detection/SnakeWatch_Speed_Test_20261106')))

    import importlib
    import sys

    _sw_module_name = "snakewatch_boundary_simulation_integrated"

    if _sw_module_name in sys.modules:
        boundary_module = importlib.reload(sys.modules[_sw_module_name])
    else:
        boundary_module = importlib.import_module(_sw_module_name)

    render_boundary_simulation = boundary_module.render_boundary_simulation

    render_boundary_simulation(
        load_model_fn=load_model,
        detect_fn=detect,
        tracker_factory=lambda fps, activation_threshold, lost_buffer, match_threshold: build_tracker(fps, activation_threshold, lost_buffer, match_threshold),
        to_sv_fn=to_sv,
        checkpoint_default=str(CHECKPOINT_DEFAULT),
        processor_id=PROCESSOR_ID,
    )


# SNAKEWATCH_HUMAN_SAFETY_TAB_V1
with tab_safety:
    TEST_DIR = Path(
        "/content/drive/MyDrive/snake_detection/Snake_detection/"
        "SnakeWatch_Speed_Test_20261106"
    )
    st.header("Human Safety Monitoring")
    st.caption(
        "Combined snake/person detection, tracking, dynamic proximity zones "
        "and temporally confirmed alerts. Distances are measured in image "
        "pixels, not metres. Research prototype only."
    )

    import sys
    sys.path.insert(0, str(TEST_DIR))
    from snakewatch_safety_processor import process_safety_video

    @st.cache_resource(show_spinner="Loading YOLO person detector…")
    def load_person_detector(model_name="yolov8n.pt"):
        from ultralytics import YOLO
        return YOLO(model_name)

    safety_upload = st.file_uploader(
        "Upload a human–snake or CCTV video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="safety_video_upload",
    )

    safety_checkpoint = st.text_input(
        "Fine-tuned snake checkpoint",
        value=str(CHECKPOINT_DEFAULT),
        key="safety_checkpoint_path",
    )

    safety_col1, safety_col2 = st.columns(2)

    with safety_col1:
        safety_stride = st.selectbox(
            "Frame processing interval",
            [1, 2, 3],
            index=0,
            format_func=lambda x: {
                1: "Every frame (best coverage)",
                2: "Every 2nd frame",
                3: "Every 3rd frame",
            }[x],
            key="safety_stride",
        )

    with safety_col2:
        safety_max_frames = st.number_input(
            "Maximum processed frames (0 = entire video)",
            min_value=0,
            max_value=100000,
            value=0,
            step=100,
            key="safety_max_frames",
        )

    st.info(
        "Snake confidence: 0.055. Tiled inference: 800 px tiles "
        "with 25% overlap."
    )

    if st.button(
        "▶ Run Human Safety Analysis",
        type="primary",
        disabled=safety_upload is None,
        key="run_safety_analysis",
    ):
        input_temp = None

        try:
            # Reuse the existing cached RT-DETRv2 model.
            processor, snake_model, device = load_model(
                safety_checkpoint.strip(), PROCESSOR_ID
            )

            # Separate YOLO model for person detection.
            person_model = load_person_detector("yolov8n.pt")

            suffix = Path(safety_upload.name).suffix or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as temp_in:
                temp_in.write(safety_upload.getvalue())
                input_temp = temp_in.name

            run_id = int(time.time())
            output_path = str(
                TEST_DIR / f"human_safety_annotated_{run_id}.mp4"
            )

            progress_bar = st.progress(0.0, text="Preparing video…")
            preview_slot = st.empty()
            status_slot = st.empty()

            def safety_progress(current, total):
                if total and total > 0:
                    progress_bar.progress(
                        min(1.0, current / total),
                        text=f"Processed {current:,} / {total:,} frames",
                    )
                else:
                    status_slot.caption(
                        f"Processed {current:,} frames"
                    )

            def safety_preview(frame_bgr, current, total):
                preview_slot.image(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )
                status_slot.caption(
                    f"Preview: processed frame {current:,} "
                    f"of {total or 'unknown'}"
                )

            result = process_safety_video(
                input_path=input_temp,
                output_path=output_path,
                processor=processor,
                snake_model=snake_model,
                device=device,
                person_model=person_model,
                tracker_factory=build_tracker,
                progress_callback=safety_progress,
                preview_callback=safety_preview,
                max_frames=int(safety_max_frames),
                frame_stride=int(safety_stride),
                snake_threshold=0.055,
            )

            progress_bar.progress(
                1.0, text="Safety analysis complete"
            )
            st.session_state["snakewatch_safety_result"] = result
            st.session_state["snakewatch_safety_output"] = output_path
            st.success("Human Safety analysis completed.")

        except Exception as exc:
            st.error(
                f"Safety analysis failed: {type(exc).__name__}: {exc}"
            )
            st.exception(exc)

        finally:
            if input_temp:
                Path(input_temp).unlink(missing_ok=True)

    # Display the last completed safety analysis.
    safety_result = st.session_state.get("snakewatch_safety_result")
    safety_output = st.session_state.get("snakewatch_safety_output")

    if safety_result:
        st.subheader("Analysis Summary")
        summary = safety_result["summary"]

        metric_cols = st.columns(4)
        metric_cols[0].metric(
            "Frames processed", f"{summary['frames_processed']:,}"
        )
        metric_cols[1].metric(
            "Processing FPS", f"{summary['processing_fps']:.2f}"
        )
        metric_cols[2].metric(
            "Warning events", summary["warning_events"]
        )
        metric_cols[3].metric(
            "Critical events", summary["critical_events"]
        )

        if safety_output and Path(safety_output).is_file():
            st.subheader("Annotated Safety Video")
            st.video(safety_output)

            st.download_button(
                "Download Annotated Safety Video",
                data=Path(safety_output).read_bytes(),
                file_name="snakewatch_safety_annotated.mp4",
                mime="video/mp4",
                key="download_safety_video",
            )

        st.subheader("Confirmed Alert Events")
        events = safety_result["events"]

        if events:
            event_df = pd.DataFrame(events)
            st.dataframe(
                event_df,
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download Alert Events CSV",
                data=event_df.to_csv(index=False).encode("utf-8"),
                file_name="snakewatch_alert_events.csv",
                mime="text/csv",
                key="download_safety_events",
            )
        else:
            st.info(
                "No temporally confirmed human-proximity alerts "
                "were generated."
            )

        with st.expander("Full Processing Metadata"):
            st.json(summary)

# END SNAKEWATCH_HUMAN_SAFETY_TAB_V1
