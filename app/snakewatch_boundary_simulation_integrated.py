import random
import json
import time
import requests
from pathlib import Path

import cv2
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas


VIDEO_DIR = Path(
    "/content/drive/MyDrive/snake_detection/Snake_detection/snake_videos"
)

BOUNDARY_FILE = Path(r"/content/drive/MyDrive/snake_detection/Snake_detection/SnakeWatch_Speed_Test_20261106/snakewatch_boundaries.json")

# SNAKEWATCH_PERSISTENT_RESULT_STORAGE_V1
import pandas as pd

RESULT_STORE = BOUNDARY_FILE.parent / "persistent_results"

def save_video_result(result, video_number, selected_names):
    RESULT_STORE.mkdir(parents=True, exist_ok=True)
    folder = RESULT_STORE / f"video_{video_number}"
    folder.mkdir(parents=True, exist_ok=True)

    result["observations"].to_csv(folder / "observations.csv", index=False)
    result["events"].to_csv(folder / "events.csv", index=False)

    metadata = {
        "video_name": result["video_name"],
        "frames_processed": int(result["frames_processed"]),
        "detections_tracked": int(result["detections_tracked"]),
        "fps": float(result["fps"]),
        "annotated_video_path": result.get("annotated_video_path", ""),
        "evidence_dir": result.get("evidence_dir", ""),
        "evidence_clips_created": int(result.get("evidence_clips_created", 0)),
        "selected_names": selected_names,
        "video_number": video_number,
    }

    (folder / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    manifest = {
        "selected_names": selected_names,
        "completed_videos": sorted(
            int(x.name.split("_")[1])
            for x in RESULT_STORE.glob("video_*")
            if x.is_dir() and (x / "metadata.json").exists()
        ),
    }
    (RESULT_STORE / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def load_persisted_results(selected_names):
    manifest_path = RESULT_STORE / "manifest.json"
    if not manifest_path.exists():
        return []

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if [Path(p).name for p in manifest.get("selected_names", [])] != [Path(p).name for p in selected_names]:
            return []

        restored = []
        for number in manifest.get("completed_videos", []):
            folder = RESULT_STORE / f"video_{number}"
            metadata_path = folder / "metadata.json"
            if not metadata_path.exists():
                continue

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if [Path(p).name for p in metadata.get("selected_names", [])] != [Path(p).name for p in selected_names]:
                continue

            observations_path = folder / "observations.csv"
            events_path = folder / "events.csv"
            if not observations_path.exists() or not events_path.exists():
                continue

            try:
                metadata["observations"] = pd.read_csv(observations_path)
            except pd.errors.EmptyDataError:
                metadata["observations"] = pd.DataFrame()

            try:
                metadata["events"] = pd.read_csv(events_path)
            except pd.errors.EmptyDataError:
                metadata["events"] = pd.DataFrame()

            restored.append(metadata)

        return restored
    except Exception as exc:
        st.warning(f"Could not restore saved analysis results: {exc}")
        return []


VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"
}

CANVAS_MAX_WIDTH = 800


def get_video_library():
    return sorted([
        p for p in VIDEO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ])


def select_random_videos(previous_videos=None):
    videos = get_video_library()

    if len(videos) < 2:
        raise RuntimeError(
            f"At least two videos are required. Found: {len(videos)}"
        )

    previous_names = {
        Path(video).name for video in (previous_videos or [])
    }

    candidates = [
        video for video in videos
        if video.name not in previous_names
    ]

    if len(candidates) < 2:
        candidates = videos

    return random.sample(candidates, 2)


def read_preview_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    success, frame = cap.read()
    cap.release()

    if not success:
        return None, None

    original_height, original_width = frame.shape[:2]

    scale = min(1.0, CANVAS_MAX_WIDTH / original_width)

    display_width = int(original_width * scale)
    display_height = int(original_height * scale)

    frame = cv2.resize(
        frame,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA
    )

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    metadata = {
        "original_width": original_width,
        "original_height": original_height,
        "display_width": display_width,
        "display_height": display_height,
        "scale_x": original_width / display_width,
        "scale_y": original_height / display_height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": frame_count / fps if fps > 0 else 0
    }

    return Image.fromarray(frame), metadata


def extract_polygon_points(canvas_data, metadata):
    """
    Convert polygon points from displayed canvas coordinates
    into original video pixel coordinates.
    """

    if not canvas_data or not canvas_data.get("objects"):
        return []

    polygons = []

    for obj in canvas_data["objects"]:
        if obj.get("type", "").lower() != "polygon":
            continue

        points = obj.get("points", [])
        scale_x = metadata["scale_x"]
        scale_y = metadata["scale_y"]

        polygon = [
            [
                round(point["x"] * scale_x, 2),
                round(point["y"] * scale_y, 2)
            ]
            for point in points
        ]

        if len(polygon) >= 3:
            polygons.append(polygon)

    return polygons


def render_video_boundary(video_path, video_number, selection_id):

    st.markdown(f"#### Video {video_number}")

    st.caption(video_path.name)

    preview, metadata = read_preview_frame(video_path)

    if preview is None:
        st.error("Could not read this video.")
        return None

    st.caption(
        f"Resolution: {metadata['original_width']} × "
        f"{metadata['original_height']} | "
        f"FPS: {metadata['fps']:.2f} | "
        f"Duration: {metadata['duration']:.2f} seconds"
    )

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.20)",
        stroke_width=3,
        stroke_color="#FF3333",
        background_image=preview,
        background_color="#FFFFFF",
        update_streamlit=True,
        height=metadata["display_height"],
        width=metadata["display_width"],
        drawing_mode="polygon",
        key=f"sw_boundary_{selection_id}_{video_number}",
    )


    polygons = extract_polygon_points(
        canvas_result.json_data,
        metadata
    )

    return {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "metadata": metadata,
        "polygons": polygons,
    }


def save_boundaries(boundaries):

    output = {
        "simulation": "SnakeWatch Boundary Simulation",
        "videos": boundaries
    }

    BOUNDARY_FILE.write_text(
        json.dumps(output, indent=4),
        encoding="utf-8"
    )


def render_boundary_simulation(
    load_model_fn=None,
    detect_fn=None,
    tracker_factory=None,
    to_sv_fn=None,
    checkpoint_default=None,
    processor_id=None,
):

    st.subheader("SnakeWatch – Boundary Simulation")

    st.write(
        "Two videos are selected randomly. Draw an independent "
        "polygonal danger zone on each video."
    )

    import json

    selection_file = BOUNDARY_FILE.parent / "selected_videos.json"
    BOUNDARY_FILE.parent.mkdir(parents=True, exist_ok=True)

    if "sw_sim_videos" not in st.session_state:
        saved_paths = []

        # Restore the most recently selected pair.
        if selection_file.exists():
            try:
                saved_data = json.loads(selection_file.read_text())
                saved_paths = saved_data.get("video_paths", [])
            except (json.JSONDecodeError, OSError):
                saved_paths = []

        # Recover the existing pair from saved boundaries if needed.
        if len(saved_paths) != 2 or not all(Path(x).is_file() for x in saved_paths):
            if BOUNDARY_FILE.exists():
                try:
                    boundary_data = json.loads(BOUNDARY_FILE.read_text())
                    boundary_videos = boundary_data.get("videos", [])
                    recovered = [
                        item.get("video_path") for item in boundary_videos
                    ]
                    if len(recovered) == 2 and all(
                        x and Path(x).is_file() for x in recovered
                    ):
                        saved_paths = recovered
                except (json.JSONDecodeError, OSError, AttributeError):
                    saved_paths = []

        if len(saved_paths) == 2 and all(Path(x).is_file() for x in saved_paths):
            st.session_state.sw_sim_videos = [Path(x) for x in saved_paths]
        else:
            st.session_state.sw_sim_videos = select_random_videos()
            saved_paths = [str(x) for x in st.session_state.sw_sim_videos]

        selection_file.write_text(json.dumps({
            "video_paths": [str(x) for x in st.session_state.sw_sim_videos]
        }, indent=2))

        st.session_state.sw_sim_selection_id = 1

    if st.button("Reshuffle Videos", key="sw_sim_reselect"):

        previous_videos = st.session_state.sw_sim_videos

        st.session_state.sw_sim_videos = select_random_videos(
            previous_videos=previous_videos
        )

        # Persist the new pair before rerunning the Streamlit app.
        selection_file.write_text(json.dumps({
            "video_paths": [str(x) for x in st.session_state.sw_sim_videos]
        }, indent=2))

        st.session_state.sw_sim_selection_id += 1

        st.session_state.sw_saved_boundaries = None
        st.session_state.pop("sw_movement_results", None)

        st.rerun()

    videos = st.session_state.sw_sim_videos
    selection_id = st.session_state.sw_sim_selection_id

    col1, col2 = st.columns(2)

    with col1:
        boundary1 = render_video_boundary(
            videos[0], 1, selection_id
        )

    with col2:
        boundary2 = render_video_boundary(
            videos[1], 2, selection_id
        )

    st.divider()

    st.subheader("Boundary Configuration")

    st.info(
        "Draw at least three points for each danger-zone polygon. "
        "The coordinates are converted back to the original video "
        "resolution."
    )

    if st.button(
        "Save Boundaries",
        type="primary",
        key="sw_save_boundaries"
    ):

        if boundary1 is None or boundary2 is None:
            st.error("One or both video previews could not be loaded.")

        elif not boundary1["polygons"] or not boundary2["polygons"]:
            st.warning(
                "Please draw a polygon on BOTH videos before saving."
            )

        else:

            saved = [
                boundary1,
                boundary2
            ]

            save_boundaries(saved)

            st.session_state.sw_saved_boundaries = saved

            st.success("Both danger boundaries saved successfully!")

    saved = st.session_state.get("sw_saved_boundaries")

    if saved:

        st.subheader("Saved Danger Boundaries")

        for i, item in enumerate(saved, 1):

            st.markdown(f"**Video {i}: {item['video_name']}**")

            st.write(
                f"Original resolution: "
                f"{item['metadata']['original_width']} × "
                f"{item['metadata']['original_height']}"
            )

            for polygon_number, polygon in enumerate(
                item["polygons"], 1
            ):
                st.write(
                    f"Polygon {polygon_number}: "
                    f"{len(polygon)} vertices"
                )

        st.caption(f"Saved to: {BOUNDARY_FILE}")


    # SNAKEWATCH_REAL_MOVEMENT_ANALYSIS_V1
    st.divider()
    st.subheader("Movement and Boundary Analysis")

    st.caption(
        "Runs RT-DETRv2 and ByteTrack independently on both saved videos. "
        "Distance and velocity are reported in image pixels."
    )

    detection_preset = st.selectbox(
        "Detection sensitivity",
        ["High recall (0.035)", "Balanced (0.040)", "Conservative (0.055)"],
        index=0,
        help="Thresholds are based on image validation. Video-level performance is being validated separately.",
        key="sw_detection_preset",
    )

    detection_thresholds = {
        "High recall (0.035)": 0.035,
        "Balanced (0.040)": 0.040,
        "Conservative (0.055)": 0.055,
    }

    detection_threshold = detection_thresholds[detection_preset]

    st.caption(
        f"Detection threshold applied: {detection_threshold:.3f}"
    )

    max_analysis_frames = st.selectbox(
        "Analysis frame limit",
        [0, 150, 300, 500],
        index=0,
        format_func=lambda value: (
            "Entire video" if value == 0 else f"{value} frames"
        ),
        key="sw_movement_frame_limit",
    )

    if st.button(
        "Run Movement Analysis",
        type="primary",
        key="sw_run_movement_analysis",
    ):
        if not all([
            load_model_fn,
            detect_fn,
            tracker_factory,
            to_sv_fn,
            checkpoint_default,
            processor_id,
        ]):
            st.error("Model and tracking functions were not connected.")
        else:
            try:
                import importlib.util

                engine_path = Path(__file__).with_name(
                    "snakewatch_movement_analysis.py"
                )

                spec = importlib.util.spec_from_file_location(
                    "snakewatch_movement_analysis",
                    engine_path,
                )

                engine = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(engine)

                if not BOUNDARY_FILE.is_file():
                    raise FileNotFoundError(
                        "Save both video boundaries before running analysis."
                    )

                saved_data = json.loads(
                    BOUNDARY_FILE.read_text(encoding="utf-8")
                )

                video_configs = saved_data.get("videos", [])

                if len(video_configs) != 2:
                    raise ValueError(
                        "Expected exactly two saved video configurations."
                    )

                with st.spinner("Loading RT-DETRv2 model..."):
                    processor, model, device = load_model_fn(
                        checkpoint_default,
                        processor_id,
                    )

                st.info(f"Model device: {device}")

                all_results = []

                for video_number, config in enumerate(video_configs, 1):
                    st.markdown(
                        f"### Video {video_number}: {config['video_name']}"
                    )

                    progress_bar = st.progress(0.0)
                    status = st.empty()

                    def update_progress(done, total):
                        if total > 0:
                            progress_bar.progress(
                                min(1.0, done / total)
                            )
                        status.caption(
                            f"Processed {done} frames"
                            + (f" of {total}" if total else "")
                        )

                    live_polygons = config.get('polygons', [])
                    import cv2

                    # SNAKEWATCH_LIVE_FRAME_DISPLAY_V1
                    live_frame_placeholder = st.empty()
                    live_metrics_placeholder = st.empty()
                    live_table_placeholder = st.empty()
                    live_timing = {
                        "time": time.monotonic(),
                        "frame": 0,
                    }

                    def update_live_frame(
                        frame, frame_index, source_fps,
                        total_frames, detections, alert_state
                    ):
                        now = time.monotonic()
                        elapsed = max(
                            now - live_timing["time"], 0.001
                        )
                        processing_fps = (
                            (frame_index - live_timing["frame"])
                            / elapsed
                        )

                        live_timing["time"] = now
                        live_timing["frame"] = frame_index

                        # Draw the selected danger zone on the preview copy.
                        preview_frame = frame.copy()

                        for polygon in live_polygons:
                            points = np.array(
                                polygon, dtype=np.int32
                            ).reshape((-1, 1, 2))

                            if len(points) >= 3:
                                cv2.polylines(
                                    preview_frame,
                                    [points],
                                    True,
                                    (0, 0, 255),
                                    3,
                                )

                        live_frame_placeholder.image(
                            preview_frame[:, :, ::-1],
                            channels="RGB",
                            caption=(
                                f"Live Boundary Simulation | "
                                f"Frame {frame_index + 1}"
                            ),
                            use_container_width=True,
                        )

                        progress_text = (
                            f"{frame_index + 1:,} / "
                            f"{total_frames:,}"
                            if total_frames else f"{frame_index + 1:,}"
                        )

                        with live_metrics_placeholder.container():
                            metric_cols = st.columns(5)
                            metric_cols[0].metric(
                                "Processed frames", progress_text
                            )
                            metric_cols[1].metric(
                                "Processing FPS",
                                f"{processing_fps:.1f}"
                            )
                            metric_cols[2].metric(
                                "Tracked snakes", len(detections)
                            )
                            metric_cols[3].metric(
                                "Source FPS", f"{source_fps:.1f}"
                            )
                            metric_cols[4].metric(
                                "Alert state", alert_state
                            )

                        if detections:
                            live_table_placeholder.dataframe(
                                detections,
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            live_table_placeholder.info(
                                "No tracked snakes in the current frame."
                            )

                    # SNAKEWATCH_BROWSER_AUDIO_ALERT_CALLBACK_V1
                    audio_state_url = "http" + chr(58) + chr(47) * 2 + "127.0.0.1:8503/state"
                    last_audio_payload = {}
                    last_audio_post_time = 0.0

                    def update_audio_alert(alert):
                        nonlocal last_audio_payload, last_audio_post_time

                        payload = {
                            "state": alert["state"],
                            "video_name": alert.get("video_name"),
                            "frame_index": alert.get("frame_index"),
                            "video_time_s": alert.get("video_time_s"),
                            "inside_track_ids": sorted(
                                alert.get("inside_track_ids", [])
                            ),
                            "approaching_track_ids": sorted(
                                alert.get("approaching_track_ids", [])
                            ),
                        }

                        # Compare alert state and tracked IDs, not frame time.
                        alert_signature = (
                            payload["state"],
                            payload["video_name"],
                            tuple(payload["inside_track_ids"]),
                            tuple(payload["approaching_track_ids"]),
                        )

                        now = time.monotonic()

                        if (
                            alert_signature != last_audio_payload
                            or now - last_audio_post_time >= 0.5
                        ):
                            try:
                                response = requests.post(
                                    audio_state_url,
                                    json=payload,
                                    timeout=0.5,
                                )
                                response.raise_for_status()
                                last_audio_payload = alert_signature
                                last_audio_post_time = now
                            except requests.RequestException:
                                pass

                    try:
                        requests.post(
                            audio_state_url,
                            json={
                                "state": "CLEAR",
                                "video_name": config.get("video_name"),
                                "frame_index": None,
                                "video_time_s": 0.0,
                                "inside_track_ids": [],
                                "approaching_track_ids": [],
                            },
                            timeout=0.5,
                        ).raise_for_status()
                    except requests.RequestException:
                        pass

                    config["evidence_root"] = str(BOUNDARY_FILE.parent / "evidence")
                    _sw_analysis_start = __import__("time").perf_counter()
                    result = engine.analyze_video(
                        video_config=config,
                        processor=processor,
                        model=model,
                        device=device,
                        detect_fn=detect_fn,
                        tracker_factory=tracker_factory,
                        to_sv_fn=to_sv_fn,
                        threshold=detection_threshold,
                        nms_iou=0.50,
                        max_frames=int(max_analysis_frames),
                        progress_callback=update_progress,
                        alert_callback=update_audio_alert,
                        frame_callback=update_live_frame,
                    )
                    _sw_analysis_seconds = __import__("time").perf_counter() - _sw_analysis_start
                    st.session_state.setdefault("sw_stage_timings", {})["Analysis + evidence generation"] = _sw_analysis_seconds
                    st.caption(f"Profiling — Video {video_number}: analysis and evidence generation: {_sw_analysis_seconds:.2f} seconds")

                    # SNAKEWATCH_SEPARATE_ENGINE_TIMINGS_V1
                    st.session_state.setdefault("sw_engine_timings", {})[
                        result.get("video_name", f"Video {video_number}")
                    ] = {
                        "Detection, tracking and movement analysis": result.get(
                            "analysis_seconds", 0.0
                        ),
                        "Annotated video and evidence generation": result.get(
                            "evidence_generation_seconds", 0.0
                        ),
                    }


                    try:
                        requests.post(
                            audio_state_url,
                            json={
                                "state": "CLEAR",
                                "video_name": result.get("video_name"),
                                "frame_index": None,
                                "video_time_s": None,
                                "inside_track_ids": [],
                                "approaching_track_ids": [],
                            },
                            timeout=0.5,
                        ).raise_for_status()
                    except requests.RequestException:
                        pass

                    all_results.append(result)

                    selected_names = [
                        str(Path(item["video_path"]).resolve())
                        for item in video_configs
                    ]
                    _sw_save_start = __import__("time").perf_counter()
                    save_video_result(result, video_number, selected_names)
                    _sw_save_seconds = __import__("time").perf_counter() - _sw_save_start
                    st.session_state.setdefault("sw_stage_timings", {})["Google Drive persistence"] = _sw_save_seconds
                    st.caption(f"Profiling — Video {video_number}: Google Drive persistence: {_sw_save_seconds:.2f} seconds")
                    st.session_state["sw_movement_results"] = list(all_results)

                    progress_bar.progress(1.0)

                    st.success(
                        f"Completed: {result['frames_processed']} frames · "
                        f"{result['detections_tracked']} tracked observations"
                    )

                    observations = result["observations"]
                    events = result["events"]

                    if observations.empty:
                        st.warning("No tracked detections were returned for this video.")
                    else:
                        st.caption(
                            f"{len(observations):,} tracked observations collected. "
                            "See the Field Safety Summary below."
                        )

                    if events.empty:
                        st.info("No boundary events were recorded for this video.")
                    else:
                        st.caption(f"{len(events):,} boundary event(s) recorded.")

                # Refresh the dashboard only after both videos finish.
                if len(all_results) == len(video_configs):
                    st.rerun()

            except Exception as exc:
                st.error(
                    f"Movement analysis failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                st.exception(exc)




# SNAKEWATCH_REFRESH_RESTORATION_V2
def render_movement_summary(all_results):
    st.session_state["sw_movement_results"] = all_results

    import pandas as pd

    observation_parts = [
        result["observations"] for result in all_results
        if not result["observations"].empty
    ]
    event_parts = [
        result["events"] for result in all_results
        if not result["events"].empty
    ]
    combined = (
        pd.concat(observation_parts, ignore_index=True)
        if observation_parts else pd.DataFrame()
    )
    events_combined = (
        pd.concat(event_parts, ignore_index=True)
        if event_parts else pd.DataFrame()
    )

    st.divider()
    st.header("SnakeWatch Field Safety Summary")
    st.caption(
        "Plain-language results from the analysed videos. "
        "Recorded events are review prompts, not verified snake intrusions."
    )

    total_frames = sum(r["frames_processed"] for r in all_results)
    total_observations = sum(r["detections_tracked"] for r in all_results)
    event_count = len(events_combined)
    entries = int((events_combined["event"] == "BOUNDARY ENTRY").sum()) if event_count else 0
    exits = int((events_combined["event"] == "BOUNDARY EXIT").sum()) if event_count else 0
    first_inside = int((events_combined["event"] == "FIRST OBSERVED INSIDE").sum()) if event_count else 0

    metrics = st.columns(4)
    metrics[0].metric("Videos analysed", f"{len(all_results)}/2")
    metrics[1].metric("Frames processed", f"{total_frames:,}")
    metrics[2].metric("Tracked observations", f"{total_observations:,}")
    metrics[3].metric("Boundary events", f"{event_count:,}")

    if event_count:
        st.warning(
            f"Boundary activity recorded: {entries} possible entries, "
            f"{first_inside} first-observed-inside events, and {exits} exits. "
            "Review the video before treating an event as a confirmed intrusion."
        )
    else:
        st.success(
            "No boundary events were recorded. This does not prove "
            "that the area is snake-free."
        )

    counts = st.columns(3)
    counts[0].metric("Possible entries", entries)
    counts[1].metric("First observed inside", first_inside)
    counts[2].metric("Boundary exits", exits)

    st.subheader("Video-by-video summary")
    for number, result in enumerate(all_results, 1):
        ev = result["events"]
        with st.container(border=True):
            st.markdown(f"**Video {number}: {result['video_name']}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Frames", f"{result['frames_processed']:,}")
            c2.metric("Tracked observations", f"{result['detections_tracked']:,}")
            c3.metric("Boundary events", f"{len(ev):,}")
            if not ev.empty:
                st.dataframe(
                    ev.sort_values(["video_time_s", "frame_index"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No boundary events recorded for this video.")

    st.subheader("Event timeline")
    if event_count:
        timeline = events_combined.sort_values(
            ["video_name", "video_time_s", "frame_index"]
        ).copy()
        timeline["video_time"] = timeline["video_time_s"].map(
            lambda value: f"{int(value // 60):02d}:{value % 60:05.2f}"
        )
        timeline["operator_message"] = timeline["event"].map({
            "BOUNDARY ENTRY": "Tracked object crossed into the marked zone",
            "FIRST OBSERVED INSIDE": "Tracked object first appeared inside the marked zone",
            "BOUNDARY EXIT": "Tracked object crossed out of the marked zone",
        }).fillna("Boundary event recorded")
        st.dataframe(
            timeline[[
                "video_name", "video_time", "operator_message",
                "track_id", "distance_px"
            ]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No boundary events to display on the timeline.")

    with st.expander("Technical Details — movement observations and CSV exports"):
        st.caption(
            "Track IDs identify tracker trajectories, not a count of individual snakes."
        )
        if not combined.empty:
            st.markdown("**Combined movement observations**")
            st.dataframe(
                combined.tail(500),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download combined movement analysis (CSV)",
                data=combined.to_csv(index=False).encode("utf-8"),
                file_name="snakewatch_combined_movement.csv",
                mime="text/csv",
                key="sw_combined_movement_csv",
            )
        if event_count:
            st.markdown("**Combined boundary events**")
            st.dataframe(
                events_combined,
                use_container_width=True,
                hide_index=True,
            )
        st.download_button(
            "Download combined boundary events (CSV)",
            data=events_combined.to_csv(index=False).encode("utf-8"),
            file_name="snakewatch_combined_boundary_events.csv",
            mime="text/csv",
            key="sw_combined_boundary_events_csv",
        )

    st.success("Both video analyses completed.")

    # SNAKEWATCH_MAIN_PROCESSED_VIDEO_OUTPUT_V1
    st.divider()
    st.subheader("Processed Video Outputs")
    st.caption(
        "Annotated videos generated after movement analysis, "
        "including the danger-zone polygon, detections and tracking trails."
    )

    for output_number, output_result in enumerate(all_results, 1):
        output_name = output_result.get(
            "video_name", f"Video {output_number}"
        )
        output_path = output_result.get("annotated_video_path", "")

        st.markdown(f"**Video {output_number}: {output_name}**")

        if output_path and Path(output_path).is_file():
            st.video(str(output_path))
            st.caption(
                f"Annotated output: {Path(output_path).name}"
            )
        else:
            st.warning(
                f"Annotated video is unavailable for {output_name}. "
                "Check the evidence-generation output path."
            )


# SNAKEWATCH_PERSISTENT_EVIDENCE_REVIEW_V1
import json
import hashlib
import pandas as pd

st.divider()
st.header("SnakeWatch Evidence and Event Review")
st.caption(
    "Review the annotated footage and individual event clips. "
    "These are algorithm-generated review prompts, not verified "
    "snake intrusions."
)

if "sw_current_run_initialized_v1" not in st.session_state:
    st.session_state["sw_current_run_initialized_v1"] = True

# Restore the matching completed analysis after a runtime/dashboard restart.
# Existing in-session results always take priority.
if not st.session_state.get("sw_movement_results"):
    try:
        if BOUNDARY_FILE.is_file():
            _sw_saved_boundary_data = json.loads(
                BOUNDARY_FILE.read_text(encoding="utf-8")
            )

            _sw_saved_configs = _sw_saved_boundary_data.get("videos", [])

            if len(_sw_saved_configs) == 2:
                _sw_saved_names = [
                    str(Path(item["video_path"]).resolve())
                    for item in _sw_saved_configs
                ]

                _sw_restored_results = load_persisted_results(
                    _sw_saved_names
                )

                if _sw_restored_results:
                    _sw_valid_results = [
                        item for item in _sw_restored_results
                        if item.get("annotated_video_path")
                        and Path(item["annotated_video_path"]).is_file()
                    ]

                    if _sw_valid_results:
                        st.session_state["sw_movement_results"] = (
                            _sw_valid_results
                        )
                        st.info(
                            f"Restored {len(_sw_valid_results)} completed "
                            "video analysis result(s) from Google Drive. "
                            "No new inference was performed."
                        )

    except Exception as _sw_restore_error:
        st.warning(
            f"Could not restore saved analysis results: "
            f"{_sw_restore_error}"
        )

saved_results = st.session_state.get("sw_movement_results", [])

if saved_results:
    _sw_summary_start = __import__("time").perf_counter()
    render_movement_summary(saved_results)
    _sw_summary_seconds = __import__("time").perf_counter() - _sw_summary_start
    st.session_state.setdefault("sw_stage_timings", {})["Combined summary rendering"] = _sw_summary_seconds
    st.caption(f"Profiling — Combined summary preparation and rendering: {_sw_summary_seconds:.2f} seconds")
    # SNAKEWATCH_VISIBLE_PERFORMANCE_REPORT_V1
    st.subheader("Performance Report")
    st.caption("Time spent in each major processing stage")

    _sw_engine_timings = st.session_state.get("sw_engine_timings", {})
    if _sw_engine_timings:
        st.markdown("**Detailed video-processing breakdown**")
        _sw_engine_rows = []
        for _sw_video_name, _sw_stages in _sw_engine_timings.items():
            for _sw_stage_name, _sw_seconds in _sw_stages.items():
                _sw_engine_rows.append({
                    "Video": _sw_video_name,
                    "Stage": _sw_stage_name,
                    "Time (seconds)": round(float(_sw_seconds), 2),
                })
        st.dataframe(
            pd.DataFrame(_sw_engine_rows),
            hide_index=True,
            use_container_width=True,
        )

    _sw_timings = st.session_state.get("sw_stage_timings", {})
    if _sw_timings:
        _sw_perf_df = pd.DataFrame([
            {"Stage": name, "Time (seconds)": round(value, 2)}
            for name, value in _sw_timings.items()
        ])
        st.dataframe(_sw_perf_df, hide_index=True, use_container_width=True)
        st.metric("Total measured time", f"{sum(_sw_timings.values()):.2f} seconds")
    else:
        st.info("Run the Boundary Simulation to populate performance timings.")


if not saved_results:
    st.info("Run Movement Analysis to generate evidence for review.")
else:
    review_file = BOUNDARY_FILE.parent / "event_reviews.json"
    st.session_state["sw_review_render_generation"] = st.session_state.get("sw_review_render_generation", 0) + 1
    review_render_generation = st.session_state["sw_review_render_generation"]
    review_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        if review_file.exists():
            review_data = json.loads(
                review_file.read_text(encoding="utf-8")
            )
        else:
            review_data = {}
    except Exception:
        review_data = {}

    review_rows = []
    review_options = [
        "Unreviewed",
        "Confirmed",
        "False alarm",
        "Unsure"
    ]

    for video_number, result in enumerate(saved_results, 1):
        video_name = str(result.get("video_name", f"Video {video_number}"))

        with st.expander(
            f"Video {video_number}: {video_name}",
            expanded=True
        ):
            annotated_path = result.get("annotated_video_path")

            if annotated_path and Path(annotated_path).is_file():
                st.markdown("**Annotated video — polygon, detections and trails**")
                st.video(str(annotated_path))
            else:
                st.warning(
                    "Annotated video is not available for this result. "
                    "Run a fresh movement analysis to generate it."
                )

            events = result.get("events", pd.DataFrame())

            if events is None or events.empty:
                st.success("No boundary events were recorded for this video.")
                continue

            st.markdown(f"**{len(events)} event(s) available for review**")

            for event_number, (_, event) in enumerate(events.iterrows(), 1):
                event_type = str(event.get("event", "Boundary event"))
                track_id = event.get("track_id", "Unknown")
                frame_index = event.get("frame_index", "Unknown")
                event_time = event.get("video_time_s", 0)

                try:
                    time_label = (
                        f"{int(float(event_time) // 60):02d}:"
                        f"{float(event_time) % 60:05.2f}"
                    )
                except Exception:
                    time_label = str(event_time)

                event_key = (
                    f"{video_name}|{frame_index}|{track_id}|{event_type}"
                )

                widget_identity = (
                    f"{event_key}|video_{video_number}|event_{event_number}"
                )
                widget_id = hashlib.sha1(
                    widget_identity.encode("utf-8")
                ).hexdigest()[:16]

                with st.container(border=True):
                    st.markdown(
                        f"**Event {event_number}: {event_type}**"
                    )
                    st.caption(
                        f"Time: {time_label} | Track ID: {track_id}"
                    )

                    clip_path = event.get("evidence_clip", "")

                    if (
                        isinstance(clip_path, str)
                        and clip_path
                        and Path(clip_path).is_file()
                    ):
                        st.video(clip_path)
                    else:
                        st.warning(
                            "Event clip unavailable. "
                            "The annotated full video is available above."
                        )

                    previous_status = review_data.get(
                        event_key, "Unreviewed"
                    )

                    if previous_status not in review_options:
                        previous_status = "Unreviewed"

                    selected_status = st.selectbox(
                        "Operator review",
                        review_options,
                        index=review_options.index(previous_status),
                        key=f"sw_review_{review_render_generation}_{widget_id}"
                    )

                    if selected_status != previous_status:
                        review_data[event_key] = selected_status

                        temporary_review_file = review_file.with_suffix(
                            ".tmp"
                        )
                        temporary_review_file.write_text(
                            json.dumps(
                                review_data,
                                indent=2,
                                ensure_ascii=False
                            ),
                            encoding="utf-8"
                        )
                        temporary_review_file.replace(review_file)

                    review_rows.append({
                        "video_name": video_name,
                        "event": event_type,
                        "video_time_s": event_time,
                        "frame_index": frame_index,
                        "track_id": track_id,
                        "review_status": selected_status,
                        "evidence_clip": (
                            clip_path
                            if isinstance(clip_path, str) else ""
                        )
                    })

    if review_rows:
        review_df = pd.DataFrame(review_rows)

        st.divider()
        st.subheader("Review progress")

        reviewed_count = int(
            (review_df["review_status"] != "Unreviewed").sum()
        )
        total_count = len(review_df)

        st.progress(reviewed_count / total_count)
        st.caption(
            f"{reviewed_count} of {total_count} events reviewed."
        )

        st.download_button(
            "Download reviewed event report (CSV)",
            data=review_df.to_csv(index=False).encode("utf-8"),
            file_name="snakewatch_reviewed_events.csv",
            mime="text/csv",
            key=f"sw_reviewed_events_csv_{review_render_generation}"
        )

    st.caption(f"Review decisions saved in: {review_file}")

# SNAKEWATCH_STAGE_PROFILING_V1
