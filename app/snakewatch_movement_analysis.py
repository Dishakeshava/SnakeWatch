import cv2
import numpy as np
import pandas as pd

from collections import defaultdict, deque


def signed_polygon_distance(point, polygon):
    """
    Positive = inside polygon.
    Negative = outside polygon.
    Absolute value = distance to nearest polygon edge, in pixels.
    """
    contour = np.asarray(polygon, dtype=np.float32).reshape((-1, 1, 2))
    return float(
        cv2.pointPolygonTest(
            contour,
            (float(point[0]), float(point[1])),
            True
        )
    )


def estimate_motion(history):
    """
    Estimate image-plane velocity and signed-distance rate
    using linear regression over recent observations.
    """
    if len(history) < 5:
        return None

    times = np.asarray([item[0] for item in history], dtype=float)
    xs = np.asarray([item[1] for item in history], dtype=float)
    ys = np.asarray([item[2] for item in history], dtype=float)
    distances = np.asarray([item[3] for item in history], dtype=float)

    times = times - times[0]

    if times[-1] <= 0:
        return None

    vx = float(np.polyfit(times, xs, 1)[0])
    vy = float(np.polyfit(times, ys, 1)[0])
    distance_rate = float(np.polyfit(times, distances, 1)[0])

    speed = float(np.hypot(vx, vy))

    return {
        "velocity_x_px_s": vx,
        "velocity_y_px_s": vy,
        "speed_px_s": speed,
        "signed_distance_rate_px_s": distance_rate,
    }


def analyze_video(
    video_config,
    processor,
    model,
    device,
    detect_fn,
    tracker_factory,
    to_sv_fn,
    threshold=0.055,
    nms_iou=0.50,
    max_frames=0,
    progress_callback=None,
    alert_callback=None,
    frame_callback=None,
    activation_threshold=0.03,
    lost_buffer=5,
    match_threshold=0.80,
):
    """
    Process one video and detect genuine per-track boundary transitions.

    Positive signed distance = inside polygon.
    Negative signed distance = outside polygon.
    """

    video_path = video_config["video_path"]
    video_name = video_config["video_name"]
    polygons = video_config["polygons"]
    metadata = video_config["metadata"]

    if not polygons:
        raise ValueError(f"No saved polygon found for {video_name}.")

    polygon = polygons[0]

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = float(metadata.get("fps", 30.0))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    tracker = tracker_factory(
        fps,
        activation_threshold,
        lost_buffer,
        match_threshold,
    )

    histories = defaultdict(lambda: deque(maxlen=12))
    distance_histories = defaultdict(lambda: deque(maxlen=5))
    previous_distances = {}
    boundary_states = {}

    rows = []
    events = []

    frame_index = 0
    processed_frames = 0
    detections_seen = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            if max_frames > 0 and processed_frames >= max_frames:
                break

            current_time = frame_index / fps

            detections = detect_fn(
                frame,
                processor,
                model,
                device,
                threshold,
                nms_iou
            )

            tracked = tracker.update_with_detections(
                to_sv_fn(detections)
            )

            boxes = tracked.xyxy if len(tracked) else []
            ids = tracked.tracker_id if len(tracked) else None
            scores = tracked.confidence if len(tracked) else None

            frame_inside_ids = set()
            frame_approaching_ids = set()
            live_detections = []

            for i, box in enumerate(boxes):
                if ids is None or ids[i] is None:
                    continue

                track_id = int(ids[i])

                x1, y1, x2, y2 = map(float, box)

                # Use the bottom-centre of the bounding box as
                # the approximate ground-contact reference point.
                px = (x1 + x2) / 2.0
                py = y2

                signed_distance = signed_polygon_distance(
                    (px, py), polygon
                )

                distance_history = distance_histories[track_id]
                distance_history.append(signed_distance)
                smoothed_distance = float(np.median(distance_history))

                previous_state = boundary_states.get(track_id)

                if previous_state is None:
                    current_state = (
                        "INSIDE" if smoothed_distance >= 3.0 else "OUTSIDE"
                    )
                elif previous_state == "OUTSIDE":
                    current_state = (
                        "INSIDE" if smoothed_distance >= 3.0 else "OUTSIDE"
                    )
                else:
                    current_state = (
                        "OUTSIDE" if smoothed_distance <= -3.0 else "INSIDE"
                    )

                previous_state = boundary_states.get(track_id)
                previous_distance = previous_distances.get(track_id)

                if previous_state is None:
                    if current_state == "INSIDE":
                        event_type = "FIRST OBSERVED INSIDE"
                    else:
                        event_type = None

                elif previous_state == "OUTSIDE" and current_state == "INSIDE":
                    event_type = "BOUNDARY ENTRY"

                elif previous_state == "INSIDE" and current_state == "OUTSIDE":
                    event_type = "BOUNDARY EXIT"

                else:
                    event_type = None

                if event_type is not None:
                    events.append({
                        "video_name": video_name,
                        "track_id": track_id,
                        "event": event_type,
                        "video_time_s": round(current_time, 3),
                        "frame_index": frame_index,
                        "distance_px": round(abs(signed_distance), 2),
                    })

                boundary_states[track_id] = current_state

                history = histories[track_id]
                history.append((
                    current_time,
                    px,
                    py,
                    signed_distance
                ))

                motion = estimate_motion(history)

                speed = None
                direction = "INSUFFICIENT HISTORY"
                eta = None
                status = "OUTSIDE"

                if current_state == "INSIDE":
                    status = "INSIDE DANGER ZONE"
                    direction = "INSIDE ZONE"

                elif motion is not None:
                    speed = motion["speed_px_s"]
                    rate = motion["signed_distance_rate_px_s"]

                    if speed < 2.0:
                        direction = "STATIONARY"
                        status = "STATIONARY"

                    elif rate > 1.5:
                        direction = "TOWARDS BOUNDARY"
                        status = "APPROACHING"
                        eta = max(0.0, -signed_distance / rate)

                    elif rate < -1.5:
                        direction = "AWAY FROM BOUNDARY"
                        status = "MOVING AWAY"

                    else:
                        direction = "MOVEMENT UNCERTAIN"
                        status = "OUTSIDE"

                confidence = (
                    float(scores[i])
                    if scores is not None else None
                )

                if status == "INSIDE DANGER ZONE":
                    frame_inside_ids.add(track_id)
                elif status == "APPROACHING":
                    frame_approaching_ids.add(track_id)

                rows.append({
                    "video_name": video_name,
                    "frame_index": frame_index,
                    "video_time_s": round(current_time, 3),
                    "track_id": track_id,
                    "confidence": confidence,
                    "bbox_x1": round(x1, 2),
                    "bbox_y1": round(y1, 2),
                    "bbox_x2": round(x2, 2),
                    "bbox_y2": round(y2, 2),
                    "bottom_center_x_px": round(px, 2),
                    "bottom_center_y_px": round(py, 2),
                    "signed_distance_px": round(signed_distance, 2),
                    "distance_to_boundary_px": round(
                        abs(signed_distance), 2
                    ),
                    "velocity_x_px_s": (
                        round(motion["velocity_x_px_s"], 2)
                        if motion else None
                    ),
                    "velocity_y_px_s": (
                        round(motion["velocity_y_px_s"], 2)
                        if motion else None
                    ),
                    "speed_px_s": (
                        round(speed, 2)
                        if speed is not None else None
                    ),
                    "direction": direction,
                    "eta_to_boundary_s": (
                        round(eta, 2)
                        if eta is not None else None
                    ),
                    "boundary_status": status,
                })

                live_detections.append({
                    "track_id": track_id,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "boundary_status": status,
                    "direction": direction,
                    "distance_to_boundary_px": abs(signed_distance),
                    "speed_px_s": speed,
                    "eta_to_boundary_s": eta,
                })

                previous_distances[track_id] = signed_distance
                detections_seen += 1

            if frame_inside_ids:
                frame_alert_state = "INSIDE"
            elif frame_approaching_ids:
                frame_alert_state = "APPROACHING"
            else:
                frame_alert_state = "CLEAR"

            if alert_callback is not None:
                alert_callback({
                    "state": frame_alert_state,
                    "video_name": video_name,
                    "frame_index": frame_index,
                    "video_time_s": round(current_time, 3),
                    "inside_track_ids": sorted(frame_inside_ids),
                    "approaching_track_ids": sorted(frame_approaching_ids),
                })

            if (
                frame_callback is not None
                and (processed_frames % 5 == 0 or processed_frames == 1)
            ):
                frame_callback(
                    frame.copy(),
                    frame_index,
                    fps,
                    total_frames,
                    live_detections,
                    frame_alert_state,
                )

            processed_frames += 1
            frame_index += 1

            if (
                progress_callback is not None
                and processed_frames % 10 == 0
            ):
                progress_callback(processed_frames, total_frames)

    finally:
        cap.release()

    return {
        "video_name": video_name,
        "frames_processed": processed_frames,
        "detections_tracked": detections_seen,
        "fps": float(fps),
        "observations": pd.DataFrame(rows),
        "events": pd.DataFrame(events),
    }



# SNAKEWATCH_EVENT_EVIDENCE_PIPELINE_V1
# Generates annotated video and event-specific review clips.
# Existing detector, tracker and event rules remain unchanged.

import os
import re
import subprocess
from pathlib import Path
from collections import defaultdict, deque


_snakewatch_original_analyze_video = analyze_video


def _snakewatch_generate_evidence(video_config, result):
    video_path = Path(video_config["video_path"])
    polygon = video_config["polygons"][0]

    observations = result["observations"].copy()
    events = result["events"].copy()

    evidence_root = (
        video_path.parent.parent / "SnakeWatch_Results" / "evidence"
    )

    # Keep all evidence inside the project results directory.
    # The simulation's configured results directory is preferred.
    project_results = Path(video_config.get(
        "evidence_root",
        evidence_root
    ))

    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        video_path.stem
    )

    output_dir = project_results / safe_name
    event_dir = output_dir / "events"
    output_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = output_dir / "annotated_video.mp4"
    temporary_path = output_dir / "annotated_video_temp.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not reopen video for evidence generation: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = float(result.get("fps", 30.0))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions for evidence generation.")

    frame_rows = {}
    if not observations.empty:
        for frame_number, group in observations.groupby("frame_index"):
            frame_rows[int(frame_number)] = group.to_dict("records")

    trails = defaultdict(lambda: deque(maxlen=60))

    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create annotated evidence video.")

    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Draw the exact polygon used by the movement analysis.
            contour = np.asarray(
                polygon, dtype=np.int32
            ).reshape((-1, 1, 2))

            cv2.polylines(
                frame, [contour], True, (0, 0, 255), 3
            )

            # Draw tracked detections and their movement trails.
            for record in frame_rows.get(frame_index, []):
                track_id = int(record["track_id"])

                x1 = int(record["bbox_x1"])
                y1 = int(record["bbox_y1"])
                x2 = int(record["bbox_x2"])
                y2 = int(record["bbox_y2"])

                cx = int(record["bottom_center_x_px"])
                cy = int(record["bottom_center_y_px"])

                trails[track_id].append((cx, cy))

                points = list(trails[track_id])
                if len(points) >= 2:
                    cv2.polylines(
                        frame,
                        [np.asarray(points, dtype=np.int32)],
                        False,
                        (0, 255, 255),
                        2
                    )

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                )

                confidence = record.get("confidence")
                confidence_text = (
                    f"{float(confidence):.2f}"
                    if confidence is not None else "N/A"
                )

                cv2.putText(
                    frame,
                    f"Track {track_id} | {confidence_text}",
                    (max(5, x1), max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                frame,
                f"Frame {frame_index} | {frame_index / fps:.2f}s",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            writer.write(frame)
            frame_index += 1

    finally:
        cap.release()
        writer.release()

    # Convert to browser-friendly H.264 MP4 when ffmpeg is available.
    conversion = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-i", str(temporary_path),
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-movflags", "+faststart",
            str(annotated_path)
        ],
        capture_output=True,
        text=True
    )

    if conversion.returncode != 0 or not annotated_path.exists():
        os.replace(temporary_path, annotated_path)
    else:
        temporary_path.unlink(missing_ok=True)

    # Generate one six-second clip centred on each event.
    if not events.empty:
        clip_paths = []
        review_statuses = []

        for event_number, event in events.iterrows():
            event_time = max(0.0, float(event["video_time_s"]))
            start_time = max(0.0, event_time - 3.0)

            track_id = int(event["track_id"])
            event_name = re.sub(
                r"[^A-Za-z0-9_-]+",
                "_",
                str(event["event"])
            )

            clip_name = (
                f"event_{len(clip_paths) + 1:03d}_"
                f"{event_name}_track_{track_id}_"
                f"{event_time:.2f}s.mp4"
            )

            clip_path = event_dir / clip_name

            clip_result = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-y",
                    "-ss", f"{start_time:.3f}",
                    "-i", str(annotated_path),
                    "-t", "6",
                    "-an",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "veryfast",
                    "-movflags", "+faststart",
                    str(clip_path)
                ],
                capture_output=True,
                text=True
            )

            if clip_result.returncode == 0 and clip_path.exists():
                clip_paths.append(str(clip_path))
            else:
                clip_paths.append("")

            review_statuses.append("Unreviewed")

        events["evidence_clip"] = clip_paths
        events["review_status"] = review_statuses

    return {
        "observations": observations,
        "events": events,
        "annotated_video_path": str(annotated_path),
        "evidence_dir": str(output_dir),
        "evidence_clips_created": (
            int(events["evidence_clip"].astype(bool).sum())
            if not events.empty else 0
        )
    }


def analyze_video(*args, **kwargs):
    import time

    _sw_analysis_start = time.perf_counter()
    result = _snakewatch_original_analyze_video(*args, **kwargs)
    _sw_analysis_seconds = time.perf_counter() - _sw_analysis_start

    video_config = (
        kwargs.get("video_config")
        if "video_config" in kwargs
        else args[0]
    )

    _sw_evidence_start = time.perf_counter()
    evidence = _snakewatch_generate_evidence(video_config, result)
    _sw_evidence_seconds = time.perf_counter() - _sw_evidence_start

    result.update(evidence)
    result["analysis_seconds"] = _sw_analysis_seconds
    result["evidence_generation_seconds"] = _sw_evidence_seconds

    return result
