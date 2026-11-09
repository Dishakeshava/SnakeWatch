"""SnakeWatch human-safety video processor.

Uses the dashboard's already-loaded fine-tuned RT-DETRv2 model, YOLO person
model, and tracker factory. Coordinates/radii are image pixels, not metres.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import supervision as sv
from torchvision.ops import nms

TILE_SIZE = 800
TILE_OVERLAP = 0.25
SNAKE_CONFIDENCE = 0.055
NMS_IOU = 0.50
HUMAN_CONFIDENCE = 0.25
WARNING_RADIUS_FACTOR = 1.5
CRITICAL_RADIUS_FACTOR = 0.75
WARNING_CONFIRM_SECONDS = 0.25
CRITICAL_CONFIRM_SECONDS = 0.15
RECOVERY_SECONDS = 0.40
ALERT_COOLDOWN_SECONDS = 2.0
HUMAN_OVERLAP_SUPPRESSION = 0.85


def detect_snakes_tiled(frame, processor, model, device,
                        threshold=SNAKE_CONFIDENCE, nms_iou=NMS_IOU):
    """Run the same overlapping-tile RT-DETRv2 inference used in experiments."""
    height, width = frame.shape[:2]
    stride = max(1, int(TILE_SIZE * (1 - TILE_OVERLAP)))
    xs = list(range(0, max(1, width - TILE_SIZE + 1), stride))
    ys = list(range(0, max(1, height - TILE_SIZE + 1), stride))
    if not xs or xs[-1] + TILE_SIZE < width:
        xs.append(max(0, width - TILE_SIZE))
    if not ys or ys[-1] + TILE_SIZE < height:
        ys.append(max(0, height - TILE_SIZE))
    all_boxes, all_scores = [], []

    with torch.inference_mode():
        for y in sorted(set(ys)):
            for x in sorted(set(xs)):
                tile = frame[y:min(y + TILE_SIZE, height), x:min(x + TILE_SIZE, width)]
                if tile.size == 0:
                    continue
                rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                inputs = processor(images=rgb, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                target_sizes = torch.tensor([[tile.shape[0], tile.shape[1]]], device=device)
                result = processor.post_process_object_detection(
                    outputs, threshold=float(threshold), target_sizes=target_sizes
                )[0]
                boxes = result["boxes"].detach().cpu().numpy()
                scores = result["scores"].detach().cpu().numpy()
                labels = result["labels"].detach().cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    if int(label) != 0:
                        continue
                    x1, y1, x2, y2 = box
                    all_boxes.append([x1 + x, y1 + y, x2 + x, y2 + y])
                    all_scores.append(float(score))

    if not all_boxes:
        return np.empty((0, 4), np.float32), np.empty((0,), np.float32)
    boxes = np.asarray(all_boxes, dtype=np.float32)
    scores = np.asarray(all_scores, dtype=np.float32)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    keep = nms(torch.as_tensor(boxes), torch.as_tensor(scores), float(nms_iou)).cpu().numpy()
    return boxes[keep], scores[keep]


def suppress_human_overlapping_snakes(snake_boxes, snake_scores, human_boxes,
                                      overlap_threshold=HUMAN_OVERLAP_SUPPRESSION):
    """Heuristic to reject snake boxes mostly contained inside a person box.

    Caution: may suppress a real snake when it is heavily occluded by a person.
    """
    if len(snake_boxes) == 0 or len(human_boxes) == 0:
        return snake_boxes, snake_scores
    kept_b, kept_s = [], []
    for box, score in zip(snake_boxes, snake_scores):
        x1, y1, x2, y2 = box
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= 0:
            continue
        suppress = False
        for hbox in human_boxes:
            hx1, hy1, hx2, hy2 = hbox
            ix1, iy1 = max(x1, hx1), max(y1, hy1)
            ix2, iy2 = min(x2, hx2), min(y2, hy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / area >= overlap_threshold:
                suppress = True
                break
        if not suppress:
            kept_b.append(box)
            kept_s.append(score)
    return np.asarray(kept_b, np.float32).reshape(-1, 4), np.asarray(kept_s, np.float32)


def _detections(boxes, scores):
    if len(boxes) == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32),
        confidence=np.asarray(scores, dtype=np.float32),
        class_id=np.zeros(len(boxes), dtype=int),
    )


def _tracks(detections):
    result = []
    ids = detections.tracker_id
    if ids is None or len(detections) == 0:
        return result
    scores = detections.confidence
    for i, (box, track_id) in enumerate(zip(detections.xyxy, ids)):
        if track_id is None or int(track_id) < 0:
            continue
        score = float(scores[i]) if scores is not None else None
        result.append({"id": int(track_id), "box": np.asarray(box, float), "confidence": score})
    return result


def _point_box_distance(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


class TemporalAlertEngine:
    """Per-human persistence state machine. Emits events only on confirmed transitions."""
    def __init__(self):
        self.states = {}

    def update(self, human_id, level, timestamp, frame_index, risk):
        state = self.states.setdefault(human_id, {
            "level": "NORMAL", "warning_since": None,
            "critical_since": None, "clear_since": None,
            "last_event": -float("inf")
        })
        event = None
        old_level = state["level"]

        if level == "CRITICAL":
            state["warning_since"] = None
            state["clear_since"] = None
            if state["critical_since"] is None:
                state["critical_since"] = timestamp
            if timestamp - state["critical_since"] >= CRITICAL_CONFIRM_SECONDS and old_level != "CRITICAL":
                state["level"] = "CRITICAL"
                if timestamp - state["last_event"] >= ALERT_COOLDOWN_SECONDS:
                    event = self._event(human_id, "CRITICAL", frame_index, timestamp, risk)
                    state["last_event"] = timestamp
        elif level == "WARNING":
            state["critical_since"] = None
            state["clear_since"] = None
            if state["warning_since"] is None:
                state["warning_since"] = timestamp
            if timestamp - state["warning_since"] >= WARNING_CONFIRM_SECONDS and old_level == "NORMAL":
                state["level"] = "WARNING"
                if timestamp - state["last_event"] >= ALERT_COOLDOWN_SECONDS:
                    event = self._event(human_id, "WARNING", frame_index, timestamp, risk)
                    state["last_event"] = timestamp
        else:
            state["warning_since"] = None
            state["critical_since"] = None
            if state["clear_since"] is None:
                state["clear_since"] = timestamp
            if timestamp - state["clear_since"] >= RECOVERY_SECONDS:
                state["level"] = "NORMAL"
                state["clear_since"] = None
        return state["level"], event

    @staticmethod
    def _event(human_id, level, frame_index, timestamp, risk):
        return {
            "frame_index": int(frame_index),
            "timestamp_seconds": round(float(timestamp), 3),
            "human_track_id": int(human_id),
            "snake_track_id": risk.get("snake_id"),
            "alert_level": level,
            "distance_pixels": round(float(risk["distance"]), 2) if risk.get("distance") is not None else None,
            "warning_radius_pixels": round(float(risk["warning_radius"]), 2),
            "critical_radius_pixels": round(float(risk["critical_radius"]), 2),
            "snake_confidence": risk.get("confidence"),
        }


def _human_risk(human, snakes):
    x1, y1, x2, y2 = human["box"]
    human_height = max(1.0, y2 - y1)
    foot = ((x1 + x2) / 2.0, y2)
    warning_r = WARNING_RADIUS_FACTOR * human_height
    critical_r = CRITICAL_RADIUS_FACTOR * human_height
    nearest = None
    for snake in snakes:
        distance = _point_box_distance(foot, snake["box"])
        if nearest is None or distance < nearest["distance"]:
            nearest = {"distance": distance, "snake_id": snake["id"], "confidence": snake["confidence"]}
    if nearest is None:
        return {"level": "NORMAL", "distance": None, "snake_id": None, "confidence": None,
                "warning_radius": warning_r, "critical_radius": critical_r}
    if nearest["distance"] <= critical_r:
        level = "CRITICAL"
    elif nearest["distance"] <= warning_r:
        level = "WARNING"
    else:
        level = "NORMAL"
    return {**nearest, "level": level, "warning_radius": warning_r, "critical_radius": critical_r}


def process_safety_video(
    input_path: str,
    output_path: str,
    processor,
    snake_model,
    device,
    person_model,
    tracker_factory: Callable,
    progress_callback: Optional[Callable] = None,
    preview_callback: Optional[Callable] = None,
    max_frames: int = 0,
    frame_stride: int = 1,
    snake_threshold: float = SNAKE_CONFIDENCE,
):
    """Process uploaded video. Returns output path, event list and summary."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open input video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Video has invalid dimensions.")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_fps = max(1.0, fps / max(1, int(frame_stride)))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output MP4. Check codec support and output path.")

    snake_tracker = tracker_factory(fps, 0.05, 30, 0.10)
    human_tracker = tracker_factory(fps, 0.25, 30, 0.10)
    alert_engine = TemporalAlertEngine()
    events = []
    frame_index = 0
    processed = 0
    frames_with_snake = 0
    frames_with_human = 0
    state_counts = {"NORMAL": 0, "WARNING": 0, "CRITICAL": 0}
    start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            current_index = frame_index
            frame_index += 1
            if current_index % max(1, int(frame_stride)) != 0:
                continue
            if max_frames > 0 and processed >= max_frames:
                break
            timestamp = current_index / fps

            # YOLO COCO class 0 = person.
            person_results = person_model.predict(
                source=frame, classes=[0], conf=HUMAN_CONFIDENCE,
                device=0 if str(device).startswith("cuda") else "cpu", verbose=False
            )
            human_boxes, human_scores = [], []
            if person_results and person_results[0].boxes is not None:
                for box in person_results[0].boxes:
                    human_boxes.append(box.xyxy[0].detach().cpu().numpy())
                    human_scores.append(float(box.conf[0].detach().cpu()))
            human_boxes = np.asarray(human_boxes, np.float32).reshape(-1, 4)
            human_scores = np.asarray(human_scores, np.float32)

            snake_boxes, snake_scores = detect_snakes_tiled(
                frame, processor, snake_model, device, snake_threshold, NMS_IOU
            )
            snake_boxes, snake_scores = suppress_human_overlapping_snakes(
                snake_boxes, snake_scores, human_boxes
            )

            tracked_humans = human_tracker.update_with_detections(_detections(human_boxes, human_scores))
            tracked_snakes = snake_tracker.update_with_detections(_detections(snake_boxes, snake_scores))
            humans, snakes = _tracks(tracked_humans), _tracks(tracked_snakes)
            if snakes:
                frames_with_snake += 1
            if humans:
                frames_with_human += 1

            annotated = frame.copy()
            # Draw snake tracks.
            for snake in snakes:
                x1, y1, x2, y2 = map(int, snake["box"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, f"SNAKE ID {snake['id']} {snake['confidence']:.3f}",
                            (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw human-specific zones and update temporal state.
            for human in humans:
                x1, y1, x2, y2 = map(int, human["box"])
                risk = _human_risk(human, snakes)
                state, event = alert_engine.update(
                    human["id"], risk["level"], timestamp, current_index, risk
                )
                state_counts[state] += 1
                color = (0, 0, 255) if state == "CRITICAL" else ((0, 165, 255) if state == "WARNING" else (0, 200, 0))
                foot = ((x1 + x2) // 2, y2)
                cv2.circle(annotated, foot, int(risk["warning_radius"]), (0, 165, 255), 2)
                cv2.circle(annotated, foot, int(risk["critical_radius"]), (0, 0, 255), 2)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"HUMAN {human['id']} | {state}",
                            (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                if event:
                    events.append(event)

            cv2.putText(annotated, f"SnakeWatch Safety | {timestamp:.2f}s",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(annotated)
            processed += 1
            if preview_callback and (processed == 1 or processed % 5 == 0):
                preview_callback(annotated, processed, total_frames)
            if progress_callback and (processed % 5 == 0 or processed == total_frames):
                progress_callback(processed, total_frames)
    finally:
        cap.release()
        writer.release()

    elapsed = max(time.perf_counter() - start, 1e-6)
    summary = {
        "frames_processed": processed,
        "source_frames": total_frames,
        "source_fps": round(float(fps), 3),
        "source_duration_seconds": round(total_frames / fps, 2) if total_frames else None,
        "processing_seconds": round(elapsed, 2),
        "processing_fps": round(processed / elapsed, 3),
        "frames_with_snake_tracks": frames_with_snake,
        "frames_with_human_tracks": frames_with_human,
        "warning_events": sum(e["alert_level"] == "WARNING" for e in events),
        "critical_events": sum(e["alert_level"] == "CRITICAL" for e in events),
        "total_alert_events": len(events),
        "human_state_observations": state_counts,
        "threshold": float(snake_threshold),
        "frame_stride": int(frame_stride),
        "notes": "Pixel-space heuristic only; not calibrated metres or a verified snake alarm."
    }
    return {"output_path": str(output_path), "events": events, "summary": summary}
