# ============================================================
# STEP 5.3 — BYTETRACK + NMS + TENTATIVE
#             + DUPLICATE RESOLUTION + SHORT PREDICTION
#             + TENTATIVE PERSISTENCE
# ============================================================

from pathlib import Path
from collections import defaultdict
import json
import time
import math

import cv2
import numpy as np
import torch
import supervision as sv

from transformers import (
    RTDetrImageProcessor,
    RTDetrV2ForObjectDetection,
)

from supervision.tracker.byte_tracker.single_object_track import (
    STrack,
    TrackState,
)

from supervision.tracker.byte_tracker import matching


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(
    "/content/drive/MyDrive/snake_detection/Snake_detection"
)

VIDEO_PATH = (
    PROJECT_DIR
    / "snake_videos"
    / "10231672-hd_1080_1920_30fps.mp4"
)

MODEL_PATH = (
    PROJECT_DIR
    / "rtdetrv2_snake_finetuned_v3_earlystop"
    / "checkpoint-8072"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "dataset"
    / "tracking_output"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_VIDEO_PATH = (
    OUTPUT_DIR
    / "tracked_video_5_3_final.mp4"
)

OUTPUT_JSON_PATH = (
    OUTPUT_DIR
    / "track_data_5_3_final.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

TRACKER_INPUT_THRESHOLD = 0.03

TRACKER_STRONG_THRESHOLD = 0.055

TRACKER_ACTIVATION_THRESHOLD = 0.03

TRACKER_LOST_TRACK_BUFFER = 5

TRACKER_MATCHING_THRESHOLD = 0.80

NMS_IOU_THRESHOLD = 0.50

DUPLICATE_IOU_THRESHOLD = 0.50

DUPLICATE_CENTER_DISTANCE_RATIO = 0.35

MAX_FRAMES = 400

TRAIL_LENGTH = 20


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print(
    "STEP 5.3 — BYTETRACK + NMS + TENTATIVE "
    "+ DUPLICATE RESOLUTION"
)
print("=" * 70)

print(
    f"\nVideo      : {VIDEO_PATH}"
)

print(
    f"Checkpoint : {MODEL_PATH}"
)

print(
    f"Device     : {DEVICE}"
)


# ============================================================
# 4. CHECK PATHS
# ============================================================

if not VIDEO_PATH.exists():

    raise FileNotFoundError(
        f"Video not found:\n{VIDEO_PATH}"
    )

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Checkpoint not found:\n{MODEL_PATH}"
    )


# ============================================================
# 5. BYTETRACK COMPATIBILITY PATCH
# ============================================================

def patched_activate(
    self,
    kalman_filter,
    frame_id,
):

    self.kalman_filter = (
        kalman_filter
    )

    self.internal_track_id = (
        self.internal_id_counter.new_id()
    )

    self.mean, self.covariance = (
        self.kalman_filter.initiate(
            self.tlwh_to_xyah(
                self._tlwh
            )
        )
    )

    self.tracklet_len = 1

    self.state = TrackState.Tracked

    if (
        self.tracklet_len
        >= self.minimum_consecutive_frames
    ):

        self.is_activated = True

        if (
            self.external_track_id
            == self.external_id_counter.NO_ID
        ):

            self.external_track_id = (
                self.external_id_counter.new_id()
            )

    self.frame_id = frame_id

    self.start_frame = frame_id


STrack.activate = patched_activate


# ============================================================
# 6. IOU-ONLY ASSOCIATION
# ============================================================

def patched_fuse_score(
    cost_matrix,
    detections,
):

    return cost_matrix


matching.fuse_score = (
    patched_fuse_score
)


print(
    "\nByteTrack compatibility patches applied:"
)

print(
    "  - Track activation fix : ENABLED"
)

print(
    "  - IoU-only association : ENABLED"
)

print(
    "  - NMS                  : ENABLED"
)

print(
    "  - Tentative candidates : ENABLED"
)

print(
    "  - Duplicate resolver   : ENABLED"
)

print(
    "  - Short prediction     : ENABLED"
)

print(
    "  - Tentative persistence: ENABLED"
)


# ============================================================
# 7. LOAD PROCESSOR
# ============================================================

processor = (
    RTDetrImageProcessor.from_pretrained(
        "PekingU/rtdetr_v2_r18vd"
    )
)

print(
    "\nProcessor loaded from:"
)

print(
    "  PekingU/rtdetr_v2_r18vd"
)


# ============================================================
# 8. LOAD MODEL
# ============================================================

model = (
    RTDetrV2ForObjectDetection
    .from_pretrained(
        MODEL_PATH,
        num_labels=1,
        id2label={
            0: "snake"
        },
        label2id={
            "snake": 0
        },
        ignore_mismatched_sizes=True,
    )
)

model = model.to(
    DEVICE
)

model.eval()

print(
    "\nModel loaded successfully."
)


# ============================================================
# 9. CREATE BYTE TRACK
# ============================================================

tracker = sv.ByteTrack(
    track_activation_threshold=(
        TRACKER_ACTIVATION_THRESHOLD
    ),

    lost_track_buffer=(
        TRACKER_LOST_TRACK_BUFFER
    ),

    minimum_matching_threshold=(
        TRACKER_MATCHING_THRESHOLD
    ),

    frame_rate=30,
)

tracker.det_thresh = (
    TRACKER_STRONG_THRESHOLD
)


print(
    "\nByteTrack internal configuration:"
)

print(
    "  minimum_consecutive_frames :",
    tracker.minimum_consecutive_frames
)

print(
    "  track_activation_threshold :",
    tracker.track_activation_threshold
)

print(
    "  det_thresh                 :",
    tracker.det_thresh
)


# ============================================================
# 10. HELPER — CENTER
# ============================================================

def bbox_center(
    bbox
):

    x1, y1, x2, y2 = bbox

    return [
        float(
            (x1 + x2) / 2
        ),
        float(
            (y1 + y2) / 2
        ),
    ]


# ============================================================
# 11. HELPER — IOU
# ============================================================

def calculate_iou(
    box_a,
    box_b,
):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    intersection = (
        iw * ih
    )

    area_a = (
        max(
            0.0,
            ax2 - ax1
        )
        *
        max(
            0.0,
            ay2 - ay1
        )
    )

    area_b = (
        max(
            0.0,
            bx2 - bx1
        )
        *
        max(
            0.0,
            by2 - by1
        )
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:

        return 0.0

    return (
        intersection / union
    )


# ============================================================
# 12. HELPER — CENTER DISTANCE RATIO
# ============================================================

def center_distance_ratio(
    box_a,
    box_b,
):

    center_a = bbox_center(
        box_a
    )

    center_b = bbox_center(
        box_b
    )

    distance = math.sqrt(
        (
            center_a[0]
            - center_b[0]
        ) ** 2
        +
        (
            center_a[1]
            - center_b[1]
        ) ** 2
    )

    width_a = max(
        1.0,
        box_a[2] - box_a[0]
    )

    height_a = max(
        1.0,
        box_a[3] - box_a[1]
    )

    width_b = max(
        1.0,
        box_b[2] - box_b[0]
    )

    height_b = max(
        1.0,
        box_b[3] - box_b[1]
    )

    reference_size = max(
        1.0,
        (
            width_a
            + height_a
            + width_b
            + height_b
        ) / 4.0
    )

    return (
        distance
        / reference_size
    )


# ============================================================
# 13. NMS
# ============================================================

def apply_nms(
    detections,
    iou_threshold,
):

    if not detections:

        return []

    boxes = []
    scores = []

    for detection in detections:

        boxes.append(
            detection["bbox"]
        )

        scores.append(
            float(
                detection["confidence"]
            )
        )

    boxes_xywh = []

    for box in boxes:

        x1, y1, x2, y2 = box

        boxes_xywh.append(
            [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1),
            ]
        )

    indices = cv2.dnn.NMSBoxes(
        boxes_xywh,
        scores,
        score_threshold=(
            TRACKER_INPUT_THRESHOLD
        ),
        nms_threshold=(
            iou_threshold
        ),
    )

    if indices is None:

        return []

    indices = np.array(
        indices
    ).reshape(-1)

    return [
        detections[int(i)]
        for i in indices
    ]


# ============================================================
# 14. DETECTOR
# ============================================================

@torch.no_grad()
def detect_snakes(
    frame
):

    image = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value
        in inputs.items()
    }

    outputs = model(
        **inputs
    )

    target_sizes = torch.tensor(
        [
            [
                frame.shape[0],
                frame.shape[1],
            ]
        ],
        device=DEVICE,
    )

    results = (
        processor
        .post_process_object_detection(
            outputs,
            threshold=(
                TRACKER_INPUT_THRESHOLD
            ),
            target_sizes=target_sizes,
        )[0]
    )

    detections = []

    for score, label, box in zip(
        results["scores"],
        results["labels"],
        results["boxes"],
    ):

        class_id = int(
            label.item()
        )

        if class_id != 0:

            continue

        confidence = float(
            score.item()
        )

        if (
            confidence
            < TRACKER_INPUT_THRESHOLD
        ):

            continue

        bbox = [
            float(v)
            for v in box.tolist()
        ]

        detections.append(
            {
                "bbox": bbox,

                "confidence":
                    confidence,

                "class_id":
                    0,

                "class_name":
                    "snake",
            }
        )

    return detections


# ============================================================
# 15. CONVERT TO SUPERVISION
# ============================================================

def detections_to_supervision(
    detections
):

    if not detections:

        return sv.Detections.empty()

    xyxy = np.array(
        [
            d["bbox"]
            for d in detections
        ],
        dtype=np.float32,
    )

    confidence = np.array(
        [
            d["confidence"]
            for d in detections
        ],
        dtype=np.float32,
    )

    class_id = np.zeros(
        len(detections),
        dtype=int,
    )

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
    )


# ============================================================
# 16. SINGLE-SNAKE DUPLICATE RESOLVER
# ============================================================

class SingleSnakeTrackResolver:

    def __init__(
        self,
        iou_threshold,
        center_distance_ratio,
    ):

        self.iou_threshold = (
            iou_threshold
        )

        self.center_distance_ratio = (
            center_distance_ratio
        )

        self.suppressed_ids = set()

        self.track_histories = (
            defaultdict(list)
        )

    def update(
        self,
        track_id,
        bbox,
    ):

        self.track_histories[
            track_id
        ].append(
            bbox
        )

    def history_length(
        self,
        track_id,
    ):

        return len(
            self.track_histories.get(
                track_id,
                []
            )
        )

    def resolve(
        self,
        active_tracks,
    ):

        if len(active_tracks) <= 1:

            return active_tracks

        suppressed = set()

        for i in range(
            len(active_tracks)
        ):

            track_a = (
                active_tracks[i]
            )

            id_a = int(
                track_a["track_id"]
            )

            if id_a in suppressed:

                continue

            for j in range(
                i + 1,
                len(active_tracks)
            ):

                track_b = (
                    active_tracks[j]
                )

                id_b = int(
                    track_b["track_id"]
                )

                if id_b in suppressed:

                    continue

                iou = calculate_iou(
                    track_a["bbox"],
                    track_b["bbox"],
                )

                distance_ratio = (
                    center_distance_ratio(
                        track_a["bbox"],
                        track_b["bbox"],
                    )
                )

                duplicate = (
                    iou
                    >= self.iou_threshold
                    or
                    distance_ratio
                    <= self.center_distance_ratio
                )

                if not duplicate:

                    continue

                history_a = (
                    self.history_length(
                        id_a
                    )
                )

                history_b = (
                    self.history_length(
                        id_b
                    )
                )

                if history_a >= history_b:

                    suppressed.add(
                        id_b
                    )

                else:

                    suppressed.add(
                        id_a
                    )

        self.suppressed_ids.update(
            suppressed
        )

        return [
            track
            for track in active_tracks
            if int(
                track["track_id"]
            )
            not in suppressed
        ]


resolver = (
    SingleSnakeTrackResolver(
        iou_threshold=(
            DUPLICATE_IOU_THRESHOLD
        ),
        center_distance_ratio=(
            DUPLICATE_CENTER_DISTANCE_RATIO
        ),
    )
)


# ============================================================
# 17. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open video:\n"
        f"{VIDEO_PATH}"
    )


video_fps = cap.get(
    cv2.CAP_PROP_FPS
)

if video_fps <= 0:

    video_fps = 30.0


video_width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

video_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_video_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print(
    f"\nVideo FPS: "
    f"{video_fps:.2f}"
)

print(
    f"Video size: "
    f"{video_width} x "
    f"{video_height}"
)

print(
    f"Total video frames: "
    f"{total_video_frames}"
)


# ============================================================
# 18. VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO_PATH),
    fourcc,
    video_fps,
    (
        video_width,
        video_height,
    ),
)

if not writer.isOpened():

    cap.release()

    raise RuntimeError(
        "Could not create output video."
    )


# ============================================================
# 19. TRACK STATE
# ============================================================

track_last_seen = {}

trail_history = (
    defaultdict(list)
)

# ------------------------------------------------------------
# IMPORTANT
#
# all_records remains EXACTLY the same data stream used
# by the validated tracker.
# ------------------------------------------------------------

all_records = []

# ------------------------------------------------------------
# NEW:
# Separate persistent copy of tentative observations.
# This does NOT affect tracking.
# ------------------------------------------------------------

tentative_records = []

unique_track_ids = set()

suppressed_ids = set()


# ============================================================
# 20. COUNTERS
# ============================================================

frames_processed = 0

raw_detector_candidates = 0

after_nms_candidates = 0

weak_candidates = 0

tentative_observations = 0

detection_observations = 0

prediction_observations = 0

frames_with_detections = 0

frames_with_strong_detections = 0

frames_with_tentative = 0

frames_with_predictions = 0

frames_with_tracks = 0

total_tracked_observations = 0


# ============================================================
# 21. MAIN LOOP
# ============================================================

start_time = time.time()


while (
    frames_processed
    < MAX_FRAMES
):

    ret, frame = cap.read()

    if not ret:

        break

    frame_id = (
        frames_processed
    )

    timestamp = (
        frame_id
        / video_fps
    )


    # ========================================================
    # DETECTION
    # ========================================================

    detections = detect_snakes(
        frame
    )

    raw_detector_candidates += (
        len(detections)
    )

    if detections:

        frames_with_detections += 1


    # ========================================================
    # NMS
    # ========================================================

    detections_nms = apply_nms(
        detections,
        NMS_IOU_THRESHOLD
    )

    after_nms_candidates += (
        len(detections_nms)
    )


    # ========================================================
    # STRONG / WEAK SPLIT
    # ========================================================

    strong_detections = []

    weak_detections = []

    for detection in detections_nms:

        if (
            detection["confidence"]
            >= TRACKER_STRONG_THRESHOLD
        ):

            strong_detections.append(
                detection
            )

        else:

            weak_detections.append(
                detection
            )

    weak_candidates += (
        len(weak_detections)
    )

    if strong_detections:

        frames_with_strong_detections += 1


    # ========================================================
    # TENTATIVE LOGIC
    #
    # IMPORTANT:
    #
    # This is the validated behavior.
    #
    # DO NOT replace this with:
    #
    #     max(weak_detections)
    #
    # or store every weak candidate.
    #
    # The existing validated tentative logic determines
    # whether a weak candidate becomes a tentative observation.
    # ========================================================

    tentative_detection = None

    if weak_detections:

        tentative_detection = max(
            weak_detections,
            key=lambda d:
                d["confidence"]
        )


    # --------------------------------------------------------
    # Create the SAME tentative record used by the
    # visualization.
    # --------------------------------------------------------

    if tentative_detection is not None:

        bbox = [
            float(v)
            for v
            in tentative_detection[
                "bbox"
            ]
        ]

        center = bbox_center(
            bbox
        )

        confidence = float(
            tentative_detection[
                "confidence"
            ]
        )

        tentative_record = {

            "frame_id":
                int(frame_id),

            "timestamp":
                float(timestamp),

            "track_id":
                None,

            "bbox":
                bbox,

            "center":
                center,

            "confidence":
                confidence,

            "class_id":
                int(
                    tentative_detection[
                        "class_id"
                    ]
                ),

            "class_name":
                "snake",

            "observation_type":
                "tentative",

            "detection_strength":
                "weak",

            "track_state":
                "tentative",
        }

        # ----------------------------------------------------
        # Existing record stream
        # ----------------------------------------------------

        all_records.append(
            tentative_record
        )

        # ----------------------------------------------------
        # NEW persistence copy
        #
        # This does NOT participate in tracking.
        # ----------------------------------------------------

        tentative_records.append(
            tentative_record
        )

        tentative_observations += 1

        frames_with_tentative += 1

        # ----------------------------------------------------
        # Draw tentative box
        # ----------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            bbox
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            (
                "TENTATIVE "
                f"{confidence:.3f}"
            ),
            (
                x1,
                max(
                    25,
                    y1 - 8
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )


    # ========================================================
    # BYTE TRACK
    # ========================================================

    sv_detections = (
        detections_to_supervision(
            detections_nms
        )
    )

    tracked_detections = (
        tracker.update_with_detections(
            sv_detections
        )
    )


    # ========================================================
    # EXTRACT TRACKS
    # ========================================================

    active_tracks = []

    if (
        len(
            tracked_detections
        )
        > 0
    ):

        boxes = (
            tracked_detections.xyxy
        )

        confidences = (
            tracked_detections.confidence
        )

        tracker_ids = (
            tracked_detections.tracker_id
        )

        if tracker_ids is not None:

            for i, box in enumerate(
                boxes
            ):

                track_id_value = (
                    tracker_ids[i]
                )

                if (
                    track_id_value
                    is None
                ):

                    continue

                track_id = int(
                    track_id_value
                )

                bbox = [
                    float(v)
                    for v in box
                ]

                confidence = float(
                    confidences[i]
                )

                active_tracks.append(
                    {
                        "track_id":
                            track_id,

                        "bbox":
                            bbox,

                        "confidence":
                            confidence,
                    }
                )


    # ========================================================
    # DUPLICATE RESOLUTION
    # ========================================================

    active_tracks = (
        resolver.resolve(
            active_tracks
        )
    )

    suppressed_ids.update(
        resolver.suppressed_ids
    )


    # ========================================================
    # CONFIRMED DETECTIONS
    # ========================================================

    confirmed_this_frame = 0

    for track in active_tracks:

        track_id = int(
            track["track_id"]
        )

        bbox = [
            float(v)
            for v in track["bbox"]
        ]

        confidence = float(
            track["confidence"]
        )

        center = bbox_center(
            bbox
        )

        unique_track_ids.add(
            track_id
        )

        track_last_seen[
            track_id
        ] = frame_id

        resolver.update(
            track_id,
            bbox
        )

        strength = (
            "strong"
            if confidence
            >= TRACKER_STRONG_THRESHOLD
            else "weak"
        )

        record = {

            "frame_id":
                int(frame_id),

            "timestamp":
                float(timestamp),

            "track_id":
                int(track_id),

            "bbox":
                bbox,

            "center":
                center,

            "confidence":
                confidence,

            "class_id":
                0,

            "class_name":
                "snake",

            "observation_type":
                "detection",

            "detection_strength":
                strength,

            "track_state":
                "tracked",
        }

        all_records.append(
            record
        )

        detection_observations += 1

        total_tracked_observations += 1

        confirmed_this_frame += 1


        # ----------------------------------------------------
        # GREEN BOX
        # ----------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            bbox
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        cv2.putText(
            frame,
            (
                f"Snake | ID {track_id} "
                f"| {confidence:.3f}"
            ),
            (
                x1,
                max(
                    30,
                    y1 - 10
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 255, 0),
            2,
        )


        # ----------------------------------------------------
        # DETECTION-ONLY TRAIL
        # ----------------------------------------------------

        trail_history[
            track_id
        ].append(
            (
                int(center[0]),
                int(center[1])
            )
        )

        if (
            len(
                trail_history[
                    track_id
                ]
            )
            > TRAIL_LENGTH
        ):

            trail_history[
                track_id
            ] = (
                trail_history[
                    track_id
                ][
                    -TRAIL_LENGTH:
                ]
            )


    if confirmed_this_frame:

        frames_with_tracks += 1


    # ========================================================
    # DRAW TRAILS
    # ========================================================

    for (
        track_id,
        points
    ) in trail_history.items():

        if (
            track_id
            in resolver.suppressed_ids
        ):

            continue

        if len(points) < 2:

            continue

        for p1, p2 in zip(
            points[:-1],
            points[1:],
        ):

            cv2.line(
                frame,
                p1,
                p2,
                (255, 0, 0),
                2,
            )


    # ========================================================
    # LOST TRACK PREDICTIONS
    # ========================================================

    current_prediction_ids = set()

    if hasattr(
        tracker,
        "lost_tracks"
    ):

        for lost_track in (
            tracker.lost_tracks
        ):

            try:

                track_id = int(
                    lost_track.external_track_id
                )

            except Exception:

                continue

            if track_id < 0:

                continue

            if (
                track_id
                in resolver.suppressed_ids
            ):

                continue

            last_seen = (
                track_last_seen.get(
                    track_id,
                    frame_id
                )
            )

            missed = (
                frame_id
                - last_seen
            )

            if (
                missed <= 0
                or
                missed
                > TRACKER_LOST_TRACK_BUFFER
            ):

                continue

            try:

                tlwh = (
                    lost_track.tlwh
                )

                x, y, w, h = [
                    float(v)
                    for v in tlwh
                ]

                bbox = [
                    x,
                    y,
                    x + w,
                    y + h,
                ]

                center = bbox_center(
                    bbox
                )

            except Exception:

                continue


            prediction_record = {

                "frame_id":
                    int(frame_id),

                "timestamp":
                    float(timestamp),

                "track_id":
                    int(track_id),

                "bbox":
                    bbox,

                "center":
                    center,

                "confidence":
                    None,

                "class_id":
                    0,

                "class_name":
                    "snake",

                "observation_type":
                    "prediction",

                "detection_strength":
                    None,

                "track_state":
                    "lost",
            }

            all_records.append(
                prediction_record
            )

            prediction_observations += 1

            total_tracked_observations += 1

            current_prediction_ids.add(
                track_id
            )

            unique_track_ids.add(
                track_id
            )


            # ------------------------------------------------
            # ORANGE PREDICTION BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                bbox
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 165, 255),
                3,
            )

            cv2.putText(
                frame,
                (
                    f"Snake | ID {track_id} "
                    "| PRED"
                ),
                (
                    x1,
                    max(
                        30,
                        y1 - 10
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
            )


    if current_prediction_ids:

        frames_with_predictions += 1


    # ========================================================
    # FRAME STATUS
    # ========================================================

    cv2.putText(
        frame,
        (
            f"Frame {frame_id} | "
            f"{timestamp:.2f}s"
        ),
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        (
            f"Confirmed: "
            f"{confirmed_this_frame}"
        ),
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        (
            f"Candidates: "
            f"{len(detections_nms)}"
        ),
        (20, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )


    # ========================================================
    # WRITE FRAME
    # ========================================================

    writer.write(
        frame
    )

    frames_processed += 1


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        frames_processed % 50 == 0
        or
        frames_processed == 1
    ):

        elapsed = (
            time.time()
            - start_time
        )

        effective_fps = (
            frames_processed
            / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"Processed "
            f"{frames_processed}/"
            f"{MAX_FRAMES}"
            f" | FPS "
            f"{effective_fps:.2f}"
        )


# ============================================================
# 22. CLEANUP
# ============================================================

cap.release()

writer.release()


# ============================================================
# 23. FINAL STATISTICS
# ============================================================

elapsed = (
    time.time()
    - start_time
)

effective_fps = (
    frames_processed
    / elapsed
    if elapsed > 0
    else 0
)


print()
print("=" * 70)
print(
    "STEP 5.3 REVISED COMPLETE"
)
print("=" * 70)

print(
    f"Frames processed           : "
    f"{frames_processed}"
)

print(
    f"Raw detector candidates   : "
    f"{raw_detector_candidates}"
)

print(
    f"After NMS                 : "
    f"{after_nms_candidates}"
)

print(
    f"Weak/tentative candidates : "
    f"{weak_candidates}"
)

print(
    f"Tentative observations    : "
    f"{tentative_observations}"
)

print(
    f"Detection observations    : "
    f"{detection_observations}"
)

print(
    f"Prediction observations   : "
    f"{prediction_observations}"
)

print(
    f"Frames with detections    : "
    f"{frames_with_detections}"
)

print(
    f"Frames with strong detect.: "
    f"{frames_with_strong_detections}"
)

print(
    f"Frames with tentative     : "
    f"{frames_with_tentative}"
)

print(
    f"Frames with predictions   : "
    f"{frames_with_predictions}"
)

print(
    f"Frames with tracks        : "
    f"{frames_with_tracks}"
)

print(
    f"Total tracked observations: "
    f"{total_tracked_observations}"
)

print(
    f"Unique Track IDs          : "
    f"{len(unique_track_ids)}"
)

print(
    f"Track IDs                 : "
    f"{sorted(unique_track_ids)}"
)

print(
    f"Suppressed duplicate IDs  : "
    f"{sorted(suppressed_ids)}"
)

print(
    f"Effective processing FPS  : "
    f"{effective_fps:.2f}"
)


# ============================================================
# 24. SAVE JSON
# ============================================================

json_output = {

    "video":
        str(VIDEO_PATH),

    "fps":
        float(video_fps),

    "frames_processed":
        int(frames_processed),

    "raw_detector_candidates":
        int(raw_detector_candidates),

    "after_nms_candidates":
        int(after_nms_candidates),

    "weak_candidates":
        int(weak_candidates),

    "tentative_observations":
        int(tentative_observations),

    "detection_observations":
        int(detection_observations),

    "prediction_observations":
        int(prediction_observations),

    "frames_with_detections":
        int(frames_with_detections),

    "frames_with_strong_detections":
        int(frames_with_strong_detections),

    "frames_with_tentative":
        int(frames_with_tentative),

    "frames_with_predictions":
        int(frames_with_predictions),

    "frames_with_tracks":
        int(frames_with_tracks),

    "total_tracked_observations":
        int(total_tracked_observations),

    "unique_track_ids":
        sorted(
            int(x)
            for x in unique_track_ids
        ),

    "track_count":
        int(
            len(unique_track_ids)
        ),

    "suppressed_duplicate_ids":
        sorted(
            int(x)
            for x in suppressed_ids
        ),

    "configuration": {

        "tracker_input_threshold":
            TRACKER_INPUT_THRESHOLD,

        "tracker_strong_threshold":
            TRACKER_STRONG_THRESHOLD,

        "tracker_activation_threshold":
            TRACKER_ACTIVATION_THRESHOLD,

        "tracker_lost_track_buffer":
            TRACKER_LOST_TRACK_BUFFER,

        "tracker_matching_threshold":
            TRACKER_MATCHING_THRESHOLD,

        "nms_iou_threshold":
            NMS_IOU_THRESHOLD,

        "duplicate_iou_threshold":
            DUPLICATE_IOU_THRESHOLD,

        "duplicate_center_distance_ratio":
            DUPLICATE_CENTER_DISTANCE_RATIO,
    },

    # --------------------------------------------------------
    # EXISTING TRACK DATA
    # --------------------------------------------------------

    "tracks":
        all_records,

    # --------------------------------------------------------
    # NEW PERSISTENT TENTATIVE DATA
    # --------------------------------------------------------

    "tentative_records":
        tentative_records,
}


with open(
    OUTPUT_JSON_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        json_output,
        f,
        indent=2,
    )


# ============================================================
# 25. OUTPUT
# ============================================================

print()
print(
    "JSON STRUCTURE:"
)

print(
    "  tracks            = existing all_records"
)

print(
    "  tentative_records = separate tentative copy"
)

print()
print(
    "OUTPUTS"
)

print(
    f"Video : {OUTPUT_VIDEO_PATH}"
)

print(
    f"JSON  : {OUTPUT_JSON_PATH}"
)

print()
print(
    "STEP 5.3 COMPLETE"
)

print("=" * 70)












# # ============================================================
# # STEP 5.3 — BYTETRACK + NMS + SINGLE TENTATIVE BOX
# #            + DUPLICATE RESOLUTION + SHORT PREDICTION
# # ============================================================

# from pathlib import Path
# import json
# import time
# import cv2
# import numpy as np
# import torch
# import supervision as sv

# from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection


# # ============================================================
# # 1. PATHS
# # ============================================================

# PROJECT_DIR = Path(
#     "/content/drive/MyDrive/snake_detection/Snake_detection"
# )

# VIDEO_PATH = (
#     PROJECT_DIR
#     / "snake_videos"
#     / "10231672-hd_1080_1920_30fps.mp4"
# )

# MODEL_PATH = (
#     PROJECT_DIR
#     / "rtdetrv2_snake_finetuned_v3_earlystop"
#     / "checkpoint-8072"
# )

# OUTPUT_DIR = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
# )

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# OUTPUT_VIDEO_PATH = (
#     OUTPUT_DIR / "tracked_video_5_3_final.mp4"
# )

# OUTPUT_JSON_PATH = (
#     OUTPUT_DIR / "track_data_5_3_final.json"
# )


# # ============================================================
# # 2. SETTINGS
# # ============================================================

# # Detector candidates
# TRACKER_INPUT_THRESHOLD = 0.03

# # Strong / confirmed detection threshold
# TRACKER_STRONG_THRESHOLD = 0.055

# # IMPORTANT:
# # Keep this at 0.03 for the current architecture.
# #
# # This means detections >= 0.03 are allowed to enter tracking.
# TRACKER_ACTIVATION_THRESHOLD = 0.03

# # Lost-track prediction buffer
# TRACKER_LOST_TRACK_BUFFER = 5

# # IoU association threshold
# TRACKER_MATCHING_THRESHOLD = 0.80

# # NMS
# NMS_IOU_THRESHOLD = 0.50

# # Duplicate track resolver
# DUPLICATE_IOU_THRESHOLD = 0.50
# DUPLICATE_CENTER_DISTANCE_RATIO = 0.35

# # Test only first 400 frames
# MAX_FRAMES = 400

# # Trail length
# TRAIL_LENGTH = 20

# # Tentative visualization:
# # We show ONLY ONE weak candidate per frame.
# SHOW_TENTATIVE_BOX = True


# # ============================================================
# # 3. DEVICE
# # ============================================================

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# print("=" * 70)
# print("STEP 5.3 — BYTETRACK + NMS + TENTATIVE + DUPLICATE RESOLUTION")
# print("=" * 70)

# print(f"\nVideo      : {VIDEO_PATH}")
# print(f"Checkpoint : {MODEL_PATH}")
# print(f"Device     : {DEVICE}")


# # ============================================================
# # 4. CHECK PATHS
# # ============================================================

# if not VIDEO_PATH.exists():
#     raise FileNotFoundError(
#         f"Video not found:\n{VIDEO_PATH}"
#     )

# if not MODEL_PATH.exists():
#     raise FileNotFoundError(
#         f"Model checkpoint not found:\n{MODEL_PATH}"
#     )

# print("\nCheckpoint contents:")
# for item in sorted(MODEL_PATH.iterdir()):
#     print(" ", item.name)


# # ============================================================
# # 5. SUPERVISION COMPATIBILITY PATCHES
# # ============================================================

# from supervision.tracker.byte_tracker.single_object_track import (
#     STrack,
#     TrackState,
# )


# # ------------------------------------------------------------
# # Patch 1:
# # Allow tracks to activate regardless of frame_id == 1.
# # ------------------------------------------------------------

# def patched_activate(self, kalman_filter, frame_id):

#     self.kalman_filter = kalman_filter

#     self.internal_track_id = (
#         self.internal_id_counter.new_id()
#     )

#     self.mean, self.covariance = (
#         self.kalman_filter.initiate(
#             self.tlwh_to_xyah(self._tlwh)
#         )
#     )

#     self.tracklet_len = 1

#     self.state = TrackState.Tracked

#     if self.tracklet_len >= self.minimum_consecutive_frames:

#         self.is_activated = True

#         if (
#             self.external_track_id
#             == self.external_id_counter.NO_ID
#         ):
#             self.external_track_id = (
#                 self.external_id_counter.new_id()
#             )

#     self.frame_id = frame_id
#     self.start_frame = frame_id


# STrack.activate = patched_activate


# # ------------------------------------------------------------
# # Patch 2:
# # Use pure IoU association rather than confidence-weighted
# # fusion.
# # ------------------------------------------------------------

# from supervision.tracker.byte_tracker import matching


# def patched_fuse_score(cost_matrix, detections):
#     return cost_matrix


# matching.fuse_score = patched_fuse_score


# print("\nByteTrack compatibility patches applied:")
# print("  - Track activation fix : ENABLED")
# print("  - IoU-only association : ENABLED")
# print("  - NMS                  : ENABLED")
# print("  - Tentative candidates : ENABLED")
# print("  - Duplicate resolver   : ENABLED")
# print("  - Short prediction     : ENABLED")


# # ============================================================
# # 6. LOAD PROCESSOR
# # ============================================================

# print("\nLoading RT-DETRv2 processor...")

# PROCESSOR_SOURCE = "PekingU/rtdetr_v2_r18vd"

# processor = RTDetrImageProcessor.from_pretrained(
#     PROCESSOR_SOURCE,
#     do_resize=True,
#     size={
#         "height": 640,
#         "width": 640
#     },
#     do_rescale=True,
#     rescale_factor=1 / 255,
#     do_normalize=False,
#     do_pad=False
# )

# print("Processor loaded.")


# # ============================================================
# # 7. LOAD MODEL
# # ============================================================

# print("\nLoading RT-DETRv2 model...")

# model = RTDetrV2ForObjectDetection.from_pretrained(
#     str(MODEL_PATH),
#     local_files_only=True,
#     num_labels=1,
#     id2label={
#         0: "snake"
#     },
#     label2id={
#         "snake": 0
#     },
#     ignore_mismatched_sizes=True
# )

# model.to(DEVICE)
# model.eval()

# print("Model loaded successfully.")
# print(f"Device: {DEVICE}")


# # ============================================================
# # 8. DETECTION FUNCTION
# # ============================================================

# @torch.no_grad()
# def detect_snakes(frame):
#     """
#     Run RT-DETRv2 on one frame.

#     Returns:
#         list of dictionaries:
#         {
#             bbox: [x1, y1, x2, y2],
#             confidence: float,
#             class_id: 0,
#             class_name: "snake"
#         }
#     """

#     image_rgb = cv2.cvtColor(
#         frame,
#         cv2.COLOR_BGR2RGB
#     )

#     inputs = processor(
#         images=image_rgb,
#         return_tensors="pt"
#     )

#     inputs = {
#         k: v.to(DEVICE)
#         for k, v in inputs.items()
#     }

#     outputs = model(**inputs)

#     target_sizes = torch.tensor(
#         [[frame.shape[0], frame.shape[1]]],
#         device=DEVICE
#     )

#     results = processor.post_process_object_detection(
#         outputs,
#         threshold=TRACKER_INPUT_THRESHOLD,
#         target_sizes=target_sizes
#     )[0]

#     detections = []

#     for score, label, box in zip(
#         results["scores"],
#         results["labels"],
#         results["boxes"]
#     ):

#         class_id = int(label.item())

#         # Only snake class
#         if class_id != 0:
#             continue

#         confidence = float(score.item())

#         x1, y1, x2, y2 = (
#             box.detach()
#             .cpu()
#             .numpy()
#             .tolist()
#         )

#         detections.append({
#             "bbox": [
#                 float(x1),
#                 float(y1),
#                 float(x2),
#                 float(y2)
#             ],
#             "confidence": confidence,
#             "class_id": 0,
#             "class_name": "snake"
#         })

#     return detections


# # ============================================================
# # 9. GEOMETRY HELPERS
# # ============================================================

# def bbox_iou(box_a, box_b):

#     ax1, ay1, ax2, ay2 = box_a
#     bx1, by1, bx2, by2 = box_b

#     ix1 = max(ax1, bx1)
#     iy1 = max(ay1, by1)
#     ix2 = min(ax2, bx2)
#     iy2 = min(ay2, by2)

#     iw = max(0.0, ix2 - ix1)
#     ih = max(0.0, iy2 - iy1)

#     intersection = iw * ih

#     area_a = max(0.0, ax2 - ax1) * max(
#         0.0, ay2 - ay1
#     )

#     area_b = max(0.0, bx2 - bx1) * max(
#         0.0, by2 - by1
#     )

#     union = area_a + area_b - intersection

#     if union <= 0:
#         return 0.0

#     return intersection / union


# def bbox_center(box):

#     x1, y1, x2, y2 = box

#     return (
#         (x1 + x2) / 2.0,
#         (y1 + y2) / 2.0
#     )


# def bbox_area(box):

#     x1, y1, x2, y2 = box

#     return max(
#         0.0,
#         x2 - x1
#     ) * max(
#         0.0,
#         y2 - y1
#     )


# def center_distance(box_a, box_b):

#     ca = bbox_center(box_a)
#     cb = bbox_center(box_b)

#     return float(
#         np.sqrt(
#             (ca[0] - cb[0]) ** 2
#             +
#             (ca[1] - cb[1]) ** 2
#         )
#     )


# def normalized_center_distance(
#     box_a,
#     box_b,
#     frame_width,
#     frame_height
# ):

#     distance = center_distance(
#         box_a,
#         box_b
#     )

#     diagonal = np.sqrt(
#         frame_width ** 2
#         +
#         frame_height ** 2
#     )

#     if diagonal <= 0:
#         return 1.0

#     return distance / diagonal


# # ============================================================
# # 10. MANUAL NMS
# # ============================================================

# def apply_nms(
#     detections,
#     iou_threshold=NMS_IOU_THRESHOLD
# ):

#     if not detections:
#         return []

#     detections = sorted(
#         detections,
#         key=lambda d: d["confidence"],
#         reverse=True
#     )

#     kept = []

#     while detections:

#         best = detections.pop(0)

#         kept.append(best)

#         remaining = []

#         for candidate in detections:

#             iou = bbox_iou(
#                 best["bbox"],
#                 candidate["bbox"]
#             )

#             if iou < iou_threshold:
#                 remaining.append(candidate)

#         detections = remaining

#     return kept


# # ============================================================
# # 11. SINGLE-SNAKE DUPLICATE RESOLVER
# # ============================================================

# class SingleSnakeTrackResolver:

#     """
#     The selected test video contains one snake.

#     If two confirmed ByteTrack IDs appear to represent
#     the same physical snake, suppress the weaker track.

#     Duplicate criteria:

#         IoU >= 0.50
#         OR
#         normalized center distance <= 0.35

#     The track with more observations is preferred.
#     """

#     def __init__(
#         self,
#         iou_threshold=0.50,
#         center_distance_ratio=0.35
#     ):

#         self.iou_threshold = iou_threshold
#         self.center_distance_ratio = (
#             center_distance_ratio
#         )

#         self.suppressed_ids = set()

#     def resolve(
#         self,
#         active_tracks,
#         frame_width,
#         frame_height
#     ):

#         valid_tracks = []

#         for track in active_tracks:

#             if track.get("track_id") is None:
#                 continue

#             if track["track_id"] in self.suppressed_ids:
#                 continue

#             valid_tracks.append(track)

#         suppressed_this_frame = set()

#         for i in range(len(valid_tracks)):

#             for j in range(i + 1, len(valid_tracks)):

#                 a = valid_tracks[i]
#                 b = valid_tracks[j]

#                 iou = bbox_iou(
#                     a["bbox"],
#                     b["bbox"]
#                 )

#                 center_ratio = (
#                     normalized_center_distance(
#                         a["bbox"],
#                         b["bbox"],
#                         frame_width,
#                         frame_height
#                     )
#                 )

#                 same_object = (
#                     iou >= self.iou_threshold
#                     or
#                     center_ratio
#                     <= self.center_distance_ratio
#                 )

#                 if not same_object:
#                     continue

#                 a_count = a.get(
#                     "observation_count",
#                     0
#                 )

#                 b_count = b.get(
#                     "observation_count",
#                     0
#                 )

#                 if a_count >= b_count:

#                     suppressed_id = b["track_id"]

#                 else:

#                     suppressed_id = a["track_id"]

#                 suppressed_this_frame.add(
#                     suppressed_id
#                 )

#         self.suppressed_ids.update(
#             suppressed_this_frame
#         )

#         return suppressed_this_frame


# # ============================================================
# # 12. OPEN VIDEO
# # ============================================================

# cap = cv2.VideoCapture(
#     str(VIDEO_PATH)
# )

# if not cap.isOpened():
#     raise RuntimeError(
#         f"Could not open video:\n{VIDEO_PATH}"
#     )

# video_fps = cap.get(
#     cv2.CAP_PROP_FPS
# )

# if video_fps <= 0:
#     video_fps = 30.0

# video_width = int(
#     cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# )

# video_height = int(
#     cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# )

# total_video_frames = int(
#     cap.get(cv2.CAP_PROP_FRAME_COUNT)
# )


# print("\nVIDEO")
# print(f"Resolution {video_width} x {video_height}")
# print(f"FPS        {video_fps:.2f}")
# print(f"Total      {total_video_frames}")
# print(f"Test limit {MAX_FRAMES}")


# # ============================================================
# # 13. INITIALIZE BYTETRACK
# # ============================================================

# tracker = sv.ByteTrack(
#     track_activation_threshold=(
#         TRACKER_ACTIVATION_THRESHOLD
#     ),
#     lost_track_buffer=(
#         TRACKER_LOST_TRACK_BUFFER
#     ),
#     minimum_matching_threshold=(
#         TRACKER_MATCHING_THRESHOLD
#     ),
#     frame_rate=video_fps,
# )


# # ------------------------------------------------------------
# # Explicitly retain the requested internal detection
# # threshold for compatibility/diagnostics.
# # ------------------------------------------------------------

# tracker.det_thresh = TRACKER_STRONG_THRESHOLD


# print("\nByteTrack initialized.")

# print(
#     f"Input threshold       : "
#     f"{TRACKER_INPUT_THRESHOLD:.3f}"
# )

# print(
#     f"Strong threshold      : "
#     f"{TRACKER_STRONG_THRESHOLD:.3f}"
# )

# print(
#     f"Activation threshold  : "
#     f"{TRACKER_ACTIVATION_THRESHOLD:.3f}"
# )

# print(
#     f"NMS IoU               : "
#     f"{NMS_IOU_THRESHOLD:.2f}"
# )

# print(
#     f"Duplicate IoU         : "
#     f"{DUPLICATE_IOU_THRESHOLD:.2f}"
# )

# print(
#     f"Duplicate center      : "
#     f"{DUPLICATE_CENTER_DISTANCE_RATIO:.2f}"
# )

# print(
#     f"Lost buffer           : "
#     f"{TRACKER_LOST_TRACK_BUFFER}"
# )

# print(
#     f"Matching threshold    : "
#     f"{TRACKER_MATCHING_THRESHOLD:.2f}"
# )

# print("Association mode      : IoU-only")
# print("Prediction recording  : ENABLED")
# print("Tentative candidates  : ENABLED")
# print("Tentative display     : ONE BOX MAX")


# # ============================================================
# # 14. BYTE TRACK DIAGNOSTIC VALUES
# # ============================================================

# print("\nByteTrack internal configuration:")

# if hasattr(
#     tracker,
#     "minimum_consecutive_frames"
# ):

#     print(
#         "  minimum_consecutive_frames :",
#         tracker.minimum_consecutive_frames
#     )

# if hasattr(
#     tracker,
#     "track_activation_threshold"
# ):

#     print(
#         "  track_activation_threshold  :",
#         tracker.track_activation_threshold
#     )

# if hasattr(
#     tracker,
#     "det_thresh"
# ):

#     print(
#         "  det_thresh                  :",
#         tracker.det_thresh
#     )


# # ============================================================
# # 15. VIDEO WRITER
# # ============================================================

# fourcc = cv2.VideoWriter_fourcc(
#     *"mp4v"
# )

# writer = cv2.VideoWriter(
#     str(OUTPUT_VIDEO_PATH),
#     fourcc,
#     video_fps,
#     (
#         video_width,
#         video_height
#     )
# )

# if not writer.isOpened():
#     cap.release()

#     raise RuntimeError(
#         "Could not create output video."
#     )


# # ============================================================
# # 16. TRACK STATE
# # ============================================================

# resolver = SingleSnakeTrackResolver(
#     iou_threshold=DUPLICATE_IOU_THRESHOLD,
#     center_distance_ratio=(
#         DUPLICATE_CENTER_DISTANCE_RATIO
#     )
# )

# track_histories = {}

# track_observation_counts = {}

# track_last_seen = {}

# track_first_seen = {}

# track_last_bbox = {}

# track_last_confidence = {}

# track_states = {}

# all_records = []

# unique_track_ids = set()

# suppressed_ids = set()

# trail_history = {}


# # ============================================================
# # 17. COUNTERS
# # ============================================================

# frames_processed = 0

# raw_detector_candidates = 0

# after_nms_candidates = 0

# weak_candidates = 0

# tentative_observations = 0

# detection_observations = 0

# prediction_observations = 0

# frames_with_detections = 0

# frames_with_strong_detections = 0

# frames_with_predictions = 0

# frames_with_tracks = 0

# total_tracked_observations = 0


# # ============================================================
# # 18. MAIN VIDEO LOOP
# # ============================================================

# start_time = time.time()

# while frames_processed < MAX_FRAMES:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     frame_id = frames_processed

#     timestamp = (
#         frame_id / video_fps
#     )

#     # --------------------------------------------------------
#     # DETECTION
#     # --------------------------------------------------------

#     detections = detect_snakes(frame)

#     raw_detector_candidates += len(
#         detections
#     )

#     if detections:
#         frames_with_detections += 1

#     # --------------------------------------------------------
#     # NMS
#     # --------------------------------------------------------

#     detections_nms = apply_nms(
#         detections,
#         NMS_IOU_THRESHOLD
#     )

#     after_nms_candidates += len(
#         detections_nms
#     )

#     # --------------------------------------------------------
#     # Separate strong and weak candidates.
#     # --------------------------------------------------------

#     strong_detections = []

#     weak_detections = []

#     for det in detections_nms:

#         if (
#             det["confidence"]
#             >= TRACKER_STRONG_THRESHOLD
#         ):

#             strong_detections.append(det)

#         else:

#             weak_detections.append(det)

#     weak_candidates += len(
#         weak_detections
#     )

#     if strong_detections:

#         frames_with_strong_detections += 1

#     # --------------------------------------------------------
#     # SELECT ONLY ONE WEAK/TENTATIVE BOX FOR DISPLAY
#     #
#     # This prevents the giant yellow "everything" problem.
#     #
#     # We choose the highest-confidence weak candidate.
#     # --------------------------------------------------------

#     tentative_detection = None

#     if weak_detections:

#         tentative_detection = max(
#             weak_detections,
#             key=lambda d: d["confidence"]
#         )

#     # --------------------------------------------------------
#     # BYTE TRACK INPUT
#     #
#     # IMPORTANT:
#     # All NMS detections >= .03 are sent into ByteTrack.
#     #
#     # We do NOT only send strong detections.
#     # --------------------------------------------------------

#     sv_xyxy = []

#     sv_confidence = []

#     sv_class_id = []

#     for det in detections_nms:

#         sv_xyxy.append(
#             det["bbox"]
#         )

#         sv_confidence.append(
#             det["confidence"]
#         )

#         sv_class_id.append(
#             det["class_id"]
#         )

#     if sv_xyxy:

#         xyxy_array = np.asarray(
#             sv_xyxy,
#             dtype=np.float32
#         )

#         confidence_array = np.asarray(
#             sv_confidence,
#             dtype=np.float32
#         )

#         class_id_array = np.asarray(
#             sv_class_id,
#             dtype=np.int32
#         )

#         sv_detections = sv.Detections(
#             xyxy=xyxy_array,
#             confidence=confidence_array,
#             class_id=class_id_array
#         )

#     else:

#         sv_detections = sv.Detections.empty()

#     # --------------------------------------------------------
#     # UPDATE TRACKER
#     # --------------------------------------------------------

#     tracked = tracker.update_with_detections(
#         sv_detections
#     )

#     # --------------------------------------------------------
#     # ACTIVE TRACKS
#     # --------------------------------------------------------

#     active_tracks = []

#     if len(tracked) > 0:

#         for i in range(
#             len(tracked)
#         ):

#             track_id = int(
#                 tracked.tracker_id[i]
#             )

#             bbox = (
#                 tracked.xyxy[i]
#                 .astype(float)
#                 .tolist()
#             )

#             confidence = None

#             if (
#                 tracked.confidence
#                 is not None
#             ):

#                 confidence = float(
#                     tracked.confidence[i]
#                 )

#             active_tracks.append({
#                 "track_id": track_id,
#                 "bbox": bbox,
#                 "confidence": confidence,
#                 "observation_count":
#                     track_observation_counts.get(
#                         track_id,
#                         0
#                     )
#             })

#     # --------------------------------------------------------
#     # DUPLICATE RESOLUTION
#     # --------------------------------------------------------

#     frame_suppressed = resolver.resolve(
#         active_tracks,
#         video_width,
#         video_height
#     )

#     suppressed_ids.update(
#         frame_suppressed
#     )

#     # --------------------------------------------------------
#     # DRAW TENTATIVE BOX
#     #
#     # Only draw if there is no confirmed track covering
#     # essentially the same location.
#     # --------------------------------------------------------

#     tentative_drawn = False

#     if (
#         SHOW_TENTATIVE_BOX
#         and tentative_detection is not None
#     ):

#         tentative_box = (
#             tentative_detection["bbox"]
#         )

#         overlap_with_confirmed = False

#         for active in active_tracks:

#             tid = active["track_id"]

#             if tid in frame_suppressed:
#                 continue

#             iou = bbox_iou(
#                 tentative_box,
#                 active["bbox"]
#             )

#             if iou >= 0.50:

#                 overlap_with_confirmed = True
#                 break

#         if not overlap_with_confirmed:

#             x1, y1, x2, y2 = map(
#                 int,
#                 tentative_box
#             )

#             cv2.rectangle(
#                 frame,
#                 (x1, y1),
#                 (x2, y2),
#                 (0, 255, 255),
#                 2
#             )

#             label = (
#                 f"Tentative "
#                 f"{tentative_detection['confidence']:.3f}"
#             )

#             cv2.putText(
#                 frame,
#                 label,
#                 (x1, max(25, y1 - 8)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.65,
#                 (0, 255, 255),
#                 2,
#                 cv2.LINE_AA
#             )

#             tentative_observations += 1

#             tentative_record = {
#                 "frame_id": frame_id,
#                 "timestamp": timestamp,
#                 "track_id": None,
#                 "bbox": [
#                     float(v)
#                     for v in tentative_box
#                 ],
#                 "center": [
#                     float(v)
#                     for v in bbox_center(
#                         tentative_box
#                     )
#                 ],
#                 "confidence":
#                     float(
#                         tentative_detection[
#                             "confidence"
#                         ]
#                     ),
#                 "class_id": 0,
#                 "class_name": "snake",
#                 "observation_type":
#                     "tentative",
#                 "track_state":
#                     "tentative"
#             }

#             all_records.append(
#                 tentative_record
#             )

#             tentative_drawn = True

#     # --------------------------------------------------------
#     # DRAW CONFIRMED TRACKS
#     # --------------------------------------------------------

#     confirmed_this_frame = 0

#     for active in active_tracks:

#         track_id = active["track_id"]

#         if track_id in frame_suppressed:
#             continue

#         bbox = active["bbox"]

#         confidence = active["confidence"]

#         # Track observation count
#         track_observation_counts[
#             track_id
#         ] = (
#             track_observation_counts.get(
#                 track_id,
#                 0
#             ) + 1
#         )

#         unique_track_ids.add(
#             track_id
#         )

#         track_last_seen[
#             track_id
#         ] = frame_id

#         if track_id not in track_first_seen:

#             track_first_seen[
#                 track_id
#             ] = frame_id

#         track_last_bbox[
#             track_id
#         ] = bbox

#         track_last_confidence[
#             track_id
#         ] = confidence

#         track_states[
#             track_id
#         ] = "tracked"

#         # ----------------------------------------------------
#         # DRAW GREEN CONFIRMED BOX
#         # ----------------------------------------------------

#         x1, y1, x2, y2 = map(
#             int,
#             bbox
#         )

#         cv2.rectangle(
#             frame,
#             (x1, y1),
#             (x2, y2),
#             (0, 255, 0),
#             3
#         )

#         label = (
#             f"Snake | ID {track_id}"
#         )

#         if confidence is not None:

#             label += (
#                 f" | {confidence:.3f}"
#             )

#         cv2.putText(
#             frame,
#             label,
#             (x1, max(30, y1 - 10)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.70,
#             (0, 255, 0),
#             2,
#             cv2.LINE_AA
#         )

#         # ----------------------------------------------------
#         # DETECTION RECORD
#         # ----------------------------------------------------

#         detection_observations += 1

#         total_tracked_observations += 1

#         center = bbox_center(
#             bbox
#         )

#         record = {
#             "frame_id": frame_id,
#             "timestamp": timestamp,
#             "track_id": track_id,
#             "bbox": [
#                 float(v)
#                 for v in bbox
#             ],
#             "center": [
#                 float(center[0]),
#                 float(center[1])
#             ],
#             "confidence":
#                 None
#                 if confidence is None
#                 else float(confidence),
#             "class_id": 0,
#             "class_name": "snake",
#             "observation_type":
#                 "detection",
#             "detection_strength":
#                 (
#                     "strong"
#                     if (
#                         confidence is not None
#                         and
#                         confidence
#                         >= TRACKER_STRONG_THRESHOLD
#                     )
#                     else "weak"
#                 ),
#             "track_state":
#                 "tracked"
#         }

#         all_records.append(
#             record
#         )

#         # ----------------------------------------------------
#         # TRAIL
#         #
#         # Only real detection positions are used.
#         # This avoids artificial trail movement from Kalman
#         # predictions.
#         # ----------------------------------------------------

#         if track_id not in trail_history:

#             trail_history[
#                 track_id
#             ] = []

#         trail_history[
#             track_id
#         ].append(
#             (
#                 int(center[0]),
#                 int(center[1])
#             )
#         )

#         if len(
#             trail_history[track_id]
#         ) > TRAIL_LENGTH:

#             trail_history[
#                 track_id
#             ] = trail_history[
#                 track_id
#             ][-TRAIL_LENGTH:]

#         confirmed_this_frame += 1

#     # --------------------------------------------------------
#     # DRAW TRAILS
#     # --------------------------------------------------------

#     for track_id, points in (
#         trail_history.items()
#     ):

#         if track_id in frame_suppressed:
#             continue

#         if len(points) < 2:
#             continue

#         for p1, p2 in zip(
#             points[:-1],
#             points[1:]
#         ):

#             cv2.line(
#                 frame,
#                 p1,
#                 p2,
#                 (255, 0, 0),
#                 2
#             )

#     # --------------------------------------------------------
#     # LOST TRACK PREDICTIONS
#     # --------------------------------------------------------

#     current_prediction_ids = set()

#     if hasattr(
#         tracker,
#         "lost_tracks"
#     ):

#         for lost_track in tracker.lost_tracks:

#             try:

#                 track_id = int(
#                     lost_track.external_track_id
#                 )

#             except Exception:

#                 continue

#             if track_id < 0:
#                 continue

#             if track_id in suppressed_ids:
#                 continue

#             # Only retain short prediction history.
#             last_seen = track_last_seen.get(
#                 track_id,
#                 frame_id
#             )

#             missed = (
#                 frame_id - last_seen
#             )

#             if (
#                 missed <= 0
#                 or
#                 missed > TRACKER_LOST_TRACK_BUFFER
#             ):
#                 continue

#             try:

#                 predicted_tlwh = (
#                     lost_track.tlwh
#                 )

#                 px, py, pw, ph = (
#                     map(
#                         float,
#                         predicted_tlwh
#                     )
#                 )

#                 predicted_bbox = [
#                     px,
#                     py,
#                     px + pw,
#                     py + ph
#                 ]

#             except Exception:

#                 continue

#             current_prediction_ids.add(
#                 track_id
#             )

#             # ------------------------------------------------
#             # DRAW ORANGE PREDICTION
#             # ------------------------------------------------

#             x1, y1, x2, y2 = map(
#                 int,
#                 predicted_bbox
#             )

#             cv2.rectangle(
#                 frame,
#                 (x1, y1),
#                 (x2, y2),
#                 (0, 165, 255),
#                 2
#             )

#             cv2.putText(
#                 frame,
#                 (
#                     f"Predicted | ID {track_id}"
#                     f" | miss {missed}"
#                 ),
#                 (x1, max(25, y1 - 8)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.60,
#                 (0, 165, 255),
#                 2,
#                 cv2.LINE_AA
#             )

#             center = bbox_center(
#                 predicted_bbox
#             )

#             prediction_observations += 1

#             total_tracked_observations += 1

#             prediction_record = {
#                 "frame_id": frame_id,
#                 "timestamp": timestamp,
#                 "track_id": track_id,
#                 "bbox": [
#                     float(v)
#                     for v in predicted_bbox
#                 ],
#                 "center": [
#                     float(center[0]),
#                     float(center[1])
#                 ],
#                 "confidence": None,
#                 "class_id": 0,
#                 "class_name": "snake",
#                 "observation_type":
#                     "prediction",
#                 "detection_strength":
#                     None,
#                 "track_state":
#                     "lost"
#             }

#             all_records.append(
#                 prediction_record
#             )

#     if current_prediction_ids:

#         frames_with_predictions += 1

#     # --------------------------------------------------------
#     # FRAME STATUS
#     # --------------------------------------------------------

#     if confirmed_this_frame > 0:

#         frames_with_tracks += 1

#     # --------------------------------------------------------
#     # STATUS TEXT
#     # --------------------------------------------------------

#     cv2.putText(
#         frame,
#         (
#             f"Frame {frame_id} | "
#             f"{timestamp:.2f}s"
#         ),
#         (20, 35),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.75,
#         (255, 255, 255),
#         2,
#         cv2.LINE_AA
#     )

#     cv2.putText(
#         frame,
#         (
#             f"Confirmed: "
#             f"{confirmed_this_frame}"
#         ),
#         (20, 68),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.65,
#         (255, 255, 255),
#         2,
#         cv2.LINE_AA
#     )

#     cv2.putText(
#         frame,
#         (
#             f"Candidates: "
#             f"{len(detections_nms)}"
#         ),
#         (20, 98),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.65,
#         (255, 255, 255),
#         2,
#         cv2.LINE_AA
#     )

#     writer.write(frame)

#     frames_processed += 1

#     # --------------------------------------------------------
#     # PROGRESS
#     # --------------------------------------------------------

#     if (
#         frames_processed % 50 == 0
#         or
#         frames_processed == 1
#     ):

#         elapsed = (
#             time.time() - start_time
#         )

#         effective_fps = (
#             frames_processed / elapsed
#             if elapsed > 0
#             else 0
#         )

#         print(
#             f"Processed "
#             f"{frames_processed}/{MAX_FRAMES} "
#             f"| FPS {effective_fps:.2f}"
#         )


# # ============================================================
# # 19. CLEANUP
# # ============================================================

# cap.release()
# writer.release()


# # ============================================================
# # 20. FINAL STATISTICS
# # ============================================================

# elapsed = (
#     time.time() - start_time
# )

# effective_fps = (
#     frames_processed / elapsed
#     if elapsed > 0
#     else 0
# )

# print("\n" + "=" * 70)
# print("STEP 5.3 REVISED COMPLETE")
# print("=" * 70)

# print(
#     f"Frames processed           : "
#     f"{frames_processed}"
# )

# print(
#     f"Raw detector candidates   : "
#     f"{raw_detector_candidates}"
# )

# print(
#     f"After NMS                 : "
#     f"{after_nms_candidates}"
# )

# print(
#     f"Weak/tentative candidates : "
#     f"{weak_candidates}"
# )

# print(
#     f"Tentative observations    : "
#     f"{tentative_observations}"
# )

# print(
#     f"Detection observations    : "
#     f"{detection_observations}"
# )

# print(
#     f"Prediction observations   : "
#     f"{prediction_observations}"
# )

# print(
#     f"Frames with detections    : "
#     f"{frames_with_detections}"
# )

# print(
#     f"Frames with strong detect.: "
#     f"{frames_with_strong_detections}"
# )

# print(
#     f"Frames with predictions   : "
#     f"{frames_with_predictions}"
# )

# print(
#     f"Frames with tracks        : "
#     f"{frames_with_tracks}"
# )

# print(
#     f"Total tracked observations: "
#     f"{total_tracked_observations}"
# )

# print(
#     f"Unique Track IDs          : "
#     f"{len(unique_track_ids)}"
# )

# print(
#     f"Track IDs                 : "
#     f"{sorted(unique_track_ids)}"
# )

# print(
#     f"Suppressed duplicate IDs  : "
#     f"{sorted(suppressed_ids)}"
# )

# print(
#     f"Effective processing FPS  : "
#     f"{effective_fps:.2f}"
# )

# print("\nOUTPUTS")

# print(
#     f"Video : {OUTPUT_VIDEO_PATH}"
# )

# print(
#     f"JSON  : {OUTPUT_JSON_PATH}"
# )


# # ============================================================
# # 21. SAVE JSON
# # ============================================================

# tracks_dict = {}

# for record in all_records:

#     track_id = record["track_id"]

#     if track_id is None:
#         continue

#     track_id = int(track_id)

#     if track_id not in tracks_dict:

#         tracks_dict[track_id] = []

#     tracks_dict[track_id].append(
#         record
#     )


# # Sort each track chronologically
# for track_id in tracks_dict:

#     tracks_dict[track_id].sort(
#         key=lambda r: r["frame_id"]
#     )


# json_output = {
#     "video": str(VIDEO_PATH),

#     "fps": float(video_fps),

#     "frames_processed":
#         int(frames_processed),

#     "total_detector_candidates":
#         int(raw_detector_candidates),

#     "after_nms_candidates":
#         int(after_nms_candidates),

#     "weak_candidates":
#         int(weak_candidates),

#     "tentative_observations":
#         int(tentative_observations),

#     "total_detection_observations":
#         int(detection_observations),

#     "total_prediction_observations":
#         int(prediction_observations),

#     "frames_with_detections":
#         int(frames_with_detections),

#     "frames_with_strong_detections":
#         int(frames_with_strong_detections),

#     "frames_with_predictions":
#         int(frames_with_predictions),

#     "frames_with_tracks":
#         int(frames_with_tracks),

#     "total_tracked_observations":
#         int(total_tracked_observations),

#     "unique_track_ids":
#         sorted(
#             int(x)
#             for x in unique_track_ids
#         ),

#     "track_count":
#         int(len(unique_track_ids)),

#     "suppressed_duplicate_ids":
#         sorted(
#             int(x)
#             for x in suppressed_ids
#         ),

#     "configuration": {
#         "tracker_input_threshold":
#             TRACKER_INPUT_THRESHOLD,

#         "tracker_strong_threshold":
#             TRACKER_STRONG_THRESHOLD,

#         "tracker_activation_threshold":
#             TRACKER_ACTIVATION_THRESHOLD,

#         "tracker_lost_track_buffer":
#             TRACKER_LOST_TRACK_BUFFER,

#         "tracker_matching_threshold":
#             TRACKER_MATCHING_THRESHOLD,

#         "nms_iou_threshold":
#             NMS_IOU_THRESHOLD,

#         "duplicate_iou_threshold":
#             DUPLICATE_IOU_THRESHOLD,

#         "duplicate_center_distance_ratio":
#             DUPLICATE_CENTER_DISTANCE_RATIO
#     },

#     "tracks": [
#         {
#             "track_id": int(track_id),
#             "observation_count":
#                 len(records),
#             "history": records
#         }
#         for track_id, records
#         in sorted(
#             tracks_dict.items()
#         )
#     ]
# }


# with open(
#     OUTPUT_JSON_PATH,
#     "w"
# ) as f:

#     json.dump(
#         json_output,
#         f,
#         indent=2
#     )


# print("\nSTEP 5.3 COMPLETE.")
# print("=" * 70)




















# """
# =====================================================================
# STEP 5.3 — BYTETRACK + TENTATIVE TRACK CONFIRMATION
# =====================================================================

# Purpose:
#     RT-DETRv2 detections -> tentative tracking -> ByteTrack IDs.

# Detection thresholds:

#     < 0.03
#         ignored by Step 5.2

#     0.03 <= confidence < 0.055
#         weak / tentative candidate

#     >= 0.055
#         strong detector evidence

# Tentative tracking:

#     Weak candidates are NOT immediately trusted.

#     A weak candidate must remain spatially consistent across
#     multiple consecutive frames before it is promoted to a
#     confirmed track.

# Confirmed tracks are then handled by ByteTrack.

# During detector gaps:
#     ByteTrack Kalman-predicted states are preserved.

# Observation types:
#     detection
#     prediction
#     tentative
# """

# from pathlib import Path
# import json
# import sys
# import importlib.util

# import cv2
# import numpy as np
# import supervision as sv

# from supervision.tracker.byte_tracker.core import (
#     STrack,
#     TrackState,
# )

# from supervision.tracker.byte_tracker import matching


# # ============================================================
# # FIX 1 — ALLOW TRACK ACTIVATION ON ANY FRAME
# # ============================================================

# def patched_activate(
#     self,
#     kalman_filter,
#     frame_id,
# ):
#     self.kalman_filter = kalman_filter

#     self.internal_track_id = (
#         self.internal_id_counter.new_id()
#     )

#     self.mean, self.covariance = (
#         self.kalman_filter.initiate(
#             self.tlwh_to_xyah(self._tlwh)
#         )
#     )

#     self.tracklet_len = 1

#     self.state = TrackState.Tracked

#     if (
#         self.tracklet_len
#         >= self.minimum_consecutive_frames
#     ):
#         self.is_activated = True

#         self.external_track_id = (
#             self.external_id_counter.new_id()
#         )

#     self.frame_id = frame_id
#     self.start_frame = frame_id


# STrack.activate = patched_activate


# # ============================================================
# # FIX 2 — IOU-ONLY ASSOCIATION
# # ============================================================

# def patched_fuse_score(
#     cost_matrix,
#     stracks,
# ):
#     return cost_matrix


# matching.fuse_score = patched_fuse_score


# print(
#     "ByteTrack compatibility patches applied:"
# )

# print(
#     "  - Track activation fix : ENABLED"
# )

# print(
#     "  - IoU-only association : ENABLED"
# )

# print(
#     "  - Lost-track prediction: ENABLED"
# )

# print(
#     "  - Tentative confirmation: ENABLED"
# )


# # ============================================================
# # PROJECT PATHS
# # ============================================================

# PROJECT_DIR = Path(
#     "/content/drive/MyDrive/snake_detection/Snake_detection"
# )

# VIDEO_PATH = (
#     PROJECT_DIR
#     / "snake_videos"
#     / "10231672-hd_1080_1920_30fps.mp4"
# )

# OUTPUT_VIDEO_PATH = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
#     / "tracked_video_5_3.mp4"
# )

# TRACK_DATA_PATH = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
#     / "track_data_5_3.json"
# )


# # ============================================================
# # LOAD STEP 5.2 ADAPTER
# # ============================================================

# ADAPTER_PATH = (
#     PROJECT_DIR
#     / "step5_2_rtdetr_bytetrack_adapter.py"
# )

# if not ADAPTER_PATH.exists():
#     raise FileNotFoundError(
#         f"Step 5.2 adapter not found:\n{ADAPTER_PATH}"
#     )


# spec = importlib.util.spec_from_file_location(
#     "step5_2_rtdetr_bytetrack_adapter",
#     ADAPTER_PATH,
# )

# adapter = importlib.util.module_from_spec(spec)

# sys.modules[
#     "step5_2_rtdetr_bytetrack_adapter"
# ] = adapter

# spec.loader.exec_module(adapter)

# load_detector = adapter.load_detector
# detect_snakes = adapter.detect_snakes


# # ============================================================
# # LOAD STEP 5.1 VIDEO READER
# # ============================================================

# VIDEO_READER_PATH = (
#     PROJECT_DIR
#     / "step5_1_video_input.py"
# )

# if not VIDEO_READER_PATH.exists():
#     raise FileNotFoundError(
#         f"Step 5.1 video reader not found:\n"
#         f"{VIDEO_READER_PATH}"
#     )


# spec_reader = (
#     importlib.util.spec_from_file_location(
#         "step5_1_video_input",
#         VIDEO_READER_PATH,
#     )
# )

# video_module = (
#     importlib.util.module_from_spec(
#         spec_reader
#     )
# )

# sys.modules[
#     "step5_1_video_input"
# ] = video_module

# spec_reader.loader.exec_module(
#     video_module
# )

# VideoFrameReader = (
#     video_module.VideoFrameReader
# )


# # ============================================================
# # THRESHOLDS
# # ============================================================

# # Matches Step 5.2
# TRACKER_INPUT_THRESHOLD = 0.03

# # Frozen strong detector threshold
# TRACKER_STRONG_THRESHOLD = 0.055

# # ------------------------------------------------------------
# # Tentative confirmation
# # ------------------------------------------------------------

# # Number of consecutive supporting frames required
# # before a weak candidate becomes confirmed.
# TENTATIVE_CONFIRMATION_FRAMES = 3

# # Minimum IoU between consecutive candidate boxes.
# TENTATIVE_MIN_IOU = 0.30

# # Maximum number of frames a tentative candidate can
# # survive without receiving a matching candidate.
# TENTATIVE_MAX_MISSED_FRAMES = 5

# # Maximum center movement between consecutive frames.
# # Expressed as a fraction of the candidate box diagonal.
# TENTATIVE_MAX_CENTER_DISTANCE_RATIO = 1.5


# # ============================================================
# # BYTE TRACK SETTINGS
# # ============================================================

# TRACKER_LOST_TRACK_BUFFER = 30

# TRACKER_MATCHING_THRESHOLD = 0.80

# MAX_FRAMES = 400


# # ============================================================
# # IOU
# # ============================================================

# def calculate_iou(
#     box_a,
#     box_b,
# ):
#     """
#     Calculate IoU between two xyxy bounding boxes.
#     """

#     ax1, ay1, ax2, ay2 = box_a
#     bx1, by1, bx2, by2 = box_b

#     inter_x1 = max(
#         ax1,
#         bx1,
#     )

#     inter_y1 = max(
#         ay1,
#         by1,
#     )

#     inter_x2 = min(
#         ax2,
#         bx2,
#     )

#     inter_y2 = min(
#         ay2,
#         by2,
#     )

#     inter_w = max(
#         0.0,
#         inter_x2 - inter_x1,
#     )

#     inter_h = max(
#         0.0,
#         inter_y2 - inter_y1,
#     )

#     intersection = (
#         inter_w * inter_h
#     )

#     area_a = (
#         max(0.0, ax2 - ax1)
#         *
#         max(0.0, ay2 - ay1)
#     )

#     area_b = (
#         max(0.0, bx2 - bx1)
#         *
#         max(0.0, by2 - by1)
#     )

#     union = (
#         area_a
#         +
#         area_b
#         -
#         intersection
#     )

#     if union <= 0:
#         return 0.0

#     return intersection / union


# # ============================================================
# # BOX CENTER
# # ============================================================

# def box_center(box):

#     x1, y1, x2, y2 = box

#     return (
#         (x1 + x2) / 2.0,
#         (y1 + y2) / 2.0,
#     )


# # ============================================================
# # CENTER DISTANCE
# # ============================================================

# def center_distance(
#     box_a,
#     box_b,
# ):
#     """
#     Euclidean distance between box centers.
#     """

#     ax, ay = box_center(box_a)
#     bx, by = box_center(box_b)

#     return float(
#         np.sqrt(
#             (ax - bx) ** 2
#             +
#             (ay - by) ** 2
#         )
#     )


# # ============================================================
# # BOX DIAGONAL
# # ============================================================

# def box_diagonal(box):

#     x1, y1, x2, y2 = box

#     width = max(
#         1.0,
#         x2 - x1,
#     )

#     height = max(
#         1.0,
#         y2 - y1,
#     )

#     return float(
#         np.sqrt(
#             width ** 2
#             +
#             height ** 2
#         )
#     )


# # ============================================================
# # TENTATIVE CANDIDATE
# # ============================================================

# class TentativeCandidate:

#     def __init__(
#         self,
#         candidate_id,
#         detection,
#         frame_id,
#         timestamp,
#     ):

#         self.candidate_id = (
#             candidate_id
#         )

#         self.first_frame = (
#             frame_id
#         )

#         self.last_frame = (
#             frame_id
#         )

#         self.last_timestamp = (
#             timestamp
#         )

#         self.bbox = list(
#             detection["bbox"]
#         )

#         self.confidence = float(
#             detection["confidence"]
#         )

#         self.hit_count = 1

#         self.missed_frames = 0

#         self.confirmed = False

#         self.history = [
#             {
#                 "frame_id": frame_id,
#                 "timestamp": timestamp,
#                 "bbox": list(
#                     detection["bbox"]
#                 ),
#                 "confidence": float(
#                     detection["confidence"]
#                 ),
#             }
#         ]


# # ============================================================
# # TENTATIVE MATCH
# # ============================================================

# def candidate_matches(
#     candidate,
#     detection,
# ):
#     """
#     Determine whether a new weak detection is
#     spatially consistent with a tentative candidate.
#     """

#     new_box = detection["bbox"]

#     iou = calculate_iou(
#         candidate.bbox,
#         new_box,
#     )

#     distance = center_distance(
#         candidate.bbox,
#         new_box,
#     )

#     diagonal = box_diagonal(
#         candidate.bbox
#     )

#     max_distance = (
#         diagonal
#         *
#         TENTATIVE_MAX_CENTER_DISTANCE_RATIO
#     )

#     # Primary requirement:
#     # boxes must overlap reasonably.
#     #
#     # Secondary allowance:
#     # if IoU is low because the snake moves/deforms,
#     # the center still has to remain spatially close.

#     if iou >= TENTATIVE_MIN_IOU:
#         return True

#     if distance <= max_distance:
#         return True

#     return False


# # ============================================================
# # UPDATE TENTATIVE CANDIDATE
# # ============================================================

# def update_candidate(
#     candidate,
#     detection,
#     frame_id,
#     timestamp,
# ):

#     candidate.bbox = list(
#         detection["bbox"]
#     )

#     candidate.confidence = float(
#         detection["confidence"]
#     )

#     candidate.last_frame = (
#         frame_id
#     )

#     candidate.last_timestamp = (
#         timestamp
#     )

#     candidate.hit_count += 1

#     candidate.missed_frames = 0

#     candidate.history.append(
#         {
#             "frame_id": frame_id,
#             "timestamp": timestamp,
#             "bbox": list(
#                 detection["bbox"]
#             ),
#             "confidence": float(
#                 detection["confidence"]
#             ),
#         }
#     )


# # ============================================================
# # CREATE TRACKER
# # ============================================================

# def create_tracker(fps):

#     # --------------------------------------------------------
#     # ByteTrack itself receives the weak candidates.
#     #
#     # New-track threshold remains 0.055.
#     #
#     # Tentative tracks are handled separately above this layer.
#     # --------------------------------------------------------

#     tracker = sv.ByteTrack(
#         track_activation_threshold=(
#             TRACKER_INPUT_THRESHOLD
#         ),

#         lost_track_buffer=(
#             TRACKER_LOST_TRACK_BUFFER
#         ),

#         minimum_matching_threshold=(
#             TRACKER_MATCHING_THRESHOLD
#         ),

#         frame_rate=fps,
#     )

#     tracker.det_thresh = (
#         TRACKER_STRONG_THRESHOLD
#     )

#     return tracker


# # ============================================================
# # CONVERT DETECTIONS
# # ============================================================

# def detections_to_supervision(
#     detections
# ):

#     if not detections:
#         return sv.Detections.empty()

#     xyxy = np.array(
#         [
#             d["bbox"]
#             for d in detections
#         ],
#         dtype=np.float32,
#     )

#     confidence = np.array(
#         [
#             d["confidence"]
#             for d in detections
#         ],
#         dtype=np.float32,
#     )

#     class_id = np.array(
#         [
#             d["class_id"]
#             for d in detections
#         ],
#         dtype=np.int32,
#     )

#     return sv.Detections(
#         xyxy=xyxy,
#         confidence=confidence,
#         class_id=class_id,
#     )


# # ============================================================
# # TRACK ID
# # ============================================================

# def get_track_id(track):

#     external_id = getattr(
#         track,
#         "external_track_id",
#         None,
#     )

#     if external_id is not None:
#         return int(
#             external_id
#         )

#     internal_id = getattr(
#         track,
#         "internal_track_id",
#         None,
#     )

#     if internal_id is not None:
#         return int(
#             internal_id
#         )

#     return None


# # ============================================================
# # TRACK BBOX
# # ============================================================

# def get_track_bbox(track):

#     tlbr = getattr(
#         track,
#         "tlbr",
#         None,
#     )

#     if tlbr is not None:

#         try:

#             values = np.asarray(
#                 tlbr,
#                 dtype=np.float32,
#             ).reshape(-1)

#             if len(values) == 4:

#                 return [
#                     float(values[0]),
#                     float(values[1]),
#                     float(values[2]),
#                     float(values[3]),
#                 ]

#         except Exception:
#             pass

#     tlwh = getattr(
#         track,
#         "tlwh",
#         None,
#     )

#     if tlwh is not None:

#         try:

#             values = np.asarray(
#                 tlwh,
#                 dtype=np.float32,
#             ).reshape(-1)

#             if len(values) == 4:

#                 x, y, w, h = values

#                 return [
#                     float(x),
#                     float(y),
#                     float(x + w),
#                     float(y + h),
#                 ]

#         except Exception:
#             pass

#     return None


# # ============================================================
# # VALID TRACK
# # ============================================================

# def valid_track(track):

#     track_id = get_track_id(
#         track
#     )

#     if track_id is None:
#         return False

#     bbox = get_track_bbox(
#         track
#     )

#     if bbox is None:
#         return False

#     x1, y1, x2, y2 = bbox

#     if x2 <= x1:
#         return False

#     if y2 <= y1:
#         return False

#     return True


# # ============================================================
# # DRAW DETECTIONS
# # ============================================================

# def draw_detection_tracks(
#     frame,
#     tracked_detections,
# ):

#     output = frame.copy()

#     if len(
#         tracked_detections
#     ) == 0:
#         return output

#     boxes = (
#         tracked_detections.xyxy
#     )

#     confidences = (
#         tracked_detections.confidence
#     )

#     track_ids = (
#         tracked_detections.tracker_id
#     )

#     for i, box in enumerate(
#         boxes
#     ):

#         x1, y1, x2, y2 = map(
#             int,
#             box,
#         )

#         confidence = float(
#             confidences[i]
#         )

#         track_id = int(
#             track_ids[i]
#         )

#         if (
#             confidence
#             >= TRACKER_STRONG_THRESHOLD
#         ):

#             color = (
#                 0,
#                 255,
#                 0,
#             )

#             label_type = "STRONG"

#         else:

#             color = (
#                 255,
#                 255,
#                 0,
#             )

#             label_type = "WEAK"

#         cv2.rectangle(
#             output,
#             (x1, y1),
#             (x2, y2),
#             color,
#             2,
#         )

#         label = (
#             f"Snake | ID: {track_id} | "
#             f"{confidence:.3f} | "
#             f"{label_type}"
#         )

#         cv2.putText(
#             output,
#             label,
#             (
#                 x1,
#                 max(
#                     25,
#                     y1 - 10
#                 ),
#             ),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             color,
#             2,
#         )

#     return output


# # ============================================================
# # DRAW PREDICTIONS
# # ============================================================

# def draw_prediction_tracks(
#     frame,
#     prediction_records,
# ):

#     output = frame

#     for record in (
#         prediction_records
#     ):

#         x1, y1, x2, y2 = map(
#             int,
#             record["bbox"],
#         )

#         track_id = int(
#             record["track_id"]
#         )

#         color = (
#             0,
#             165,
#             255,
#         )

#         cv2.rectangle(
#             output,
#             (x1, y1),
#             (x2, y2),
#             color,
#             2,
#         )

#         label = (
#             f"Snake | ID: "
#             f"{track_id} | PRED"
#         )

#         cv2.putText(
#             output,
#             label,
#             (
#                 x1,
#                 max(
#                     25,
#                     y1 - 10
#                 ),
#             ),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             color,
#             2,
#         )

#     return output


# # ============================================================
# # DRAW TENTATIVE CANDIDATES
# # ============================================================

# def draw_tentative_candidates(
#     frame,
#     tentative_candidates,
# ):

#     output = frame

#     for candidate in (
#         tentative_candidates
#     ):

#         x1, y1, x2, y2 = map(
#             int,
#             candidate.bbox,
#         )

#         # Magenta = tentative
#         color = (
#             255,
#             0,
#             255,
#         )

#         cv2.rectangle(
#             output,
#             (x1, y1),
#             (x2, y2),
#             color,
#             2,
#         )

#         label = (
#             f"TENTATIVE "
#             f"{candidate.hit_count}/"
#             f"{TENTATIVE_CONFIRMATION_FRAMES}"
#             f" | {candidate.confidence:.3f}"
#         )

#         cv2.putText(
#             output,
#             label,
#             (
#                 x1,
#                 max(
#                     25,
#                     y1 - 10
#                 ),
#             ),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.55,
#             color,
#             2,
#         )

#     return output


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     print(
#         "\n" + "=" * 70
#     )

#     print(
#         "STEP 5.3 — BYTETRACK + "
#         "TENTATIVE TRACK CONFIRMATION"
#     )

#     print(
#         "=" * 70
#     )

#     # --------------------------------------------------------
#     # VIDEO
#     # --------------------------------------------------------

#     if not VIDEO_PATH.exists():

#         raise FileNotFoundError(
#             f"Video not found:\n{VIDEO_PATH}"
#         )

#     print(
#         "\nVideo:"
#     )

#     print(
#         VIDEO_PATH
#     )

#     # --------------------------------------------------------
#     # DETECTOR
#     # --------------------------------------------------------

#     print(
#         "\nLoading RT-DETRv2..."
#     )

#     processor, model = (
#         load_detector()
#     )

#     print(
#         "Detector loaded successfully."
#     )

#     # --------------------------------------------------------
#     # VIDEO READER
#     # --------------------------------------------------------

#     reader = VideoFrameReader(
#         video_path=VIDEO_PATH,
#         frame_skip=1,
#     )

#     video_info = (
#         reader.get_video_info()
#     )

#     fps = video_info["fps"]

#     print(
#         "\nVIDEO"
#     )

#     print(
#         "-" * 70
#     )

#     print(
#         f"Resolution   : "
#         f"{video_info['width']} x "
#         f"{video_info['height']}"
#     )

#     print(
#         f"FPS          : "
#         f"{fps:.2f}"
#     )

#     print(
#         f"Total frames : "
#         f"{video_info['total_frames']}"
#     )

#     print(
#         f"Test limit   : "
#         f"{MAX_FRAMES}"
#     )

#     # --------------------------------------------------------
#     # TRACKER
#     # --------------------------------------------------------

#     tracker = create_tracker(
#         fps
#     )

#     print(
#         "\nByteTrack initialized."
#     )

#     print(
#         f"Tracker input threshold : "
#         f"{TRACKER_INPUT_THRESHOLD}"
#     )

#     print(
#         f"Strong threshold        : "
#         f"{TRACKER_STRONG_THRESHOLD}"
#     )

#     print(
#         f"Tentative confirmation  : "
#         f"{TENTATIVE_CONFIRMATION_FRAMES}"
#         f" frames"
#     )

#     print(
#         f"Tentative minimum IoU   : "
#         f"{TENTATIVE_MIN_IOU}"
#     )

#     print(
#         f"Tentative max missed    : "
#         f"{TENTATIVE_MAX_MISSED_FRAMES}"
#     )

#     print(
#         f"Lost track buffer       : "
#         f"{TRACKER_LOST_TRACK_BUFFER}"
#     )

#     print(
#         f"Matching threshold      : "
#         f"{TRACKER_MATCHING_THRESHOLD}"
#     )

#     print(
#         "Association mode        : IoU-only"
#     )

#     print(
#         "Prediction recording    : ENABLED"
#     )

#     # --------------------------------------------------------
#     # OUTPUT
#     # --------------------------------------------------------

#     OUTPUT_VIDEO_PATH.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     TRACK_DATA_PATH.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     fourcc = cv2.VideoWriter_fourcc(
#         *"mp4v"
#     )

#     writer = cv2.VideoWriter(
#         str(OUTPUT_VIDEO_PATH),
#         fourcc,
#         fps,
#         (
#             video_info["width"],
#             video_info["height"],
#         ),
#     )

#     if not writer.isOpened():

#         reader.release()

#         raise RuntimeError(
#             "Could not create output video."
#         )

#     # --------------------------------------------------------
#     # DATA
#     # --------------------------------------------------------

#     track_data = []

#     frames_processed = 0

#     total_detections = 0

#     strong_detection_count = 0

#     weak_detection_count = 0

#     frames_with_detections = 0

#     frames_with_strong_detections = 0

#     frames_with_tracks = 0

#     total_tracked_observations = 0

#     total_detection_observations = 0

#     total_prediction_observations = 0

#     total_tentative_observations = 0

#     frames_with_predictions = 0

#     frames_with_tentative = 0

#     confirmed_track_ids = set()

#     unique_track_ids = set()

#     # --------------------------------------------------------
#     # Tentative candidates
#     # --------------------------------------------------------

#     tentative_candidates = []

#     next_tentative_id = 1

#     # ========================================================
#     # PROCESS VIDEO
#     # ========================================================

#     print(
#         "\nPROCESSING VIDEO"
#     )

#     print(
#         "-" * 70
#     )

#     try:

#         for item in reader.frames():

#             frame = item["frame"]

#             frame_id = item["frame_id"]

#             timestamp = item["timestamp"]

#             # ------------------------------------------------
#             # DETECTOR
#             # ------------------------------------------------

#             detections = detect_snakes(
#                 frame,
#                 processor,
#                 model,
#             )

#             detection_count = len(
#                 detections
#             )

#             total_detections += (
#                 detection_count
#             )

#             if detection_count > 0:

#                 frames_with_detections += 1

#             # ------------------------------------------------
#             # STRONG / WEAK
#             # ------------------------------------------------

#             strong_detections = []

#             weak_detections = []

#             for detection in detections:

#                 confidence = float(
#                     detection["confidence"]
#                 )

#                 if (
#                     confidence
#                     >= TRACKER_STRONG_THRESHOLD
#                 ):

#                     strong_detections.append(
#                         detection
#                     )

#                     strong_detection_count += 1

#                 else:

#                     weak_detections.append(
#                         detection
#                     )

#                     weak_detection_count += 1

#             if strong_detections:

#                 frames_with_strong_detections += 1

#             # =================================================
#             # UPDATE TENTATIVE CANDIDATES
#             # =================================================

#             matched_candidate_ids = set()

#             tentative_records = []

#             # -------------------------------------------------
#             # Strong detections immediately invalidate the need
#             # for tentative promotion in the same spatial region.
#             #
#             # We still let ByteTrack handle the strong detection.
#             # -------------------------------------------------

#             candidate_detections = (
#                 weak_detections
#             )

#             # -------------------------------------------------
#             # Match each weak detection to the best tentative
#             # candidate.
#             # -------------------------------------------------

#             for detection in (
#                 candidate_detections
#             ):

#                 best_candidate = None

#                 best_score = -1.0

#                 for candidate in (
#                     tentative_candidates
#                 ):

#                     if (
#                         candidate.candidate_id
#                         in matched_candidate_ids
#                     ):
#                         continue

#                     if not candidate_matches(
#                         candidate,
#                         detection,
#                     ):
#                         continue

#                     iou = calculate_iou(
#                         candidate.bbox,
#                         detection["bbox"],
#                     )

#                     if iou > best_score:

#                         best_score = iou

#                         best_candidate = (
#                             candidate
#                         )

#                 # ---------------------------------------------
#                 # Existing tentative candidate
#                 # ---------------------------------------------

#                 if best_candidate is not None:

#                     update_candidate(
#                         best_candidate,
#                         detection,
#                         frame_id,
#                         timestamp,
#                     )

#                     matched_candidate_ids.add(
#                         best_candidate.candidate_id
#                     )

#                 # ---------------------------------------------
#                 # New tentative candidate
#                 # ---------------------------------------------

#                 else:

#                     new_candidate = (
#                         TentativeCandidate(
#                             next_tentative_id,
#                             detection,
#                             frame_id,
#                             timestamp,
#                         )
#                     )

#                     tentative_candidates.append(
#                         new_candidate
#                     )

#                     next_tentative_id += 1

#             # -------------------------------------------------
#             # Handle missed tentative candidates
#             # -------------------------------------------------

#             surviving_candidates = []

#             for candidate in (
#                 tentative_candidates
#             ):

#                 if (
#                     candidate.candidate_id
#                     in matched_candidate_ids
#                 ):
#                     surviving_candidates.append(
#                         candidate
#                     )
#                     continue

#                 candidate.missed_frames += 1

#                 if (
#                     candidate.missed_frames
#                     <= TENTATIVE_MAX_MISSED_FRAMES
#                 ):

#                     surviving_candidates.append(
#                         candidate
#                     )

#             tentative_candidates = (
#                 surviving_candidates
#             )

#             # -------------------------------------------------
#             # Confirm candidates
#             # -------------------------------------------------

#             for candidate in (
#                 tentative_candidates
#             ):

#                 if (
#                     candidate.confirmed
#                 ):
#                     continue

#                 if (
#                     candidate.hit_count
#                     >= TENTATIVE_CONFIRMATION_FRAMES
#                 ):

#                     candidate.confirmed = True

#             # -------------------------------------------------
#             # Store tentative observations
#             #
#             # These are shown only while they are not confirmed.
#             # -------------------------------------------------

#             for candidate in (
#                 tentative_candidates
#             ):

#                 if candidate.confirmed:
#                     continue

#                 tentative_record = {

#                     "frame_id":
#                         frame_id,

#                     "timestamp":
#                         timestamp,

#                     "track_id":
#                         None,

#                     "candidate_id":
#                         candidate.candidate_id,

#                     "bbox":
#                         list(candidate.bbox),

#                     "center":
#                         list(
#                             box_center(
#                                 candidate.bbox
#                             )
#                         ),

#                     "confidence":
#                         float(
#                             candidate.confidence
#                         ),

#                     "class_id":
#                         0,

#                     "class_name":
#                         "snake",

#                     "observation_type":
#                         "tentative",

#                     "detection_strength":
#                         "weak",

#                     "track_state":
#                         "tentative",
#                 }

#                 tentative_records.append(
#                     tentative_record
#                 )

#                 total_tentative_observations += 1

#             if tentative_records:

#                 frames_with_tentative += 1

#             # =================================================
#             # BYTE TRACK
#             # =================================================

#             sv_detections = (
#                 detections_to_supervision(
#                     detections
#                 )
#             )

#             tracked_detections = (
#                 tracker.update_with_detections(
#                     sv_detections
#                 )
#             )

#             # =================================================
#             # DETECTION-ASSOCIATED TRACKS
#             # =================================================

#             detection_track_ids = set()

#             current_detection_records = []

#             if len(
#                 tracked_detections
#             ) > 0:

#                 boxes = (
#                     tracked_detections.xyxy
#                 )

#                 confidences = (
#                     tracked_detections.confidence
#                 )

#                 track_ids = (
#                     tracked_detections.tracker_id
#                 )

#                 for i, box in enumerate(
#                     boxes
#                 ):

#                     x1, y1, x2, y2 = map(
#                         float,
#                         box,
#                     )

#                     confidence = float(
#                         confidences[i]
#                     )

#                     track_id = int(
#                         track_ids[i]
#                     )

#                     detection_track_ids.add(
#                         track_id
#                     )

#                     unique_track_ids.add(
#                         track_id
#                     )

#                     confirmed_track_ids.add(
#                         track_id
#                     )

#                     center_x = (
#                         x1 + x2
#                     ) / 2.0

#                     center_y = (
#                         y1 + y2
#                     ) / 2.0

#                     strength = (
#                         "strong"
#                         if confidence
#                         >= TRACKER_STRONG_THRESHOLD
#                         else "weak"
#                     )

#                     record = {

#                         "frame_id":
#                             frame_id,

#                         "timestamp":
#                             timestamp,

#                         "track_id":
#                             track_id,

#                         "candidate_id":
#                             None,

#                         "bbox": [
#                             x1,
#                             y1,
#                             x2,
#                             y2,
#                         ],

#                         "center": [
#                             center_x,
#                             center_y,
#                         ],

#                         "confidence":
#                             confidence,

#                         "class_id":
#                             0,

#                         "class_name":
#                             "snake",

#                         "observation_type":
#                             "detection",

#                         "detection_strength":
#                             strength,

#                         "track_state":
#                             "tracked",
#                     }

#                     current_detection_records.append(
#                         record
#                     )

#                     track_data.append(
#                         record
#                     )

#                     total_tracked_observations += 1

#                     total_detection_observations += 1

#             # =================================================
#             # KALMAN PREDICTIONS
#             # =================================================

#             prediction_records = []

#             lost_tracks = getattr(
#                 tracker,
#                 "lost_tracks",
#                 [],
#             )

#             for track in lost_tracks:

#                 if not valid_track(track):
#                     continue

#                 track_id = get_track_id(
#                     track
#                 )

#                 if track_id in (
#                     detection_track_ids
#                 ):
#                     continue

#                 bbox = get_track_bbox(
#                     track
#                 )

#                 if bbox is None:
#                     continue

#                 x1, y1, x2, y2 = bbox

#                 center_x = (
#                     x1 + x2
#                 ) / 2.0

#                 center_y = (
#                     y1 + y2
#                 ) / 2.0

#                 record = {

#                     "frame_id":
#                         frame_id,

#                     "timestamp":
#                         timestamp,

#                     "track_id":
#                         int(track_id),

#                     "candidate_id":
#                         None,

#                     "bbox": [
#                         float(x1),
#                         float(y1),
#                         float(x2),
#                         float(y2),
#                     ],

#                     "center": [
#                         float(center_x),
#                         float(center_y),
#                     ],

#                     "confidence":
#                         None,

#                     "class_id":
#                         0,

#                     "class_name":
#                         "snake",

#                     "observation_type":
#                         "prediction",

#                     "detection_strength":
#                         None,

#                     "track_state":
#                         "lost",
#                 }

#                 prediction_records.append(
#                     record
#                 )

#                 track_data.append(
#                     record
#                 )

#                 unique_track_ids.add(
#                     int(track_id)
#                 )

#                 total_tracked_observations += 1

#                 total_prediction_observations += 1

#             if prediction_records:

#                 frames_with_predictions += 1

#             # =================================================
#             # RECORD TENTATIVE DATA
#             # =================================================

#             for record in (
#                 tentative_records
#             ):

#                 track_data.append(
#                     record
#                 )

#             # =================================================
#             # FRAME TRACK COUNT
#             # =================================================

#             current_track_count = (
#                 len(
#                     current_detection_records
#                 )
#                 +
#                 len(
#                     prediction_records
#                 )
#             )

#             if current_track_count > 0:

#                 frames_with_tracks += 1

#             # =================================================
#             # DRAW
#             # =================================================

#             output_frame = (
#                 draw_tentative_candidates(
#                     frame,
#                     tentative_candidates,
#                 )
#             )

#             output_frame = (
#                 draw_detection_tracks(
#                     output_frame,
#                     tracked_detections,
#                 )
#             )

#             output_frame = (
#                 draw_prediction_tracks(
#                     output_frame,
#                     prediction_records,
#                 )
#             )

#             # =================================================
#             # STATUS
#             # =================================================

#             cv2.putText(
#                 output_frame,
#                 f"Frame: {frame_id}",
#                 (20, 35),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (255, 255, 255),
#                 2,
#             )

#             cv2.putText(
#                 output_frame,
#                 f"Tracks: "
#                 f"{current_track_count}",
#                 (20, 70),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (255, 255, 255),
#                 2,
#             )

#             if prediction_records:

#                 status = (
#                     "PREDICTION"
#                 )

#                 status_color = (
#                     0,
#                     165,
#                     255,
#                 )

#             elif current_detection_records:

#                 status = (
#                     "DETECTION"
#                 )

#                 status_color = (
#                     0,
#                     255,
#                     0,
#                 )

#             elif tentative_records:

#                 status = (
#                     "TENTATIVE"
#                 )

#                 status_color = (
#                     255,
#                     0,
#                     255,
#                 )

#             else:

#                 status = (
#                     "NO TRACK"
#                 )

#                 status_color = (
#                     255,
#                     255,
#                     255,
#                 )

#             cv2.putText(
#                 output_frame,
#                 status,
#                 (20, 105),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.65,
#                 status_color,
#                 2,
#             )

#             # =================================================
#             # WRITE
#             # =================================================

#             writer.write(
#                 output_frame
#             )

#             frames_processed += 1

#             # =================================================
#             # PROGRESS
#             # =================================================

#             if (
#                 frames_processed % 10
#                 == 0
#             ):

#                 print(
#                     f"Frame "
#                     f"{frame_id:05d} | "
#                     f"Candidates: "
#                     f"{detection_count:2d} | "
#                     f"Strong: "
#                     f"{len(strong_detections):2d} | "
#                     f"Weak: "
#                     f"{len(weak_detections):2d} | "
#                     f"Tracks: "
#                     f"{len(current_detection_records):2d} | "
#                     f"Tentative: "
#                     f"{len(tentative_records):2d} | "
#                     f"Predictions: "
#                     f"{len(prediction_records):2d}"
#                 )

#             # =================================================
#             # TEST LIMIT
#             # =================================================

#             if (
#                 MAX_FRAMES is not None
#                 and frames_processed
#                 >= MAX_FRAMES
#             ):
#                 break

#     finally:

#         reader.release()

#         writer.release()

#     # ========================================================
#     # SAVE JSON
#     # ========================================================

#     output_data = {

#         "video":
#             str(VIDEO_PATH),

#         "fps":
#             fps,

#         "frames_processed":
#             frames_processed,

#         "total_detections":
#             total_detections,

#         "strong_detection_count":
#             strong_detection_count,

#         "weak_detection_count":
#             weak_detection_count,

#         "frames_with_detections":
#             frames_with_detections,

#         "frames_with_strong_detections":
#             frames_with_strong_detections,

#         "frames_with_tentative":
#             frames_with_tentative,

#         "frames_with_tracks":
#             frames_with_tracks,

#         "total_tracked_observations":
#             total_tracked_observations,

#         "total_detection_observations":
#             total_detection_observations,

#         "total_prediction_observations":
#             total_prediction_observations,

#         "total_tentative_observations":
#             total_tentative_observations,

#         "frames_with_predictions":
#             frames_with_predictions,

#         "unique_track_ids":
#             sorted(
#                 unique_track_ids
#             ),

#         "track_count":
#             len(unique_track_ids),

#         "tentative_confirmation_frames":
#             TENTATIVE_CONFIRMATION_FRAMES,

#         "tentative_min_iou":
#             TENTATIVE_MIN_IOU,

#         "tentative_max_missed_frames":
#             TENTATIVE_MAX_MISSED_FRAMES,

#         "tracks":
#             track_data,
#     }

#     with open(
#         TRACK_DATA_PATH,
#         "w",
#         encoding="utf-8",
#     ) as f:

#         json.dump(
#             output_data,
#             f,
#             indent=2,
#         )

#     # ========================================================
#     # SUMMARY
#     # ========================================================

#     print(
#         "\n" + "=" * 70
#     )

#     print(
#         "STEP 5.3 COMPLETE"
#     )

#     print(
#         "=" * 70
#     )

#     print(
#         f"Frames processed           : "
#         f"{frames_processed}"
#     )

#     print(
#         f"Total detector candidates  : "
#         f"{total_detections}"
#     )

#     print(
#         f"Strong detections          : "
#         f"{strong_detection_count}"
#     )

#     print(
#         f"Weak candidates            : "
#         f"{weak_detection_count}"
#     )

#     print(
#         f"Frames with detections     : "
#         f"{frames_with_detections}"
#     )

#     print(
#         f"Frames with strong detect. : "
#         f"{frames_with_strong_detections}"
#     )

#     print(
#         f"Tentative observations     : "
#         f"{total_tentative_observations}"
#     )

#     print(
#         f"Frames with tentative      : "
#         f"{frames_with_tentative}"
#     )

#     print(
#         f"Detection observations     : "
#         f"{total_detection_observations}"
#     )

#     print(
#         f"Prediction observations    : "
#         f"{total_prediction_observations}"
#     )

#     print(
#         f"Frames with predictions    : "
#         f"{frames_with_predictions}"
#     )

#     print(
#         f"Frames with tracks         : "
#         f"{frames_with_tracks}"
#     )

#     print(
#         f"Unique Track IDs           : "
#         f"{len(unique_track_ids)}"
#     )

#     print(
#         f"Track IDs                  : "
#         f"{sorted(unique_track_ids)}"
#     )

#     print(
#         "\nTracked video:"
#     )

#     print(
#         OUTPUT_VIDEO_PATH
#     )

#     print(
#         "\nTrack data:"
#     )

#     print(
#         TRACK_DATA_PATH
#     )

#     print(
#         "\nThreshold logic:"
#     )

#     print(
#         "  < 0.03       = ignored"
#     )

#     print(
#         "  0.03–0.055   = tentative / weak"
#     )

#     print(
#         "  >= 0.055     = strong detector evidence"
#     )

#     print(
#         f"  confirmation = "
#         f"{TENTATIVE_CONFIRMATION_FRAMES} "
#         f"consecutive supporting frames"
#     )


# # ============================================================
# # RUN
# # ============================================================

# if __name__ == "__main__":
#     main()



































# # """
# # =====================================================================
# # STEP 5.3 — BYTETRACK INTEGRATION
# # =====================================================================

# # Purpose:
# #     RT-DETRv2 detections -> ByteTrack persistent IDs.

# # Tracker input:
# #     Confidence >= 0.03

# # Strong detection:
# #     Confidence >= 0.055

# # Weak tracker candidate:
# #     0.03 <= confidence < 0.055

# # Important:
# #     Weak candidates may help an EXISTING track continue,
# #     but weak candidates must NOT create a new confirmed track.

# # During detector gaps:
# #     ByteTrack Kalman-predicted states are preserved.

# # Observation types:
# #     detection  = current RT-DETRv2 detection associated with a track
# #     prediction = ByteTrack/Kalman prediction during detector gap
# # """

# # from pathlib import Path
# # import json
# # import sys
# # import importlib.util

# # import cv2
# # import numpy as np
# # import supervision as sv

# # from supervision.tracker.byte_tracker.core import (
# #     STrack,
# #     TrackState,
# # )

# # from supervision.tracker.byte_tracker import matching


# # # ============================================================
# # # FIX 1 — ALLOW TRACK ACTIVATION ON ANY FRAME
# # # ============================================================

# # def patched_activate(
# #     self,
# #     kalman_filter,
# #     frame_id,
# # ):
# #     self.kalman_filter = kalman_filter

# #     self.internal_track_id = (
# #         self.internal_id_counter.new_id()
# #     )

# #     self.mean, self.covariance = (
# #         self.kalman_filter.initiate(
# #             self.tlwh_to_xyah(self._tlwh)
# #         )
# #     )

# #     self.tracklet_len = 1

# #     self.state = TrackState.Tracked

# #     if (
# #         self.tracklet_len
# #         >= self.minimum_consecutive_frames
# #     ):
# #         self.is_activated = True

# #         self.external_track_id = (
# #             self.external_id_counter.new_id()
# #         )

# #     self.frame_id = frame_id
# #     self.start_frame = frame_id


# # STrack.activate = patched_activate


# # # ============================================================
# # # FIX 2 — IOU-ONLY ASSOCIATION
# # # ============================================================

# # def patched_fuse_score(
# #     cost_matrix,
# #     stracks,
# # ):
# #     # Do NOT multiply IoU similarity by detector confidence.
# #     #
# #     # Example:
# #     # IoU = 0.97
# #     # confidence = 0.04
# #     #
# #     # Original fusion can destroy the excellent spatial match.
# #     #
# #     # We therefore use pure IoU association.

# #     return cost_matrix


# # matching.fuse_score = patched_fuse_score


# # print(
# #     "ByteTrack compatibility patches applied:"
# # )

# # print(
# #     "  - Track activation fix : ENABLED"
# # )

# # print(
# #     "  - IoU-only association : ENABLED"
# # )

# # print(
# #     "  - Lost-track prediction: ENABLED"
# # )


# # # ============================================================
# # # PROJECT PATHS
# # # ============================================================

# # PROJECT_DIR = Path(
# #     "/content/drive/MyDrive/snake_detection/Snake_detection"
# # )

# # VIDEO_PATH = (
# #     PROJECT_DIR
# #     / "snake_videos"
# #     / "10231672-hd_1080_1920_30fps.mp4"
# # )

# # OUTPUT_VIDEO_PATH = (
# #     PROJECT_DIR
# #     / "dataset"
# #     / "tracking_output"
# #     / "tracked_video_5_3.mp4"
# # )

# # TRACK_DATA_PATH = (
# #     PROJECT_DIR
# #     / "dataset"
# #     / "tracking_output"
# #     / "track_data_5_3.json"
# # )


# # # ============================================================
# # # LOAD STEP 5.2 ADAPTER
# # # ============================================================

# # ADAPTER_PATH = (
# #     PROJECT_DIR
# #     / "step5_2_rtdetr_bytetrack_adapter.py"
# # )

# # if not ADAPTER_PATH.exists():
# #     raise FileNotFoundError(
# #         f"Step 5.2 adapter not found:\n{ADAPTER_PATH}"
# #     )


# # spec = importlib.util.spec_from_file_location(
# #     "step5_2_rtdetr_bytetrack_adapter",
# #     ADAPTER_PATH,
# # )

# # adapter = importlib.util.module_from_spec(spec)

# # sys.modules[
# #     "step5_2_rtdetr_bytetrack_adapter"
# # ] = adapter

# # spec.loader.exec_module(adapter)

# # load_detector = adapter.load_detector
# # detect_snakes = adapter.detect_snakes


# # # ============================================================
# # # LOAD STEP 5.1 VIDEO READER
# # # ============================================================

# # VIDEO_READER_PATH = (
# #     PROJECT_DIR
# #     / "step5_1_video_input.py"
# # )

# # if not VIDEO_READER_PATH.exists():
# #     raise FileNotFoundError(
# #         f"Step 5.1 video reader not found:\n"
# #         f"{VIDEO_READER_PATH}"
# #     )


# # spec_reader = (
# #     importlib.util.spec_from_file_location(
# #         "step5_1_video_input",
# #         VIDEO_READER_PATH,
# #     )
# # )

# # video_module = (
# #     importlib.util.module_from_spec(
# #         spec_reader
# #     )
# # )

# # sys.modules[
# #     "step5_1_video_input"
# # ] = video_module

# # spec_reader.loader.exec_module(
# #     video_module
# # )

# # VideoFrameReader = (
# #     video_module.VideoFrameReader
# # )


# # # ============================================================
# # # BYTE TRACK SETTINGS
# # # ============================================================

# # # Candidates from Step 5.2 are allowed into ByteTrack
# # TRACKER_INPUT_THRESHOLD = 0.03

# # # Only detections >= 0.055 may CREATE a new confirmed track
# # TRACKER_NEW_TRACK_THRESHOLD = 0.055

# # # How long ByteTrack keeps lost tracks alive
# # TRACKER_LOST_TRACK_BUFFER = 30

# # # IoU association threshold
# # TRACKER_MATCHING_THRESHOLD = 0.80

# # # First test only
# # MAX_FRAMES = 400


# # # ============================================================
# # # CREATE TRACKER
# # # ============================================================

# # def create_tracker(fps):

# #     tracker = sv.ByteTrack(
# #         track_activation_threshold=(
# #             TRACKER_INPUT_THRESHOLD
# #         ),

# #         lost_track_buffer=(
# #             TRACKER_LOST_TRACK_BUFFER
# #         ),

# #         minimum_matching_threshold=(
# #             TRACKER_MATCHING_THRESHOLD
# #         ),

# #         frame_rate=fps,
# #     )

# #     # --------------------------------------------------------
# #     # IMPORTANT
# #     #
# #     # Supervision normally initializes det_thresh above the
# #     # activation threshold.
# #     #
# #     # We explicitly set it to 0.055.
# #     #
# #     # Therefore:
# #     #
# #     # 0.03–0.054999
# #     #     → can participate in tracking
# #     #     → cannot create a new track
# #     #
# #     # >= 0.055
# #     #     → can create a new track
# #     # --------------------------------------------------------

# #     tracker.det_thresh = (
# #         TRACKER_NEW_TRACK_THRESHOLD
# #     )

# #     return tracker


# # # ============================================================
# # # CONVERT DETECTIONS
# # # ============================================================

# # def detections_to_supervision(
# #     detections
# # ):

# #     if not detections:
# #         return sv.Detections.empty()

# #     xyxy = np.array(
# #         [
# #             d["bbox"]
# #             for d in detections
# #         ],
# #         dtype=np.float32,
# #     )

# #     confidence = np.array(
# #         [
# #             d["confidence"]
# #             for d in detections
# #         ],
# #         dtype=np.float32,
# #     )

# #     class_id = np.array(
# #         [
# #             d["class_id"]
# #             for d in detections
# #         ],
# #         dtype=np.int32,
# #     )

# #     return sv.Detections(
# #         xyxy=xyxy,
# #         confidence=confidence,
# #         class_id=class_id,
# #     )


# # # ============================================================
# # # TRACK ID
# # # ============================================================

# # def get_track_id(track):

# #     external_id = getattr(
# #         track,
# #         "external_track_id",
# #         None,
# #     )

# #     if external_id is not None:
# #         return int(external_id)

# #     internal_id = getattr(
# #         track,
# #         "internal_track_id",
# #         None,
# #     )

# #     if internal_id is not None:
# #         return int(internal_id)

# #     return None


# # # ============================================================
# # # TRACK BBOX
# # # ============================================================

# # def get_track_bbox(track):

# #     # Preferred: tlbr
# #     tlbr = getattr(
# #         track,
# #         "tlbr",
# #         None,
# #     )

# #     if tlbr is not None:

# #         try:

# #             values = np.asarray(
# #                 tlbr,
# #                 dtype=np.float32,
# #             ).reshape(-1)

# #             if len(values) == 4:

# #                 return [
# #                     float(values[0]),
# #                     float(values[1]),
# #                     float(values[2]),
# #                     float(values[3]),
# #                 ]

# #         except Exception:
# #             pass

# #     # Fallback: tlwh
# #     tlwh = getattr(
# #         track,
# #         "tlwh",
# #         None,
# #     )

# #     if tlwh is not None:

# #         try:

# #             values = np.asarray(
# #                 tlwh,
# #                 dtype=np.float32,
# #             ).reshape(-1)

# #             if len(values) == 4:

# #                 x, y, w, h = values

# #                 return [
# #                     float(x),
# #                     float(y),
# #                     float(x + w),
# #                     float(y + h),
# #                 ]

# #         except Exception:
# #             pass

# #     return None


# # # ============================================================
# # # CHECK ACTIVE / VALID TRACK
# # # ============================================================

# # def valid_track(track):

# #     track_id = get_track_id(track)

# #     if track_id is None:
# #         return False

# #     bbox = get_track_bbox(track)

# #     if bbox is None:
# #         return False

# #     x1, y1, x2, y2 = bbox

# #     if x2 <= x1:
# #         return False

# #     if y2 <= y1:
# #         return False

# #     return True


# # # ============================================================
# # # DRAW DETECTION TRACKS
# # # ============================================================

# # def draw_detection_tracks(
# #     frame,
# #     tracked_detections,
# # ):

# #     output = frame.copy()

# #     if len(tracked_detections) == 0:
# #         return output

# #     boxes = tracked_detections.xyxy

# #     confidences = (
# #         tracked_detections.confidence
# #     )

# #     track_ids = (
# #         tracked_detections.tracker_id
# #     )

# #     for i, box in enumerate(boxes):

# #         x1, y1, x2, y2 = map(
# #             int,
# #             box,
# #         )

# #         confidence = float(
# #             confidences[i]
# #         )

# #         track_id = int(
# #             track_ids[i]
# #         )

# #         # Strong vs weak visualization
# #         if confidence >= TRACKER_NEW_TRACK_THRESHOLD:

# #             color = (0, 255, 0)
# #             label_type = "STRONG"

# #         else:

# #             color = (255, 255, 0)
# #             label_type = "WEAK"

# #         cv2.rectangle(
# #             output,
# #             (x1, y1),
# #             (x2, y2),
# #             color,
# #             2,
# #         )

# #         label = (
# #             f"Snake | ID: {track_id} | "
# #             f"{confidence:.3f} | {label_type}"
# #         )

# #         cv2.putText(
# #             output,
# #             label,
# #             (
# #                 x1,
# #                 max(25, y1 - 10),
# #             ),
# #             cv2.FONT_HERSHEY_SIMPLEX,
# #             0.6,
# #             color,
# #             2,
# #         )

# #     return output


# # # ============================================================
# # # DRAW PREDICTIONS
# # # ============================================================

# # def draw_prediction_tracks(
# #     frame,
# #     prediction_records,
# # ):

# #     output = frame

# #     for record in prediction_records:

# #         x1, y1, x2, y2 = map(
# #             int,
# #             record["bbox"],
# #         )

# #         track_id = int(
# #             record["track_id"]
# #         )

# #         # Orange = Kalman prediction
# #         color = (
# #             0,
# #             165,
# #             255,
# #         )

# #         cv2.rectangle(
# #             output,
# #             (x1, y1),
# #             (x2, y2),
# #             color,
# #             2,
# #         )

# #         label = (
# #             f"Snake | ID: {track_id} | PRED"
# #         )

# #         cv2.putText(
# #             output,
# #             label,
# #             (
# #                 x1,
# #                 max(25, y1 - 10),
# #             ),
# #             cv2.FONT_HERSHEY_SIMPLEX,
# #             0.6,
# #             color,
# #             2,
# #         )

# #     return output


# # # ============================================================
# # # MAIN
# # # ============================================================

# # def main():

# #     print(
# #         "\n" + "=" * 70
# #     )

# #     print(
# #         "STEP 5.3 — BYTETRACK INTEGRATION"
# #     )

# #     print(
# #         "=" * 70
# #     )

# #     # --------------------------------------------------------
# #     # VIDEO
# #     # --------------------------------------------------------

# #     if not VIDEO_PATH.exists():

# #         raise FileNotFoundError(
# #             f"Video not found:\n{VIDEO_PATH}"
# #         )

# #     print(
# #         "\nVideo:"
# #     )

# #     print(
# #         VIDEO_PATH
# #     )

# #     # --------------------------------------------------------
# #     # DETECTOR
# #     # --------------------------------------------------------

# #     print(
# #         "\nLoading RT-DETRv2..."
# #     )

# #     processor, model = (
# #         load_detector()
# #     )

# #     print(
# #         "Detector loaded successfully."
# #     )

# #     # --------------------------------------------------------
# #     # VIDEO READER
# #     # --------------------------------------------------------

# #     reader = VideoFrameReader(
# #         video_path=VIDEO_PATH,
# #         frame_skip=1,
# #     )

# #     video_info = (
# #         reader.get_video_info()
# #     )

# #     fps = video_info["fps"]

# #     print(
# #         "\nVIDEO"
# #     )

# #     print(
# #         "-" * 70
# #     )

# #     print(
# #         f"Resolution   : "
# #         f"{video_info['width']} x "
# #         f"{video_info['height']}"
# #     )

# #     print(
# #         f"FPS          : "
# #         f"{fps:.2f}"
# #     )

# #     print(
# #         f"Total frames : "
# #         f"{video_info['total_frames']}"
# #     )

# #     print(
# #         f"Test limit   : "
# #         f"{MAX_FRAMES}"
# #     )

# #     # --------------------------------------------------------
# #     # TRACKER
# #     # --------------------------------------------------------

# #     tracker = create_tracker(
# #         fps
# #     )

# #     print(
# #         "\nByteTrack initialized."
# #     )

# #     print(
# #         f"Tracker input threshold : "
# #         f"{TRACKER_INPUT_THRESHOLD}"
# #     )

# #     print(
# #         f"New-track threshold     : "
# #         f"{TRACKER_NEW_TRACK_THRESHOLD}"
# #     )

# #     print(
# #         f"Internal det threshold  : "
# #         f"{tracker.det_thresh}"
# #     )

# #     print(
# #         f"Lost track buffer       : "
# #         f"{TRACKER_LOST_TRACK_BUFFER}"
# #     )

# #     print(
# #         f"Matching threshold      : "
# #         f"{TRACKER_MATCHING_THRESHOLD}"
# #     )

# #     print(
# #         "Association mode        : IoU-only"
# #     )

# #     print(
# #         "Prediction recording    : ENABLED"
# #     )

# #     # --------------------------------------------------------
# #     # OUTPUT DIRECTORIES
# #     # --------------------------------------------------------

# #     OUTPUT_VIDEO_PATH.parent.mkdir(
# #         parents=True,
# #         exist_ok=True,
# #     )

# #     TRACK_DATA_PATH.parent.mkdir(
# #         parents=True,
# #         exist_ok=True,
# #     )

# #     # --------------------------------------------------------
# #     # VIDEO WRITER
# #     # --------------------------------------------------------

# #     fourcc = cv2.VideoWriter_fourcc(
# #         *"mp4v"
# #     )

# #     writer = cv2.VideoWriter(
# #         str(OUTPUT_VIDEO_PATH),
# #         fourcc,
# #         fps,
# #         (
# #             video_info["width"],
# #             video_info["height"],
# #         ),
# #     )

# #     if not writer.isOpened():

# #         reader.release()

# #         raise RuntimeError(
# #             "Could not create output video."
# #         )

# #     # --------------------------------------------------------
# #     # DATA
# #     # --------------------------------------------------------

# #     track_data = []

# #     frames_processed = 0

# #     total_detections = 0

# #     frames_with_detections = 0

# #     frames_with_tracks = 0

# #     total_tracked_observations = 0

# #     total_detection_observations = 0

# #     total_prediction_observations = 0

# #     frames_with_predictions = 0

# #     strong_detection_count = 0

# #     weak_detection_count = 0

# #     frames_with_strong_detections = 0

# #     unique_track_ids = set()

# #     # ========================================================
# #     # PROCESS VIDEO
# #     # ========================================================

# #     print(
# #         "\nPROCESSING VIDEO"
# #     )

# #     print(
# #         "-" * 70
# #     )

# #     try:

# #         for item in reader.frames():

# #             frame = item["frame"]

# #             frame_id = item["frame_id"]

# #             timestamp = item["timestamp"]

# #             # ------------------------------------------------
# #             # DETECTION
# #             # ------------------------------------------------

# #             detections = detect_snakes(
# #                 frame,
# #                 processor,
# #                 model,
# #             )

# #             detection_count = len(
# #                 detections
# #             )

# #             total_detections += (
# #                 detection_count
# #             )

# #             if detection_count > 0:

# #                 frames_with_detections += 1

# #             # ------------------------------------------------
# #             # CLASSIFY STRONG / WEAK
# #             # ------------------------------------------------

# #             frame_has_strong = False

# #             for detection in detections:

# #                 confidence = float(
# #                     detection["confidence"]
# #                 )

# #                 if (
# #                     confidence
# #                     >= TRACKER_NEW_TRACK_THRESHOLD
# #                 ):

# #                     strong_detection_count += 1
# #                     frame_has_strong = True

# #                 else:

# #                     weak_detection_count += 1

# #             if frame_has_strong:

# #                 frames_with_strong_detections += 1

# #             # ------------------------------------------------
# #             # SUPERVISION DETECTIONS
# #             # ------------------------------------------------

# #             sv_detections = (
# #                 detections_to_supervision(
# #                     detections
# #                 )
# #             )

# #             # ------------------------------------------------
# #             # BYTE TRACK
# #             # ------------------------------------------------

# #             tracked_detections = (
# #                 tracker.update_with_detections(
# #                     sv_detections
# #                 )
# #             )

# #             # ------------------------------------------------
# #             # DETECTION-ASSOCIATED IDS
# #             # ------------------------------------------------

# #             detection_track_ids = set()

# #             current_detection_records = []

# #             if len(
# #                 tracked_detections
# #             ) > 0:

# #                 boxes = (
# #                     tracked_detections.xyxy
# #                 )

# #                 confidences = (
# #                     tracked_detections.confidence
# #                 )

# #                 track_ids = (
# #                     tracked_detections.tracker_id
# #                 )

# #                 for i, box in enumerate(
# #                     boxes
# #                 ):

# #                     x1, y1, x2, y2 = map(
# #                         float,
# #                         box,
# #                     )

# #                     confidence = float(
# #                         confidences[i]
# #                     )

# #                     track_id = int(
# #                         track_ids[i]
# #                     )

# #                     detection_track_ids.add(
# #                         track_id
# #                     )

# #                     unique_track_ids.add(
# #                         track_id
# #                     )

# #                     center_x = (
# #                         x1 + x2
# #                     ) / 2.0

# #                     center_y = (
# #                         y1 + y2
# #                     ) / 2.0

# #                     observation_strength = (
# #                         "strong"
# #                         if confidence
# #                         >= TRACKER_NEW_TRACK_THRESHOLD
# #                         else "weak"
# #                     )

# #                     record = {

# #                         "frame_id":
# #                             frame_id,

# #                         "timestamp":
# #                             timestamp,

# #                         "track_id":
# #                             track_id,

# #                         "bbox": [
# #                             x1,
# #                             y1,
# #                             x2,
# #                             y2,
# #                         ],

# #                         "center": [
# #                             center_x,
# #                             center_y,
# #                         ],

# #                         "confidence":
# #                             confidence,

# #                         "class_id":
# #                             0,

# #                         "class_name":
# #                             "snake",

# #                         "observation_type":
# #                             "detection",

# #                         "detection_strength":
# #                             observation_strength,

# #                         "track_state":
# #                             "tracked",
# #                     }

# #                     current_detection_records.append(
# #                         record
# #                     )

# #                     track_data.append(
# #                         record
# #                     )

# #                     total_tracked_observations += 1

# #                     total_detection_observations += 1

# #             # ------------------------------------------------
# #             # READ LOST TRACKS EVERY FRAME
# #             # ------------------------------------------------

# #             prediction_records = []

# #             lost_tracks = getattr(
# #                 tracker,
# #                 "lost_tracks",
# #                 [],
# #             )

# #             for track in lost_tracks:

# #                 if not valid_track(track):
# #                     continue

# #                 track_id = get_track_id(
# #                     track
# #                 )

# #                 # Don't duplicate a track that has
# #                 # a current detector association.
# #                 if track_id in detection_track_ids:
# #                     continue

# #                 bbox = get_track_bbox(
# #                     track
# #                 )

# #                 if bbox is None:
# #                     continue

# #                 x1, y1, x2, y2 = bbox

# #                 center_x = (
# #                     x1 + x2
# #                 ) / 2.0

# #                 center_y = (
# #                     y1 + y2
# #                 ) / 2.0

# #                 record = {

# #                     "frame_id":
# #                         frame_id,

# #                     "timestamp":
# #                         timestamp,

# #                     "track_id":
# #                         int(track_id),

# #                     "bbox": [
# #                         float(x1),
# #                         float(y1),
# #                         float(x2),
# #                         float(y2),
# #                     ],

# #                     "center": [
# #                         float(center_x),
# #                         float(center_y),
# #                     ],

# #                     "confidence":
# #                         None,

# #                     "class_id":
# #                         0,

# #                     "class_name":
# #                         "snake",

# #                     "observation_type":
# #                         "prediction",

# #                     "detection_strength":
# #                         None,

# #                     "track_state":
# #                         "lost",
# #                 }

# #                 prediction_records.append(
# #                     record
# #                 )

# #                 track_data.append(
# #                     record
# #                 )

# #                 unique_track_ids.add(
# #                     int(track_id)
# #                 )

# #                 total_tracked_observations += 1

# #                 total_prediction_observations += 1

# #             if prediction_records:

# #                 frames_with_predictions += 1

# #             # ------------------------------------------------
# #             # FRAME TRACK COUNT
# #             # ------------------------------------------------

# #             current_track_count = (
# #                 len(current_detection_records)
# #                 + len(prediction_records)
# #             )

# #             if current_track_count > 0:

# #                 frames_with_tracks += 1

# #             # ------------------------------------------------
# #             # DRAW DETECTIONS
# #             # ------------------------------------------------

# #             output_frame = (
# #                 draw_detection_tracks(
# #                     frame,
# #                     tracked_detections,
# #                 )
# #             )

# #             # ------------------------------------------------
# #             # DRAW PREDICTIONS
# #             # ------------------------------------------------

# #             output_frame = (
# #                 draw_prediction_tracks(
# #                     output_frame,
# #                     prediction_records,
# #                 )
# #             )

# #             # ------------------------------------------------
# #             # FRAME STATUS
# #             # ------------------------------------------------

# #             cv2.putText(
# #                 output_frame,
# #                 f"Frame: {frame_id}",
# #                 (20, 35),
# #                 cv2.FONT_HERSHEY_SIMPLEX,
# #                 0.8,
# #                 (255, 255, 255),
# #                 2,
# #             )

# #             cv2.putText(
# #                 output_frame,
# #                 f"Tracks: {current_track_count}",
# #                 (20, 70),
# #                 cv2.FONT_HERSHEY_SIMPLEX,
# #                 0.8,
# #                 (255, 255, 255),
# #                 2,
# #             )

# #             if prediction_records:

# #                 status = (
# #                     "PREDICTION"
# #                 )

# #                 status_color = (
# #                     0,
# #                     165,
# #                     255,
# #                 )

# #             elif current_detection_records:

# #                 status = (
# #                     "DETECTION"
# #                 )

# #                 status_color = (
# #                     0,
# #                     255,
# #                     0,
# #                 )

# #             else:

# #                 status = (
# #                     "NO TRACK"
# #                 )

# #                 status_color = (
# #                     255,
# #                     255,
# #                     255,
# #                 )

# #             cv2.putText(
# #                 output_frame,
# #                 status,
# #                 (20, 105),
# #                 cv2.FONT_HERSHEY_SIMPLEX,
# #                 0.65,
# #                 status_color,
# #                 2,
# #             )

# #             # ------------------------------------------------
# #             # WRITE
# #             # ------------------------------------------------

# #             writer.write(
# #                 output_frame
# #             )

# #             frames_processed += 1

# #             # ------------------------------------------------
# #             # PROGRESS
# #             # ------------------------------------------------

# #             if (
# #                 frames_processed % 10
# #                 == 0
# #             ):

# #                 print(
# #                     f"Frame "
# #                     f"{frame_id:05d} | "
# #                     f"Detections: "
# #                     f"{detection_count:2d} | "
# #                     f"Detection tracks: "
# #                     f"{len(current_detection_records):2d} | "
# #                     f"Predictions: "
# #                     f"{len(prediction_records):2d}"
# #                 )

# #             # ------------------------------------------------
# #             # TEST LIMIT
# #             # ------------------------------------------------

# #             if (
# #                 MAX_FRAMES is not None
# #                 and frames_processed
# #                 >= MAX_FRAMES
# #             ):
# #                 break

# #     finally:

# #         reader.release()

# #         writer.release()

# #     # ========================================================
# #     # SAVE JSON
# #     # ========================================================

# #     output_data = {

# #         "video":
# #             str(VIDEO_PATH),

# #         "fps":
# #             fps,

# #         "frames_processed":
# #             frames_processed,

# #         "total_detections":
# #             total_detections,

# #         "frames_with_detections":
# #             frames_with_detections,

# #         "strong_detection_count":
# #             strong_detection_count,

# #         "weak_detection_count":
# #             weak_detection_count,

# #         "frames_with_strong_detections":
# #             frames_with_strong_detections,

# #         "frames_with_tracks":
# #             frames_with_tracks,

# #         "total_tracked_observations":
# #             total_tracked_observations,

# #         "total_detection_observations":
# #             total_detection_observations,

# #         "total_prediction_observations":
# #             total_prediction_observations,

# #         "frames_with_predictions":
# #             frames_with_predictions,

# #         "unique_track_ids":
# #             sorted(
# #                 unique_track_ids
# #             ),

# #         "track_count":
# #             len(unique_track_ids),

# #         "tracks":
# #             track_data,
# #     }

# #     with open(
# #         TRACK_DATA_PATH,
# #         "w",
# #         encoding="utf-8",
# #     ) as f:

# #         json.dump(
# #             output_data,
# #             f,
# #             indent=2,
# #         )

# #     # ========================================================
# #     # SUMMARY
# #     # ========================================================

# #     print(
# #         "\n" + "=" * 70
# #     )

# #     print(
# #         "STEP 5.3 COMPLETE"
# #     )

# #     print(
# #         "=" * 70
# #     )

# #     print(
# #         f"Frames processed           : "
# #         f"{frames_processed}"
# #     )

# #     print(
# #         f"Total detector candidates  : "
# #         f"{total_detections}"
# #     )

# #     print(
# #         f"Strong detections          : "
# #         f"{strong_detection_count}"
# #     )

# #     print(
# #         f"Weak tracker candidates    : "
# #         f"{weak_detection_count}"
# #     )

# #     print(
# #         f"Frames with detections     : "
# #         f"{frames_with_detections}"
# #     )

# #     print(
# #         f"Frames with strong detect. : "
# #         f"{frames_with_strong_detections}"
# #     )

# #     print(
# #         f"Frames with tracks         : "
# #         f"{frames_with_tracks}"
# #     )

# #     print(
# #         f"Tracked observations       : "
# #         f"{total_tracked_observations}"
# #     )

# #     print(
# #         f"Detection observations     : "
# #         f"{total_detection_observations}"
# #     )

# #     print(
# #         f"Prediction observations    : "
# #         f"{total_prediction_observations}"
# #     )

# #     print(
# #         f"Frames with predictions    : "
# #         f"{frames_with_predictions}"
# #     )

# #     print(
# #         f"Unique Track IDs           : "
# #         f"{len(unique_track_ids)}"
# #     )

# #     print(
# #         f"Track IDs                  : "
# #         f"{sorted(unique_track_ids)}"
# #     )

# #     print(
# #         "\nTracked video:"
# #     )

# #     print(
# #         OUTPUT_VIDEO_PATH
# #     )

# #     print(
# #         "\nTrack data:"
# #     )

# #     print(
# #         TRACK_DATA_PATH
# #     )

# #     print(
# #         "\nThreshold logic:"
# #     )

# #     print(
# #         "  >= 0.055 = strong detection / can create track"
# #     )

# #     print(
# #         "  0.03-0.055 = weak tracker candidate"
# #     )

# #     print(
# #         "  prediction = ByteTrack Kalman state"
# #     )


# # # ============================================================
# # # RUN
# # # ============================================================

# # if __name__ == "__main__":
# #     main()










# # # """
# # # =====================================================================
# # # STEP 5.3 — BYTETRACK INTEGRATION
# # # =====================================================================

# # # Purpose:
# # #     RT-DETRv2 detections -> ByteTrack persistent IDs.

# # # Tracker input:
# # #     Confidence >= 0.03

# # # Strong detection:
# # #     Confidence >= 0.055

# # # Weak tracker candidate:
# # #     0.03 <= confidence < 0.055

# # # Important:
# # #     Weak candidates may help an EXISTING track continue,
# # #     but weak candidates must NOT create a new confirmed track.

# # # During detector gaps:
# # #     ByteTrack Kalman-predicted states are preserved.

# # # Observation types:
# # #     detection  = current RT-DETRv2 detection associated with a track
# # #     prediction = ByteTrack/Kalman prediction during detector gap
# # # """

# # # from pathlib import Path
# # # import json
# # # import sys
# # # import importlib.util

# # # import cv2
# # # import numpy as np
# # # import supervision as sv

# # # from supervision.tracker.byte_tracker.core import (
# # #     STrack,
# # #     TrackState,
# # # )

# # # from supervision.tracker.byte_tracker import matching


# # # # ============================================================
# # # # FIX 1 — ALLOW TRACK ACTIVATION ON ANY FRAME
# # # # ============================================================

# # # def patched_activate(
# # #     self,
# # #     kalman_filter,
# # #     frame_id,
# # # ):
# # #     self.kalman_filter = kalman_filter

# # #     self.internal_track_id = (
# # #         self.internal_id_counter.new_id()
# # #     )

# # #     self.mean, self.covariance = (
# # #         self.kalman_filter.initiate(
# # #             self.tlwh_to_xyah(self._tlwh)
# # #         )
# # #     )

# # #     self.tracklet_len = 1

# # #     self.state = TrackState.Tracked

# # #     if (
# # #         self.tracklet_len
# # #         >= self.minimum_consecutive_frames
# # #     ):
# # #         self.is_activated = True

# # #         self.external_track_id = (
# # #             self.external_id_counter.new_id()
# # #         )

# # #     self.frame_id = frame_id
# # #     self.start_frame = frame_id


# # # STrack.activate = patched_activate


# # # # ============================================================
# # # # FIX 2 — IOU-ONLY ASSOCIATION
# # # # ============================================================

# # # def patched_fuse_score(
# # #     cost_matrix,
# # #     stracks,
# # # ):
# # #     # Do NOT multiply IoU similarity by detector confidence.
# # #     #
# # #     # Example:
# # #     # IoU = 0.97
# # #     # confidence = 0.04
# # #     #
# # #     # Original fusion can destroy the excellent spatial match.
# # #     #
# # #     # We therefore use pure IoU association.

# # #     return cost_matrix


# # # matching.fuse_score = patched_fuse_score


# # # print(
# # #     "ByteTrack compatibility patches applied:"
# # # )

# # # print(
# # #     "  - Track activation fix : ENABLED"
# # # )

# # # print(
# # #     "  - IoU-only association : ENABLED"
# # # )

# # # print(
# # #     "  - Lost-track prediction: ENABLED"
# # # )


# # # # ============================================================
# # # # PROJECT PATHS
# # # # ============================================================

# # # PROJECT_DIR = Path(
# # #     "/content/drive/MyDrive/snake_detection/Snake_detection"
# # # )

# # # VIDEO_PATH = (
# # #     PROJECT_DIR
# # #     / "snake_videos"
# # #     / "10231672-hd_1080_1920_30fps.mp4"
# # # )

# # # OUTPUT_VIDEO_PATH = (
# # #     PROJECT_DIR
# # #     / "dataset"
# # #     / "tracking_output"
# # #     / "tracked_video_5_3.mp4"
# # # )

# # # TRACK_DATA_PATH = (
# # #     PROJECT_DIR
# # #     / "dataset"
# # #     / "tracking_output"
# # #     / "track_data_5_3.json"
# # # )


# # # # ============================================================
# # # # LOAD STEP 5.2 ADAPTER
# # # # ============================================================

# # # ADAPTER_PATH = (
# # #     PROJECT_DIR
# # #     / "step5_2_rtdetr_bytetrack_adapter.py"
# # # )

# # # if not ADAPTER_PATH.exists():
# # #     raise FileNotFoundError(
# # #         f"Step 5.2 adapter not found:\n{ADAPTER_PATH}"
# # #     )


# # # spec = importlib.util.spec_from_file_location(
# # #     "step5_2_rtdetr_bytetrack_adapter",
# # #     ADAPTER_PATH,
# # # )

# # # adapter = importlib.util.module_from_spec(spec)

# # # sys.modules[
# # #     "step5_2_rtdetr_bytetrack_adapter"
# # # ] = adapter

# # # spec.loader.exec_module(adapter)

# # # load_detector = adapter.load_detector
# # # detect_snakes = adapter.detect_snakes


# # # # ============================================================
# # # # LOAD STEP 5.1 VIDEO READER
# # # # ============================================================

# # # VIDEO_READER_PATH = (
# # #     PROJECT_DIR
# # #     / "step5_1_video_input.py"
# # # )

# # # if not VIDEO_READER_PATH.exists():
# # #     raise FileNotFoundError(
# # #         f"Step 5.1 video reader not found:\n"
# # #         f"{VIDEO_READER_PATH}"
# # #     )


# # # spec_reader = (
# # #     importlib.util.spec_from_file_location(
# # #         "step5_1_video_input",
# # #         VIDEO_READER_PATH,
# # #     )
# # # )

# # # video_module = (
# # #     importlib.util.module_from_spec(
# # #         spec_reader
# # #     )
# # # )

# # # sys.modules[
# # #     "step5_1_video_input"
# # # ] = video_module

# # # spec_reader.loader.exec_module(
# # #     video_module
# # # )

# # # VideoFrameReader = (
# # #     video_module.VideoFrameReader
# # # )


# # # # ============================================================
# # # # BYTE TRACK SETTINGS
# # # # ============================================================

# # # # Candidates from Step 5.2 are allowed into ByteTrack
# # # TRACKER_INPUT_THRESHOLD = 0.03

# # # # Only detections >= 0.055 may CREATE a new confirmed track
# # # TRACKER_NEW_TRACK_THRESHOLD = 0.055

# # # # How long ByteTrack keeps lost tracks alive
# # # TRACKER_LOST_TRACK_BUFFER = 30

# # # # IoU association threshold
# # # TRACKER_MATCHING_THRESHOLD = 0.80

# # # # First test only
# # # MAX_FRAMES = 400


# # # # ============================================================
# # # # CREATE TRACKER
# # # # ============================================================

# # # def create_tracker(fps):

# # #     tracker = sv.ByteTrack(
# # #         track_activation_threshold=(
# # #             TRACKER_INPUT_THRESHOLD
# # #         ),

# # #         lost_track_buffer=(
# # #             TRACKER_LOST_TRACK_BUFFER
# # #         ),

# # #         minimum_matching_threshold=(
# # #             TRACKER_MATCHING_THRESHOLD
# # #         ),

# # #         frame_rate=fps,
# # #     )

# # #     # --------------------------------------------------------
# # #     # IMPORTANT
# # #     #
# # #     # Supervision normally initializes det_thresh above the
# # #     # activation threshold.
# # #     #
# # #     # We explicitly set it to 0.055.
# # #     #
# # #     # Therefore:
# # #     #
# # #     # 0.03–0.054999
# # #     #     → can participate in tracking
# # #     #     → cannot create a new track
# # #     #
# # #     # >= 0.055
# # #     #     → can create a new track
# # #     # --------------------------------------------------------

# # #     tracker.det_thresh = (
# # #         TRACKER_NEW_TRACK_THRESHOLD
# # #     )

# # #     return tracker


# # # # ============================================================
# # # # CONVERT DETECTIONS
# # # # ============================================================

# # # def detections_to_supervision(
# # #     detections
# # # ):

# # #     if not detections:
# # #         return sv.Detections.empty()

# # #     xyxy = np.array(
# # #         [
# # #             d["bbox"]
# # #             for d in detections
# # #         ],
# # #         dtype=np.float32,
# # #     )

# # #     confidence = np.array(
# # #         [
# # #             d["confidence"]
# # #             for d in detections
# # #         ],
# # #         dtype=np.float32,
# # #     )

# # #     class_id = np.array(
# # #         [
# # #             d["class_id"]
# # #             for d in detections
# # #         ],
# # #         dtype=np.int32,
# # #     )

# # #     return sv.Detections(
# # #         xyxy=xyxy,
# # #         confidence=confidence,
# # #         class_id=class_id,
# # #     )


# # # # ============================================================
# # # # TRACK ID
# # # # ============================================================

# # # def get_track_id(track):

# # #     external_id = getattr(
# # #         track,
# # #         "external_track_id",
# # #         None,
# # #     )

# # #     if external_id is not None:
# # #         return int(external_id)

# # #     internal_id = getattr(
# # #         track,
# # #         "internal_track_id",
# # #         None,
# # #     )

# # #     if internal_id is not None:
# # #         return int(internal_id)

# # #     return None


# # # # ============================================================
# # # # TRACK BBOX
# # # # ============================================================

# # # def get_track_bbox(track):

# # #     # Preferred: tlbr
# # #     tlbr = getattr(
# # #         track,
# # #         "tlbr",
# # #         None,
# # #     )

# # #     if tlbr is not None:

# # #         try:

# # #             values = np.asarray(
# # #                 tlbr,
# # #                 dtype=np.float32,
# # #             ).reshape(-1)

# # #             if len(values) == 4:

# # #                 return [
# # #                     float(values[0]),
# # #                     float(values[1]),
# # #                     float(values[2]),
# # #                     float(values[3]),
# # #                 ]

# # #         except Exception:
# # #             pass

# # #     # Fallback: tlwh
# # #     tlwh = getattr(
# # #         track,
# # #         "tlwh",
# # #         None,
# # #     )

# # #     if tlwh is not None:

# # #         try:

# # #             values = np.asarray(
# # #                 tlwh,
# # #                 dtype=np.float32,
# # #             ).reshape(-1)

# # #             if len(values) == 4:

# # #                 x, y, w, h = values

# # #                 return [
# # #                     float(x),
# # #                     float(y),
# # #                     float(x + w),
# # #                     float(y + h),
# # #                 ]

# # #         except Exception:
# # #             pass

# # #     return None


# # # # ============================================================
# # # # CHECK ACTIVE / VALID TRACK
# # # # ============================================================

# # # def valid_track(track):

# # #     track_id = get_track_id(track)

# # #     if track_id is None:
# # #         return False

# # #     bbox = get_track_bbox(track)

# # #     if bbox is None:
# # #         return False

# # #     x1, y1, x2, y2 = bbox

# # #     if x2 <= x1:
# # #         return False

# # #     if y2 <= y1:
# # #         return False

# # #     return True


# # # # ============================================================
# # # # DRAW DETECTION TRACKS
# # # # ============================================================

# # # def draw_detection_tracks(
# # #     frame,
# # #     tracked_detections,
# # # ):

# # #     output = frame.copy()

# # #     if len(tracked_detections) == 0:
# # #         return output

# # #     boxes = tracked_detections.xyxy

# # #     confidences = (
# # #         tracked_detections.confidence
# # #     )

# # #     track_ids = (
# # #         tracked_detections.tracker_id
# # #     )

# # #     for i, box in enumerate(boxes):

# # #         x1, y1, x2, y2 = map(
# # #             int,
# # #             box,
# # #         )

# # #         confidence = float(
# # #             confidences[i]
# # #         )

# # #         track_id = int(
# # #             track_ids[i]
# # #         )

# # #         # Strong vs weak visualization
# # #         if confidence >= TRACKER_NEW_TRACK_THRESHOLD:

# # #             color = (0, 255, 0)
# # #             label_type = "STRONG"

# # #         else:

# # #             color = (255, 255, 0)
# # #             label_type = "WEAK"

# # #         cv2.rectangle(
# # #             output,
# # #             (x1, y1),
# # #             (x2, y2),
# # #             color,
# # #             2,
# # #         )

# # #         label = (
# # #             f"Snake | ID: {track_id} | "
# # #             f"{confidence:.3f} | {label_type}"
# # #         )

# # #         cv2.putText(
# # #             output,
# # #             label,
# # #             (
# # #                 x1,
# # #                 max(25, y1 - 10),
# # #             ),
# # #             cv2.FONT_HERSHEY_SIMPLEX,
# # #             0.6,
# # #             color,
# # #             2,
# # #         )

# # #     return output


# # # # ============================================================
# # # # DRAW PREDICTIONS
# # # # ============================================================

# # # def draw_prediction_tracks(
# # #     frame,
# # #     prediction_records,
# # # ):

# # #     output = frame

# # #     for record in prediction_records:

# # #         x1, y1, x2, y2 = map(
# # #             int,
# # #             record["bbox"],
# # #         )

# # #         track_id = int(
# # #             record["track_id"]
# # #         )

# # #         # Orange = Kalman prediction
# # #         color = (
# # #             0,
# # #             165,
# # #             255,
# # #         )

# # #         cv2.rectangle(
# # #             output,
# # #             (x1, y1),
# # #             (x2, y2),
# # #             color,
# # #             2,
# # #         )

# # #         label = (
# # #             f"Snake | ID: {track_id} | PRED"
# # #         )

# # #         cv2.putText(
# # #             output,
# # #             label,
# # #             (
# # #                 x1,
# # #                 max(25, y1 - 10),
# # #             ),
# # #             cv2.FONT_HERSHEY_SIMPLEX,
# # #             0.6,
# # #             color,
# # #             2,
# # #         )

# # #     return output


# # # # ============================================================
# # # # MAIN
# # # # ============================================================

# # # def main():

# # #     print(
# # #         "\n" + "=" * 70
# # #     )

# # #     print(
# # #         "STEP 5.3 — BYTETRACK INTEGRATION"
# # #     )

# # #     print(
# # #         "=" * 70
# # #     )

# # #     # --------------------------------------------------------
# # #     # VIDEO
# # #     # --------------------------------------------------------

# # #     if not VIDEO_PATH.exists():

# # #         raise FileNotFoundError(
# # #             f"Video not found:\n{VIDEO_PATH}"
# # #         )

# # #     print(
# # #         "\nVideo:"
# # #     )

# # #     print(
# # #         VIDEO_PATH
# # #     )

# # #     # --------------------------------------------------------
# # #     # DETECTOR
# # #     # --------------------------------------------------------

# # #     print(
# # #         "\nLoading RT-DETRv2..."
# # #     )

# # #     processor, model = (
# # #         load_detector()
# # #     )

# # #     print(
# # #         "Detector loaded successfully."
# # #     )

# # #     # --------------------------------------------------------
# # #     # VIDEO READER
# # #     # --------------------------------------------------------

# # #     reader = VideoFrameReader(
# # #         video_path=VIDEO_PATH,
# # #         frame_skip=1,
# # #     )

# # #     video_info = (
# # #         reader.get_video_info()
# # #     )

# # #     fps = video_info["fps"]

# # #     print(
# # #         "\nVIDEO"
# # #     )

# # #     print(
# # #         "-" * 70
# # #     )

# # #     print(
# # #         f"Resolution   : "
# # #         f"{video_info['width']} x "
# # #         f"{video_info['height']}"
# # #     )

# # #     print(
# # #         f"FPS          : "
# # #         f"{fps:.2f}"
# # #     )

# # #     print(
# # #         f"Total frames : "
# # #         f"{video_info['total_frames']}"
# # #     )

# # #     print(
# # #         f"Test limit   : "
# # #         f"{MAX_FRAMES}"
# # #     )

# # #     # --------------------------------------------------------
# # #     # TRACKER
# # #     # --------------------------------------------------------

# # #     tracker = create_tracker(
# # #         fps
# # #     )

# # #     print(
# # #         "\nByteTrack initialized."
# # #     )

# # #     print(
# # #         f"Tracker input threshold : "
# # #         f"{TRACKER_INPUT_THRESHOLD}"
# # #     )

# # #     print(
# # #         f"New-track threshold     : "
# # #         f"{TRACKER_NEW_TRACK_THRESHOLD}"
# # #     )

# # #     print(
# # #         f"Internal det threshold  : "
# # #         f"{tracker.det_thresh}"
# # #     )

# # #     print(
# # #         f"Lost track buffer       : "
# # #         f"{TRACKER_LOST_TRACK_BUFFER}"
# # #     )

# # #     print(
# # #         f"Matching threshold      : "
# # #         f"{TRACKER_MATCHING_THRESHOLD}"
# # #     )

# # #     print(
# # #         "Association mode        : IoU-only"
# # #     )

# # #     print(
# # #         "Prediction recording    : ENABLED"
# # #     )

# # #     # --------------------------------------------------------
# # #     # OUTPUT DIRECTORIES
# # #     # --------------------------------------------------------

# # #     OUTPUT_VIDEO_PATH.parent.mkdir(
# # #         parents=True,
# # #         exist_ok=True,
# # #     )

# # #     TRACK_DATA_PATH.parent.mkdir(
# # #         parents=True,
# # #         exist_ok=True,
# # #     )

# # #     # --------------------------------------------------------
# # #     # VIDEO WRITER
# # #     # --------------------------------------------------------

# # #     fourcc = cv2.VideoWriter_fourcc(
# # #         *"mp4v"
# # #     )

# # #     writer = cv2.VideoWriter(
# # #         str(OUTPUT_VIDEO_PATH),
# # #         fourcc,
# # #         fps,
# # #         (
# # #             video_info["width"],
# # #             video_info["height"],
# # #         ),
# # #     )

# # #     if not writer.isOpened():

# # #         reader.release()

# # #         raise RuntimeError(
# # #             "Could not create output video."
# # #         )

# # #     # --------------------------------------------------------
# # #     # DATA
# # #     # --------------------------------------------------------

# # #     track_data = []

# # #     frames_processed = 0

# # #     total_detections = 0

# # #     frames_with_detections = 0

# # #     frames_with_tracks = 0

# # #     total_tracked_observations = 0

# # #     total_detection_observations = 0

# # #     total_prediction_observations = 0

# # #     frames_with_predictions = 0

# # #     strong_detection_count = 0

# # #     weak_detection_count = 0

# # #     frames_with_strong_detections = 0

# # #     unique_track_ids = set()

# # #     # ========================================================
# # #     # PROCESS VIDEO
# # #     # ========================================================

# # #     print(
# # #         "\nPROCESSING VIDEO"
# # #     )

# # #     print(
# # #         "-" * 70
# # #     )

# # #     try:

# # #         for item in reader.frames():

# # #             frame = item["frame"]

# # #             frame_id = item["frame_id"]

# # #             timestamp = item["timestamp"]

# # #             # ------------------------------------------------
# # #             # DETECTION
# # #             # ------------------------------------------------

# # #             detections = detect_snakes(
# # #                 frame,
# # #                 processor,
# # #                 model,
# # #             )

# # #             detection_count = len(
# # #                 detections
# # #             )

# # #             total_detections += (
# # #                 detection_count
# # #             )

# # #             if detection_count > 0:

# # #                 frames_with_detections += 1

# # #             # ------------------------------------------------
# # #             # CLASSIFY STRONG / WEAK
# # #             # ------------------------------------------------

# # #             frame_has_strong = False

# # #             for detection in detections:

# # #                 confidence = float(
# # #                     detection["confidence"]
# # #                 )

# # #                 if (
# # #                     confidence
# # #                     >= TRACKER_NEW_TRACK_THRESHOLD
# # #                 ):

# # #                     strong_detection_count += 1
# # #                     frame_has_strong = True

# # #                 else:

# # #                     weak_detection_count += 1

# # #             if frame_has_strong:

# # #                 frames_with_strong_detections += 1

# # #             # ------------------------------------------------
# # #             # SUPERVISION DETECTIONS
# # #             # ------------------------------------------------

# # #             sv_detections = (
# # #                 detections_to_supervision(
# # #                     detections
# # #                 )
# # #             )

# # #             # ------------------------------------------------
# # #             # BYTE TRACK
# # #             # ------------------------------------------------

# # #             tracked_detections = (
# # #                 tracker.update_with_detections(
# # #                     sv_detections
# # #                 )
# # #             )

# # #             # ------------------------------------------------
# # #             # DETECTION-ASSOCIATED IDS
# # #             # ------------------------------------------------

# # #             detection_track_ids = set()

# # #             current_detection_records = []

# # #             if len(
# # #                 tracked_detections
# # #             ) > 0:

# # #                 boxes = (
# # #                     tracked_detections.xyxy
# # #                 )

# # #                 confidences = (
# # #                     tracked_detections.confidence
# # #                 )

# # #                 track_ids = (
# # #                     tracked_detections.tracker_id
# # #                 )

# # #                 for i, box in enumerate(
# # #                     boxes
# # #                 ):

# # #                     x1, y1, x2, y2 = map(
# # #                         float,
# # #                         box,
# # #                     )

# # #                     confidence = float(
# # #                         confidences[i]
# # #                     )

# # #                     track_id = int(
# # #                         track_ids[i]
# # #                     )

# # #                     detection_track_ids.add(
# # #                         track_id
# # #                     )

# # #                     unique_track_ids.add(
# # #                         track_id
# # #                     )

# # #                     center_x = (
# # #                         x1 + x2
# # #                     ) / 2.0

# # #                     center_y = (
# # #                         y1 + y2
# # #                     ) / 2.0

# # #                     observation_strength = (
# # #                         "strong"
# # #                         if confidence
# # #                         >= TRACKER_NEW_TRACK_THRESHOLD
# # #                         else "weak"
# # #                     )

# # #                     record = {

# # #                         "frame_id":
# # #                             frame_id,

# # #                         "timestamp":
# # #                             timestamp,

# # #                         "track_id":
# # #                             track_id,

# # #                         "bbox": [
# # #                             x1,
# # #                             y1,
# # #                             x2,
# # #                             y2,
# # #                         ],

# # #                         "center": [
# # #                             center_x,
# # #                             center_y,
# # #                         ],

# # #                         "confidence":
# # #                             confidence,

# # #                         "class_id":
# # #                             0,

# # #                         "class_name":
# # #                             "snake",

# # #                         "observation_type":
# # #                             "detection",

# # #                         "detection_strength":
# # #                             observation_strength,

# # #                         "track_state":
# # #                             "tracked",
# # #                     }

# # #                     current_detection_records.append(
# # #                         record
# # #                     )

# # #                     track_data.append(
# # #                         record
# # #                     )

# # #                     total_tracked_observations += 1

# # #                     total_detection_observations += 1

# # #             # ------------------------------------------------
# # #             # READ LOST TRACKS EVERY FRAME
# # #             # ------------------------------------------------

# # #             prediction_records = []

# # #             lost_tracks = getattr(
# # #                 tracker,
# # #                 "lost_tracks",
# # #                 [],
# # #             )

# # #             for track in lost_tracks:

# # #                 if not valid_track(track):
# # #                     continue

# # #                 track_id = get_track_id(
# # #                     track
# # #                 )

# # #                 # Don't duplicate a track that has
# # #                 # a current detector association.
# # #                 if track_id in detection_track_ids:
# # #                     continue

# # #                 bbox = get_track_bbox(
# # #                     track
# # #                 )

# # #                 if bbox is None:
# # #                     continue

# # #                 x1, y1, x2, y2 = bbox

# # #                 center_x = (
# # #                     x1 + x2
# # #                 ) / 2.0

# # #                 center_y = (
# # #                     y1 + y2
# # #                 ) / 2.0

# # #                 record = {

# # #                     "frame_id":
# # #                         frame_id,

# # #                     "timestamp":
# # #                         timestamp,

# # #                     "track_id":
# # #                         int(track_id),

# # #                     "bbox": [
# # #                         float(x1),
# # #                         float(y1),
# # #                         float(x2),
# # #                         float(y2),
# # #                     ],

# # #                     "center": [
# # #                         float(center_x),
# # #                         float(center_y),
# # #                     ],

# # #                     "confidence":
# # #                         None,

# # #                     "class_id":
# # #                         0,

# # #                     "class_name":
# # #                         "snake",

# # #                     "observation_type":
# # #                         "prediction",

# # #                     "detection_strength":
# # #                         None,

# # #                     "track_state":
# # #                         "lost",
# # #                 }

# # #                 prediction_records.append(
# # #                     record
# # #                 )

# # #                 track_data.append(
# # #                     record
# # #                 )

# # #                 unique_track_ids.add(
# # #                     int(track_id)
# # #                 )

# # #                 total_tracked_observations += 1

# # #                 total_prediction_observations += 1

# # #             if prediction_records:

# # #                 frames_with_predictions += 1

# # #             # ------------------------------------------------
# # #             # FRAME TRACK COUNT
# # #             # ------------------------------------------------

# # #             current_track_count = (
# # #                 len(current_detection_records)
# # #                 + len(prediction_records)
# # #             )

# # #             if current_track_count > 0:

# # #                 frames_with_tracks += 1

# # #             # ------------------------------------------------
# # #             # DRAW DETECTIONS
# # #             # ------------------------------------------------

# # #             output_frame = (
# # #                 draw_detection_tracks(
# # #                     frame,
# # #                     tracked_detections,
# # #                 )
# # #             )

# # #             # ------------------------------------------------
# # #             # DRAW PREDICTIONS
# # #             # ------------------------------------------------

# # #             output_frame = (
# # #                 draw_prediction_tracks(
# # #                     output_frame,
# # #                     prediction_records,
# # #                 )
# # #             )

# # #             # ------------------------------------------------
# # #             # FRAME STATUS
# # #             # ------------------------------------------------

# # #             cv2.putText(
# # #                 output_frame,
# # #                 f"Frame: {frame_id}",
# # #                 (20, 35),
# # #                 cv2.FONT_HERSHEY_SIMPLEX,
# # #                 0.8,
# # #                 (255, 255, 255),
# # #                 2,
# # #             )

# # #             cv2.putText(
# # #                 output_frame,
# # #                 f"Tracks: {current_track_count}",
# # #                 (20, 70),
# # #                 cv2.FONT_HERSHEY_SIMPLEX,
# # #                 0.8,
# # #                 (255, 255, 255),
# # #                 2,
# # #             )

# # #             if prediction_records:

# # #                 status = (
# # #                     "PREDICTION"
# # #                 )

# # #                 status_color = (
# # #                     0,
# # #                     165,
# # #                     255,
# # #                 )

# # #             elif current_detection_records:

# # #                 status = (
# # #                     "DETECTION"
# # #                 )

# # #                 status_color = (
# # #                     0,
# # #                     255,
# # #                     0,
# # #                 )

# # #             else:

# # #                 status = (
# # #                     "NO TRACK"
# # #                 )

# # #                 status_color = (
# # #                     255,
# # #                     255,
# # #                     255,
# # #                 )

# # #             cv2.putText(
# # #                 output_frame,
# # #                 status,
# # #                 (20, 105),
# # #                 cv2.FONT_HERSHEY_SIMPLEX,
# # #                 0.65,
# # #                 status_color,
# # #                 2,
# # #             )

# # #             # ------------------------------------------------
# # #             # WRITE
# # #             # ------------------------------------------------

# # #             writer.write(
# # #                 output_frame
# # #             )

# # #             frames_processed += 1

# # #             # ------------------------------------------------
# # #             # PROGRESS
# # #             # ------------------------------------------------

# # #             if (
# # #                 frames_processed % 10
# # #                 == 0
# # #             ):

# # #                 print(
# # #                     f"Frame "
# # #                     f"{frame_id:05d} | "
# # #                     f"Detections: "
# # #                     f"{detection_count:2d} | "
# # #                     f"Detection tracks: "
# # #                     f"{len(current_detection_records):2d} | "
# # #                     f"Predictions: "
# # #                     f"{len(prediction_records):2d}"
# # #                 )

# # #             # ------------------------------------------------
# # #             # TEST LIMIT
# # #             # ------------------------------------------------

# # #             if (
# # #                 MAX_FRAMES is not None
# # #                 and frames_processed
# # #                 >= MAX_FRAMES
# # #             ):
# # #                 break

# # #     finally:

# # #         reader.release()

# # #         writer.release()

# # #     # ========================================================
# # #     # SAVE JSON
# # #     # ========================================================

# # #     output_data = {

# # #         "video":
# # #             str(VIDEO_PATH),

# # #         "fps":
# # #             fps,

# # #         "frames_processed":
# # #             frames_processed,

# # #         "total_detections":
# # #             total_detections,

# # #         "frames_with_detections":
# # #             frames_with_detections,

# # #         "strong_detection_count":
# # #             strong_detection_count,

# # #         "weak_detection_count":
# # #             weak_detection_count,

# # #         "frames_with_strong_detections":
# # #             frames_with_strong_detections,

# # #         "frames_with_tracks":
# # #             frames_with_tracks,

# # #         "total_tracked_observations":
# # #             total_tracked_observations,

# # #         "total_detection_observations":
# # #             total_detection_observations,

# # #         "total_prediction_observations":
# # #             total_prediction_observations,

# # #         "frames_with_predictions":
# # #             frames_with_predictions,

# # #         "unique_track_ids":
# # #             sorted(
# # #                 unique_track_ids
# # #             ),

# # #         "track_count":
# # #             len(unique_track_ids),

# # #         "tracks":
# # #             track_data,
# # #     }

# # #     with open(
# # #         TRACK_DATA_PATH,
# # #         "w",
# # #         encoding="utf-8",
# # #     ) as f:

# # #         json.dump(
# # #             output_data,
# # #             f,
# # #             indent=2,
# # #         )

# # #     # ========================================================
# # #     # SUMMARY
# # #     # ========================================================

# # #     print(
# # #         "\n" + "=" * 70
# # #     )

# # #     print(
# # #         "STEP 5.3 COMPLETE"
# # #     )

# # #     print(
# # #         "=" * 70
# # #     )

# # #     print(
# # #         f"Frames processed           : "
# # #         f"{frames_processed}"
# # #     )

# # #     print(
# # #         f"Total detector candidates  : "
# # #         f"{total_detections}"
# # #     )

# # #     print(
# # #         f"Strong detections          : "
# # #         f"{strong_detection_count}"
# # #     )

# # #     print(
# # #         f"Weak tracker candidates    : "
# # #         f"{weak_detection_count}"
# # #     )

# # #     print(
# # #         f"Frames with detections     : "
# # #         f"{frames_with_detections}"
# # #     )

# # #     print(
# # #         f"Frames with strong detect. : "
# # #         f"{frames_with_strong_detections}"
# # #     )

# # #     print(
# # #         f"Frames with tracks         : "
# # #         f"{frames_with_tracks}"
# # #     )

# # #     print(
# # #         f"Tracked observations       : "
# # #         f"{total_tracked_observations}"
# # #     )

# # #     print(
# # #         f"Detection observations     : "
# # #         f"{total_detection_observations}"
# # #     )

# # #     print(
# # #         f"Prediction observations    : "
# # #         f"{total_prediction_observations}"
# # #     )

# # #     print(
# # #         f"Frames with predictions    : "
# # #         f"{frames_with_predictions}"
# # #     )

# # #     print(
# # #         f"Unique Track IDs           : "
# # #         f"{len(unique_track_ids)}"
# # #     )

# # #     print(
# # #         f"Track IDs                  : "
# # #         f"{sorted(unique_track_ids)}"
# # #     )

# # #     print(
# # #         "\nTracked video:"
# # #     )

# # #     print(
# # #         OUTPUT_VIDEO_PATH
# # #     )

# # #     print(
# # #         "\nTrack data:"
# # #     )

# # #     print(
# # #         TRACK_DATA_PATH
# # #     )

# # #     print(
# # #         "\nThreshold logic:"
# # #     )

# # #     print(
# # #         "  >= 0.055 = strong detection / can create track"
# # #     )

# # #     print(
# # #         "  0.03-0.055 = weak tracker candidate"
# # #     )

# # #     print(
# # #         "  prediction = ByteTrack Kalman state"
# # #     )


# # # # ============================================================
# # # # RUN
# # # # ============================================================

# # # if __name__ == "__main__":
# # #     main()