# ============================================================
# STEP 5.6 — TRACKING VISUALIZATION & VALIDATION
# ============================================================

"""
Purpose:
    Visually validate the finalized tracking pipeline while also
    showing weak/tentative detector candidates before they become
    confirmed detections.

Visualization:
    DASHED YELLOW BOX = tentative / weak detector candidate
    SOLID TRACK BOX   = confirmed detector observation
    DASHED TRACK BOX  = Kalman prediction

Important:
    Tentative observations are VISUALIZATION ONLY.
    They are NOT added to Step 5.4, Step 5.5, or any downstream
    movement, spatial, intrusion, or alert analysis.

Priority on a frame:
    confirmed detection > tentative candidate > prediction

Trail:
    confirmed detector detections only.
"""

from pathlib import Path
import cv2
import json
from collections import defaultdict


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(
    "/content/drive/MyDrive/snake_detection/Snake_detection"
)


# ============================================================
# FILE PATHS — 24638 VIDEO
# ============================================================

VIDEO_PATH = (
    PROJECT_DIR
    / "snake_videos"
    / "24638-344119107_medium.mp4"
)

STEP_5_3_PATH = (
    PROJECT_DIR
    / "dataset"
    / "tracking_output"
    / "24638-344119107_medium"
    / "track_data_5_3_final.json"
)

STEP_5_5_PATH = (
    PROJECT_DIR
    / "dataset"
    / "tracking_output"
    / "24638-344119107_medium"
    / "track_data_5_5_history.json"
)

OUTPUT_VIDEO_PATH = (
    PROJECT_DIR
    / "dataset"
    / "tracking_output"
    / "24638-344119107_medium"
    / "tracking_visualization_5_6_tentative.mp4"
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_FRAMES = 400
TRAIL_LENGTH = 30


# ============================================================
# COLORS — OpenCV uses BGR
# ============================================================

TRACK_COLORS = [
    (0, 255, 0),        # green
    (0, 165, 255),      # orange
    (255, 0, 0),        # blue
    (0, 0, 255),        # red
    (255, 255, 0),      # cyan
    (255, 0, 255),      # magenta
    (0, 255, 255),      # yellow
    (128, 0, 128),      # purple
]

TENTATIVE_COLOR = (0, 255, 255)   # yellow


# ============================================================
# TRACK COLOR
# ============================================================

def color_for_track(track_id):

    return TRACK_COLORS[
        int(track_id) % len(TRACK_COLORS)
    ]


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"\nFile not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# CLIP POINT
# ============================================================

def clip_point(
    x,
    y,
    width,
    height
):

    x = max(
        0,
        min(
            int(x),
            width - 1
        )
    )

    y = max(
        0,
        min(
            int(y),
            height - 1
        )
    )

    return x, y


# ============================================================
# CLIP BBOX
# ============================================================

def clip_bbox(
    bbox,
    width,
    height
):

    x1, y1, x2, y2 = bbox

    x1 = max(
        0,
        min(
            int(x1),
            width - 1
        )
    )

    y1 = max(
        0,
        min(
            int(y1),
            height - 1
        )
    )

    x2 = max(
        0,
        min(
            int(x2),
            width - 1
        )
    )

    y2 = max(
        0,
        min(
            int(y2),
            height - 1
        )
    )

    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


# ============================================================
# DRAW DASHED RECTANGLE
# ============================================================

def draw_dashed_rectangle(
    frame,
    top_left,
    bottom_right,
    color,
    thickness=2,
    dash_length=12
):

    x1, y1 = top_left
    x2, y2 = bottom_right

    # Horizontal lines
    for x in range(
        x1,
        x2,
        dash_length * 2
    ):

        cv2.line(
            frame,
            (x, y1),
            (
                min(x + dash_length, x2),
                y1
            ),
            color,
            thickness
        )

        cv2.line(
            frame,
            (x, y2),
            (
                min(x + dash_length, x2),
                y2
            ),
            color,
            thickness
        )

    # Vertical lines
    for y in range(
        y1,
        y2,
        dash_length * 2
    ):

        cv2.line(
            frame,
            (x1, y),
            (
                x1,
                min(y + dash_length, y2)
            ),
            color,
            thickness
        )

        cv2.line(
            frame,
            (x2, y),
            (
                x2,
                min(y + dash_length, y2)
            ),
            color,
            thickness
        )


# ============================================================
# EXTRACT TENTATIVE OBSERVATIONS FROM STEP 5.3
# ============================================================

def extract_step_5_3_tentative_observations(data):

    """
    Extract actual tentative detector records from Step 5.3.

    Current Step 5.3 stores them in:
        data["tentative_records"]

    Tentative observations are visualization-only.
    """

    tentative_records = data.get(
        "tentative_records",
        None
    )

    # --------------------------------------------------------
    # Current Step 5.3 format
    # --------------------------------------------------------

    if isinstance(
        tentative_records,
        list
    ):

        observations = []

        for record in tentative_records:

            if not isinstance(
                record,
                dict
            ):
                continue

            if str(
                record.get(
                    "observation_type",
                    ""
                )
            ).lower() != "tentative":
                continue

            try:

                observations.append(
                    {
                        "frame_id": int(
                            record["frame_id"]
                        ),

                        "timestamp": float(
                            record["timestamp"]
                        ),

                        "bbox": [
                            float(v)
                            for v in record["bbox"]
                        ],

                        "center": [
                            float(
                                record["center"][0]
                            ),
                            float(
                                record["center"][1]
                            ),
                        ],

                        "confidence": (
                            float(
                                record["confidence"]
                            )
                            if record.get(
                                "confidence"
                            ) is not None
                            else None
                        ),

                        "observation_type":
                            "tentative",

                        "track_id": None,

                        "segment_id": None,
                    }
                )

            except (
                KeyError,
                TypeError,
                ValueError,
                IndexError
            ):

                continue

        return observations

    # --------------------------------------------------------
    # Compatibility fallback for older Step 5.3 JSON
    # --------------------------------------------------------

    observations = []

    for record in data.get(
        "tracks",
        []
    ):

        if not isinstance(
            record,
            dict
        ):
            continue

        if str(
            record.get(
                "observation_type",
                ""
            )
        ).lower() != "tentative":
            continue

        try:

            observations.append(
                {
                    "frame_id": int(
                        record["frame_id"]
                    ),

                    "timestamp": float(
                        record["timestamp"]
                    ),

                    "bbox": [
                        float(v)
                        for v in record["bbox"]
                    ],

                    "center": [
                        float(
                            record["center"][0]
                        ),
                        float(
                            record["center"][1]
                        ),
                    ],

                    "confidence": (
                        float(
                            record["confidence"]
                        )
                        if record.get(
                            "confidence"
                        ) is not None
                        else None
                    ),

                    "observation_type":
                        "tentative",

                    "track_id": None,

                    "segment_id": None,
                }
            )

        except (
            KeyError,
            TypeError,
            ValueError,
            IndexError
        ):

            continue

    return observations


# ============================================================
# EXTRACT STEP 5.5 OBSERVATIONS
# ============================================================

def extract_step_5_5_observations(data):

    """
    Extract confirmed detections and Kalman predictions
    from Step 5.5 lifecycle histories.
    """

    observations = []

    for segment in data.get(
        "tracks",
        []
    ):

        track_id = int(
            segment["track_id"]
        )

        segment_id = int(
            segment["segment_id"]
        )

        history = segment.get(
            "history",
            []
        )

        for record in history:

            observation_type = str(
                record.get(
                    "observation_type",
                    ""
                )
            ).lower()

            if observation_type not in {
                "detection",
                "prediction"
            }:
                continue

            observation = dict(
                record
            )

            observation[
                "track_id"
            ] = track_id

            observation[
                "segment_id"
            ] = segment_id

            observations.append(
                observation
            )

    return observations


# ============================================================
# ORGANIZE CONFIRMED OBSERVATIONS BY FRAME
# ============================================================

def organize_confirmed_by_frame(
    confirmed_observations
):

    frame_data = defaultdict(list)

    for record in confirmed_observations:

        frame_id = int(
            record["frame_id"]
        )

        frame_data[
            frame_id
        ].append(
            record
        )

    return frame_data


# ============================================================
# ORGANIZE TENTATIVE OBSERVATIONS BY FRAME
# ============================================================

def organize_tentative_by_frame(
    tentative_observations
):

    frame_data = defaultdict(list)

    for record in tentative_observations:

        frame_id = int(
            record["frame_id"]
        )

        frame_data[
            frame_id
        ].append(
            record
        )

    return frame_data


# ============================================================
# BUILD TRACK HISTORIES
# ============================================================

def build_track_histories(
    confirmed_observations
):

    histories = defaultdict(list)

    for record in confirmed_observations:

        key = (
            int(record["track_id"]),
            int(record["segment_id"])
        )

        histories[key].append(
            record
        )

    for key in histories:

        histories[key].sort(
            key=lambda r: int(
                r["frame_id"]
            )
        )

    return histories


# ============================================================
# DRAW DETECTION TRAIL
# ============================================================

def draw_detection_trail(
    frame,
    history,
    color
):

    """
    Trail uses ONLY actual detector detections.
    Predictions are intentionally excluded.
    """

    detections = [
        record
        for record in history
        if str(
            record.get(
                "observation_type",
                ""
            )
        ).lower() == "detection"
    ]

    if len(detections) < 2:
        return

    points = []

    height, width = frame.shape[:2]

    for record in detections[
        -TRAIL_LENGTH:
    ]:

        center = record.get(
            "center"
        )

        if center is None:
            continue

        x, y = center

        x, y = clip_point(
            x,
            y,
            width,
            height
        )

        points.append(
            (x, y)
        )

    for i in range(
        1,
        len(points)
    ):

        cv2.line(
            frame,
            points[i - 1],
            points[i],
            color,
            2
        )


# ============================================================
# DRAW TENTATIVE DETECTION
# ============================================================

def draw_tentative(
    frame,
    record
):

    """
    Draw a weak/tentative detector candidate.

    Tentative = dashed yellow box.
    No Track ID is assigned.
    """

    height, width = frame.shape[:2]

    x1, y1, x2, y2 = clip_bbox(
        record["bbox"],
        width,
        height
    )

    confidence = record.get(
        "confidence"
    )

    if confidence is not None:

        confidence = float(
            confidence
        )

    draw_dashed_rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        TENTATIVE_COLOR,
        thickness=2,
        dash_length=10
    )

    center = record.get(
        "center"
    )

    if center is not None:

        cx, cy = clip_point(
            center[0],
            center[1],
            width,
            height
        )

        cv2.circle(
            frame,
            (cx, cy),
            4,
            TENTATIVE_COLOR,
            -1
        )

    if confidence is not None:

        label = (
            "TENTATIVE | "
            f"CONF {confidence:.3f}"
        )

    else:

        label = "TENTATIVE"

    cv2.putText(
        frame,
        label,
        (
            max(x1, 5),
            max(y1 - 10, 25)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TENTATIVE_COLOR,
        2
    )


# ============================================================
# DRAW CONFIRMED DETECTION
# ============================================================

def draw_detection(
    frame,
    record,
    history
):

    height, width = frame.shape[:2]

    x1, y1, x2, y2 = clip_bbox(
        record["bbox"],
        width,
        height
    )

    track_id = int(
        record["track_id"]
    )

    confidence = float(
        record["confidence"]
    )

    color = color_for_track(
        track_id
    )

    # Solid bounding box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

    # Center
    center = record.get(
        "center"
    )

    if center is not None:

        cx, cy = clip_point(
            center[0],
            center[1],
            width,
            height
        )

        cv2.circle(
            frame,
            (cx, cy),
            5,
            color,
            -1
        )

    # Label
    label = (
        f"Snake | ID: {track_id} "
        f"| DET {confidence:.3f}"
    )

    cv2.putText(
        frame,
        label,
        (
            max(x1, 5),
            max(y1 - 10, 25)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )

    # Detection-only trail
    draw_detection_trail(
        frame,
        history,
        color
    )


# ============================================================
# DRAW KALMAN PREDICTION
# ============================================================

def draw_prediction(
    frame,
    record,
    history
):

    height, width = frame.shape[:2]

    x1, y1, x2, y2 = clip_bbox(
        record["bbox"],
        width,
        height
    )

    track_id = int(
        record["track_id"]
    )

    color = color_for_track(
        track_id
    )

    # Dashed prediction box
    draw_dashed_rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness=2,
        dash_length=12
    )

    # Center
    center = record.get(
        "center"
    )

    if center is not None:

        cx, cy = clip_point(
            center[0],
            center[1],
            width,
            height
        )

        cv2.circle(
            frame,
            (cx, cy),
            5,
            color,
            2
        )

    # Label
    label = (
        f"Snake | ID: {track_id} "
        f"| PRED"
    )

    cv2.putText(
        frame,
        label,
        (
            max(x1, 5),
            max(y1 - 10, 25)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )

    # Detection-only trail
    draw_detection_trail(
        frame,
        history,
        color
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("STEP 5.6 — TRACKING VISUALIZATION & VALIDATION")
    print("=" * 70)

    # ========================================================
    # LOAD STEP 5.3
    # ========================================================

    print("\nLoading Step 5.3...")

    step_5_3_data = load_json(
        STEP_5_3_PATH
    )

    tentative_observations = (
        extract_step_5_3_tentative_observations(
            step_5_3_data
        )
    )

    tentative_count = len(
        tentative_observations
    )

    print(
        "Step 5.3 tentative observations: "
        f"{tentative_count}"
    )

    # ========================================================
    # LOAD STEP 5.5
    # ========================================================

    print("\nLoading Step 5.5...")

    step_5_5_data = load_json(
        STEP_5_5_PATH
    )

    confirmed_observations = (
        extract_step_5_5_observations(
            step_5_5_data
        )
    )

    print(
        "Step 5.5 confirmed observations: "
        f"{len(confirmed_observations)}"
    )

    # ========================================================
    # ORGANIZE DATA
    # ========================================================

    confirmed_frame_data = (
        organize_confirmed_by_frame(
            confirmed_observations
        )
    )

    tentative_frame_data = (
        organize_tentative_by_frame(
            tentative_observations
        )
    )

    track_histories = (
        build_track_histories(
            confirmed_observations
        )
    )

    # ========================================================
    # TRACK INFORMATION
    # ========================================================

    unique_track_ids = sorted(
        set(
            int(record["track_id"])
            for record in confirmed_observations
        )
    )

    unique_segment_keys = sorted(
        set(
            (
                int(record["track_id"]),
                int(record["segment_id"])
            )
            for record in confirmed_observations
        )
    )

    detection_count = sum(
        1
        for record in confirmed_observations
        if str(
            record.get(
                "observation_type",
                ""
            )
        ).lower() == "detection"
    )

    prediction_count = sum(
        1
        for record in confirmed_observations
        if str(
            record.get(
                "observation_type",
                ""
            )
        ).lower() == "prediction"
    )

    print(
        "\nConfirmed ByteTrack IDs: "
        f"{unique_track_ids}"
    )

    print(
        "Confirmed lifecycle segments: "
        f"{unique_segment_keys}"
    )

    print(
        "Detection observations: "
        f"{detection_count}"
    )

    print(
        "Prediction observations: "
        f"{prediction_count}"
    )

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    print("\nOpening video...")

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"\nVideo not found:\n{VIDEO_PATH}"
        )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "\nCould not open video:\n"
            f"{VIDEO_PATH}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(
        f"Video FPS: {fps:.2f}"
    )

    print(
        "Video size: "
        f"{width} x {height}"
    )

    print(
        "Total video frames: "
        f"{total_frames}"
    )

    # ========================================================
    # OUTPUT VIDEO
    # ========================================================

    OUTPUT_VIDEO_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "\nCould not create output video:\n"
            f"{OUTPUT_VIDEO_PATH}"
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    frame_id = 0

    frames_with_observations = 0
    frames_with_confirmed = 0
    frames_with_tentative = 0

    total_tentative_visualizations = 0
    total_detection_visualizations = 0
    total_prediction_visualizations = 0

    maximum_simultaneous_tentative = 0
    maximum_simultaneous_confirmed = 0

    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_id >= MAX_FRAMES:
                break

            tentative = tentative_frame_data.get(
                frame_id,
                []
            )

            confirmed = confirmed_frame_data.get(
                frame_id,
                []
            )

            # ------------------------------------------------
            # STATISTICS
            # ------------------------------------------------

            if tentative:

                frames_with_tentative += 1

                maximum_simultaneous_tentative = max(
                    maximum_simultaneous_tentative,
                    len(tentative)
                )

            if confirmed:

                frames_with_confirmed += 1

                maximum_simultaneous_confirmed = max(
                    maximum_simultaneous_confirmed,
                    len(confirmed)
                )

            if tentative or confirmed:
                frames_with_observations += 1

            # ------------------------------------------------
            # DRAW TENTATIVE OBSERVATIONS
            # ------------------------------------------------

            # If a confirmed detection exists on this frame,
            # the confirmed box has priority and the tentative
            # candidate is not drawn over it.

            if not confirmed:

                for record in tentative:

                    draw_tentative(
                        frame,
                        record
                    )

                    total_tentative_visualizations += 1

            # ------------------------------------------------
            # DRAW CONFIRMED OBSERVATIONS
            # ------------------------------------------------

            for record in confirmed:

                key = (
                    int(record["track_id"]),
                    int(record["segment_id"])
                )

                history = track_histories[
                    key
                ]

                observation_type = str(
                    record[
                        "observation_type"
                    ]
                ).lower()

                if observation_type == "detection":

                    draw_detection(
                        frame,
                        record,
                        history
                    )

                    total_detection_visualizations += 1

                elif observation_type == "prediction":

                    draw_prediction(
                        frame,
                        record,
                        history
                    )

                    total_prediction_visualizations += 1

            # ------------------------------------------------
            # FRAME INFORMATION
            # ------------------------------------------------

            info = (
                f"Frame: {frame_id} | "
                f"Tentative: {len(tentative)} | "
                f"Confirmed: {len(confirmed)}"
            )

            cv2.putText(
                frame,
                info,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # LEGEND
            # ------------------------------------------------

            cv2.putText(
                frame,
                "DASHED YELLOW = TENTATIVE / WEAK DETECTION",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "SOLID = CONFIRMED DETECTION",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "DASHED TRACK COLOR = KALMAN PREDICTION",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "TRAIL = CONFIRMED DETECTIONS ONLY",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # WRITE FRAME
            # ------------------------------------------------

            writer.write(
                frame
            )

            frame_id += 1

    finally:

        cap.release()
        writer.release()

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "-" * 70)
    print("TRACKING VISUALIZATION COMPLETE")
    print("-" * 70)

    print(
        "Frames processed: "
        f"{frame_id}"
    )

    print(
        "Frames with tentative observations: "
        f"{frames_with_tentative}"
    )

    print(
        "Tentative visualizations: "
        f"{total_tentative_visualizations}"
    )

    print(
        "Maximum simultaneous tentative: "
        f"{maximum_simultaneous_tentative}"
    )

    print(
        "Frames with confirmed observations: "
        f"{frames_with_confirmed}"
    )

    print(
        "Detection visualizations: "
        f"{total_detection_visualizations}"
    )

    print(
        "Prediction visualizations: "
        f"{total_prediction_visualizations}"
    )

    print(
        "Maximum simultaneous confirmed: "
        f"{maximum_simultaneous_confirmed}"
    )

    print(
        "Step 5.3 tentative observation count: "
        f"{tentative_count}"
    )

    print("\nOutput video:")
    print(OUTPUT_VIDEO_PATH)

    print("\n" + "=" * 70)
    print("STEP 5.6 COMPLETE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
