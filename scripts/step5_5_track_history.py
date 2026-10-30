# ============================================================
# STEP 5.5 — TRACK HISTORY & DATA PREPARATION
# ============================================================
#
# Input:
#   track_data_5_4_persistent.json
#
# Output:
#   track_data_5_5_history.json
#
# Purpose:
#   Prepare clean per-track-segment temporal data for the
#   upcoming Movement & Trajectory Analysis module.
#
# Rules:
#   - Tentative observations are excluded.
#   - Detection observations are preserved.
#   - Prediction observations are preserved.
#   - Predictions are NOT treated as detections.
#   - Duplicate records at the same frame are resolved with
#     detection taking priority over prediction.
#   - Track identity comes from the parent lifecycle segment.
#
# This module does NOT calculate:
#   - velocity
#   - acceleration
#   - direction
#   - trajectory
#   - Time-to-Intrusion
# ============================================================


from pathlib import Path
import json


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/content/drive/MyDrive/snake_detection/Snake_detection"
)

INPUT_PATH = (
    PROJECT_DIR
    / "dataset"
    / "tracking_output"
    / "track_data_5_4_persistent.json"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "dataset"
    / "tracking_output"
    / "track_data_5_5_history.json"
)

# Minimum number of observations required for
# downstream analysis.
MIN_OBSERVATIONS = 5


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"\nInput file not found:\n"
            f"{INPUT_PATH}\n\n"
            f"Run Step 5.4 first."
        )

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# 3. CLEAN ONE HISTORY RECORD
# ============================================================

def clean_history_record(
    item
):
    """
    Clean one observation from a Step 5.4 lifecycle history.

    Returns:
        Cleaned observation dictionary
        OR None if the observation should be excluded.
    """

    observation_type = str(
        item.get(
            "observation_type",
            "detection",
        )
    ).lower()

    # --------------------------------------------------------
    # Tentative observations are not part of confirmed history
    # --------------------------------------------------------

    if observation_type == "tentative":

        return None

    # --------------------------------------------------------
    # Only detection and prediction are valid here
    # --------------------------------------------------------

    if observation_type not in {
        "detection",
        "prediction",
    }:

        return None

    # --------------------------------------------------------
    # Frame
    # --------------------------------------------------------

    if "frame_id" not in item:

        raise ValueError(
            "History record is missing frame_id."
        )

    frame_id = int(
        item["frame_id"]
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if "timestamp" not in item:

        raise ValueError(
            f"Frame {frame_id} is missing timestamp."
        )

    timestamp = float(
        item["timestamp"]
    )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    bbox = item.get(
        "bbox"
    )

    if bbox is None:

        raise ValueError(
            f"Frame {frame_id} is missing bbox."
        )

    if len(bbox) != 4:

        raise ValueError(
            f"Invalid bbox at frame "
            f"{frame_id}: {bbox}"
        )

    clean_bbox = [
        float(value)
        for value in bbox
    ]

    # --------------------------------------------------------
    # Center
    # --------------------------------------------------------

    center = item.get(
        "center"
    )

    if center is None:

        raise ValueError(
            f"Frame {frame_id} is missing center."
        )

    if len(center) != 2:

        raise ValueError(
            f"Invalid center at frame "
            f"{frame_id}: {center}"
        )

    clean_center = [
        float(center[0]),
        float(center[1]),
    ]

    # --------------------------------------------------------
    # Confidence
    #
    # Detection:
    #     numeric confidence
    #
    # Prediction:
    #     None
    # --------------------------------------------------------

    raw_confidence = item.get(
        "confidence"
    )

    if raw_confidence is None:

        confidence = None

    else:

        confidence = float(
            raw_confidence
        )

    # --------------------------------------------------------
    # Validate confidence semantics
    # --------------------------------------------------------

    if (
        observation_type == "prediction"
        and confidence is not None
    ):

        raise ValueError(
            f"Prediction at frame "
            f"{frame_id} has non-None confidence."
        )

    if (
        observation_type == "detection"
        and confidence is None
    ):

        raise ValueError(
            f"Detection at frame "
            f"{frame_id} has None confidence."
        )

    # --------------------------------------------------------
    # Detection strength
    # --------------------------------------------------------

    detection_strength = item.get(
        "detection_strength"
    )

    # --------------------------------------------------------
    # Track state
    # --------------------------------------------------------

    track_state = item.get(
        "track_state",
        "UNKNOWN",
    )

    # --------------------------------------------------------
    # Class information
    # --------------------------------------------------------

    class_id = item.get(
        "class_id"
    )

    if class_id is not None:

        class_id = int(
            class_id
        )

    class_name = item.get(
        "class_name",
        "snake",
    )

    # --------------------------------------------------------
    # Clean record
    # --------------------------------------------------------

    return {

        "frame_id":
            frame_id,

        "timestamp":
            timestamp,

        "bbox":
            clean_bbox,

        "center":
            clean_center,

        "confidence":
            confidence,

        "class_id":
            class_id,

        "class_name":
            class_name,

        "observation_type":
            observation_type,

        "detection_strength":
            detection_strength,

        "track_state":
            track_state,
    }


# ============================================================
# 4. CLEAN ONE TRACK HISTORY
# ============================================================

def clean_track_history(
    history
):
    """
    Clean the history belonging to one lifecycle segment.

    Steps:
        1. Sort chronologically.
        2. Remove tentative/unsupported observations.
        3. Remove duplicate frame records.
        4. Detection takes priority over prediction.
        5. Restore chronological order.
    """

    # --------------------------------------------------------
    # Sort raw history chronologically
    # --------------------------------------------------------

    history = sorted(
        history,
        key=lambda item: int(
            item["frame_id"]
        )
    )

    # --------------------------------------------------------
    # Store one valid record per frame
    # --------------------------------------------------------

    candidates_by_frame = {}

    for item in history:

        cleaned_item = clean_history_record(
            item
        )

        # Tentative / unsupported
        if cleaned_item is None:

            continue

        frame_id = cleaned_item[
            "frame_id"
        ]

        # ----------------------------------------------------
        # First record for this frame
        # ----------------------------------------------------

        if frame_id not in candidates_by_frame:

            candidates_by_frame[
                frame_id
            ] = cleaned_item

            continue

        # ----------------------------------------------------
        # Duplicate frame
        #
        # Detection > prediction
        # ----------------------------------------------------

        existing = candidates_by_frame[
            frame_id
        ]

        existing_type = existing[
            "observation_type"
        ]

        new_type = cleaned_item[
            "observation_type"
        ]

        if (
            new_type == "detection"
            and existing_type == "prediction"
        ):

            candidates_by_frame[
                frame_id
            ] = cleaned_item

        # If both are the same observation type,
        # retain the first valid record.

    # --------------------------------------------------------
    # Restore chronological order
    # --------------------------------------------------------

    cleaned = list(
        candidates_by_frame.values()
    )

    cleaned.sort(
        key=lambda item: item[
            "frame_id"
        ]
    )

    return cleaned


# ============================================================
# 5. PREPARE ONE LIFECYCLE SEGMENT
# ============================================================

def prepare_track(
    track
):
    """
    Convert one Step 5.4 lifecycle segment into
    Step 5.5 analysis-ready data.
    """

    # --------------------------------------------------------
    # Identity comes from PARENT lifecycle segment
    # --------------------------------------------------------

    track_id = int(
        track["track_id"]
    )

    segment_id = int(
        track["segment_id"]
    )

    managed_track_id = str(
        track["managed_track_id"]
    )

    # --------------------------------------------------------
    # Clean history
    # --------------------------------------------------------

    history = clean_track_history(
        track.get(
            "history",
            []
        )
    )

    # --------------------------------------------------------
    # No valid history
    # --------------------------------------------------------

    if not history:

        return None

    # --------------------------------------------------------
    # Find actual detector observations
    # --------------------------------------------------------

    detection_history = [
        item
        for item in history
        if item["observation_type"]
        == "detection"
    ]

    # A lifecycle segment without an actual
    # detector observation is not valid.
    if not detection_history:

        return None

    # --------------------------------------------------------
    # First / last history record
    # --------------------------------------------------------

    first_history = history[0]

    last_history = history[-1]

    # --------------------------------------------------------
    # First / last REAL detection
    #
    # These are preferred for movement analysis.
    # --------------------------------------------------------

    first_detection = (
        detection_history[0]
    )

    last_detection = (
        detection_history[-1]
    )

    # --------------------------------------------------------
    # Positions
    # --------------------------------------------------------

    first_history_position = (
        first_history["center"]
    )

    last_history_position = (
        last_history["center"]
    )

    first_detection_position = (
        first_detection["center"]
    )

    last_detection_position = (
        last_detection["center"]
    )

    # --------------------------------------------------------
    # Complete history time range
    # --------------------------------------------------------

    start_time = float(
        first_history["timestamp"]
    )

    end_time = float(
        last_history["timestamp"]
    )

    duration_seconds = max(
        0.0,
        end_time - start_time,
    )

    # --------------------------------------------------------
    # Actual detection time range
    # --------------------------------------------------------

    detection_start_time = float(
        first_detection["timestamp"]
    )

    detection_end_time = float(
        last_detection["timestamp"]
    )

    detection_duration_seconds = max(
        0.0,
        detection_end_time
        - detection_start_time,
    )

    # --------------------------------------------------------
    # Observation counts
    # --------------------------------------------------------

    observation_count = len(
        history
    )

    detection_count = len(
        detection_history
    )

    prediction_count = sum(
        1
        for item in history
        if item["observation_type"]
        == "prediction"
    )

    # --------------------------------------------------------
    # Track quality
    # --------------------------------------------------------

    if (
        observation_count
        >= MIN_OBSERVATIONS
    ):

        history_status = (
            "sufficient_for_analysis"
        )

    else:

        history_status = (
            "insufficient_for_analysis"
        )

    # --------------------------------------------------------
    # Confidence statistics
    #
    # ONLY actual detections.
    # Predictions have confidence=None.
    # --------------------------------------------------------

    detection_confidences = [
        item["confidence"]
        for item in detection_history
        if item["confidence"] is not None
    ]

    if detection_confidences:

        average_confidence = (
            sum(detection_confidences)
            /
            len(detection_confidences)
        )

        max_confidence = max(
            detection_confidences
        )

        min_confidence = min(
            detection_confidences
        )

    else:

        average_confidence = 0.0
        max_confidence = 0.0
        min_confidence = 0.0

    # --------------------------------------------------------
    # Persistence statistics inherited from Step 5.4
    # --------------------------------------------------------

    total_missed_frames = int(
        track.get(
            "total_missed_frames",
            0,
        )
    )

    longest_missed_sequence = int(
        track.get(
            "longest_missed_sequence",
            0,
        )
    )

    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    return {

        # ----------------------------------------------------
        # Identity
        # ----------------------------------------------------

        "track_id":
            track_id,

        "segment_id":
            segment_id,

        "managed_track_id":
            managed_track_id,

        # ----------------------------------------------------
        # Lifecycle
        # ----------------------------------------------------

        "state":
            track.get(
                "state",
                "UNKNOWN",
            ),

        # ----------------------------------------------------
        # Complete history frame range
        # ----------------------------------------------------

        "first_frame":
            history[0]["frame_id"],

        "last_frame":
            history[-1]["frame_id"],

        # ----------------------------------------------------
        # REAL detection frame range
        # ----------------------------------------------------

        "first_detection_frame":
            first_detection["frame_id"],

        "last_detection_frame":
            last_detection["frame_id"],

        # ----------------------------------------------------
        # Complete history time range
        # ----------------------------------------------------

        "start_timestamp":
            start_time,

        "end_timestamp":
            end_time,

        "duration_seconds":
            duration_seconds,

        # ----------------------------------------------------
        # Detection time range
        # ----------------------------------------------------

        "detection_start_timestamp":
            detection_start_time,

        "detection_end_timestamp":
            detection_end_time,

        "detection_duration_seconds":
            detection_duration_seconds,

        # ----------------------------------------------------
        # Observation statistics
        # ----------------------------------------------------

        "observation_count":
            observation_count,

        "detection_count":
            detection_count,

        "prediction_count":
            prediction_count,

        # ----------------------------------------------------
        # Confidence statistics
        # ----------------------------------------------------

        "average_confidence":
            float(average_confidence),

        "max_confidence":
            float(max_confidence),

        "min_confidence":
            float(min_confidence),

        # ----------------------------------------------------
        # Persistence statistics
        # ----------------------------------------------------

        "total_missed_frames":
            total_missed_frames,

        "longest_missed_sequence":
            longest_missed_sequence,

        # ----------------------------------------------------
        # History quality
        # ----------------------------------------------------

        "history_status":
            history_status,

        # ----------------------------------------------------
        # Complete-history positions
        #
        # These may include a prediction endpoint.
        # ----------------------------------------------------

        "start_position":
            [
                float(
                    first_history_position[0]
                ),
                float(
                    first_history_position[1]
                ),
            ],

        "end_position":
            [
                float(
                    last_history_position[0]
                ),
                float(
                    last_history_position[1]
                ),
            ],

        # ----------------------------------------------------
        # REAL detector positions
        #
        # These should be preferred by downstream
        # movement analysis.
        # ----------------------------------------------------

        "first_detection_position":
            [
                float(
                    first_detection_position[0]
                ),
                float(
                    first_detection_position[1]
                ),
            ],

        "last_detection_position":
            [
                float(
                    last_detection_position[0]
                ),
                float(
                    last_detection_position[1]
                ),
            ],

        # ----------------------------------------------------
        # Complete cleaned history
        # ----------------------------------------------------

        "history":
            history,
    }


# ============================================================
# 6. PREPARE ALL LIFECYCLE SEGMENTS
# ============================================================

def prepare_all_tracks(
    data
):

    managed_tracks = data.get(
        "tracks",
        []
    )

    prepared_tracks = []

    # --------------------------------------------------------
    # Process every lifecycle segment
    # --------------------------------------------------------

    for track in managed_tracks:

        prepared = prepare_track(
            track
        )

        if prepared is not None:

            prepared_tracks.append(
                prepared
            )

    # --------------------------------------------------------
    # Chronological order
    # --------------------------------------------------------

    prepared_tracks.sort(
        key=lambda track: (
            track["first_frame"],
            track["track_id"],
            track["segment_id"],
        )
    )

    return prepared_tracks


# ============================================================
# 7. SAVE OUTPUT
# ============================================================

def save_output(
    original_data,
    tracks,
    tentative_count,
):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Aggregate counts
    # --------------------------------------------------------

    total_detections = sum(
        track["detection_count"]
        for track in tracks
    )

    total_predictions = sum(
        track["prediction_count"]
        for track in tracks
    )

    # --------------------------------------------------------
    # Output structure
    # --------------------------------------------------------

    output = {

        # ----------------------------------------------------
        # Video information
        # ----------------------------------------------------

        "video":
            original_data.get(
                "video"
            ),

        "fps":
            original_data.get(
                "fps"
            ),

        "frames_processed":
            original_data.get(
                "frames_processed"
            ),

        # ----------------------------------------------------
        # ByteTrack information
        # ----------------------------------------------------

        "byte_track_count":
            original_data.get(
                "byte_track_count",
                0,
            ),

        "byte_track_ids":
            original_data.get(
                "byte_track_ids",
                [],
            ),

        # ----------------------------------------------------
        # Lifecycle information
        # ----------------------------------------------------

        "lifecycle_segment_count":
            len(tracks),

        # ----------------------------------------------------
        # Observation counts
        # ----------------------------------------------------

        "detection_observation_count":
            total_detections,

        "prediction_observation_count":
            total_predictions,

        "tentative_observation_count":
            tentative_count,

        # ----------------------------------------------------
        # Analysis setting
        # ----------------------------------------------------

        "minimum_observations":
            MIN_OBSERVATIONS,

        # ----------------------------------------------------
        # Prepared tracks
        # ----------------------------------------------------

        "tracks":
            tracks,
    }

    # --------------------------------------------------------
    # Write JSON
    # --------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "STEP 5.5 — "
        "TRACK HISTORY & DATA PREPARATION"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_data()

    print(
        f"\nInput ByteTrack IDs : "
        f"{data.get('byte_track_count', 0)}"
    )

    print(
        f"Input segments      : "
        f"{data.get('lifecycle_segment_count', 0)}"
    )

    print(
        f"Minimum observations: "
        f"{MIN_OBSERVATIONS}"
    )

    # --------------------------------------------------------
    # Count raw observation types
    # --------------------------------------------------------

    raw_detection_count = 0

    raw_prediction_count = 0

    # Step 5.4 stores tentative observations
    # as a top-level integer count.

    raw_tentative_count = int(
        data.get(
            "tentative_observation_count",
            0,
        )
    )

    for track in data.get(
        "tracks",
        [],
    ):

        for item in track.get(
            "history",
            [],
        ):

            observation_type = str(
                item.get(
                    "observation_type",
                    "detection",
                )
            ).lower()

            if observation_type == "detection":

                raw_detection_count += 1

            elif observation_type == "prediction":

                raw_prediction_count += 1

    print(
        "\nINPUT OBSERVATION TYPES"
    )

    print(
        "-" * 70
    )

    print(
        f"Detection observations : "
        f"{raw_detection_count}"
    )

    print(
        f"Prediction observations: "
        f"{raw_prediction_count}"
    )

    print(
        f"Tentative observations : "
        f"{raw_tentative_count}"
    )

    print(
        "\nTentative observations "
        "will NOT be included in "
        "confirmed track history."
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    print(
        "\nPreparing clean track histories..."
    )

    tracks = prepare_all_tracks(
        data
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_output(
        data,
        tracks,
        raw_tentative_count,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    sufficient = 0

    insufficient = 0

    total_detections = 0

    total_predictions = 0

    for track in tracks:

        if (
            track["history_status"]
            == "sufficient_for_analysis"
        ):

            sufficient += 1

        else:

            insufficient += 1

        total_detections += (
            track["detection_count"]
        )

        total_predictions += (
            track["prediction_count"]
        )

    print(
        "\nTRACK HISTORY SUMMARY"
    )

    print(
        "-" * 70
    )

    print(
        f"Segments prepared      : "
        f"{len(tracks)}"
    )

    print(
        f"Sufficient segments    : "
        f"{sufficient}"
    )

    print(
        f"Short segments          : "
        f"{insufficient}"
    )

    print(
        f"Detection history      : "
        f"{total_detections}"
    )

    print(
        f"Prediction history     : "
        f"{total_predictions}"
    )

    print(
        f"Tentative excluded     : "
        f"{raw_tentative_count}"
    )

    # --------------------------------------------------------
    # Track details
    # --------------------------------------------------------

    print(
        "\nTRACK SEGMENTS"
    )

    print(
        "-" * 70
    )

    for track in tracks:

        print(
            f"ID {track['track_id']:3d} | "
            f"Segment {track['segment_id']:2d} | "
            f"Frames "
            f"{track['first_frame']} → "
            f"{track['last_frame']} | "
            f"Detection frames "
            f"{track['first_detection_frame']} → "
            f"{track['last_detection_frame']} | "
            f"Duration "
            f"{track['duration_seconds']:.2f}s | "
            f"Obs "
            f"{track['observation_count']:3d} | "
            f"Det "
            f"{track['detection_count']:3d} | "
            f"Pred "
            f"{track['prediction_count']:3d} | "
            f"{track['history_status']}"
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print(
        "\nOUTPUT"
    )

    print(
        "-" * 70
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "STEP 5.5 COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# 9. RUN
# ============================================================

if __name__ == "__main__":

    main()















# # ============================================================
# # STEP 5.5 — TRACK HISTORY & DATA PREPARATION
# # ============================================================

# """
# Purpose:
#     Prepare clean per-track-segment temporal data for the
#     upcoming Movement & Trajectory Analysis module.

# Input:
#     track_data_5_4_persistent.json

# Output:
#     track_data_5_5_history.json

# For every lifecycle segment, we store:
#     - ByteTrack ID
#     - lifecycle segment ID
#     - managed track ID
#     - frame number
#     - timestamp
#     - bounding box
#     - center point
#     - confidence
#     - observation type
#     - track state

# Observation types:
#     detection  -> confirmed detector observation
#     prediction -> Kalman prediction during detector gap

# Tentative observations are intentionally excluded because
# they were not treated as confirmed tracks in Step 5.4.

# This module does NOT calculate:
#     - velocity
#     - acceleration
#     - direction
#     - trajectory
#     - Time-to-Intrusion

# Short tracks are retained and marked as:
#     sufficient_for_analysis
#     OR
#     insufficient_for_analysis
# """

# from pathlib import Path
# import json


# # ============================================================
# # PROJECT PATH
# # ============================================================

# PROJECT_DIR = Path(
#     "/content/drive/MyDrive/snake_detection/Snake_detection"
# )


# # ============================================================
# # PATHS
# # ============================================================

# INPUT_PATH = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
#     / "track_data_5_4_persistent.json"
# )

# OUTPUT_PATH = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
#     / "track_data_5_5_history.json"
# )


# # ============================================================
# # MINIMUM TRACK LENGTH
# # ============================================================

# MIN_OBSERVATIONS = 5


# # ============================================================
# # LOAD DATA
# # ============================================================

# def load_data():

#     if not INPUT_PATH.exists():

#         raise FileNotFoundError(
#             f"\nInput file not found:\n"
#             f"{INPUT_PATH}\n\n"
#             f"Run Step 5.4 first."
#         )

#     with open(
#         INPUT_PATH,
#         "r",
#         encoding="utf-8",
#     ) as f:

#         return json.load(f)


# # ============================================================
# # CLEAN ONE OBSERVATION
# # ============================================================

# def clean_observation(item):
#     """
#     Convert one Step 5.4 observation into a clean
#     Step 5.5 history record.

#     Supported observation types:
#         detection
#         prediction

#     Tentative observations are rejected.
#     """

#     observation_type = str(
#         item.get(
#             "observation_type",
#             "detection"
#         )
#     ).lower()


#     # --------------------------------------------------------
#     # Tentative observations are NOT confirmed history
#     # --------------------------------------------------------

#     if observation_type == "tentative":
#         return None


#     # --------------------------------------------------------
#     # Only accept known confirmed observation types
#     # --------------------------------------------------------

#     if observation_type not in {
#         "detection",
#         "prediction",
#     }:
#         return None


#     # --------------------------------------------------------
#     # Frame
#     # --------------------------------------------------------

#     frame_id = int(
#         item["frame_id"]
#     )


#     # --------------------------------------------------------
#     # Timestamp
#     # --------------------------------------------------------

#     timestamp = float(
#         item["timestamp"]
#     )


#     # --------------------------------------------------------
#     # Bounding box
#     # --------------------------------------------------------

#     bbox = item["bbox"]

#     clean_bbox = [
#         float(value)
#         for value in bbox
#     ]


#     # --------------------------------------------------------
#     # Center
#     # --------------------------------------------------------

#     center = item["center"]

#     clean_center = [
#         float(center[0]),
#         float(center[1]),
#     ]


#     # --------------------------------------------------------
#     # Confidence
#     #
#     # Detection:
#     #     numeric confidence
#     #
#     # Prediction:
#     #     None
#     # --------------------------------------------------------

#     raw_confidence = item.get(
#         "confidence"
#     )

#     if raw_confidence is None:

#         confidence = None

#     else:

#         confidence = float(
#             raw_confidence
#         )


#     # --------------------------------------------------------
#     # Track ID
#     # --------------------------------------------------------

#     raw_track_id = item.get(
#         "track_id"
#     )

#     if raw_track_id is None:
#         return None

#     track_id = int(
#         raw_track_id
#     )


#     # --------------------------------------------------------
#     # Track state
#     # --------------------------------------------------------

#     track_state = item.get(
#         "track_state",
#         "UNKNOWN"
#     )


#     # --------------------------------------------------------
#     # Clean record
#     # --------------------------------------------------------

#     return {
#         "frame_id": frame_id,

#         "timestamp": timestamp,

#         "bbox": clean_bbox,

#         "center": clean_center,

#         "confidence": confidence,

#         "observation_type": (
#             observation_type
#         ),

#         "track_state": (
#             track_state
#         ),
#     }


# # ============================================================
# # CLEAN HISTORY
# # ============================================================

# def clean_track_history(history):
#     """
#     Sort and clean observations belonging to one
#     lifecycle segment.

#     Tentative observations are excluded.

#     Duplicate frame records are removed.
#     """

#     # --------------------------------------------------------
#     # Sort by frame
#     # --------------------------------------------------------

#     history = sorted(
#         history,
#         key=lambda item: item["frame_id"]
#     )


#     cleaned = []

#     previous_frame = None


#     # --------------------------------------------------------
#     # Process observations
#     # --------------------------------------------------------

#     for item in history:

#         cleaned_item = clean_observation(
#             item
#         )

#         # ----------------------------------------------------
#         # Ignore tentative/unsupported observations
#         # ----------------------------------------------------

#         if cleaned_item is None:
#             continue


#         frame_id = cleaned_item[
#             "frame_id"
#         ]


#         # ----------------------------------------------------
#         # Avoid duplicate frame records
#         # ----------------------------------------------------

#         if (
#             previous_frame is not None
#             and frame_id == previous_frame
#         ):
#             continue


#         previous_frame = frame_id


#         # ----------------------------------------------------
#         # Store cleaned record
#         # ----------------------------------------------------

#         cleaned.append(
#             cleaned_item
#         )


#     return cleaned


# # ============================================================
# # PREPARE ONE TRACK SEGMENT
# # ============================================================

# def prepare_track(track):
#     """
#     Convert one managed lifecycle segment into a clean
#     analysis-ready representation.
#     """

#     # --------------------------------------------------------
#     # Identity
#     # --------------------------------------------------------

#     track_id = int(
#         track["track_id"]
#     )

#     segment_id = int(
#         track["segment_id"]
#     )

#     managed_track_id = str(
#         track["managed_track_id"]
#     )


#     # --------------------------------------------------------
#     # Clean history
#     # --------------------------------------------------------

#     history = clean_track_history(
#         track.get(
#             "history",
#             []
#         )
#     )


#     # --------------------------------------------------------
#     # No valid observations
#     # --------------------------------------------------------

#     if not history:
#         return None


#     # --------------------------------------------------------
#     # Start/end information
#     # --------------------------------------------------------

#     first = history[0]

#     last = history[-1]


#     first_position = first[
#         "center"
#     ]

#     last_position = last[
#         "center"
#     ]


#     # --------------------------------------------------------
#     # Duration
#     # --------------------------------------------------------

#     start_time = float(
#         first["timestamp"]
#     )

#     end_time = float(
#         last["timestamp"]
#     )

#     duration_seconds = max(
#         0.0,
#         end_time - start_time
#     )


#     # --------------------------------------------------------
#     # Observation statistics
#     # --------------------------------------------------------

#     observation_count = len(
#         history
#     )

#     detection_count = sum(
#         1
#         for item in history
#         if item["observation_type"]
#         == "detection"
#     )

#     prediction_count = sum(
#         1
#         for item in history
#         if item["observation_type"]
#         == "prediction"
#     )


#     # --------------------------------------------------------
#     # Track quality
#     # --------------------------------------------------------

#     if (
#         observation_count
#         >= MIN_OBSERVATIONS
#     ):

#         history_status = (
#             "sufficient_for_analysis"
#         )

#     else:

#         history_status = (
#             "insufficient_for_analysis"
#         )


#     # --------------------------------------------------------
#     # Prepare output
#     # --------------------------------------------------------

#     return {

#         # ----------------------------------------------------
#         # Identity
#         # ----------------------------------------------------

#         "track_id": track_id,

#         "segment_id": segment_id,

#         "managed_track_id": (
#             managed_track_id
#         ),


#         # ----------------------------------------------------
#         # Lifecycle
#         # ----------------------------------------------------

#         "state": track.get(
#             "state",
#             "UNKNOWN"
#         ),


#         # ----------------------------------------------------
#         # Frame range
#         # ----------------------------------------------------

#         "first_frame": (
#             history[0]["frame_id"]
#         ),

#         "last_frame": (
#             history[-1]["frame_id"]
#         ),


#         # ----------------------------------------------------
#         # Time range
#         # ----------------------------------------------------

#         "start_timestamp": (
#             start_time
#         ),

#         "end_timestamp": (
#             end_time
#         ),

#         "duration_seconds": (
#             duration_seconds
#         ),


#         # ----------------------------------------------------
#         # Observation statistics
#         # ----------------------------------------------------

#         "observation_count": (
#             observation_count
#         ),

#         "detection_count": (
#             detection_count
#         ),

#         "prediction_count": (
#             prediction_count
#         ),


#         # ----------------------------------------------------
#         # Confidence statistics
#         #
#         # Only detector observations have confidence.
#         # ----------------------------------------------------

#         "average_confidence": float(
#             track.get(
#                 "average_confidence",
#                 0.0
#             )
#         ),

#         "max_confidence": float(
#             track.get(
#                 "max_confidence",
#                 0.0
#             )
#         ),

#         "min_confidence": float(
#             track.get(
#                 "min_confidence",
#                 0.0
#             )
#         ),


#         # ----------------------------------------------------
#         # Persistence statistics
#         # ----------------------------------------------------

#         "total_missed_frames": int(
#             track.get(
#                 "total_missed_frames",
#                 0
#             )
#         ),

#         "longest_missed_sequence": int(
#             track.get(
#                 "longest_missed_sequence",
#                 0
#             )
#         ),


#         # ----------------------------------------------------
#         # History quality
#         # ----------------------------------------------------

#         "history_status": (
#             history_status
#         ),


#         # ----------------------------------------------------
#         # Start/end positions
#         # ----------------------------------------------------

#         "start_position": [
#             float(first_position[0]),
#             float(first_position[1]),
#         ],

#         "end_position": [
#             float(last_position[0]),
#             float(last_position[1]),
#         ],


#         # ----------------------------------------------------
#         # Complete cleaned history
#         # ----------------------------------------------------

#         "history": history,
#     }


# # ============================================================
# # PREPARE ALL TRACK SEGMENTS
# # ============================================================

# def prepare_all_tracks(data):

#     managed_tracks = data.get(
#         "tracks",
#         []
#     )

#     prepared_tracks = []


#     # --------------------------------------------------------
#     # Prepare each lifecycle segment
#     # --------------------------------------------------------

#     for track in managed_tracks:

#         prepared = prepare_track(
#             track
#         )

#         if prepared is not None:

#             prepared_tracks.append(
#                 prepared
#             )


#     # --------------------------------------------------------
#     # Sort chronologically
#     # --------------------------------------------------------

#     prepared_tracks.sort(
#         key=lambda track: (
#             track["first_frame"],
#             track["track_id"],
#             track["segment_id"],
#         )
#     )


#     return prepared_tracks


# # ============================================================
# # SAVE OUTPUT
# # ============================================================

# def save_output(
#     original_data,
#     tracks,
#     tentative_count,
# ):

#     OUTPUT_PATH.parent.mkdir(
#         parents=True,
#         exist_ok=True
#     )


#     output = {

#         # ----------------------------------------------------
#         # Video information
#         # ----------------------------------------------------

#         "video": original_data.get(
#             "video"
#         ),

#         "fps": original_data.get(
#             "fps"
#         ),

#         "frames_processed": (
#             original_data.get(
#                 "frames_processed"
#             )
#         ),


#         # ----------------------------------------------------
#         # Track information
#         # ----------------------------------------------------

#         "byte_track_count": (
#             original_data.get(
#                 "byte_track_count"
#             )
#         ),

#         "lifecycle_segment_count": (
#             len(tracks)
#         ),


#         # ----------------------------------------------------
#         # Observation information
#         # ----------------------------------------------------

#         "detection_observation_count": (
#             sum(
#                 track["detection_count"]
#                 for track in tracks
#             )
#         ),

#         "prediction_observation_count": (
#             sum(
#                 track["prediction_count"]
#                 for track in tracks
#             )
#         ),

#         "tentative_observation_count": (
#             tentative_count
#         ),


#         # ----------------------------------------------------
#         # Analysis setting
#         # ----------------------------------------------------

#         "minimum_observations": (
#             MIN_OBSERVATIONS
#         ),


#         # ----------------------------------------------------
#         # Prepared tracks
#         # ----------------------------------------------------

#         "tracks": tracks,
#     }


#     with open(
#         OUTPUT_PATH,
#         "w",
#         encoding="utf-8"
#     ) as f:

#         json.dump(
#             output,
#             f,
#             indent=2
#         )


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     print("=" * 70)

#     print(
#         "STEP 5.5 — "
#         "TRACK HISTORY & DATA PREPARATION"
#     )

#     print("=" * 70)


#     # --------------------------------------------------------
#     # LOAD
#     # --------------------------------------------------------

#     data = load_data()


#     print(
#         f"\nInput ByteTrack IDs : "
#         f"{data.get('byte_track_count', 0)}"
#     )

#     print(
#         f"Input segments      : "
#         f"{data.get('lifecycle_segment_count', 0)}"
#     )

#     print(
#         f"Minimum observations: "
#         f"{MIN_OBSERVATIONS}"
#     )


#     # --------------------------------------------------------
#     # Count observation types in Step 5.4
#     # --------------------------------------------------------

#     tentative_count = 0
#     detection_count = 0
#     prediction_count = 0

#     for track in data.get(
#         "tracks",
#         []
#     ):

#         for item in track.get(
#             "history",
#             []
#         ):

#             observation_type = str(
#                 item.get(
#                     "observation_type",
#                     "detection"
#                 )
#             ).lower()


#             if observation_type == "tentative":

#                 tentative_count += 1

#             elif observation_type == "detection":

#                 detection_count += 1

#             elif observation_type == "prediction":

#                 prediction_count += 1


#     print(
#         "\nINPUT OBSERVATION TYPES"
#     )

#     print("-" * 70)

#     print(
#         f"Detection observations : "
#         f"{detection_count}"
#     )

#     print(
#         f"Prediction observations: "
#         f"{prediction_count}"
#     )

#     print(
#         f"Tentative observations : "
#         f"{tentative_count}"
#     )

#     print(
#         "\nTentative observations "
#         "will NOT be included in "
#         "confirmed track history."
#     )


#     # --------------------------------------------------------
#     # PREPARE
#     # --------------------------------------------------------

#     print(
#         "\nPreparing clean track histories..."
#     )

#     tracks = prepare_all_tracks(
#         data
#     )


#     # --------------------------------------------------------
#     # SAVE
#     # --------------------------------------------------------

#     save_output(
#         data,
#         tracks,
#         tentative_count
#     )


#     # --------------------------------------------------------
#     # SUMMARY
#     # --------------------------------------------------------

#     sufficient = 0
#     insufficient = 0

#     total_detections = 0
#     total_predictions = 0


#     for track in tracks:

#         if (
#             track["history_status"]
#             == "sufficient_for_analysis"
#         ):

#             sufficient += 1

#         else:

#             insufficient += 1


#         total_detections += (
#             track["detection_count"]
#         )

#         total_predictions += (
#             track["prediction_count"]
#         )


#     print(
#         "\nTRACK HISTORY SUMMARY"
#     )

#     print("-" * 70)

#     print(
#         f"Segments prepared      : "
#         f"{len(tracks)}"
#     )

#     print(
#         f"Sufficient segments    : "
#         f"{sufficient}"
#     )

#     print(
#         f"Short segments          : "
#         f"{insufficient}"
#     )

#     print(
#         f"Detection history      : "
#         f"{total_detections}"
#     )

#     print(
#         f"Prediction history     : "
#         f"{total_predictions}"
#     )

#     print(
#         f"Tentative excluded     : "
#         f"{tentative_count}"
#     )


#     # --------------------------------------------------------
#     # TRACK DETAILS
#     # --------------------------------------------------------

#     print(
#         "\nTRACK SEGMENTS"
#     )

#     print("-" * 70)


#     for track in tracks:

#         print(
#             f"ID {track['track_id']:3d} | "
#             f"Segment {track['segment_id']:2d} | "
#             f"Frames "
#             f"{track['first_frame']} → "
#             f"{track['last_frame']} | "
#             f"Duration "
#             f"{track['duration_seconds']:.2f}s | "
#             f"Obs "
#             f"{track['observation_count']:3d} | "
#             f"Det "
#             f"{track['detection_count']:3d} | "
#             f"Pred "
#             f"{track['prediction_count']:3d} | "
#             f"{track['history_status']}"
#         )


#     # --------------------------------------------------------
#     # OUTPUT
#     # --------------------------------------------------------

#     print(
#         "\nOUTPUT"
#     )

#     print("-" * 70)

#     print(
#         OUTPUT_PATH
#     )

#     print(
#         "\n" + "=" * 70
#     )

#     print(
#         "STEP 5.5 COMPLETE"
#     )

#     print("=" * 70)


# # ============================================================
# # RUN
# # ============================================================

# if __name__ == "__main__":

#     main()