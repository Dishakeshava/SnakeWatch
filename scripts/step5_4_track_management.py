# ============================================================
# STEP 5.4 — TRACK MANAGEMENT & PERSISTENCE
# ============================================================
#
# Input:
#   Step 5.3 track_data_5_3_final.json
#
# Output:
#   track_data_5_4_persistent.json
#
# Rules:
#   - Tentative observations are NOT treated as tracks.
#   - Predictions are NOT treated as detections.
#   - Only actual detector observations determine
#     lifecycle continuity.
#   - A track persists for at most MAX_MISSED_FRAMES.
#   - Predictions are recorded only inside the persistence
#     window.
#
# This module does NOT calculate:
#   - velocity
#   - acceleration
#   - trajectory
#   - temporal verification
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

# ============================================================
# EXACT Step 5.3 output for 24638 video
# ============================================================

TRACKING_OUTPUT_DIR = (
    PROJECT_DIR
    / "dataset"
    / "tracking_output"
    / "24638-344119107_medium"
)

TRACK_DATA_PATH = (
    TRACKING_OUTPUT_DIR
    / "track_data_5_3_final.json"
)

# ============================================================
# Step 5.4 output
# ============================================================

OUTPUT_PATH = (
    TRACKING_OUTPUT_DIR
    / "track_data_5_4_persistent.json"
)

# At ~29.97 FPS:
# 9 frames ≈ 0.30 seconds
MAX_MISSED_FRAMES = 9

FPS = 30.0


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_track_data():

    if not TRACK_DATA_PATH.exists():

        raise FileNotFoundError(
            f"\nTrack data not found:\n"
            f"{TRACK_DATA_PATH}\n\n"
            f"Run Step 5.3 first."
        )

    with open(
        TRACK_DATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# 3. CLASSIFY RECORDS
# ============================================================

def classify_records(data):

    detections = []
    predictions = []

    # ========================================================
    # STEP 5.3 RECORD EXTRACTION
    #
    # Supports:
    # 1. Current Step 5.3 format:
    #       tracks = [observation records]
    #
    # 2. Older format:
    #       tracks = [
    #           {
    #               "history": [...]
    #           }
    #       ]
    # ========================================================

    track_items = data.get("tracks", [])

    for item in track_items:

        # ----------------------------------------------------
        # CASE 1:
        # Track contains a "history" list
        # ----------------------------------------------------

        if isinstance(item, dict) and "history" in item:

            records = item.get("history", [])

        # ----------------------------------------------------
        # CASE 2:
        # Current Step 5.3:
        # item itself is an observation record
        # ----------------------------------------------------

        else:

            records = [item]

        # ----------------------------------------------------
        # Classify each observation
        # ----------------------------------------------------

        for record in records:

            if not isinstance(record, dict):
                continue

            observation_type = record.get(
                "observation_type"
            )

            # ----------------------------------------------
            # Actual detector observation
            # ----------------------------------------------

            if observation_type == "detection":

                if record.get("track_id") is not None:

                    detections.append(record)

            # ----------------------------------------------
            # Kalman prediction
            # ----------------------------------------------

            elif observation_type == "prediction":

                if record.get("track_id") is not None:

                    predictions.append(record)

    # ========================================================
    # TENTATIVE OBSERVATIONS
    # ========================================================

    tentative_records = data.get(
        "tentative_records",
        []
    )

    if isinstance(tentative_records, list):

        tentative_count = len(tentative_records)

    else:

        tentative_count = int(
            data.get(
                "tentative_observations",
                0
            )
        )

    # ========================================================
    # RETURN
    # ========================================================

    return (
        detections,
        predictions,
        tentative_count,
    )

# ============================================================
# 4. GROUP DETECTIONS BY TRACK ID
# ============================================================

def group_detections_by_track_id(
    detection_records,
):

    grouped = {}

    for record in detection_records:

        track_id = int(
            record["track_id"]
        )

        if track_id not in grouped:

            grouped[track_id] = []

        grouped[track_id].append(
            record
        )

    # Chronological order
    for track_id in grouped:

        grouped[track_id].sort(
            key=lambda r: int(
                r["frame_id"]
            )
        )

    return grouped


# ============================================================
# 5. GROUP PREDICTIONS BY TRACK ID
# ============================================================

def group_predictions_by_track_id(
    prediction_records,
):

    grouped = {}

    for record in prediction_records:

        track_id = int(
            record["track_id"]
        )

        if track_id not in grouped:

            grouped[track_id] = []

        grouped[track_id].append(
            record
        )

    # Chronological order
    for track_id in grouped:

        grouped[track_id].sort(
            key=lambda r: int(
                r["frame_id"]
            )
        )

    return grouped


# ============================================================
# 6. TRACK SEGMENT
# ============================================================

class TrackSegment:

    def __init__(
        self,
        track_id,
        segment_id,
        first_detection,
    ):

        # ----------------------------------------------------
        # Identity
        # ----------------------------------------------------

        self.track_id = int(
            track_id
        )

        self.segment_id = int(
            segment_id
        )

        self.managed_track_id = (
            f"{self.track_id}_{self.segment_id}"
        )

        # ----------------------------------------------------
        # Lifecycle
        # ----------------------------------------------------

        self.state = "NEW"

        # ----------------------------------------------------
        # Frame information
        # ----------------------------------------------------

        self.first_frame = int(
            first_detection["frame_id"]
        )

        self.last_frame = int(
            first_detection["frame_id"]
        )

        self.last_detection_frame = int(
            first_detection["frame_id"]
        )

        self.last_timestamp = float(
            first_detection["timestamp"]
        )

        # ----------------------------------------------------
        # Histories
        # ----------------------------------------------------

        self.detections = []

        self.predictions = []

        # ----------------------------------------------------
        # Missed-frame statistics
        # ----------------------------------------------------

        self.consecutive_missed_frames = 0

        self.total_missed_frames = 0

        self.longest_missed_sequence = 0

        # ----------------------------------------------------
        # First detection
        # ----------------------------------------------------

        self.add_detection(
            first_detection
        )

        self.state = "ACTIVE"


    # ========================================================
    # ADD DETECTION
    # ========================================================

    def add_detection(
        self,
        record,
    ):

        frame_id = int(
            record["frame_id"]
        )

        self.detections.append(
            record
        )

        # A real detector observation restores
        # the lifecycle.
        self.consecutive_missed_frames = 0

        self.last_detection_frame = (
            frame_id
        )

        self.last_frame = max(
            self.last_frame,
            frame_id,
        )

        self.last_timestamp = float(
            record["timestamp"]
        )

        self.state = "ACTIVE"


    # ========================================================
    # ADD PREDICTION
    # ========================================================

    def add_prediction(
        self,
        record,
    ):

        self.predictions.append(
            record
        )

        frame_id = int(
            record["frame_id"]
        )

        self.last_frame = max(
            self.last_frame,
            frame_id,
        )

        # IMPORTANT:
        #
        # Prediction does NOT reset missed-frame count.
        #
        # It is only an estimated position during a
        # detector gap.


    # ========================================================
    # MARK MISSED
    # ========================================================

    def mark_missed(self):

        self.consecutive_missed_frames += 1

        self.total_missed_frames += 1

        self.longest_missed_sequence = max(
            self.longest_missed_sequence,
            self.consecutive_missed_frames,
        )

        if (
            self.consecutive_missed_frames
            >= MAX_MISSED_FRAMES
        ):

            self.state = "REMOVED"

        else:

            self.state = "TEMPORARILY_LOST"


    # ========================================================
    # FINALIZE
    # ========================================================

    def finalize(
        self,
        fps,
    ):

        # If the segment has already reached the
        # persistence limit, it remains REMOVED.
        if (
            self.consecutive_missed_frames
            >= MAX_MISSED_FRAMES
        ):

            self.state = "REMOVED"

        elif self.state != "REMOVED":

            self.state = "COMPLETED"

        return self.to_dict(
            fps
        )


    # ========================================================
    # SERIALIZE
    # ========================================================

    def to_dict(
        self,
        fps,
    ):

        # Combine detection and prediction history.
        history = (
            list(self.detections)
            +
            list(self.predictions)
        )

        # Chronological order.
        history.sort(
            key=lambda r: int(
                r["frame_id"]
            )
        )

        duration_frames = (
            self.last_frame
            - self.first_frame
            + 1
        )

        return {

            "track_id":
                self.track_id,

            "segment_id":
                self.segment_id,

            "managed_track_id":
                self.managed_track_id,

            "state":
                self.state,

            "first_frame":
                self.first_frame,

            "last_frame":
                self.last_frame,

            "duration_frames":
                duration_frames,

            "duration_seconds":
                duration_frames / fps,

            "observation_count":
                len(self.detections),

            "prediction_count":
                len(self.predictions),

            "total_observations":
                len(history),

            "total_missed_frames":
                self.total_missed_frames,

            "longest_missed_sequence":
                self.longest_missed_sequence,

            "history":
                history,
        }


# ============================================================
# 7. BUILD LIFECYCLE SEGMENTS FOR ONE TRACK
# ============================================================

def build_segments_for_track(
    track_id,
    detection_records,
    prediction_records,
    fps,
    total_frames,
):

    segments = []

    current_segment = None

    segment_number = 0

    # --------------------------------------------------------
    # Prediction lookup by frame
    # --------------------------------------------------------

    predictions_by_frame = {}

    for record in prediction_records:

        frame_id = int(
            record["frame_id"]
        )

        predictions_by_frame[
            frame_id
        ] = record

    # --------------------------------------------------------
    # Process ACTUAL detections chronologically
    # --------------------------------------------------------

    for detection in detection_records:

        frame_id = int(
            detection["frame_id"]
        )

        # ----------------------------------------------------
        # FIRST DETECTION
        # ----------------------------------------------------

        if current_segment is None:

            segment_number += 1

            current_segment = TrackSegment(
                track_id=track_id,
                segment_id=segment_number,
                first_detection=detection,
            )

            continue

        previous_frame = (
            current_segment.last_detection_frame
        )

        # Number of completely missed frames
        # between two actual detections.
        missed_frames = (
            frame_id
            - previous_frame
            - 1
        )

        # ----------------------------------------------------
        # CASE 1:
        # Detection returns before persistence expires.
        # ----------------------------------------------------

        if missed_frames < MAX_MISSED_FRAMES:

            for gap_frame in range(
                previous_frame + 1,
                frame_id,
            ):

                # Record Kalman prediction if available.
                if gap_frame in predictions_by_frame:

                    current_segment.add_prediction(
                        predictions_by_frame[
                            gap_frame
                        ]
                    )

                # Every frame without a detector observation
                # counts as missed.
                current_segment.mark_missed()

            # Real detector observation returns.
            current_segment.add_detection(
                detection
            )

        # ----------------------------------------------------
        # CASE 2:
        # Persistence expired.
        #
        # Close old segment after MAX_MISSED_FRAMES.
        # Start a NEW lifecycle segment.
        # ----------------------------------------------------

        else:

            # Process only the allowed persistence window.
            for gap_frame in range(
                previous_frame + 1,
                previous_frame
                + MAX_MISSED_FRAMES
                + 1,
            ):

                if gap_frame in predictions_by_frame:

                    current_segment.add_prediction(
                        predictions_by_frame[
                            gap_frame
                        ]
                    )

                current_segment.mark_missed()

            # Finalize old lifecycle.
            segments.append(
                current_segment
            )

            # Start a new lifecycle.
            segment_number += 1

            current_segment = TrackSegment(
                track_id=track_id,
                segment_id=segment_number,
                first_detection=detection,
            )

    # --------------------------------------------------------
    # FINAL SEGMENT
    # --------------------------------------------------------

    if current_segment is not None:

        trailing_missed_frames = (
            total_frames
            - 1
            - current_segment.last_detection_frame
        )

        # Never process more than the persistence window.
        frames_to_process = min(
            trailing_missed_frames,
            MAX_MISSED_FRAMES,
        )

        # Process trailing missed frames.
        for gap_frame in range(
            current_segment.last_detection_frame + 1,
            current_segment.last_detection_frame
            + frames_to_process
            + 1,
        ):

            if gap_frame in predictions_by_frame:

                current_segment.add_prediction(
                    predictions_by_frame[
                        gap_frame
                    ]
                )

            current_segment.mark_missed()

        # ----------------------------------------------------
        # Final state
        # ----------------------------------------------------

        if (
            trailing_missed_frames
            >= MAX_MISSED_FRAMES
        ):

            current_segment.state = "REMOVED"

        else:

            current_segment.state = "COMPLETED"

        segments.append(
            current_segment
        )

    return segments


# ============================================================
# 8. BUILD ALL LIFECYCLE SEGMENTS
# ============================================================

def build_all_segments(
    detections_by_id,
    predictions_by_id,
    fps,
    total_frames,
):

    managed_tracks = []

    all_track_ids = sorted(
        set(
            list(detections_by_id.keys())
            +
            list(predictions_by_id.keys())
        )
    )

    for track_id in all_track_ids:

        detections = (
            detections_by_id.get(
                track_id,
                []
            )
        )

        predictions = (
            predictions_by_id.get(
                track_id,
                []
            )
        )

        # A lifecycle segment must have at least
        # one actual detector observation.
        if not detections:

            continue

        segments = build_segments_for_track(
            track_id=track_id,
            detection_records=detections,
            prediction_records=predictions,
            fps=fps,
            total_frames=total_frames,
        )

        # Keep OBJECTS here.
        #
        # Serialization happens only once later.
        managed_tracks.extend(
            segments
        )

    return managed_tracks


# ============================================================
# 9. SAVE RESULTS
# ============================================================

def save_results(
    original_data,
    managed_tracks,
    tentative_count,
):

    fps = float(
        original_data.get(
            "fps",
            30.0
        )
    )

    # Convert TrackSegment objects -> dictionaries
    # exactly ONCE.
    serialized_tracks = [
        segment.to_dict(fps)
        for segment in managed_tracks
    ]

    completed_count = sum(
        1
        for track in serialized_tracks
        if track["state"] == "COMPLETED"
    )

    removed_count = sum(
        1
        for track in serialized_tracks
        if track["state"] == "REMOVED"
    )

    unique_byte_track_ids = sorted(
        set(
            track["track_id"]
            for track in serialized_tracks
        )
    )

    output = {

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

        "byte_track_count":
            len(unique_byte_track_ids),

        "byte_track_ids":
            unique_byte_track_ids,

        "lifecycle_segment_count":
            len(serialized_tracks),

        "completed_segment_count":
            completed_count,

        "removed_segment_count":
            removed_count,

        "tentative_observation_count":
            tentative_count,

        "max_missed_frames":
            MAX_MISSED_FRAMES,

        "tracks":
            serialized_tracks,
    }

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

    return serialized_tracks


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "STEP 5.4 — "
        "TRACK MANAGEMENT & PERSISTENCE"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    data = load_track_data()

    global FPS

    FPS = float(
        data.get(
            "fps",
            30.0,
        )
    )

    frames_processed = int(
        data.get(
            "frames_processed",
            0,
        )
    )

    print(
        f"\nInput JSON : "
        f"{TRACK_DATA_PATH}"
    )

    print(
        f"Frames processed : "
        f"{frames_processed}"
    )

    print(
        f"FPS              : "
        f"{FPS:.2f}"
    )

    print(
        f"Max missed frames: "
        f"{MAX_MISSED_FRAMES}"
    )

    # --------------------------------------------------------
    # CLASSIFY
    # --------------------------------------------------------

    (
        detections,
        predictions,
        tentative_count,
    ) = classify_records(
        data
    )

    print(
        "\nINPUT OBSERVATIONS"
    )

    print(
        "-" * 70
    )

    print(
        f"Detection observations  : "
        f"{len(detections)}"
    )

    print(
        f"Prediction observations : "
        f"{len(predictions)}"
    )

    print(
        f"Tentative observations  : "
        f"{tentative_count}"
    )

    # --------------------------------------------------------
    # GROUP
    # --------------------------------------------------------

    detections_by_id = (
        group_detections_by_track_id(
            detections
        )
    )

    predictions_by_id = (
        group_predictions_by_track_id(
            predictions
        )
    )

    print(
        f"\nByteTrack IDs: "
        f"{sorted(detections_by_id.keys())}"
    )

    # --------------------------------------------------------
    # BUILD LIFECYCLES
    # --------------------------------------------------------

    print(
        "\nManaging track lifecycles..."
    )

    managed_tracks = build_all_segments(
        detections_by_id,
        predictions_by_id,
        FPS,
        frames_processed,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    serialized_tracks = save_results(
        data,
        managed_tracks,
        tentative_count,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    completed_count = sum(
        1
        for track in serialized_tracks
        if track["state"] == "COMPLETED"
    )

    removed_count = sum(
        1
        for track in serialized_tracks
        if track["state"] == "REMOVED"
    )

    unique_ids = sorted(
        set(
            track["track_id"]
            for track in serialized_tracks
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "STEP 5.4 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Lifecycle segments  : "
        f"{len(serialized_tracks)}"
    )

    print(
        f"Completed segments  : "
        f"{completed_count}"
    )

    print(
        f"Removed segments    : "
        f"{removed_count}"
    )

    print(
        f"Tentative records   : "
        f"{tentative_count}"
    )

    print(
        f"ByteTrack IDs       : "
        f"{unique_ids}"
    )

    # --------------------------------------------------------
    # INDIVIDUAL SEGMENTS
    # --------------------------------------------------------

    print(
        "\nTRACK SEGMENTS"
    )

    print(
        "-" * 70
    )

    for track in serialized_tracks:

        print(
            f"ID {track['track_id']:3d} "
            f"| Segment {track['segment_id']:2d} "
            f"| Frames "
            f"{track['first_frame']} → "
            f"{track['last_frame']} "
            f"| Detections: "
            f"{track['observation_count']:3d} "
            f"| Predictions: "
            f"{track['prediction_count']:3d} "
            f"| Missed: "
            f"{track['total_missed_frames']:3d} "
            f"| Longest gap: "
            f"{track['longest_missed_sequence']:3d} "
            f"| State: "
            f"{track['state']}"
        )

    # --------------------------------------------------------
    # OUTPUT
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


# ============================================================
# 11. RUN
# ============================================================

if __name__ == "__main__":

    main()








# # ============================================================
# # STEP 5.4 — TRACK MANAGEMENT & PERSISTENCE
# # ============================================================
# #
# # Input:
# #   Step 5.3 track_data_5_3_final.json
# #
# # Output:
# #   track_data_5_4_persistent.json
# #
# # Rules:
# #   - Tentative observations are NOT treated as tracks.
# #   - Predictions are NOT treated as detections.
# #   - Only actual detector observations determine
# #     lifecycle continuity.
# #   - A track persists for at most MAX_MISSED_FRAMES.
# #   - Predictions are recorded only inside the persistence
# #     window.
# #
# # This module does NOT calculate:
# #   - velocity
# #   - acceleration
# #   - trajectory
# #   - temporal verification
# #   - Time-to-Intrusion
# # ============================================================


# from pathlib import Path
# import json


# # ============================================================
# # 1. CONFIGURATION
# # ============================================================

# PROJECT_DIR = Path(
#     "/content/drive/MyDrive/snake_detection/Snake_detection"
# )

# # EXACT frozen Step 5.3 output
# TRACK_DATA_PATH = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
#     / "track_data_5_3_final.json"
# )

# # Step 5.4 output
# OUTPUT_PATH = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
#     / "track_data_5_4_persistent.json"
# )

# # At ~29.97 FPS:
# # 9 frames ≈ 0.30 seconds
# MAX_MISSED_FRAMES = 9

# FPS = 30.0


# # ============================================================
# # 2. LOAD DATA
# # ============================================================

# def load_track_data():

#     if not TRACK_DATA_PATH.exists():

#         raise FileNotFoundError(
#             f"\nTrack data not found:\n"
#             f"{TRACK_DATA_PATH}\n\n"
#             f"Run Step 5.3 first."
#         )

#     with open(
#         TRACK_DATA_PATH,
#         "r",
#         encoding="utf-8",
#     ) as f:

#         return json.load(f)


# # ============================================================
# # 3. CLASSIFY RECORDS
# # ============================================================

# def classify_records(data):

#     detections = []
#     predictions = []

#     # --------------------------------------------------------
#     # Confirmed track histories
#     # --------------------------------------------------------

#     for track in data.get(
#         "tracks",
#         []
#     ):

#         for record in track.get(
#             "history",
#             []
#         ):

#             observation_type = record.get(
#                 "observation_type",
#                 "detection",
#             )

#             # ------------------------------------------------
#             # Actual detector observation
#             # ------------------------------------------------

#             if observation_type == "detection":

#                 if record.get(
#                     "track_id"
#                 ) is not None:

#                     detections.append(
#                         record
#                     )

#             # ------------------------------------------------
#             # Kalman prediction
#             # ------------------------------------------------

#             elif observation_type == "prediction":

#                 if record.get(
#                     "track_id"
#                 ) is not None:

#                     predictions.append(
#                         record
#                     )

#     # --------------------------------------------------------
#     # Tentative observations
#     #
#     # In frozen Step 5.3 JSON this is an INTEGER COUNT,
#     # not a list of records.
#     # --------------------------------------------------------

#     tentative_count = int(
#         data.get(
#             "tentative_observations",
#             0
#         )
#     )

#     return (
#         detections,
#         predictions,
#         tentative_count,
#     )


# # ============================================================
# # 4. GROUP DETECTIONS BY TRACK ID
# # ============================================================

# def group_detections_by_track_id(
#     detection_records,
# ):

#     grouped = {}

#     for record in detection_records:

#         track_id = int(
#             record["track_id"]
#         )

#         if track_id not in grouped:

#             grouped[track_id] = []

#         grouped[track_id].append(
#             record
#         )

#     # Chronological order
#     for track_id in grouped:

#         grouped[track_id].sort(
#             key=lambda r: int(
#                 r["frame_id"]
#             )
#         )

#     return grouped


# # ============================================================
# # 5. GROUP PREDICTIONS BY TRACK ID
# # ============================================================

# def group_predictions_by_track_id(
#     prediction_records,
# ):

#     grouped = {}

#     for record in prediction_records:

#         track_id = int(
#             record["track_id"]
#         )

#         if track_id not in grouped:

#             grouped[track_id] = []

#         grouped[track_id].append(
#             record
#         )

#     # Chronological order
#     for track_id in grouped:

#         grouped[track_id].sort(
#             key=lambda r: int(
#                 r["frame_id"]
#             )
#         )

#     return grouped


# # ============================================================
# # 6. TRACK SEGMENT
# # ============================================================

# class TrackSegment:

#     def __init__(
#         self,
#         track_id,
#         segment_id,
#         first_detection,
#     ):

#         # ----------------------------------------------------
#         # Identity
#         # ----------------------------------------------------

#         self.track_id = int(
#             track_id
#         )

#         self.segment_id = int(
#             segment_id
#         )

#         self.managed_track_id = (
#             f"{self.track_id}_{self.segment_id}"
#         )

#         # ----------------------------------------------------
#         # Lifecycle
#         # ----------------------------------------------------

#         self.state = "NEW"

#         # ----------------------------------------------------
#         # Frame information
#         # ----------------------------------------------------

#         self.first_frame = int(
#             first_detection["frame_id"]
#         )

#         self.last_frame = int(
#             first_detection["frame_id"]
#         )

#         self.last_detection_frame = int(
#             first_detection["frame_id"]
#         )

#         self.last_timestamp = float(
#             first_detection["timestamp"]
#         )

#         # ----------------------------------------------------
#         # Histories
#         # ----------------------------------------------------

#         self.detections = []

#         self.predictions = []

#         # ----------------------------------------------------
#         # Missed-frame statistics
#         # ----------------------------------------------------

#         self.consecutive_missed_frames = 0

#         self.total_missed_frames = 0

#         self.longest_missed_sequence = 0

#         # ----------------------------------------------------
#         # First detection
#         # ----------------------------------------------------

#         self.add_detection(
#             first_detection
#         )

#         self.state = "ACTIVE"


#     # ========================================================
#     # ADD DETECTION
#     # ========================================================

#     def add_detection(
#         self,
#         record,
#     ):

#         frame_id = int(
#             record["frame_id"]
#         )

#         self.detections.append(
#             record
#         )

#         # A real detector observation restores
#         # the lifecycle.
#         self.consecutive_missed_frames = 0

#         self.last_detection_frame = (
#             frame_id
#         )

#         self.last_frame = max(
#             self.last_frame,
#             frame_id,
#         )

#         self.last_timestamp = float(
#             record["timestamp"]
#         )

#         self.state = "ACTIVE"


#     # ========================================================
#     # ADD PREDICTION
#     # ========================================================

#     def add_prediction(
#         self,
#         record,
#     ):

#         self.predictions.append(
#             record
#         )

#         frame_id = int(
#             record["frame_id"]
#         )

#         self.last_frame = max(
#             self.last_frame,
#             frame_id,
#         )

#         # IMPORTANT:
#         #
#         # Prediction does NOT reset missed-frame count.
#         #
#         # It is only an estimated position during a
#         # detector gap.


#     # ========================================================
#     # MARK MISSED
#     # ========================================================

#     def mark_missed(self):

#         self.consecutive_missed_frames += 1

#         self.total_missed_frames += 1

#         self.longest_missed_sequence = max(
#             self.longest_missed_sequence,
#             self.consecutive_missed_frames,
#         )

#         if (
#             self.consecutive_missed_frames
#             >= MAX_MISSED_FRAMES
#         ):

#             self.state = "REMOVED"

#         else:

#             self.state = "TEMPORARILY_LOST"


#     # ========================================================
#     # FINALIZE
#     # ========================================================

#     def finalize(
#         self,
#         fps,
#     ):

#         # If the segment has already reached the
#         # persistence limit, it remains REMOVED.
#         if (
#             self.consecutive_missed_frames
#             >= MAX_MISSED_FRAMES
#         ):

#             self.state = "REMOVED"

#         elif self.state != "REMOVED":

#             self.state = "COMPLETED"

#         return self.to_dict(
#             fps
#         )


#     # ========================================================
#     # SERIALIZE
#     # ========================================================

#     def to_dict(
#         self,
#         fps,
#     ):

#         # Combine detection and prediction history.
#         history = (
#             list(self.detections)
#             +
#             list(self.predictions)
#         )

#         # Chronological order.
#         history.sort(
#             key=lambda r: int(
#                 r["frame_id"]
#             )
#         )

#         duration_frames = (
#             self.last_frame
#             - self.first_frame
#             + 1
#         )

#         return {

#             "track_id":
#                 self.track_id,

#             "segment_id":
#                 self.segment_id,

#             "managed_track_id":
#                 self.managed_track_id,

#             "state":
#                 self.state,

#             "first_frame":
#                 self.first_frame,

#             "last_frame":
#                 self.last_frame,

#             "duration_frames":
#                 duration_frames,

#             "duration_seconds":
#                 duration_frames / fps,

#             "observation_count":
#                 len(self.detections),

#             "prediction_count":
#                 len(self.predictions),

#             "total_observations":
#                 len(history),

#             "total_missed_frames":
#                 self.total_missed_frames,

#             "longest_missed_sequence":
#                 self.longest_missed_sequence,

#             "history":
#                 history,
#         }


# # ============================================================
# # 7. BUILD LIFECYCLE SEGMENTS FOR ONE TRACK
# # ============================================================

# def build_segments_for_track(
#     track_id,
#     detection_records,
#     prediction_records,
#     fps,
#     total_frames,
# ):

#     segments = []

#     current_segment = None

#     segment_number = 0

#     # --------------------------------------------------------
#     # Prediction lookup by frame
#     # --------------------------------------------------------

#     predictions_by_frame = {}

#     for record in prediction_records:

#         frame_id = int(
#             record["frame_id"]
#         )

#         predictions_by_frame[
#             frame_id
#         ] = record

#     # --------------------------------------------------------
#     # Process ACTUAL detections chronologically
#     # --------------------------------------------------------

#     for detection in detection_records:

#         frame_id = int(
#             detection["frame_id"]
#         )

#         # ----------------------------------------------------
#         # FIRST DETECTION
#         # ----------------------------------------------------

#         if current_segment is None:

#             segment_number += 1

#             current_segment = TrackSegment(
#                 track_id=track_id,
#                 segment_id=segment_number,
#                 first_detection=detection,
#             )

#             continue

#         previous_frame = (
#             current_segment.last_detection_frame
#         )

#         # Number of completely missed frames
#         # between two actual detections.
#         missed_frames = (
#             frame_id
#             - previous_frame
#             - 1
#         )

#         # ----------------------------------------------------
#         # CASE 1:
#         # Detection returns before persistence expires.
#         # ----------------------------------------------------

#         if missed_frames < MAX_MISSED_FRAMES:

#             for gap_frame in range(
#                 previous_frame + 1,
#                 frame_id,
#             ):

#                 # Record Kalman prediction if available.
#                 if gap_frame in predictions_by_frame:

#                     current_segment.add_prediction(
#                         predictions_by_frame[
#                             gap_frame
#                         ]
#                     )

#                 # Every frame without a detector observation
#                 # counts as missed.
#                 current_segment.mark_missed()

#             # Real detector observation returns.
#             current_segment.add_detection(
#                 detection
#             )

#         # ----------------------------------------------------
#         # CASE 2:
#         # Persistence expired.
#         #
#         # Close old segment after MAX_MISSED_FRAMES.
#         # Start a NEW lifecycle segment.
#         # ----------------------------------------------------

#         else:

#             # Process only the allowed persistence window.
#             for gap_frame in range(
#                 previous_frame + 1,
#                 previous_frame
#                 + MAX_MISSED_FRAMES
#                 + 1,
#             ):

#                 if gap_frame in predictions_by_frame:

#                     current_segment.add_prediction(
#                         predictions_by_frame[
#                             gap_frame
#                         ]
#                     )

#                 current_segment.mark_missed()

#             # Finalize old lifecycle.
#             segments.append(
#                 current_segment
#             )

#             # Start a new lifecycle.
#             segment_number += 1

#             current_segment = TrackSegment(
#                 track_id=track_id,
#                 segment_id=segment_number,
#                 first_detection=detection,
#             )

#     # --------------------------------------------------------
#     # FINAL SEGMENT
#     # --------------------------------------------------------

#     if current_segment is not None:

#         trailing_missed_frames = (
#             total_frames
#             - 1
#             - current_segment.last_detection_frame
#         )

#         # Never process more than the persistence window.
#         frames_to_process = min(
#             trailing_missed_frames,
#             MAX_MISSED_FRAMES,
#         )

#         # Process trailing missed frames.
#         for gap_frame in range(
#             current_segment.last_detection_frame + 1,
#             current_segment.last_detection_frame
#             + frames_to_process
#             + 1,
#         ):

#             if gap_frame in predictions_by_frame:

#                 current_segment.add_prediction(
#                     predictions_by_frame[
#                         gap_frame
#                     ]
#                 )

#             current_segment.mark_missed()

#         # ----------------------------------------------------
#         # Final state
#         # ----------------------------------------------------

#         if (
#             trailing_missed_frames
#             >= MAX_MISSED_FRAMES
#         ):

#             current_segment.state = "REMOVED"

#         else:

#             current_segment.state = "COMPLETED"

#         segments.append(
#             current_segment
#         )

#     return segments


# # ============================================================
# # 8. BUILD ALL LIFECYCLE SEGMENTS
# # ============================================================

# def build_all_segments(
#     detections_by_id,
#     predictions_by_id,
#     fps,
#     total_frames,
# ):

#     managed_tracks = []

#     all_track_ids = sorted(
#         set(
#             list(detections_by_id.keys())
#             +
#             list(predictions_by_id.keys())
#         )
#     )

#     for track_id in all_track_ids:

#         detections = (
#             detections_by_id.get(
#                 track_id,
#                 []
#             )
#         )

#         predictions = (
#             predictions_by_id.get(
#                 track_id,
#                 []
#             )
#         )

#         # A lifecycle segment must have at least
#         # one actual detector observation.
#         if not detections:

#             continue

#         segments = build_segments_for_track(
#             track_id=track_id,
#             detection_records=detections,
#             prediction_records=predictions,
#             fps=fps,
#             total_frames=total_frames,
#         )

#         # Keep OBJECTS here.
#         #
#         # Serialization happens only once later.
#         managed_tracks.extend(
#             segments
#         )

#     return managed_tracks


# # ============================================================
# # 9. SAVE RESULTS
# # ============================================================

# def save_results(
#     original_data,
#     managed_tracks,
#     tentative_count,
# ):

#     fps = float(
#         original_data.get(
#             "fps",
#             30.0
#         )
#     )

#     # Convert TrackSegment objects -> dictionaries
#     # exactly ONCE.
#     serialized_tracks = [
#         segment.to_dict(fps)
#         for segment in managed_tracks
#     ]

#     completed_count = sum(
#         1
#         for track in serialized_tracks
#         if track["state"] == "COMPLETED"
#     )

#     removed_count = sum(
#         1
#         for track in serialized_tracks
#         if track["state"] == "REMOVED"
#     )

#     unique_byte_track_ids = sorted(
#         set(
#             track["track_id"]
#             for track in serialized_tracks
#         )
#     )

#     output = {

#         "video":
#             original_data.get(
#                 "video"
#             ),

#         "fps":
#             original_data.get(
#                 "fps"
#             ),

#         "frames_processed":
#             original_data.get(
#                 "frames_processed"
#             ),

#         "byte_track_count":
#             len(unique_byte_track_ids),

#         "byte_track_ids":
#             unique_byte_track_ids,

#         "lifecycle_segment_count":
#             len(serialized_tracks),

#         "completed_segment_count":
#             completed_count,

#         "removed_segment_count":
#             removed_count,

#         "tentative_observation_count":
#             tentative_count,

#         "max_missed_frames":
#             MAX_MISSED_FRAMES,

#         "tracks":
#             serialized_tracks,
#     }

#     with open(
#         OUTPUT_PATH,
#         "w",
#         encoding="utf-8",
#     ) as f:

#         json.dump(
#             output,
#             f,
#             indent=2,
#         )

#     return serialized_tracks


# # ============================================================
# # 10. MAIN
# # ============================================================

# def main():

#     print("=" * 70)

#     print(
#         "STEP 5.4 — "
#         "TRACK MANAGEMENT & PERSISTENCE"
#     )

#     print("=" * 70)

#     # --------------------------------------------------------
#     # LOAD
#     # --------------------------------------------------------

#     data = load_track_data()

#     global FPS

#     FPS = float(
#         data.get(
#             "fps",
#             30.0,
#         )
#     )

#     frames_processed = int(
#         data.get(
#             "frames_processed",
#             0,
#         )
#     )

#     print(
#         f"\nInput JSON : "
#         f"{TRACK_DATA_PATH}"
#     )

#     print(
#         f"Frames processed : "
#         f"{frames_processed}"
#     )

#     print(
#         f"FPS              : "
#         f"{FPS:.2f}"
#     )

#     print(
#         f"Max missed frames: "
#         f"{MAX_MISSED_FRAMES}"
#     )

#     # --------------------------------------------------------
#     # CLASSIFY
#     # --------------------------------------------------------

#     (
#         detections,
#         predictions,
#         tentative_count,
#     ) = classify_records(
#         data
#     )

#     print(
#         "\nINPUT OBSERVATIONS"
#     )

#     print(
#         "-" * 70
#     )

#     print(
#         f"Detection observations  : "
#         f"{len(detections)}"
#     )

#     print(
#         f"Prediction observations : "
#         f"{len(predictions)}"
#     )

#     print(
#         f"Tentative observations  : "
#         f"{tentative_count}"
#     )

#     # --------------------------------------------------------
#     # GROUP
#     # --------------------------------------------------------

#     detections_by_id = (
#         group_detections_by_track_id(
#             detections
#         )
#     )

#     predictions_by_id = (
#         group_predictions_by_track_id(
#             predictions
#         )
#     )

#     print(
#         f"\nByteTrack IDs: "
#         f"{sorted(detections_by_id.keys())}"
#     )

#     # --------------------------------------------------------
#     # BUILD LIFECYCLES
#     # --------------------------------------------------------

#     print(
#         "\nManaging track lifecycles..."
#     )

#     managed_tracks = build_all_segments(
#         detections_by_id,
#         predictions_by_id,
#         FPS,
#         frames_processed,
#     )

#     # --------------------------------------------------------
#     # SAVE
#     # --------------------------------------------------------

#     serialized_tracks = save_results(
#         data,
#         managed_tracks,
#         tentative_count,
#     )

#     # --------------------------------------------------------
#     # SUMMARY
#     # --------------------------------------------------------

#     completed_count = sum(
#         1
#         for track in serialized_tracks
#         if track["state"] == "COMPLETED"
#     )

#     removed_count = sum(
#         1
#         for track in serialized_tracks
#         if track["state"] == "REMOVED"
#     )

#     unique_ids = sorted(
#         set(
#             track["track_id"]
#             for track in serialized_tracks
#         )
#     )

#     print(
#         "\n" + "=" * 70
#     )

#     print(
#         "STEP 5.4 COMPLETE"
#     )

#     print(
#         "=" * 70
#     )

#     print(
#         f"Lifecycle segments  : "
#         f"{len(serialized_tracks)}"
#     )

#     print(
#         f"Completed segments  : "
#         f"{completed_count}"
#     )

#     print(
#         f"Removed segments    : "
#         f"{removed_count}"
#     )

#     print(
#         f"Tentative records   : "
#         f"{tentative_count}"
#     )

#     print(
#         f"ByteTrack IDs       : "
#         f"{unique_ids}"
#     )

#     # --------------------------------------------------------
#     # INDIVIDUAL SEGMENTS
#     # --------------------------------------------------------

#     print(
#         "\nTRACK SEGMENTS"
#     )

#     print(
#         "-" * 70
#     )

#     for track in serialized_tracks:

#         print(
#             f"ID {track['track_id']:3d} "
#             f"| Segment {track['segment_id']:2d} "
#             f"| Frames "
#             f"{track['first_frame']} → "
#             f"{track['last_frame']} "
#             f"| Detections: "
#             f"{track['observation_count']:3d} "
#             f"| Predictions: "
#             f"{track['prediction_count']:3d} "
#             f"| Missed: "
#             f"{track['total_missed_frames']:3d} "
#             f"| Longest gap: "
#             f"{track['longest_missed_sequence']:3d} "
#             f"| State: "
#             f"{track['state']}"
#         )

#     # --------------------------------------------------------
#     # OUTPUT
#     # --------------------------------------------------------

#     print(
#         "\nOUTPUT"
#     )

#     print(
#         "-" * 70
#     )

#     print(
#         OUTPUT_PATH
#     )

#     print(
#         "\n" + "=" * 70
#     )


# # ============================================================
# # 11. RUN
# # ============================================================

# if __name__ == "__main__":

#     main()














# # ============================================================
# # STEP 5.4 — TRACK MANAGEMENT & PERSISTENCE
# # ============================================================

# """
# Purpose:
#     Maintain application-level lifecycle of ByteTrack tracks.

# Input:
#     Step 5.3 track_data_5_3.json

# Step 5.3 observation types:

#     detection
#         Actual RT-DETR detection associated with ByteTrack.

#     prediction
#         Kalman-predicted position during detector miss.

#     tentative
#         Weak candidate that has not become a confirmed
#         ByteTrack track.

# Important:
#     - Tentative observations are NOT treated as tracks.
#     - Predictions are NOT treated as detections.
#     - Only detector observations determine lifecycle continuity.

# Lifecycle:

#     NEW
#       ↓
#     ACTIVE
#       ↓
#     TEMPORARILY_LOST
#       ↓
#     ACTIVE again

#     OR

#     REMOVED

# If the same ByteTrack ID reappears after exceeding
# MAX_MISSED_FRAMES, a new lifecycle segment is created.
# """


# from pathlib import Path
# import json


# # ============================================================
# # CONFIGURATION
# # ============================================================

# PROJECT_DIR = Path(
#     "/content/drive/MyDrive/snake_detection/Snake_detection"
# )


# TRACK_DATA_PATH = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
#     / "track_data_5_3.json"
# )


# OUTPUT_PATH = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
#     / "track_data_5_4_persistent.json"
# )


# # At ~29.97 FPS:
# #
# # 9 frames ≈ 0.30 seconds
# #
# # If an actual detector observation is absent for
# # 9 or more frames, the current lifecycle segment
# # is considered removed.

# MAX_MISSED_FRAMES = 9


# # ============================================================
# # LOAD TRACK DATA
# # ============================================================

# def load_track_data():

#     if not TRACK_DATA_PATH.exists():

#         raise FileNotFoundError(
#             f"\nTrack data not found:\n"
#             f"{TRACK_DATA_PATH}\n\n"
#             f"Run Step 5.3 first."
#         )

#     with open(
#         TRACK_DATA_PATH,
#         "r",
#         encoding="utf-8",
#     ) as f:

#         return json.load(f)


# # ============================================================
# # FILTER OBSERVATIONS
# # ============================================================

# def classify_records(track_records):
#     """
#     Separate Step 5.3 records into:

#         detection
#         prediction
#         tentative

#     Tentative records do not have a ByteTrack ID and therefore
#     cannot participate in ByteTrack lifecycle management.
#     """

#     detections = []
#     predictions = []
#     tentative = []

#     for record in track_records:

#         observation_type = record.get(
#             "observation_type",
#             "detection",
#         )

#         if observation_type == "detection":

#             # Only valid ByteTrack detections
#             # participate in lifecycle management.
#             if record.get("track_id") is not None:
#                 detections.append(record)

#         elif observation_type == "prediction":

#             # Predictions belong to an existing
#             # ByteTrack track.
#             if record.get("track_id") is not None:
#                 predictions.append(record)

#         elif observation_type == "tentative":

#             # Tentative candidates are intentionally
#             # not treated as confirmed tracks.
#             tentative.append(record)

#     return (
#         detections,
#         predictions,
#         tentative,
#     )


# # ============================================================
# # GROUP DETECTIONS BY BYTE TRACK ID
# # ============================================================

# def group_detections_by_track_id(
#     detection_records,
# ):

#     grouped = {}

#     for record in detection_records:

#         track_id = int(
#             record["track_id"]
#         )

#         if track_id not in grouped:

#             grouped[track_id] = []

#         grouped[track_id].append(
#             record
#         )

#     # Chronological order
#     for track_id in grouped:

#         grouped[track_id].sort(
#             key=lambda r: r["frame_id"]
#         )

#     return grouped


# # ============================================================
# # GROUP PREDICTIONS BY BYTE TRACK ID
# # ============================================================

# def group_predictions_by_track_id(
#     prediction_records,
# ):

#     grouped = {}

#     for record in prediction_records:

#         track_id = int(
#             record["track_id"]
#         )

#         if track_id not in grouped:

#             grouped[track_id] = []

#         grouped[track_id].append(
#             record
#         )

#     for track_id in grouped:

#         grouped[track_id].sort(
#             key=lambda r: r["frame_id"]
#         )

#     return grouped


# # ============================================================
# # TRACK SEGMENT
# # ============================================================

# class TrackSegment:

#     def __init__(
#         self,
#         track_id,
#         segment_id,
#         first_detection,
#     ):

#         # ----------------------------------------------------
#         # Identity
#         # ----------------------------------------------------

#         self.track_id = int(
#             track_id
#         )

#         self.segment_id = int(
#             segment_id
#         )

#         self.managed_track_id = (
#             f"{self.track_id}_{self.segment_id}"
#         )

#         # ----------------------------------------------------
#         # Lifecycle
#         # ----------------------------------------------------

#         self.state = "NEW"

#         # ----------------------------------------------------
#         # Frame information
#         # ----------------------------------------------------

#         self.first_frame = int(
#             first_detection["frame_id"]
#         )

#         self.last_frame = int(
#             first_detection["frame_id"]
#         )

#         self.last_timestamp = float(
#             first_detection["timestamp"]
#         )

#         # ----------------------------------------------------
#         # Detection statistics
#         # ----------------------------------------------------

#         self.observation_count = 0

#         self.total_confidence = 0.0

#         self.max_confidence = 0.0

#         self.min_confidence = 1.0

#         # ----------------------------------------------------
#         # Prediction statistics
#         # ----------------------------------------------------

#         self.prediction_count = 0

#         # ----------------------------------------------------
#         # Missed-frame statistics
#         # ----------------------------------------------------

#         self.consecutive_missed_frames = 0

#         self.total_missed_frames = 0

#         self.longest_missed_sequence = 0

#         # ----------------------------------------------------
#         # Current information
#         # ----------------------------------------------------

#         self.current_bbox = None

#         self.current_center = None

#         self.current_confidence = None

#         # ----------------------------------------------------
#         # Complete history
#         # ----------------------------------------------------

#         self.history = []


#     # ========================================================
#     # ADD DETECTION
#     # ========================================================

#     def add_detection(
#         self,
#         record,
#     ):

#         frame_id = int(
#             record["frame_id"]
#         )

#         timestamp = float(
#             record["timestamp"]
#         )

#         confidence = float(
#             record["confidence"]
#         )

#         bbox = list(
#             record["bbox"]
#         )

#         # ----------------------------------------------------
#         # Detection means the object was observed.
#         # ----------------------------------------------------

#         self.consecutive_missed_frames = 0

#         self.state = "ACTIVE"

#         self.last_frame = frame_id

#         self.last_timestamp = timestamp

#         # ----------------------------------------------------
#         # Statistics
#         # ----------------------------------------------------

#         self.observation_count += 1

#         self.total_confidence += confidence

#         self.max_confidence = max(
#             self.max_confidence,
#             confidence,
#         )

#         self.min_confidence = min(
#             self.min_confidence,
#             confidence,
#         )

#         # ----------------------------------------------------
#         # Current bounding box
#         # ----------------------------------------------------

#         self.current_bbox = bbox

#         x1, y1, x2, y2 = bbox

#         center_x = (
#             float(x1) + float(x2)
#         ) / 2.0

#         center_y = (
#             float(y1) + float(y2)
#         ) / 2.0

#         self.current_center = [
#             center_x,
#             center_y,
#         ]

#         self.current_confidence = (
#             confidence
#         )

#         # ----------------------------------------------------
#         # History
#         # ----------------------------------------------------

#         self.history.append(
#             {
#                 "frame_id": frame_id,
#                 "timestamp": timestamp,
#                 "bbox": bbox,
#                 "center": self.current_center,
#                 "confidence": confidence,
#                 "class_id": record.get(
#                     "class_id",
#                     0,
#                 ),
#                 "class_name": record.get(
#                     "class_name",
#                     "snake",
#                 ),
#                 "observation_type":
#                     "detection",
#                 "detection_strength":
#                     record.get(
#                         "detection_strength"
#                     ),
#                 "track_state":
#                     "active",
#             }
#         )


#     # ========================================================
#     # ADD PREDICTION
#     # ========================================================

#     def add_prediction(
#         self,
#         record,
#     ):

#         frame_id = int(
#             record["frame_id"]
#         )

#         timestamp = float(
#             record["timestamp"]
#         )

#         bbox = list(
#             record["bbox"]
#         )

#         x1, y1, x2, y2 = bbox

#         center_x = (
#             float(x1) + float(x2)
#         ) / 2.0

#         center_y = (
#             float(y1) + float(y2)
#         ) / 2.0

#         self.prediction_count += 1

#         # ----------------------------------------------------
#         # IMPORTANT:
#         #
#         # A prediction does NOT reset missed frames.
#         # It means the detector did not observe the object.
#         # ----------------------------------------------------

#         self.history.append(
#             {
#                 "frame_id": frame_id,
#                 "timestamp": timestamp,
#                 "bbox": bbox,
#                 "center": [
#                     center_x,
#                     center_y,
#                 ],
#                 "confidence": None,
#                 "class_id": record.get(
#                     "class_id",
#                     0,
#                 ),
#                 "class_name": record.get(
#                     "class_name",
#                     "snake",
#                 ),
#                 "observation_type":
#                     "prediction",
#                 "detection_strength":
#                     None,
#                 "track_state":
#                     "lost",
#             }
#         )


#     # ========================================================
#     # MARK MISSED
#     # ========================================================

#     def mark_missed(
#         self,
#     ):

#         self.consecutive_missed_frames += 1

#         self.total_missed_frames += 1

#         self.longest_missed_sequence = max(
#             self.longest_missed_sequence,
#             self.consecutive_missed_frames,
#         )

#         if (
#             self.consecutive_missed_frames
#             >= MAX_MISSED_FRAMES
#         ):

#             self.state = "REMOVED"

#         else:

#             self.state = "TEMPORARILY_LOST"


#     # ========================================================
#     # FINALIZE
#     # ========================================================

#     def finalize(
#         self,
#         fps,
#     ):

#         # ----------------------------------------------------
#         # Duration
#         # ----------------------------------------------------

#         duration_frames = (
#             self.last_frame
#             - self.first_frame
#             + 1
#         )

#         duration_seconds = (
#             duration_frames / fps
#         )

#         # ----------------------------------------------------
#         # Average confidence
#         # ----------------------------------------------------

#         if self.observation_count > 0:

#             average_confidence = (
#                 self.total_confidence
#                 /
#                 self.observation_count
#             )

#         else:

#             average_confidence = 0.0

#         # ----------------------------------------------------
#         # Output
#         # ----------------------------------------------------

#         return {

#             # Identity
#             "track_id":
#                 self.track_id,

#             "segment_id":
#                 self.segment_id,

#             "managed_track_id":
#                 self.managed_track_id,

#             # Lifecycle
#             "state":
#                 self.state,

#             # Frames
#             "first_frame":
#                 self.first_frame,

#             "last_frame":
#                 self.last_frame,

#             "duration_frames":
#                 duration_frames,

#             "duration_seconds":
#                 duration_seconds,

#             # Detection statistics
#             "observation_count":
#                 self.observation_count,

#             "average_confidence":
#                 average_confidence,

#             "max_confidence":
#                 self.max_confidence,

#             "min_confidence":
#                 self.min_confidence,

#             # Prediction statistics
#             "prediction_count":
#                 self.prediction_count,

#             # Missed statistics
#             "total_missed_frames":
#                 self.total_missed_frames,

#             "longest_missed_sequence":
#                 self.longest_missed_sequence,

#             # Current information
#             "current_bbox":
#                 self.current_bbox,

#             "current_center":
#                 self.current_center,

#             "current_confidence":
#                 self.current_confidence,

#             # Complete history
#             "history":
#                 self.history,
#         }


# # ============================================================
# # BUILD ONE SEGMENT
# # ============================================================

# def build_segments_for_track(
#     track_id,
#     detection_records,
#     prediction_records,
#     fps,
#     total_frames,
# ):

#     segments = []

#     current_segment = None

#     segment_number = 0

#     # --------------------------------------------------------
#     # Predictions indexed by frame
#     # --------------------------------------------------------

#     predictions_by_frame = {}

#     for record in prediction_records:

#         frame_id = int(
#             record["frame_id"]
#         )

#         predictions_by_frame[
#             frame_id
#         ] = record

#     # --------------------------------------------------------
#     # Process ONLY actual detections
#     # chronologically.
#     # --------------------------------------------------------

#     for detection in detection_records:

#         frame_id = int(
#             detection["frame_id"]
#         )

#         # ----------------------------------------------------
#         # First detection
#         # ----------------------------------------------------

#         if current_segment is None:

#             segment_number += 1

#             current_segment = TrackSegment(
#                 track_id=track_id,
#                 segment_id=segment_number,
#                 first_detection=detection,
#             )

#             current_segment.add_detection(
#                 detection
#             )

#             continue

#         # ----------------------------------------------------
#         # Gap since previous REAL detection
#         # ----------------------------------------------------

#         previous_frame = (
#             current_segment.last_frame
#         )

#         missed_frames = (
#             frame_id
#             - previous_frame
#             - 1
#         )

#         # ----------------------------------------------------
#         # Case 1:
#         #
#         # Gap is within persistence window.
#         # Same lifecycle segment.
#         # ----------------------------------------------------

#         if missed_frames < MAX_MISSED_FRAMES:

#             # -----------------------------------------------
#             # Add prediction records that exist in the gap.
#             #
#             # These are useful history, but do NOT reset
#             # missed-frame counters.
#             # -----------------------------------------------

#             for gap_frame in range(
#                 previous_frame + 1,
#                 frame_id,
#             ):

#                 if (
#                     gap_frame
#                     in predictions_by_frame
#                 ):

#                     current_segment.add_prediction(
#                         predictions_by_frame[
#                             gap_frame
#                         ]
#                     )

#                 current_segment.mark_missed()

#             # -----------------------------------------------
#             # Add actual detection
#             # -----------------------------------------------

#             current_segment.add_detection(
#                 detection
#             )

#         # ----------------------------------------------------
#         # Case 2:
#         #
#         # Gap reaches/exceeds persistence threshold.
#         #
#         # Close current segment.
#         # Start a new one.
#         # ----------------------------------------------------

#         else:

#             # -----------------------------------------------
#             # Account for persistence window.
#             # -----------------------------------------------

#             for gap_frame in range(
#                 previous_frame + 1,
#                 previous_frame
#                 + MAX_MISSED_FRAMES
#                 + 1,
#             ):

#                 if (
#                     gap_frame
#                     in predictions_by_frame
#                 ):

#                     current_segment.add_prediction(
#                         predictions_by_frame[
#                             gap_frame
#                         ]
#                     )

#                 current_segment.mark_missed()

#             # -----------------------------------------------
#             # Save old segment
#             # -----------------------------------------------

#             segments.append(
#                 current_segment.finalize(
#                     fps
#                 )
#             )

#             # -----------------------------------------------
#             # New segment
#             # -----------------------------------------------

#             segment_number += 1

#             current_segment = TrackSegment(
#                 track_id=track_id,
#                 segment_id=segment_number,
#                 first_detection=detection,
#             )

#             current_segment.add_detection(
#                 detection
#             )

#     # --------------------------------------------------------
#     # Final segment
#     # --------------------------------------------------------

#     if current_segment is not None:

#         trailing_missed_frames = (
#             total_frames
#             - 1
#             - current_segment.last_frame
#         )

#         # -----------------------------------------------
#         # Add trailing predictions
#         # -----------------------------------------------

#         frames_to_process = min(
#             trailing_missed_frames,
#             MAX_MISSED_FRAMES,
#         )

#         for gap_frame in range(
#             current_segment.last_frame + 1,
#             current_segment.last_frame
#             + frames_to_process
#             + 1,
#         ):

#             if (
#                 gap_frame
#                 in predictions_by_frame
#             ):

#                 current_segment.add_prediction(
#                     predictions_by_frame[
#                         gap_frame
#                     ]
#                 )

#             current_segment.mark_missed()

#         # -----------------------------------------------
#         # Final state
#         # -----------------------------------------------

#         if (
#             trailing_missed_frames
#             >= MAX_MISSED_FRAMES
#         ):

#             current_segment.state = (
#                 "REMOVED"
#             )

#         else:

#             current_segment.state = (
#                 "COMPLETED"
#             )

#         segments.append(
#             current_segment.finalize(
#                 fps
#             )
#         )

#     return segments


# # ============================================================
# # MANAGE ALL TRACKS
# # ============================================================

# def manage_tracks(
#     data,
# ):

#     fps = float(
#         data.get(
#             "fps",
#             30.0,
#         )
#     )

#     total_frames = int(
#         data.get(
#             "frames_processed",
#             0,
#         )
#     )

#     track_records = data.get(
#         "tracks",
#         [],
#     )

#     # --------------------------------------------------------
#     # Separate observation types
#     # --------------------------------------------------------

#     (
#         detection_records,
#         prediction_records,
#         tentative_records,
#     ) = classify_records(
#         track_records
#     )

#     print(
#         "\nOBSERVATION TYPES"
#     )

#     print(
#         "-" * 70
#     )

#     print(
#         f"Detection observations : "
#         f"{len(detection_records)}"
#     )

#     print(
#         f"Prediction observations: "
#         f"{len(prediction_records)}"
#     )

#     print(
#         f"Tentative observations : "
#         f"{len(tentative_records)}"
#     )

#     print(
#         "\nTentative observations "
#         "will NOT be treated as confirmed tracks."
#     )

#     # --------------------------------------------------------
#     # Group by ByteTrack ID
#     # --------------------------------------------------------

#     grouped_detections = (
#         group_detections_by_track_id(
#             detection_records
#         )
#     )

#     grouped_predictions = (
#         group_predictions_by_track_id(
#             prediction_records
#         )
#     )

#     # --------------------------------------------------------
#     # Build lifecycle segments
#     # --------------------------------------------------------

#     managed_tracks = []

#     for track_id in sorted(
#         grouped_detections.keys()
#     ):

#         detections = (
#             grouped_detections[
#                 track_id
#             ]
#         )

#         predictions = (
#             grouped_predictions.get(
#                 track_id,
#                 [],
#             )
#         )

#         segments = (
#             build_segments_for_track(
#                 track_id=track_id,
#                 detection_records=detections,
#                 prediction_records=predictions,
#                 fps=fps,
#                 total_frames=total_frames,
#             )
#         )

#         managed_tracks.extend(
#             segments
#         )

#     # --------------------------------------------------------
#     # Sort chronologically
#     # --------------------------------------------------------

#     managed_tracks.sort(
#         key=lambda track: (
#             track["first_frame"],
#             track["track_id"],
#             track["segment_id"],
#         )
#     )

#     return (
#         managed_tracks,
#         len(tentative_records),
#     )


# # ============================================================
# # SAVE RESULTS
# # ============================================================

# def save_results(
#     original_data,
#     managed_tracks,
#     tentative_count,
# ):

#     OUTPUT_PATH.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     # --------------------------------------------------------
#     # ByteTrack IDs
#     # --------------------------------------------------------

#     unique_byte_track_ids = sorted(
#         set(
#             track["track_id"]
#             for track in managed_tracks
#         )
#     )

#     # --------------------------------------------------------
#     # States
#     # --------------------------------------------------------

#     completed_count = sum(
#         1
#         for track in managed_tracks
#         if track["state"] == "COMPLETED"
#     )

#     removed_count = sum(
#         1
#         for track in managed_tracks
#         if track["state"] == "REMOVED"
#     )

#     # --------------------------------------------------------
#     # Output
#     # --------------------------------------------------------

#     output = {

#         "video":
#             original_data.get(
#                 "video"
#             ),

#         "fps":
#             original_data.get(
#                 "fps"
#             ),

#         "frames_processed":
#             original_data.get(
#                 "frames_processed"
#             ),

#         "byte_track_count":
#             len(
#                 unique_byte_track_ids
#             ),

#         "byte_track_ids":
#             unique_byte_track_ids,

#         "lifecycle_segment_count":
#             len(managed_tracks),

#         "completed_segment_count":
#             completed_count,

#         "removed_segment_count":
#             removed_count,

#         "tentative_observation_count":
#             tentative_count,

#         "max_missed_frames":
#             MAX_MISSED_FRAMES,

#         "tracks":
#             managed_tracks,
#     }

#     with open(
#         OUTPUT_PATH,
#         "w",
#         encoding="utf-8",
#     ) as f:

#         json.dump(
#             output,
#             f,
#             indent=2,
#         )


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     print("=" * 70)

#     print(
#         "STEP 5.4 — "
#         "TRACK MANAGEMENT & PERSISTENCE"
#     )

#     print("=" * 70)

#     # --------------------------------------------------------
#     # LOAD
#     # --------------------------------------------------------

#     data = load_track_data()

#     fps = float(
#         data.get(
#             "fps",
#             30.0,
#         )
#     )

#     frames_processed = int(
#         data.get(
#             "frames_processed",
#             0,
#         )
#     )

#     print(
#         f"\nFrames processed : "
#         f"{frames_processed}"
#     )

#     print(
#         f"FPS              : "
#         f"{fps:.2f}"
#     )

#     print(
#         f"Max missed frames: "
#         f"{MAX_MISSED_FRAMES}"
#     )

#     # --------------------------------------------------------
#     # PROCESS
#     # --------------------------------------------------------

#     print(
#         "\nManaging track lifecycles..."
#     )

#     (
#         managed_tracks,
#         tentative_count,
#     ) = manage_tracks(
#         data
#     )

#     # --------------------------------------------------------
#     # SAVE
#     # --------------------------------------------------------

#     save_results(
#         data,
#         managed_tracks,
#         tentative_count,
#     )

#     # --------------------------------------------------------
#     # SUMMARY
#     # --------------------------------------------------------

#     completed_count = sum(
#         1
#         for track in managed_tracks
#         if track["state"] == "COMPLETED"
#     )

#     removed_count = sum(
#         1
#         for track in managed_tracks
#         if track["state"] == "REMOVED"
#     )

#     unique_ids = sorted(
#         set(
#             track["track_id"]
#             for track in managed_tracks
#         )
#     )

#     print(
#         "\nTRACK SUMMARY"
#     )

#     print(
#         "-" * 70
#     )

#     print(
#         f"ByteTrack IDs       : "
#         f"{len(unique_ids)}"
#     )

#     print(
#         f"Lifecycle segments  : "
#         f"{len(managed_tracks)}"
#     )

#     print(
#         f"Completed segments  : "
#         f"{completed_count}"
#     )

#     print(
#         f"Removed segments    : "
#         f"{removed_count}"
#     )

#     print(
#         f"Tentative records   : "
#         f"{tentative_count}"
#     )

#     print(
#         f"ByteTrack IDs       : "
#         f"{unique_ids}"
#     )

#     # --------------------------------------------------------
#     # INDIVIDUAL SEGMENTS
#     # --------------------------------------------------------

#     print(
#         "\nTRACK SEGMENTS"
#     )

#     print(
#         "-" * 70
#     )

#     for track in managed_tracks:

#         print(
#             f"ID {track['track_id']:3d} "
#             f"| Segment {track['segment_id']:2d} "
#             f"| Frames "
#             f"{track['first_frame']} → "
#             f"{track['last_frame']} "
#             f"| Detections: "
#             f"{track['observation_count']:3d} "
#             f"| Predictions: "
#             f"{track['prediction_count']:3d} "
#             f"| Missed: "
#             f"{track['total_missed_frames']:3d} "
#             f"| Longest gap: "
#             f"{track['longest_missed_sequence']:3d} "
#             f"| State: "
#             f"{track['state']}"
#         )

#     # --------------------------------------------------------
#     # OUTPUT
#     # --------------------------------------------------------

#     print(
#         "\nOUTPUT"
#     )

#     print(
#         "-" * 70
#     )

#     print(
#         OUTPUT_PATH
#     )

#     print(
#         "\n" + "=" * 70
#     )

#     print(
#         "STEP 5.4 COMPLETE"
#     )

#     print(
#         "=" * 70
#     )


# # ============================================================
# # RUN
# # ============================================================

# if __name__ == "__main__":

#     main()