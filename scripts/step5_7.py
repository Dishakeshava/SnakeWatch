# ============================================================
# STEP 5.7 — MOVEMENT & TRAJECTORY CALCULATION (REVISED v2)
# ============================================================

from pathlib import Path
import json
import math
from collections import Counter


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(
    "/content/drive/MyDrive/snake_detection/Snake_detection"
)

INPUT_PATH = (
    PROJECT_DIR
    / "dataset"
    / "tracking_output"
    / "24638-344119107_medium"
    / "track_data_5_5_history.json"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "dataset"
    / "tracking_output"
    / "24638-344119107_medium"
    / "track_movement_5_7_revised.json"
)


# ============================================================
# 2. PARAMETERS
# ============================================================

SMOOTHING_WINDOW = 5

# Movement is calculated over a multi-frame baseline
# instead of using adjacent frames.
MOVEMENT_BASELINE_FRAMES = 5

HIGH_RELIABILITY_DETECTION_RATIO = 0.60

DIRECTION_CHANGE_THRESHOLD_DEGREES = 45.0

# A direction change must persist for 3 consecutive
# movement estimates before being counted.
DIRECTION_CHANGE_PERSISTENCE = 3

MIN_DETECTIONS = 5


# ============================================================
# 3. LOAD DATA
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_PATH}"
    )

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


tracks = data.get("tracks", [])

fps = data.get("fps", None)

if fps is None:

    # Some versions may store video FPS differently.
    # Use the known video FPS only if the field is absent.
    fps = 29.97002997

fps = float(fps)


print("=" * 70)
print("STEP 5.7 — MOVEMENT & TRAJECTORY CALCULATION (REVISED v2)")
print("=" * 70)

print(f"\nInput file:")
print(INPUT_PATH)

print(f"\nInput tracks: {len(tracks)}")
print(f"FPS: {fps:.6f}")
print(f"Smoothing window: {SMOOTHING_WINDOW}")
print(
    f"Movement baseline: {MOVEMENT_BASELINE_FRAMES} frames"
)
print(
    f"Direction persistence: "
    f"{DIRECTION_CHANGE_PERSISTENCE} frames"
)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def euclidean_distance(p1, p2):

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    return math.sqrt(
        dx * dx + dy * dy
    )


def angle_degrees(p1, p2):

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if dx == 0 and dy == 0:
        return None

    return math.degrees(
        math.atan2(dy, dx)
    )


def normalize_angle_difference(a, b):

    if a is None or b is None:
        return None

    diff = abs(a - b)

    while diff > 180:
        diff = abs(diff - 360)

    return diff


def safe_float(value):

    if value is None:
        return None

    value = float(value)

    if math.isfinite(value):
        return value

    return None


# ============================================================
# 5. PROCESS EACH TRACK
# ============================================================

movement_tracks = []


for track in tracks:

    track_id = track.get("track_id")
    segment_id = track.get("segment_id")

    history = track.get("history", [])

    if not history:
        continue


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Step 5.5 uses "frame_id", NOT "frame".
    # --------------------------------------------------------

    history = sorted(
        history,
        key=lambda x: int(x["frame_id"])
    )


    # --------------------------------------------------------
    # Track-level counts
    # --------------------------------------------------------

    detection_count = sum(
        1
        for obs in history
        if obs.get("observation_type")
        == "detection"
    )

    prediction_count = sum(
        1
        for obs in history
        if obs.get("observation_type")
        == "prediction"
    )

    total_observations = len(history)

    prediction_ratio = (
        prediction_count / total_observations
        if total_observations > 0
        else 0.0
    )

    prediction_heavy = (
        prediction_ratio > 0.50
    )


    # --------------------------------------------------------
    # History sufficiency
    # --------------------------------------------------------

    if detection_count >= MIN_DETECTIONS:
        history_status = "sufficient"
    else:
        history_status = "insufficient"


    # ========================================================
    # 6. RAW CENTERS
    # ========================================================

    raw_centers = []

    for obs in history:

        center = obs.get("center")

        if center is None:

            raw_centers.append(None)

        else:

            raw_centers.append(
                [
                    float(center[0]),
                    float(center[1])
                ]
            )


    # ========================================================
    # 7. CAUSAL POSITION SMOOTHING
    # ========================================================

    smoothed_centers = []

    for i in range(len(raw_centers)):

        start = max(
            0,
            i - SMOOTHING_WINDOW + 1
        )

        valid_points = []

        for j in range(
            start,
            i + 1
        ):

            if raw_centers[j] is not None:

                valid_points.append(
                    raw_centers[j]
                )


        if not valid_points:

            smoothed_centers.append(None)

        else:

            x = (
                sum(
                    p[0]
                    for p in valid_points
                )
                / len(valid_points)
            )

            y = (
                sum(
                    p[1]
                    for p in valid_points
                )
                / len(valid_points)
            )

            smoothed_centers.append(
                [x, y]
            )


    # ========================================================
    # 8. MOVEMENT CALCULATION
    # ========================================================

    movement_history = []

    cumulative_distance = 0.0

    previous_speed = None
    previous_speed_time = None

    previous_direction = None

    candidate_direction = None
    candidate_persistence = 0

    direction_change_count = 0


    for i, obs in enumerate(history):

        # ----------------------------------------------------
        # IMPORTANT:
        # Step 5.5 uses frame_id.
        # ----------------------------------------------------

        frame_id = int(
            obs["frame_id"]
        )

        timestamp = float(
            obs["timestamp"]
        )

        center = raw_centers[i]

        smoothed_center = (
            smoothed_centers[i]
        )

        observation_type = obs.get(
            "observation_type",
            "unknown"
        )


        # ----------------------------------------------------
        # Temporal reliability
        # ----------------------------------------------------

        reliability_start = max(
            0,
            i - SMOOTHING_WINDOW + 1
        )

        reliability_window = history[
            reliability_start:i + 1
        ]

        window_count = len(
            reliability_window
        )

        detection_window_count = sum(
            1
            for item in reliability_window
            if item.get(
                "observation_type"
            ) == "detection"
        )

        detection_ratio = (
            detection_window_count
            / window_count
            if window_count > 0
            else 0.0
        )

        has_detection = (
            detection_window_count > 0
        )


        if not has_detection:

            reliability = "LOW"

        elif (
            detection_ratio
            >= HIGH_RELIABILITY_DETECTION_RATIO
        ):

            reliability = "HIGH"

        else:

            reliability = "MEDIUM"


        # ----------------------------------------------------
        # Defaults
        # ----------------------------------------------------

        speed_px_per_s = None

        direction_degrees = None

        acceleration_px_per_s2 = None

        step_distance_px = None


        # ====================================================
        # MULTI-FRAME MOVEMENT BASELINE
        # ====================================================

        baseline_index = (
            i - MOVEMENT_BASELINE_FRAMES
        )


        if baseline_index >= 0:

            current_point = (
                smoothed_centers[i]
            )

            previous_point = (
                smoothed_centers[
                    baseline_index
                ]
            )


            if (
                current_point is not None
                and previous_point is not None
            ):

                previous_timestamp = float(
                    history[
                        baseline_index
                    ]["timestamp"]
                )

                dt = (
                    timestamp
                    - previous_timestamp
                )


                if dt > 0:

                    dx = (
                        current_point[0]
                        - previous_point[0]
                    )

                    dy = (
                        current_point[1]
                        - previous_point[1]
                    )


                    # ----------------------------------------
                    # Distance over temporal baseline
                    # ----------------------------------------

                    step_distance_px = math.sqrt(
                        dx * dx
                        + dy * dy
                    )


                    # ----------------------------------------
                    # Robust speed
                    # ----------------------------------------

                    speed_px_per_s = (
                        step_distance_px
                        / dt
                    )


                    # ----------------------------------------
                    # Direction
                    # ----------------------------------------

                    direction_degrees = (
                        angle_degrees(
                            previous_point,
                            current_point
                        )
                    )


                    # ====================================================
                    # ACCELERATION
                    # ====================================================

                    if (
                        previous_speed is not None
                        and previous_speed_time is not None
                    ):

                        acceleration_dt = (
                            timestamp
                            - previous_speed_time
                        )

                        if acceleration_dt > 0:

                            acceleration_px_per_s2 = (
                                (
                                    speed_px_per_s
                                    - previous_speed
                                )
                                / acceleration_dt
                            )


                    previous_speed = (
                        speed_px_per_s
                    )

                    previous_speed_time = (
                        timestamp
                    )


                    # ====================================================
                    # DIRECTION CHANGE PERSISTENCE
                    # ====================================================

                    if direction_degrees is not None:

                        if previous_direction is None:

                            previous_direction = (
                                direction_degrees
                            )

                        else:

                            direction_difference = (
                                normalize_angle_difference(
                                    direction_degrees,
                                    previous_direction
                                )
                            )


                            if (
                                direction_difference
                                >=
                                DIRECTION_CHANGE_THRESHOLD_DEGREES
                            ):

                                # ------------------------------------
                                # Start or continue candidate
                                # ------------------------------------

                                if (
                                    candidate_direction
                                    is None
                                ):

                                    candidate_direction = (
                                        direction_degrees
                                    )

                                    candidate_persistence = 1

                                else:

                                    candidate_difference = (
                                        normalize_angle_difference(
                                            direction_degrees,
                                            candidate_direction
                                        )
                                    )


                                    if (
                                        candidate_difference
                                        <
                                        DIRECTION_CHANGE_THRESHOLD_DEGREES
                                    ):

                                        candidate_persistence += 1

                                    else:

                                        candidate_direction = (
                                            direction_degrees
                                        )

                                        candidate_persistence = 1


                                # ------------------------------------
                                # Confirm persistent direction change
                                # ------------------------------------

                                if (
                                    candidate_persistence
                                    >=
                                    DIRECTION_CHANGE_PERSISTENCE
                                ):

                                    direction_change_count += 1

                                    previous_direction = (
                                        direction_degrees
                                    )

                                    candidate_direction = None

                                    candidate_persistence = 0


                            else:

                                # No persistent change.
                                candidate_direction = None
                                candidate_persistence = 0

                                previous_direction = (
                                    direction_degrees
                                )


        # ====================================================
        # CUMULATIVE DISTANCE
        # ====================================================

        if step_distance_px is not None:

            # Because the 5-frame baselines overlap,
            # we do NOT add the full 5-frame displacement
            # at every frame.
            #
            # Approximate local contribution:
            # baseline displacement / baseline length.

            cumulative_distance += (
                step_distance_px
                /
                MOVEMENT_BASELINE_FRAMES
            )


        # ====================================================
        # NET DISPLACEMENT
        # ====================================================

        net_displacement_px = None

        first_center = (
            smoothed_centers[0]
        )

        if (
            first_center is not None
            and smoothed_center is not None
        ):

            net_displacement_px = (
                euclidean_distance(
                    first_center,
                    smoothed_center
                )
            )


        # ====================================================
        # SAVE FRAME MOVEMENT
        # ====================================================

        movement_history.append(
            {
                "frame": frame_id,

                "timestamp": timestamp,

                "bbox": obs.get(
                    "bbox"
                ),

                "center": center,

                "smoothed_center":
                    smoothed_center,

                "confidence": (
                    obs.get("confidence")
                    if observation_type
                    == "detection"
                    else None
                ),

                "observation_type":
                    observation_type,

                "detection_strength":
                    obs.get(
                        "detection_strength"
                    ),

                "track_state":
                    obs.get(
                        "track_state"
                    ),

                "speed_px_per_s":
                    safe_float(
                        speed_px_per_s
                    ),

                "direction_degrees":
                    safe_float(
                        direction_degrees
                    ),

                "acceleration_px_per_s2":
                    safe_float(
                        acceleration_px_per_s2
                    ),

                "step_distance_px":
                    safe_float(
                        step_distance_px
                    ),

                "cumulative_distance_px":
                    safe_float(
                        cumulative_distance
                    ),

                "net_displacement_px":
                    safe_float(
                        net_displacement_px
                    ),

                "reliability":
                    reliability,

                "window_detection_ratio":
                    round(
                        detection_ratio,
                        4
                    ),

                "movement_baseline_frames":
                    MOVEMENT_BASELINE_FRAMES
            }
        )


    # ========================================================
    # 9. TRACK-LEVEL STATISTICS
    # ========================================================

    valid_speeds = [
        item["speed_px_per_s"]
        for item in movement_history
        if item["speed_px_per_s"]
        is not None
    ]


    valid_accelerations = [
        item["acceleration_px_per_s2"]
        for item in movement_history
        if item["acceleration_px_per_s2"]
        is not None
    ]


    if valid_speeds:

        average_speed = (
            sum(valid_speeds)
            / len(valid_speeds)
        )

        maximum_speed = max(
            valid_speeds
        )

    else:

        average_speed = None
        maximum_speed = None


    if valid_accelerations:

        maximum_abs_acceleration = max(
            abs(value)
            for value in valid_accelerations
        )

    else:

        maximum_abs_acceleration = None


    if movement_history:

        total_distance = (
            movement_history[-1][
                "cumulative_distance_px"
            ]
        )

        final_net_displacement = (
            movement_history[-1][
                "net_displacement_px"
            ]
        )

    else:

        total_distance = None

        final_net_displacement = None


    # ========================================================
    # 10. RELIABILITY SUMMARY
    # ========================================================

    reliability_counts = Counter(
        item["reliability"]
        for item in movement_history
    )


    # ========================================================
    # 11. SAVE TRACK
    # ========================================================

    movement_tracks.append(
        {
            "track_id":
                track_id,

            "segment_id":
                segment_id,

            "history_status":
                history_status,

            "detection_count":
                detection_count,

            "prediction_count":
                prediction_count,

            "prediction_ratio":
                round(
                    prediction_ratio,
                    4
                ),

            "prediction_heavy":
                prediction_heavy,

            "average_speed_px_per_s":
                safe_float(
                    average_speed
                ),

            "maximum_speed_px_per_s":
                safe_float(
                    maximum_speed
                ),

            "maximum_abs_acceleration_px_per_s2":
                safe_float(
                    maximum_abs_acceleration
                ),

            "total_distance_px":
                safe_float(
                    total_distance
                ),

            "final_net_displacement_px":
                safe_float(
                    final_net_displacement
                ),

            "direction_change_count":
                direction_change_count,

            "movement":
                (
                    "movement_observed"
                    if valid_speeds
                    else "insufficient"
                ),

            "reliability_summary":
                {
                    "HIGH":
                        reliability_counts.get(
                            "HIGH",
                            0
                        ),

                    "MEDIUM":
                        reliability_counts.get(
                            "MEDIUM",
                            0
                        ),

                    "LOW":
                        reliability_counts.get(
                            "LOW",
                            0
                        )
                },

            "movement_baseline_frames":
                MOVEMENT_BASELINE_FRAMES,

            "smoothing_window":
                SMOOTHING_WINDOW,

            "direction_change_threshold_degrees":
                DIRECTION_CHANGE_THRESHOLD_DEGREES,

            "direction_change_persistence":
                DIRECTION_CHANGE_PERSISTENCE,

            "history":
                movement_history
        }
    )


# ============================================================
# 12. OUTPUT OBJECT
# ============================================================

output = {

    "step":
        "5.7",

    "step_name":
        "Movement & Trajectory Calculation",

    "version":
        "revised_temporal_baseline_v2",

    "input_file":
        str(INPUT_PATH),

    "video_fps":
        fps,

    "frames_processed":
        data.get(
            "frames_processed",
            None
        ),

    "parameters":
        {
            "smoothing_window":
                SMOOTHING_WINDOW,

            "movement_baseline_frames":
                MOVEMENT_BASELINE_FRAMES,

            "high_reliability_detection_ratio":
                HIGH_RELIABILITY_DETECTION_RATIO,

            "direction_change_threshold_degrees":
                DIRECTION_CHANGE_THRESHOLD_DEGREES,

            "direction_change_persistence":
                DIRECTION_CHANGE_PERSISTENCE,

            "minimum_detections":
                MIN_DETECTIONS
        },

    "units":
        {
            "speed":
                "pixels_per_second",

            "acceleration":
                "pixels_per_second_squared",

            "distance":
                "pixels",

            "direction":
                "degrees"
        },

    "track_count":
        len(movement_tracks),

    "tracks":
        movement_tracks
}


# ============================================================
# 13. WRITE OUTPUT
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2
    )


# ============================================================
# 14. CONSOLE SUMMARY — EDGE-CASE SAFE
# ============================================================

def format_number(value, decimals=2, suffix=""):
    """
    Safely format numerical values.

    Returns:
        'N/A' when value is None, NaN, or infinite.
        Otherwise returns a formatted number.
    """

    if value is None:
        return "N/A"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if not math.isfinite(value):
        return "N/A"

    return f"{value:.{decimals}f}{suffix}"


def safe_count(value):
    """
    Safely format integer-like counts.
    """

    if value is None:
        return 0

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


print("\n")
print("=" * 70)
print("TRACK MOVEMENT SUMMARY")
print("=" * 70)


# ------------------------------------------------------------
# Handle completely empty output safely
# ------------------------------------------------------------

if not movement_tracks:

    print("\nNo valid tracks available for movement analysis.")

    print("\nOUTPUT")
    print("-" * 70)
    print(OUTPUT_PATH)

    print("\n")
    print("=" * 70)
    print("STEP 5.7 REVISED v2 COMPLETE")
    print("=" * 70)

    print("\nIMPORTANT SANITY CHECK:")
    print(
        "No tracks were available. Movement statistics could not "
        "be calculated."
    )

else:

    # --------------------------------------------------------
    # Process every track safely
    # --------------------------------------------------------

    for track in movement_tracks:

        track_id = track.get("track_id", "N/A")
        segment_id = track.get("segment_id", "N/A")

        avg_speed = track.get(
            "average_speed_px_per_s"
        )

        max_speed = track.get(
            "maximum_speed_px_per_s"
        )

        total_dist = track.get(
            "total_distance_px"
        )

        net_disp = track.get(
            "final_net_displacement_px"
        )

        direction_changes = safe_count(
            track.get("direction_change_count", 0)
        )

        detection_count = safe_count(
            track.get("detection_count", 0)
        )

        prediction_count = safe_count(
            track.get("prediction_count", 0)
        )

        prediction_ratio = track.get(
            "prediction_ratio"
        )

        history_status = track.get(
            "history_status",
            "unknown"
        )

        reliability_summary = track.get(
            "reliability_summary",
            {}
        )

        if not isinstance(reliability_summary, dict):
            reliability_summary = {}

        high_count = safe_count(
            reliability_summary.get("HIGH", 0)
        )

        medium_count = safe_count(
            reliability_summary.get("MEDIUM", 0)
        )

        low_count = safe_count(
            reliability_summary.get("LOW", 0)
        )

        prediction_heavy = bool(
            track.get("prediction_heavy", False)
        )

        movement_status = track.get(
            "movement",
            "insufficient"
        )


        # ----------------------------------------------------
        # Main track summary
        # ----------------------------------------------------

        print(
            f"ID {str(track_id):>2} "
            f"| Seg {str(segment_id):>2} "
            f"| AvgSpeed: "
            f"{format_number(avg_speed, 2)} px/s "
            f"| MaxSpeed: "
            f"{format_number(max_speed, 2)} px/s "
            f"| TotalDist: "
            f"{format_number(total_dist, 1)}px "
            f"| NetDisp: "
            f"{format_number(net_disp, 1)}px "
            f"| DirChanges: "
            f"{direction_changes:>3} "
            f"| {history_status}"
        )


        # ----------------------------------------------------
        # Observation statistics
        # ----------------------------------------------------

        print(
            f"     Detection: "
            f"{detection_count} | "
            f"Prediction: "
            f"{prediction_count} | "
            f"PredictionRatio: "
            f"{format_number(prediction_ratio, 3)}"
        )


        # ----------------------------------------------------
        # Reliability
        # ----------------------------------------------------

        print(
            f"     Reliability → "
            f"HIGH: {high_count} | "
            f"MEDIUM: {medium_count} | "
            f"LOW: {low_count}"
        )


        # ----------------------------------------------------
        # Prediction-heavy warning
        # ----------------------------------------------------

        if prediction_heavy:

            print(
                "     WARNING: "
                "Prediction-heavy track"
            )


        # ----------------------------------------------------
        # Explicit insufficient-history message
        # ----------------------------------------------------

        if history_status != "sufficient":

            print(
                "     NOTE: "
                "Insufficient detection history for reliable "
                "movement statistics."
            )


        # ----------------------------------------------------
        # Explicit no-speed message
        # ----------------------------------------------------

        if avg_speed is None:

            print(
                "     NOTE: "
                "No valid speed estimate available."
            )


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("\n")
    print("OUTPUT")
    print("-" * 70)
    print(OUTPUT_PATH)

    print("\n")
    print("=" * 70)
    print("STEP 5.7 REVISED v2 COMPLETE")
    print("=" * 70)

    print("\nIMPORTANT SANITY CHECK:")
    print(
        "Compare the revised movement statistics against "
        "the Step 5.6 tracking visualization."
    )

    print(
        "Pay particular attention to Track 2."
    )