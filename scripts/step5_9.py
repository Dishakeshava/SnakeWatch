# ============================================================
# STEP 5.9 — INTRUSION & RISK CALCULATION
# ============================================================

from pathlib import Path
import json
import math


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(
    "/content/drive/MyDrive/snake_detection/Snake_detection"
)

# Updated for the 24638-344119107_medium video run.
INPUT_PATH = (
    PROJECT_DIR
    / "dataset"
    / "tracking_output"
    / "24638-344119107_medium"
    / "track_spatial_5_8.json"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "dataset"
    / "tracking_output"
    / "24638-344119107_medium"
    / "track_risk_5_9.json"
)


# ============================================================
# 2. PARAMETERS
# ============================================================
#
# IMPORTANT:
# These are DEVELOPMENT PARAMETERS for pixel-space risk scoring.
#
# They should be validated/tuned on representative video scenes.
# ============================================================

# Distance in pixels from the protected boundary at which a point is
# considered spatially close to intrusion.
BOUNDARY_BUFFER_PX = 50.0

# Minimum pixel speed considered meaningful movement.
APPROACHING_SPEED_THRESHOLD_PX_PER_S = 20.0

# Number of consecutive observations required before an
# approaching/boundary state becomes persistent.
MIN_PERSISTENCE_OBSERVATIONS = 3

# Number of consecutive observations required for a
# high-confidence intrusion state.
INTRUSION_PERSISTENCE_OBSERVATIONS = 3

# Reliability required for a strong risk assessment.
HIGH_RELIABILITY_RATIO = 0.60

# TTI should only be calculated when the snake has a meaningful
# inward velocity.
MIN_INWARD_VELOCITY_PX_PER_S = 10.0

# TTI is disabled in pixel-space mode.
#
# These are provisional development thresholds.
# TTI is intentionally disabled in pixel-space mode because pixels
# do not provide a physically meaningful distance scale.
USE_TTI = False


# ============================================================
# 3. LOAD STEP 5.8
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Step 5.8 output not found:\n{INPUT_PATH}"
    )

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


tracks = data.get("tracks", [])

calibration_status = data.get(
    "calibration_status",
    "UNKNOWN"
)

video_fps = safe_fps = None
try:
    video_fps = float(
        data.get(
            "video_fps",
            29.97002997
        )
    )
    if not math.isfinite(video_fps) or video_fps <= 0:
        video_fps = 29.97002997
except (TypeError, ValueError):
    video_fps = 29.97002997


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def safe_float(value):
    """
    Safely convert a value to a finite float.

    Returns None for:
    - None
    - invalid strings
    - NaN
    - positive/negative infinity
    """
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if math.isfinite(value):
        return value

    return None


def safe_int(value, default=None):
    """Safely convert a value to int."""
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value, default=False):
    """Safely normalize boolean-like values."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value_lower = value.strip().lower()

        if value_lower in {
            "true", "1", "yes", "y", "approaching"
        }:
            return True

        if value_lower in {
            "false", "0", "no", "n", "not_approaching"
        }:
            return False

    if value is None:
        return default

    return bool(value)


def get_reliability_weight(reliability):
    if reliability == "HIGH":
        return 1.0

    if reliability == "MEDIUM":
        return 0.7

    if reliability == "LOW":
        return 0.3

    return 0.0


def calculate_risk_score(
    zone,
    distance_to_boundary,
    approaching_boundary,
    velocity,
    persistence_count,
    reliability,
    tti
):
    """
    Calculate a provisional 0–100 risk score.

    This is a development-stage scoring model.
    It is NOT a physically validated risk probability.
    """

    score = 0.0

    # --------------------------------------------------------
    # Spatial component
    # --------------------------------------------------------

    if zone == "INSIDE":
        score += 60.0

    elif zone == "BOUNDARY":
        score += 40.0

    elif zone == "OUTSIDE":
        score += 10.0

    # --------------------------------------------------------
    # Boundary proximity
    # --------------------------------------------------------

    if distance_to_boundary is not None:

        if distance_to_boundary <= 25.0:
            score += 20.0

        elif distance_to_boundary <= 50.0:
            score += 15.0

        elif distance_to_boundary <= 100.0:
            score += 8.0

        elif distance_to_boundary <= 200.0:
            score += 3.0

    # --------------------------------------------------------
    # Movement toward boundary
    # --------------------------------------------------------

    if approaching_boundary:
        score += 10.0

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    if velocity is not None:

        if velocity >= 200.0:
            score += 10.0

        elif velocity >= 100.0:
            score += 6.0

        elif velocity >= APPROACHING_SPEED_THRESHOLD_PX_PER_S:
            score += 3.0

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    if persistence_count >= 5:
        score += 10.0

    elif persistence_count >= 3:
        score += 6.0

    # --------------------------------------------------------
    # Time-to-Intrusion
    # --------------------------------------------------------

    if tti is not None:

        if tti <= VERY_SHORT_TTI_SECONDS:
            score += 15.0

        elif tti <= SHORT_TTI_SECONDS:
            score += 8.0

    # --------------------------------------------------------
    # Reliability adjustment
    # --------------------------------------------------------

    reliability_weight = get_reliability_weight(
        reliability
    )

    # Keep some score even for medium/low reliability,
    # but reduce confidence in the result.
    score *= (
        0.70
        + 0.30 * reliability_weight
    )

    return min(
        100.0,
        max(0.0, score)
    )


def risk_level_from_score(
    score,
    zone,
    intrusion_confirmed
):
    if intrusion_confirmed:

        if score >= 75:
            return "CRITICAL"

        return "HIGH"

    if zone == "BOUNDARY":

        if score >= 60:
            return "HIGH"

        if score >= 35:
            return "MEDIUM"

        return "LOW"

    if zone == "OUTSIDE":

        if score >= 60:
            return "HIGH"

        if score >= 30:
            return "MEDIUM"

        return "LOW"

    if zone == "INSIDE":
        return "HIGH"

    return "LOW"


def normalize_zone(value):
    """Normalize spatial zone labels."""
    if value is None:
        return "UNKNOWN"

    zone = str(value).strip().upper()

    valid_zones = {
        "INSIDE",
        "BOUNDARY",
        "OUTSIDE"
    }

    if zone in valid_zones:
        return zone

    return "UNKNOWN"


def normalize_reliability(value):
    """Normalize reliability labels."""
    if value is None:
        return "LOW"

    reliability = str(value).strip().upper()

    if reliability in {"HIGH", "MEDIUM", "LOW"}:
        return reliability

    return "LOW"


def safe_frame_from_observation(obs):
    """Return frame number or None."""
    return safe_int(
        obs.get("frame"),
        default=None
    )


def safe_timestamp_from_observation(obs):
    """Return timestamp or None."""
    return safe_float(
        obs.get("timestamp")
    )


# ============================================================
# 5. PROCESS TRACKS
# ============================================================

risk_tracks = []

skipped_tracks = 0
skipped_observations = 0


for track in tracks:

    if not isinstance(track, dict):
        skipped_tracks += 1
        continue

    track_id = track.get(
        "track_id"
    )

    segment_id = track.get(
        "segment_id"
    )

    history = track.get(
        "history",
        []
    )

    if not isinstance(history, list) or not history:
        skipped_tracks += 1
        continue

    # --------------------------------------------------------
    # Keep only valid dictionary observations.
    # --------------------------------------------------------

    valid_history = []

    for item in history:

        if not isinstance(item, dict):
            skipped_observations += 1
            continue

        frame_value = safe_frame_from_observation(item)

        if frame_value is None:
            skipped_observations += 1
            continue

        valid_history.append(item)

    if not valid_history:
        skipped_tracks += 1
        continue

    # --------------------------------------------------------
    # Sort chronologically.
    # --------------------------------------------------------

    history = sorted(
        valid_history,
        key=lambda x: safe_frame_from_observation(x)
    )

    risk_history = []

    # --------------------------------------------------------
    # State counters
    # --------------------------------------------------------

    current_state = "NO_INTRUSION"

    state_persistence = 0

    intrusion_persistence = 0

    maximum_risk_score = 0.0

    minimum_tti = None

    intrusion_confirmed = False

    # --------------------------------------------------------
    # Track-level reliability
    # --------------------------------------------------------

    reliability_counts = {
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0
    }

    # ========================================================
    # PROCESS EACH OBSERVATION
    # ========================================================

    for obs in history:

        frame = safe_frame_from_observation(obs)

        if frame is None:
            skipped_observations += 1
            continue

        timestamp = safe_timestamp_from_observation(
            obs
        )

        if timestamp is None:
            # Keep the observation but use frame/FPS as a
            # fallback timestamp when possible.
            if video_fps > 0:
                timestamp = frame / video_fps
            else:
                timestamp = 0.0

        zone = normalize_zone(
            obs.get(
                "zone",
                "UNKNOWN"
            )
        )

        distance_to_boundary = safe_float(
            obs.get(
                "distance_to_boundary_px"
            )
        )

        velocity = safe_float(
            obs.get(
                "velocity_px_per_s"
            )
        )

        approaching_boundary = safe_bool(
            obs.get(
                "approaching_boundary"
            ),
            default=False
        )

        reliability = normalize_reliability(
            obs.get(
                "reliability",
                "LOW"
            )
        )

        if reliability in reliability_counts:
            reliability_counts[
                reliability
            ] += 1

        # ====================================================
        # 1. DETERMINE CANDIDATE STATE
        # ====================================================

        if zone == "INSIDE":

            candidate_state = "INTRUSION"

        elif zone == "BOUNDARY":

            candidate_state = "BOUNDARY_RISK"

        elif (
            zone == "OUTSIDE"
            and approaching_boundary
            and velocity is not None
            and velocity >=
                APPROACHING_SPEED_THRESHOLD_PX_PER_S
        ):

            candidate_state = "APPROACHING"

        else:

            candidate_state = "NO_INTRUSION"

        # ====================================================
        # 2. STATE PERSISTENCE
        # ====================================================

        if candidate_state == current_state:

            state_persistence += 1

        else:

            current_state = candidate_state
            state_persistence = 1

        # ----------------------------------------------------
        # Intrusion persistence
        # ----------------------------------------------------

        if candidate_state == "INTRUSION":

            intrusion_persistence += 1

        else:

            intrusion_persistence = 0

        # ====================================================
        # 3. CONFIRM INTRUSION
        # ====================================================

        if (
            intrusion_persistence
            >= INTRUSION_PERSISTENCE_OBSERVATIONS
        ):

            intrusion_confirmed = True

        elif candidate_state != "INTRUSION":

            intrusion_confirmed = False

        # ====================================================
        # 4. TIME-TO-INTRUSION
        # ====================================================
        #
        # TTI is intentionally disabled in pixel-space mode.
        # Without a metric calibration, pixel distance cannot be
        # interpreted as a physical distance, so a physically
        # meaningful time-to-intrusion cannot be computed.
        # ====================================================

        tti = None

        # ====================================================
        # 5. RISK SCORE
        # ====================================================

        risk_score = calculate_risk_score(
            zone=zone,
            distance_to_boundary=
                distance_to_boundary,
            approaching_boundary=
                approaching_boundary,
            velocity=velocity,
            persistence_count=
                state_persistence,
            reliability=reliability,
            tti=tti
        )

        maximum_risk_score = max(
            maximum_risk_score,
            risk_score
        )

        # ====================================================
        # 6. RISK LEVEL
        # ====================================================

        risk_level = risk_level_from_score(
            score=risk_score,
            zone=zone,
            intrusion_confirmed=
                intrusion_confirmed
        )

        # ====================================================
        # 7. SAVE OBSERVATION
        # ====================================================

        risk_history.append({

            "frame":
                frame,

            "timestamp":
                timestamp,

            "zone":
                zone,

            "distance_to_boundary_px":
                safe_float(
                    distance_to_boundary
                ),

            "velocity_px_per_s":
                safe_float(
                    velocity
                ),

            "approaching_boundary":
                approaching_boundary,

            "tti_s":
                safe_float(
                    tti
                ),

            "intrusion_state":
                current_state,

            "intrusion_confirmed":
                intrusion_confirmed,

            "state_persistence":
                state_persistence,

            "intrusion_persistence":
                intrusion_persistence,

            "risk_score":
                round(
                    risk_score,
                    2
                ),

            "risk_level":
                risk_level,

            "reliability":
                reliability,

            "observation_type":
                obs.get(
                    "observation_type"
                )
        })

    # ========================================================
    # TRACK-LEVEL SUMMARY
    # ========================================================

    final_observation = (
        risk_history[-1]
        if risk_history
        else None
    )

    final_state = (
        final_observation[
            "intrusion_state"
        ]
        if final_observation
        else None
    )

    final_risk_level = (
        final_observation[
            "risk_level"
        ]
        if final_observation
        else None
    )

    risk_tracks.append({

        "track_id":
            track_id,

        "segment_id":
            segment_id,

        "history_status":
            track.get(
                "history_status"
            ),

        "final_intrusion_state":
            final_state,

        "final_risk_level":
            final_risk_level,

        "intrusion_confirmed":
            intrusion_confirmed,

        "maximum_risk_score":
            round(
                maximum_risk_score,
                2
            ),

        "minimum_tti_s":
            safe_float(
                minimum_tti
            ),

        "reliability_summary":
            reliability_counts,

        "history":
            risk_history
    })


# ============================================================
# 6. OUTPUT OBJECT
# ============================================================

output = {

    "step":
        "5.9",

    "step_name":
        "Intrusion & Risk Calculation",

    "version":
        "persistent_spatial_risk_v1",

    "input_file":
        str(INPUT_PATH),

    "video_fps":
        video_fps,

    "calibration_status":
        calibration_status,

    "calibration_warning":
        (
            "Pixel-space risk mode. Spatial distance is in pixels "
            "and velocity is in pixels/second. Metric calibration "
            "is not available for this online video, so physical "
            "distance, physical velocity and TTI are not reported."
        ),

    "parameters":
        {

            "boundary_buffer_px":
                BOUNDARY_BUFFER_PX,

            "approaching_speed_threshold_mps":
                APPROACHING_SPEED_THRESHOLD_PX_PER_S,

            "min_persistence_observations":
                MIN_PERSISTENCE_OBSERVATIONS,

            "intrusion_persistence_observations":
                INTRUSION_PERSISTENCE_OBSERVATIONS,

            "high_reliability_ratio":
                HIGH_RELIABILITY_RATIO,

            "min_inward_velocity_px_per_s":
                MIN_INWARD_VELOCITY_PX_PER_S,

            "tti_enabled":
                USE_TTI
        },

    "intrusion_states":
        [
            "NO_INTRUSION",
            "APPROACHING",
            "BOUNDARY_RISK",
            "INTRUSION"
        ],

    "risk_levels":
        [
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        ],

    "track_count":
        len(risk_tracks),

    "tracks":
        risk_tracks,

    "processing_notes":
        {
            "skipped_tracks":
                skipped_tracks,

            "skipped_observations":
                skipped_observations,

            "calibration_validated":
                False,
            "metric_calibration_available":
                False,
            "tti_enabled":
                USE_TTI
        }
}


# ============================================================
# 7. WRITE OUTPUT
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
# 8. CONSOLE SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print(
    "STEP 5.9 — INTRUSION & RISK CALCULATION"
)
print("=" * 70)

print(
    f"\nInput tracks : {len(tracks)}"
)

print(
    f"Processed tracks : {len(risk_tracks)}"
)

print(
    f"Calibration  : {calibration_status}"
)

print(
    f"Boundary buffer : "
    f"{BOUNDARY_BUFFER_PX:.1f} px"
)

print(
    f"Persistence required : "
    f"{INTRUSION_PERSISTENCE_OBSERVATIONS} observations"
)

print("\n")
print("TRACK RISK SUMMARY")
print("-" * 70)


for track in risk_tracks:

    track_id = track.get(
        "track_id"
    )

    segment_id = track.get(
        "segment_id"
    )

    final_state = (
        track.get(
            "final_intrusion_state"
        )
        or "N/A"
    )

    final_risk = (
        track.get(
            "final_risk_level"
        )
        or "N/A"
    )

    maximum_score = safe_float(
        track.get(
            "maximum_risk_score"
        )
    )

    if maximum_score is None:
        maximum_score_text = "N/A"
    else:
        maximum_score_text = (
            f"{maximum_score:.2f}"
        )

    minimum_tti = track.get(
        "minimum_tti_s"
    )

    if minimum_tti is None:
        minimum_tti_text = "N/A"
    else:
        minimum_tti_text = (
            f"{minimum_tti:.3f} s"
        )

    print(
        f"ID {str(track_id):>2} "
        f"| Seg {str(segment_id):>2} "
        f"| State: "
        f"{str(final_state):<15} "
        f"| Risk: "
        f"{str(final_risk):<8}"
    )

    print(
        f"     Intrusion confirmed: "
        f"{track.get('intrusion_confirmed')}"
    )

    print(
        f"     Maximum risk score: "
        f"{maximum_score_text}"
    )

    print(
        f"     Minimum TTI: "
        f"{minimum_tti_text}"
    )

    print(
        f"     Reliability: "
        f"{track.get('reliability_summary')}"
    )


print("\n")
print("OUTPUT")
print("-" * 70)
print(OUTPUT_PATH)

print("\n")
print("=" * 70)
print("STEP 5.9 COMPLETE")
print("=" * 70)

print("\nIMPORTANT:")

print(
    "Pixel-space risk mode is active."
)

print(
    "Distances are pixels and velocities are pixels/second."
)

print(
    "Metric calibration is not available for this online video."
)

print(
    "TTI is disabled because pixel distance is not a physical distance."
)

print(
    "Step 5.10 can consume this file for alert generation."
)



















# # ============================================================
# # STEP 5.9 — INTRUSION & RISK CALCULATION
# # ============================================================

# from pathlib import Path
# import json
# import math


# # ============================================================
# # 1. PATHS
# # ============================================================

# PROJECT_DIR = Path(
#     "/content/drive/MyDrive/snake_detection/Snake_detection"
# )

# # Updated for the 24638-344119107_medium video run.
# INPUT_PATH = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
#     / "24638-344119107_medium"
#     / "track_spatial_5_8.json"
# )

# OUTPUT_PATH = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
#     / "24638-344119107_medium"
#     / "track_risk_5_9.json"
# )


# # ============================================================
# # 2. PARAMETERS
# # ============================================================
# #
# # IMPORTANT:
# # These are DEVELOPMENT PARAMETERS.
# #
# # They must be validated/tuned after real-world calibration
# # and testing.
# # ============================================================

# # Distance from the protected boundary at which a point is
# # considered spatially close to intrusion.
# BOUNDARY_BUFFER_M = 0.50

# # Minimum speed considered meaningful movement.
# APPROACHING_SPEED_THRESHOLD_MPS = 0.10

# # Number of consecutive observations required before an
# # approaching/boundary state becomes persistent.
# MIN_PERSISTENCE_OBSERVATIONS = 3

# # Number of consecutive observations required for a
# # high-confidence intrusion state.
# INTRUSION_PERSISTENCE_OBSERVATIONS = 3

# # Reliability required for a strong risk assessment.
# HIGH_RELIABILITY_RATIO = 0.60

# # TTI should only be calculated when the snake has a meaningful
# # inward velocity.
# MIN_INWARD_VELOCITY_MPS = 0.05

# # TTI limits used only for risk categorization.
# #
# # These are provisional development thresholds.
# VERY_SHORT_TTI_SECONDS = 2.0
# SHORT_TTI_SECONDS = 5.0


# # ============================================================
# # 3. LOAD STEP 5.8
# # ============================================================

# if not INPUT_PATH.exists():
#     raise FileNotFoundError(
#         f"Step 5.8 output not found:\n{INPUT_PATH}"
#     )

# with open(
#     INPUT_PATH,
#     "r",
#     encoding="utf-8"
# ) as f:
#     data = json.load(f)


# tracks = data.get("tracks", [])

# calibration_status = data.get(
#     "calibration_status",
#     "UNKNOWN"
# )

# video_fps = safe_fps = None
# try:
#     video_fps = float(
#         data.get(
#             "video_fps",
#             29.97002997
#         )
#     )
#     if not math.isfinite(video_fps) or video_fps <= 0:
#         video_fps = 29.97002997
# except (TypeError, ValueError):
#     video_fps = 29.97002997


# # ============================================================
# # 4. HELPER FUNCTIONS
# # ============================================================

# def safe_float(value):
#     """
#     Safely convert a value to a finite float.

#     Returns None for:
#     - None
#     - invalid strings
#     - NaN
#     - positive/negative infinity
#     """
#     if value is None:
#         return None

#     try:
#         value = float(value)
#     except (TypeError, ValueError):
#         return None

#     if math.isfinite(value):
#         return value

#     return None


# def safe_int(value, default=None):
#     """Safely convert a value to int."""
#     if value is None:
#         return default

#     try:
#         return int(value)
#     except (TypeError, ValueError):
#         return default


# def safe_bool(value, default=False):
#     """Safely normalize boolean-like values."""
#     if isinstance(value, bool):
#         return value

#     if isinstance(value, str):
#         value_lower = value.strip().lower()

#         if value_lower in {
#             "true", "1", "yes", "y", "approaching"
#         }:
#             return True

#         if value_lower in {
#             "false", "0", "no", "n", "not_approaching"
#         }:
#             return False

#     if value is None:
#         return default

#     return bool(value)


# def get_reliability_weight(reliability):
#     if reliability == "HIGH":
#         return 1.0

#     if reliability == "MEDIUM":
#         return 0.7

#     if reliability == "LOW":
#         return 0.3

#     return 0.0


# def calculate_risk_score(
#     zone,
#     distance_to_boundary,
#     approaching_boundary,
#     velocity,
#     persistence_count,
#     reliability,
#     tti
# ):
#     """
#     Calculate a provisional 0–100 risk score.

#     This is a development-stage scoring model.
#     It is NOT a physically validated risk probability.
#     """

#     score = 0.0

#     # --------------------------------------------------------
#     # Spatial component
#     # --------------------------------------------------------

#     if zone == "INSIDE":
#         score += 60.0

#     elif zone == "BOUNDARY":
#         score += 40.0

#     elif zone == "OUTSIDE":
#         score += 10.0

#     # --------------------------------------------------------
#     # Boundary proximity
#     # --------------------------------------------------------

#     if distance_to_boundary is not None:

#         if distance_to_boundary <= 0.25:
#             score += 20.0

#         elif distance_to_boundary <= 0.50:
#             score += 15.0

#         elif distance_to_boundary <= 1.00:
#             score += 8.0

#         elif distance_to_boundary <= 2.00:
#             score += 3.0

#     # --------------------------------------------------------
#     # Movement toward boundary
#     # --------------------------------------------------------

#     if approaching_boundary:
#         score += 10.0

#     # --------------------------------------------------------
#     # Velocity
#     # --------------------------------------------------------

#     if velocity is not None:

#         if velocity >= 2.0:
#             score += 10.0

#         elif velocity >= 1.0:
#             score += 6.0

#         elif velocity >= APPROACHING_SPEED_THRESHOLD_MPS:
#             score += 3.0

#     # --------------------------------------------------------
#     # Persistence
#     # --------------------------------------------------------

#     if persistence_count >= 5:
#         score += 10.0

#     elif persistence_count >= 3:
#         score += 6.0

#     # --------------------------------------------------------
#     # Time-to-Intrusion
#     # --------------------------------------------------------

#     if tti is not None:

#         if tti <= VERY_SHORT_TTI_SECONDS:
#             score += 15.0

#         elif tti <= SHORT_TTI_SECONDS:
#             score += 8.0

#     # --------------------------------------------------------
#     # Reliability adjustment
#     # --------------------------------------------------------

#     reliability_weight = get_reliability_weight(
#         reliability
#     )

#     # Keep some score even for medium/low reliability,
#     # but reduce confidence in the result.
#     score *= (
#         0.70
#         + 0.30 * reliability_weight
#     )

#     return min(
#         100.0,
#         max(0.0, score)
#     )


# def risk_level_from_score(
#     score,
#     zone,
#     intrusion_confirmed
# ):
#     if intrusion_confirmed:

#         if score >= 75:
#             return "CRITICAL"

#         return "HIGH"

#     if zone == "BOUNDARY":

#         if score >= 60:
#             return "HIGH"

#         if score >= 35:
#             return "MEDIUM"

#         return "LOW"

#     if zone == "OUTSIDE":

#         if score >= 60:
#             return "HIGH"

#         if score >= 30:
#             return "MEDIUM"

#         return "LOW"

#     if zone == "INSIDE":
#         return "HIGH"

#     return "LOW"


# def normalize_zone(value):
#     """Normalize spatial zone labels."""
#     if value is None:
#         return "UNKNOWN"

#     zone = str(value).strip().upper()

#     valid_zones = {
#         "INSIDE",
#         "BOUNDARY",
#         "OUTSIDE"
#     }

#     if zone in valid_zones:
#         return zone

#     return "UNKNOWN"


# def normalize_reliability(value):
#     """Normalize reliability labels."""
#     if value is None:
#         return "LOW"

#     reliability = str(value).strip().upper()

#     if reliability in {"HIGH", "MEDIUM", "LOW"}:
#         return reliability

#     return "LOW"


# def safe_frame_from_observation(obs):
#     """Return frame number or None."""
#     return safe_int(
#         obs.get("frame"),
#         default=None
#     )


# def safe_timestamp_from_observation(obs):
#     """Return timestamp or None."""
#     return safe_float(
#         obs.get("timestamp")
#     )


# # ============================================================
# # 5. PROCESS TRACKS
# # ============================================================

# risk_tracks = []

# skipped_tracks = 0
# skipped_observations = 0


# for track in tracks:

#     if not isinstance(track, dict):
#         skipped_tracks += 1
#         continue

#     track_id = track.get(
#         "track_id"
#     )

#     segment_id = track.get(
#         "segment_id"
#     )

#     history = track.get(
#         "history",
#         []
#     )

#     if not isinstance(history, list) or not history:
#         skipped_tracks += 1
#         continue

#     # --------------------------------------------------------
#     # Keep only valid dictionary observations.
#     # --------------------------------------------------------

#     valid_history = []

#     for item in history:

#         if not isinstance(item, dict):
#             skipped_observations += 1
#             continue

#         frame_value = safe_frame_from_observation(item)

#         if frame_value is None:
#             skipped_observations += 1
#             continue

#         valid_history.append(item)

#     if not valid_history:
#         skipped_tracks += 1
#         continue

#     # --------------------------------------------------------
#     # Sort chronologically.
#     # --------------------------------------------------------

#     history = sorted(
#         valid_history,
#         key=lambda x: safe_frame_from_observation(x)
#     )

#     risk_history = []

#     # --------------------------------------------------------
#     # State counters
#     # --------------------------------------------------------

#     current_state = "NO_INTRUSION"

#     state_persistence = 0

#     intrusion_persistence = 0

#     maximum_risk_score = 0.0

#     minimum_tti = None

#     intrusion_confirmed = False

#     # --------------------------------------------------------
#     # Track-level reliability
#     # --------------------------------------------------------

#     reliability_counts = {
#         "HIGH": 0,
#         "MEDIUM": 0,
#         "LOW": 0
#     }

#     # ========================================================
#     # PROCESS EACH OBSERVATION
#     # ========================================================

#     for obs in history:

#         frame = safe_frame_from_observation(obs)

#         if frame is None:
#             skipped_observations += 1
#             continue

#         timestamp = safe_timestamp_from_observation(
#             obs
#         )

#         if timestamp is None:
#             # Keep the observation but use frame/FPS as a
#             # fallback timestamp when possible.
#             if video_fps > 0:
#                 timestamp = frame / video_fps
#             else:
#                 timestamp = 0.0

#         zone = normalize_zone(
#             obs.get(
#                 "zone",
#                 "UNKNOWN"
#             )
#         )

#         distance_to_boundary = safe_float(
#             obs.get(
#                 "distance_to_boundary_m"
#             )
#         )

#         velocity = safe_float(
#             obs.get(
#                 "velocity_m_per_s"
#             )
#         )

#         approaching_boundary = safe_bool(
#             obs.get(
#                 "approaching_boundary"
#             ),
#             default=False
#         )

#         reliability = normalize_reliability(
#             obs.get(
#                 "reliability",
#                 "LOW"
#             )
#         )

#         if reliability in reliability_counts:
#             reliability_counts[
#                 reliability
#             ] += 1

#         # ====================================================
#         # 1. DETERMINE CANDIDATE STATE
#         # ====================================================

#         if zone == "INSIDE":

#             candidate_state = "INTRUSION"

#         elif zone == "BOUNDARY":

#             candidate_state = "BOUNDARY_RISK"

#         elif (
#             zone == "OUTSIDE"
#             and approaching_boundary
#             and velocity is not None
#             and velocity >=
#                 APPROACHING_SPEED_THRESHOLD_MPS
#         ):

#             candidate_state = "APPROACHING"

#         else:

#             candidate_state = "NO_INTRUSION"

#         # ====================================================
#         # 2. STATE PERSISTENCE
#         # ====================================================

#         if candidate_state == current_state:

#             state_persistence += 1

#         else:

#             current_state = candidate_state
#             state_persistence = 1

#         # ----------------------------------------------------
#         # Intrusion persistence
#         # ----------------------------------------------------

#         if candidate_state == "INTRUSION":

#             intrusion_persistence += 1

#         else:

#             intrusion_persistence = 0

#         # ====================================================
#         # 3. CONFIRM INTRUSION
#         # ====================================================

#         if (
#             intrusion_persistence
#             >= INTRUSION_PERSISTENCE_OBSERVATIONS
#         ):

#             intrusion_confirmed = True

#         elif candidate_state != "INTRUSION":

#             intrusion_confirmed = False

#         # ====================================================
#         # 4. TIME-TO-INTRUSION
#         # ====================================================
#         #
#         # IMPORTANT:
#         #
#         # We only estimate TTI when:
#         #
#         # - snake is outside
#         # - moving toward the boundary
#         # - velocity is meaningful
#         # - distance is available
#         #
#         # This is a provisional estimate because the current
#         # homography is not physically validated.
#         # ====================================================

#         tti = None

#         if (
#             zone == "OUTSIDE"
#             and approaching_boundary
#             and distance_to_boundary is not None
#             and velocity is not None
#             and velocity >=
#                 MIN_INWARD_VELOCITY_MPS
#         ):

#             # Avoid invalid division.
#             if velocity > 0:

#                 tti = (
#                     distance_to_boundary
#                     / velocity
#                 )

#                 if not math.isfinite(tti):
#                     tti = None

#                 elif tti < 0:
#                     tti = None

#             if tti is not None:

#                 if minimum_tti is None:

#                     minimum_tti = tti

#                 else:

#                     minimum_tti = min(
#                         minimum_tti,
#                         tti
#                     )

#         # ====================================================
#         # 5. RISK SCORE
#         # ====================================================

#         risk_score = calculate_risk_score(
#             zone=zone,
#             distance_to_boundary=
#                 distance_to_boundary,
#             approaching_boundary=
#                 approaching_boundary,
#             velocity=velocity,
#             persistence_count=
#                 state_persistence,
#             reliability=reliability,
#             tti=tti
#         )

#         maximum_risk_score = max(
#             maximum_risk_score,
#             risk_score
#         )

#         # ====================================================
#         # 6. RISK LEVEL
#         # ====================================================

#         risk_level = risk_level_from_score(
#             score=risk_score,
#             zone=zone,
#             intrusion_confirmed=
#                 intrusion_confirmed
#         )

#         # ====================================================
#         # 7. SAVE OBSERVATION
#         # ====================================================

#         risk_history.append({

#             "frame":
#                 frame,

#             "timestamp":
#                 timestamp,

#             "zone":
#                 zone,

#             "distance_to_boundary_m":
#                 safe_float(
#                     distance_to_boundary
#                 ),

#             "velocity_m_per_s":
#                 safe_float(
#                     velocity
#                 ),

#             "approaching_boundary":
#                 approaching_boundary,

#             "tti_s":
#                 safe_float(
#                     tti
#                 ),

#             "intrusion_state":
#                 current_state,

#             "intrusion_confirmed":
#                 intrusion_confirmed,

#             "state_persistence":
#                 state_persistence,

#             "intrusion_persistence":
#                 intrusion_persistence,

#             "risk_score":
#                 round(
#                     risk_score,
#                     2
#                 ),

#             "risk_level":
#                 risk_level,

#             "reliability":
#                 reliability,

#             "observation_type":
#                 obs.get(
#                     "observation_type"
#                 )
#         })

#     # ========================================================
#     # TRACK-LEVEL SUMMARY
#     # ========================================================

#     final_observation = (
#         risk_history[-1]
#         if risk_history
#         else None
#     )

#     final_state = (
#         final_observation[
#             "intrusion_state"
#         ]
#         if final_observation
#         else None
#     )

#     final_risk_level = (
#         final_observation[
#             "risk_level"
#         ]
#         if final_observation
#         else None
#     )

#     risk_tracks.append({

#         "track_id":
#             track_id,

#         "segment_id":
#             segment_id,

#         "history_status":
#             track.get(
#                 "history_status"
#             ),

#         "final_intrusion_state":
#             final_state,

#         "final_risk_level":
#             final_risk_level,

#         "intrusion_confirmed":
#             intrusion_confirmed,

#         "maximum_risk_score":
#             round(
#                 maximum_risk_score,
#                 2
#             ),

#         "minimum_tti_s":
#             safe_float(
#                 minimum_tti
#             ),

#         "reliability_summary":
#             reliability_counts,

#         "history":
#             risk_history
#     })


# # ============================================================
# # 6. OUTPUT OBJECT
# # ============================================================

# output = {

#     "step":
#         "5.9",

#     "step_name":
#         "Intrusion & Risk Calculation",

#     "version":
#         "persistent_spatial_risk_v1",

#     "input_file":
#         str(INPUT_PATH),

#     "video_fps":
#         video_fps,

#     "calibration_status":
#         calibration_status,

#     "calibration_warning":
#         (
#             "Risk calculations involving physical distance, "
#             "velocity and TTI are provisional because the "
#             "Step 5.8 homography is currently "
#             "UNVALIDATED_PLACEHOLDER."
#         ),

#     "parameters":
#         {

#             "boundary_buffer_m":
#                 BOUNDARY_BUFFER_M,

#             "approaching_speed_threshold_mps":
#                 APPROACHING_SPEED_THRESHOLD_MPS,

#             "min_persistence_observations":
#                 MIN_PERSISTENCE_OBSERVATIONS,

#             "intrusion_persistence_observations":
#                 INTRUSION_PERSISTENCE_OBSERVATIONS,

#             "high_reliability_ratio":
#                 HIGH_RELIABILITY_RATIO,

#             "min_inward_velocity_mps":
#                 MIN_INWARD_VELOCITY_MPS,

#             "very_short_tti_seconds":
#                 VERY_SHORT_TTI_SECONDS,

#             "short_tti_seconds":
#                 SHORT_TTI_SECONDS
#         },

#     "intrusion_states":
#         [
#             "NO_INTRUSION",
#             "APPROACHING",
#             "BOUNDARY_RISK",
#             "INTRUSION"
#         ],

#     "risk_levels":
#         [
#             "LOW",
#             "MEDIUM",
#             "HIGH",
#             "CRITICAL"
#         ],

#     "track_count":
#         len(risk_tracks),

#     "tracks":
#         risk_tracks,

#     "processing_notes":
#         {
#             "skipped_tracks":
#                 skipped_tracks,

#             "skipped_observations":
#                 skipped_observations,

#             "calibration_validated":
#                 calibration_status !=
#                 "UNVALIDATED_PLACEHOLDER"
#         }
# }


# # ============================================================
# # 7. WRITE OUTPUT
# # ============================================================

# OUTPUT_PATH.parent.mkdir(
#     parents=True,
#     exist_ok=True
# )

# with open(
#     OUTPUT_PATH,
#     "w",
#     encoding="utf-8"
# ) as f:

#     json.dump(
#         output,
#         f,
#         indent=2
#     )


# # ============================================================
# # 8. CONSOLE SUMMARY
# # ============================================================

# print("\n")
# print("=" * 70)
# print(
#     "STEP 5.9 — INTRUSION & RISK CALCULATION"
# )
# print("=" * 70)

# print(
#     f"\nInput tracks : {len(tracks)}"
# )

# print(
#     f"Processed tracks : {len(risk_tracks)}"
# )

# print(
#     f"Calibration  : {calibration_status}"
# )

# print(
#     f"Boundary buffer : "
#     f"{BOUNDARY_BUFFER_M:.2f} m"
# )

# print(
#     f"Persistence required : "
#     f"{INTRUSION_PERSISTENCE_OBSERVATIONS} observations"
# )

# print("\n")
# print("TRACK RISK SUMMARY")
# print("-" * 70)


# for track in risk_tracks:

#     track_id = track.get(
#         "track_id"
#     )

#     segment_id = track.get(
#         "segment_id"
#     )

#     final_state = (
#         track.get(
#             "final_intrusion_state"
#         )
#         or "N/A"
#     )

#     final_risk = (
#         track.get(
#             "final_risk_level"
#         )
#         or "N/A"
#     )

#     maximum_score = safe_float(
#         track.get(
#             "maximum_risk_score"
#         )
#     )

#     if maximum_score is None:
#         maximum_score_text = "N/A"
#     else:
#         maximum_score_text = (
#             f"{maximum_score:.2f}"
#         )

#     minimum_tti = track.get(
#         "minimum_tti_s"
#     )

#     if minimum_tti is None:
#         minimum_tti_text = "N/A"
#     else:
#         minimum_tti_text = (
#             f"{minimum_tti:.3f} s"
#         )

#     print(
#         f"ID {str(track_id):>2} "
#         f"| Seg {str(segment_id):>2} "
#         f"| State: "
#         f"{str(final_state):<15} "
#         f"| Risk: "
#         f"{str(final_risk):<8}"
#     )

#     print(
#         f"     Intrusion confirmed: "
#         f"{track.get('intrusion_confirmed')}"
#     )

#     print(
#         f"     Maximum risk score: "
#         f"{maximum_score_text}"
#     )

#     print(
#         f"     Minimum TTI: "
#         f"{minimum_tti_text}"
#     )

#     print(
#         f"     Reliability: "
#         f"{track.get('reliability_summary')}"
#     )


# print("\n")
# print("OUTPUT")
# print("-" * 70)
# print(OUTPUT_PATH)

# print("\n")
# print("=" * 70)
# print("STEP 5.9 COMPLETE")
# print("=" * 70)

# print("\nIMPORTANT:")

# print(
#     "Risk, velocity and TTI values are provisional "
#     "until Step 5.8 receives real physical calibration."
# )

# print(
#     "The current Step 5.8 calibration status is "
#     f"{calibration_status}."
# )

# print(
#     "Step 5.10 will consume this file for alert generation."
# )
















# # ============================================================
# # STEP 5.9 — INTRUSION & RISK CALCULATION
# # ============================================================

# from pathlib import Path
# import json
# import math


# # ============================================================
# # 1. PATHS
# # ============================================================

# PROJECT_DIR = Path(
#     "/content/drive/MyDrive/snake_detection/Snake_detection"
# )

# INPUT_PATH = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
#     / "track_spatial_5_8.json"
# )

# OUTPUT_PATH = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
#     / "track_risk_5_9.json"
# )


# # ============================================================
# # 2. PARAMETERS
# # ============================================================
# #
# # IMPORTANT:
# # These are DEVELOPMENT PARAMETERS.
# #
# # They must be validated/tuned after real-world calibration
# # and testing.
# # ============================================================

# # Distance from the protected boundary at which a point is
# # considered spatially close to intrusion.
# BOUNDARY_BUFFER_M = 0.50

# # Minimum speed considered meaningful movement.
# APPROACHING_SPEED_THRESHOLD_MPS = 0.10

# # Number of consecutive observations required before an
# # approaching/boundary state becomes persistent.
# MIN_PERSISTENCE_OBSERVATIONS = 3

# # Number of consecutive observations required for a
# # high-confidence intrusion state.
# INTRUSION_PERSISTENCE_OBSERVATIONS = 3

# # Reliability required for a strong risk assessment.
# HIGH_RELIABILITY_RATIO = 0.60

# # TTI should only be calculated when the snake has a meaningful
# # inward velocity.
# MIN_INWARD_VELOCITY_MPS = 0.05

# # TTI limits used only for risk categorization.
# #
# # These are provisional development thresholds.
# VERY_SHORT_TTI_SECONDS = 2.0
# SHORT_TTI_SECONDS = 5.0


# # ============================================================
# # 3. LOAD STEP 5.8
# # ============================================================

# if not INPUT_PATH.exists():
#     raise FileNotFoundError(
#         f"Step 5.8 output not found:\n{INPUT_PATH}"
#     )

# with open(
#     INPUT_PATH,
#     "r",
#     encoding="utf-8"
# ) as f:
#     data = json.load(f)


# tracks = data.get(
#     "tracks",
#     []
# )

# calibration_status = data.get(
#     "calibration_status",
#     "UNKNOWN"
# )

# video_fps = float(
#     data.get(
#         "video_fps",
#         29.97002997
#     )
# )


# # ============================================================
# # 4. HELPER FUNCTIONS
# # ============================================================

# def safe_float(value):

#     if value is None:
#         return None

#     value = float(value)

#     if math.isfinite(value):
#         return value

#     return None


# def get_reliability_weight(reliability):

#     if reliability == "HIGH":
#         return 1.0

#     if reliability == "MEDIUM":
#         return 0.7

#     if reliability == "LOW":
#         return 0.3

#     return 0.0


# def calculate_risk_score(
#     zone,
#     distance_to_boundary,
#     approaching_boundary,
#     velocity,
#     persistence_count,
#     reliability,
#     tti
# ):
#     """
#     Calculate a provisional 0–100 risk score.

#     This is a development-stage scoring model.
#     It is NOT a physically validated risk probability.
#     """

#     score = 0.0


#     # --------------------------------------------------------
#     # Spatial component
#     # --------------------------------------------------------

#     if zone == "INSIDE":

#         score += 60.0

#     elif zone == "BOUNDARY":

#         score += 40.0

#     elif zone == "OUTSIDE":

#         score += 10.0


#     # --------------------------------------------------------
#     # Boundary proximity
#     # --------------------------------------------------------

#     if distance_to_boundary is not None:

#         if distance_to_boundary <= 0.25:

#             score += 20.0

#         elif distance_to_boundary <= 0.50:

#             score += 15.0

#         elif distance_to_boundary <= 1.00:

#             score += 8.0

#         elif distance_to_boundary <= 2.00:

#             score += 3.0


#     # --------------------------------------------------------
#     # Movement toward boundary
#     # --------------------------------------------------------

#     if approaching_boundary:

#         score += 10.0


#     # --------------------------------------------------------
#     # Velocity
#     # --------------------------------------------------------

#     if velocity is not None:

#         if velocity >= 2.0:

#             score += 10.0

#         elif velocity >= 1.0:

#             score += 6.0

#         elif velocity >= APPROACHING_SPEED_THRESHOLD_MPS:

#             score += 3.0


#     # --------------------------------------------------------
#     # Persistence
#     # --------------------------------------------------------

#     if persistence_count >= 5:

#         score += 10.0

#     elif persistence_count >= 3:

#         score += 6.0


#     # --------------------------------------------------------
#     # Time-to-Intrusion
#     # --------------------------------------------------------

#     if tti is not None:

#         if tti <= VERY_SHORT_TTI_SECONDS:

#             score += 15.0

#         elif tti <= SHORT_TTI_SECONDS:

#             score += 8.0


#     # --------------------------------------------------------
#     # Reliability adjustment
#     # --------------------------------------------------------

#     reliability_weight = (
#         get_reliability_weight(
#             reliability
#         )
#     )

#     # Keep some score even for medium/low reliability,
#     # but reduce confidence in the result.
#     score *= (
#         0.70
#         + 0.30 * reliability_weight
#     )


#     return min(
#         100.0,
#         max(0.0, score)
#     )


# def risk_level_from_score(
#     score,
#     zone,
#     intrusion_confirmed
# ):

#     if intrusion_confirmed:

#         if score >= 75:
#             return "CRITICAL"

#         return "HIGH"


#     if zone == "BOUNDARY":

#         if score >= 60:
#             return "HIGH"

#         if score >= 35:
#             return "MEDIUM"

#         return "LOW"


#     if zone == "OUTSIDE":

#         if score >= 60:
#             return "HIGH"

#         if score >= 30:
#             return "MEDIUM"

#         return "LOW"


#     if zone == "INSIDE":

#         return "HIGH"


#     return "LOW"


# # ============================================================
# # 5. PROCESS TRACKS
# # ============================================================

# risk_tracks = []


# for track in tracks:

#     track_id = track.get(
#         "track_id"
#     )

#     segment_id = track.get(
#         "segment_id"
#     )

#     history = track.get(
#         "history",
#         []
#     )

#     if not history:
#         continue


#     history = sorted(
#         history,
#         key=lambda x: int(
#             x["frame"]
#         )
#     )


#     risk_history = []


#     # --------------------------------------------------------
#     # State counters
#     # --------------------------------------------------------

#     current_state = "NO_INTRUSION"

#     state_persistence = 0

#     intrusion_persistence = 0

#     maximum_risk_score = 0.0

#     minimum_tti = None

#     intrusion_confirmed = False


#     # --------------------------------------------------------
#     # Track-level reliability
#     # --------------------------------------------------------

#     reliability_counts = {
#         "HIGH": 0,
#         "MEDIUM": 0,
#         "LOW": 0
#     }


#     # ========================================================
#     # PROCESS EACH OBSERVATION
#     # ========================================================

#     for obs in history:

#         frame = int(
#             obs["frame"]
#         )

#         timestamp = float(
#             obs["timestamp"]
#         )

#         zone = obs.get(
#             "zone",
#             "UNKNOWN"
#         )

#         distance_to_boundary = obs.get(
#             "distance_to_boundary_m"
#         )

#         velocity = obs.get(
#             "velocity_m_per_s"
#         )

#         approaching_boundary = obs.get(
#             "approaching_boundary"
#         )

#         reliability = obs.get(
#             "reliability",
#             "LOW"
#         )

#         if reliability in reliability_counts:
#             reliability_counts[
#                 reliability
#             ] += 1


#         # ----------------------------------------------------
#         # Normalize values
#         # ----------------------------------------------------

#         if distance_to_boundary is not None:
#             distance_to_boundary = float(
#                 distance_to_boundary
#             )

#         if velocity is not None:
#             velocity = float(
#                 velocity
#             )


#         # ====================================================
#         # 1. DETERMINE CANDIDATE STATE
#         # ====================================================

#         if zone == "INSIDE":

#             candidate_state = "INTRUSION"

#         elif zone == "BOUNDARY":

#             candidate_state = "BOUNDARY_RISK"

#         elif (
#             zone == "OUTSIDE"
#             and approaching_boundary
#             and velocity is not None
#             and velocity >=
#                 APPROACHING_SPEED_THRESHOLD_MPS
#         ):

#             candidate_state = "APPROACHING"

#         else:

#             candidate_state = "NO_INTRUSION"


#         # ====================================================
#         # 2. STATE PERSISTENCE
#         # ====================================================

#         if candidate_state == current_state:

#             state_persistence += 1

#         else:

#             current_state = candidate_state
#             state_persistence = 1


#         # ----------------------------------------------------
#         # Intrusion persistence
#         # ----------------------------------------------------

#         if candidate_state == "INTRUSION":

#             intrusion_persistence += 1

#         else:

#             intrusion_persistence = 0


#         # ====================================================
#         # 3. CONFIRM INTRUSION
#         # ====================================================

#         if (
#             intrusion_persistence
#             >= INTRUSION_PERSISTENCE_OBSERVATIONS
#         ):

#             intrusion_confirmed = True

#         elif candidate_state != "INTRUSION":

#             intrusion_confirmed = False


#         # ====================================================
#         # 4. TIME-TO-INTRUSION
#         # ====================================================
#         #
#         # IMPORTANT:
#         #
#         # We only estimate TTI when:
#         #
#         #   - snake is outside
#         #   - moving toward the boundary
#         #   - velocity is meaningful
#         #   - distance is available
#         #
#         # This is a provisional estimate because the current
#         # homography is not physically validated.
#         # ====================================================

#         tti = None

#         if (
#             zone == "OUTSIDE"
#             and approaching_boundary
#             and distance_to_boundary is not None
#             and velocity is not None
#             and velocity >=
#                 MIN_INWARD_VELOCITY_MPS
#         ):

#             tti = (
#                 distance_to_boundary
#                 / velocity
#             )

#             if minimum_tti is None:
#                 minimum_tti = tti

#             else:
#                 minimum_tti = min(
#                     minimum_tti,
#                     tti
#                 )


#         # ====================================================
#         # 5. RISK SCORE
#         # ====================================================

#         risk_score = calculate_risk_score(
#             zone=zone,
#             distance_to_boundary=
#                 distance_to_boundary,
#             approaching_boundary=
#                 approaching_boundary,
#             velocity=velocity,
#             persistence_count=
#                 state_persistence,
#             reliability=reliability,
#             tti=tti
#         )


#         maximum_risk_score = max(
#             maximum_risk_score,
#             risk_score
#         )


#         # ====================================================
#         # 6. RISK LEVEL
#         # ====================================================

#         risk_level = risk_level_from_score(
#             score=risk_score,
#             zone=zone,
#             intrusion_confirmed=
#                 intrusion_confirmed
#         )


#         # ====================================================
#         # 7. SAVE OBSERVATION
#         # ====================================================

#         risk_history.append({

#             "frame":
#                 frame,

#             "timestamp":
#                 timestamp,

#             "zone":
#                 zone,

#             "distance_to_boundary_m":
#                 safe_float(
#                     distance_to_boundary
#                 ),

#             "velocity_m_per_s":
#                 safe_float(
#                     velocity
#                 ),

#             "approaching_boundary":
#                 approaching_boundary,

#             "tti_s":
#                 safe_float(
#                     tti
#                 ),

#             "intrusion_state":
#                 current_state,

#             "intrusion_confirmed":
#                 intrusion_confirmed,

#             "state_persistence":
#                 state_persistence,

#             "intrusion_persistence":
#                 intrusion_persistence,

#             "risk_score":
#                 round(
#                     risk_score,
#                     2
#                 ),

#             "risk_level":
#                 risk_level,

#             "reliability":
#                 reliability,

#             "observation_type":
#                 obs.get(
#                     "observation_type"
#                 )
#         })


#     # ========================================================
#     # TRACK-LEVEL SUMMARY
#     # ========================================================

#     final_observation = (
#         risk_history[-1]
#         if risk_history
#         else None
#     )

#     final_state = (
#         final_observation[
#             "intrusion_state"
#         ]
#         if final_observation
#         else None
#     )

#     final_risk_level = (
#         final_observation[
#             "risk_level"
#         ]
#         if final_observation
#         else None
#     )


#     risk_tracks.append({

#         "track_id":
#             track_id,

#         "segment_id":
#             segment_id,

#         "history_status":
#             track.get(
#                 "history_status"
#             ),

#         "final_intrusion_state":
#             final_state,

#         "final_risk_level":
#             final_risk_level,

#         "intrusion_confirmed":
#             intrusion_confirmed,

#         "maximum_risk_score":
#             round(
#                 maximum_risk_score,
#                 2
#             ),

#         "minimum_tti_s":
#             safe_float(
#                 minimum_tti
#             ),

#         "reliability_summary":
#             reliability_counts,

#         "history":
#             risk_history
#     })


# # ============================================================
# # 6. OUTPUT OBJECT
# # ============================================================

# output = {

#     "step":
#         "5.9",

#     "step_name":
#         "Intrusion & Risk Calculation",

#     "version":
#         "persistent_spatial_risk_v1",

#     "input_file":
#         str(INPUT_PATH),

#     "video_fps":
#         video_fps,

#     "calibration_status":
#         calibration_status,

#     "calibration_warning":
#         (
#             "Risk calculations involving physical distance, "
#             "velocity and TTI are provisional because the "
#             "Step 5.8 homography is currently "
#             "UNVALIDATED_PLACEHOLDER."
#         ),

#     "parameters":
#         {
#             "boundary_buffer_m":
#                 BOUNDARY_BUFFER_M,

#             "approaching_speed_threshold_mps":
#                 APPROACHING_SPEED_THRESHOLD_MPS,

#             "min_persistence_observations":
#                 MIN_PERSISTENCE_OBSERVATIONS,

#             "intrusion_persistence_observations":
#                 INTRUSION_PERSISTENCE_OBSERVATIONS,

#             "high_reliability_ratio":
#                 HIGH_RELIABILITY_RATIO,

#             "min_inward_velocity_mps":
#                 MIN_INWARD_VELOCITY_MPS,

#             "very_short_tti_seconds":
#                 VERY_SHORT_TTI_SECONDS,

#             "short_tti_seconds":
#                 SHORT_TTI_SECONDS
#         },

#     "intrusion_states":
#         [
#             "NO_INTRUSION",
#             "APPROACHING",
#             "BOUNDARY_RISK",
#             "INTRUSION"
#         ],

#     "risk_levels":
#         [
#             "LOW",
#             "MEDIUM",
#             "HIGH",
#             "CRITICAL"
#         ],

#     "track_count":
#         len(risk_tracks),

#     "tracks":
#         risk_tracks
# }


# # ============================================================
# # 7. WRITE OUTPUT
# # ============================================================

# OUTPUT_PATH.parent.mkdir(
#     parents=True,
#     exist_ok=True
# )

# with open(
#     OUTPUT_PATH,
#     "w",
#     encoding="utf-8"
# ) as f:

#     json.dump(
#         output,
#         f,
#         indent=2
#     )


# # ============================================================
# # 8. CONSOLE SUMMARY
# # ============================================================

# print("\n")
# print("=" * 70)
# print(
#     "STEP 5.9 — INTRUSION & RISK CALCULATION"
# )
# print("=" * 70)

# print(
#     f"\nInput tracks : {len(tracks)}"
# )

# print(
#     f"Calibration  : {calibration_status}"
# )

# print(
#     f"Boundary buffer : "
#     f"{BOUNDARY_BUFFER_M:.2f} m"
# )

# print(
#     f"Persistence required : "
#     f"{INTRUSION_PERSISTENCE_OBSERVATIONS} observations"
# )


# print("\n")
# print("TRACK RISK SUMMARY")
# print("-" * 70)


# for track in risk_tracks:

#     print(
#         f"ID {track['track_id']:>2} "
#         f"| Seg {track['segment_id']:>2} "
#         f"| State: "
#         f"{track['final_intrusion_state']:<15} "
#         f"| Risk: "
#         f"{track['final_risk_level']:<8}"
#     )

#     print(
#         f"     Intrusion confirmed: "
#         f"{track['intrusion_confirmed']}"
#     )

#     print(
#         f"     Maximum risk score: "
#         f"{track['maximum_risk_score']:.2f}"
#     )

#     print(
#         f"     Minimum TTI: "
#         f"{track['minimum_tti_s']}"
#     )


# print("\n")
# print("OUTPUT")
# print("-" * 70)
# print(OUTPUT_PATH)

# print("\n")
# print("=" * 70)
# print("STEP 5.9 COMPLETE")
# print("=" * 70)

# print("\nIMPORTANT:")
# print(
#     "Risk, velocity and TTI values are provisional "
#     "until Step 5.8 receives real physical calibration."
# )

# print(
#     "Step 5.10 will consume this file for alert generation."
# )