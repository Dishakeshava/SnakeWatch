# ============================================================
# STEP 5.10 — ALERT GENERATION
# Pixel-space final version for 24638-344119107_medium
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

INPUT_PATH = (
    PROJECT_DIR
    / "dataset"
    / "tracking_output"
    / "24638-344119107_medium"
    / "track_risk_5_9.json"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "dataset"
    / "tracking_output"
    / "24638-344119107_medium"
    / "track_alerts_5_10.json"
)


# ============================================================
# 2. ALERT PARAMETERS
# ============================================================

# Number of consecutive eligible observations required
# before generating an alert.
ALERT_TRIGGER_PERSISTENCE = 3

# Risk levels eligible to generate an alert.
ALERT_RISK_LEVELS = {
    "HIGH",
    "CRITICAL",
}

# Minimum risk score required for an alert.
ALERT_MIN_RISK_SCORE = 60.0

# Number of consecutive non-eligible observations required
# before an active alert is considered resolved.
ALERT_RESOLUTION_PERSISTENCE = 3


# ============================================================
# 3. LOAD STEP 5.9
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Step 5.9 output not found:\n{INPUT_PATH}"
    )

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

tracks = data.get("tracks", [])

calibration_status = data.get(
    "calibration_status",
    "UNKNOWN",
)

coordinate_system = data.get(
    "coordinate_system",
    "UNKNOWN",
)

spatial_units = data.get(
    "spatial_units",
    "UNKNOWN",
)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=None):
    """Safely convert a value to a finite float."""
    if value is None:
        return default

    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    if math.isfinite(value):
        return value

    return default


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

        if value_lower in {"true", "1", "yes", "y"}:
            return True

        if value_lower in {"false", "0", "no", "n"}:
            return False

    if value is None:
        return default

    return bool(value)


def normalize_intrusion_state(value):
    """Normalize Step 5.9 intrusion state."""
    if value is None:
        return "NO_INTRUSION"

    state = str(value).strip().upper()

    valid_states = {
        "NO_INTRUSION",
        "APPROACHING",
        "BOUNDARY_RISK",
        "INTRUSION",
    }

    if state in valid_states:
        return state

    return "NO_INTRUSION"


def normalize_risk_level(value):
    """Normalize Step 5.9 risk level."""
    if value is None:
        return "LOW"

    level = str(value).strip().upper()

    valid_levels = {
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }

    if level in valid_levels:
        return level

    return "LOW"


def normalize_reliability(value):
    """Normalize reliability label."""
    if value is None:
        return "LOW"

    reliability = str(value).strip().upper()

    if reliability in {"HIGH", "MEDIUM", "LOW"}:
        return reliability

    return "LOW"


def make_alert_id(track_id, segment_id, frame):
    """
    Generate a unique alert identifier.

    Format:
        T<track>_S<segment>_F<frame>
    """
    return f"T{track_id}_S{segment_id}_F{frame}"


# ============================================================
# 5. PROCESS TRACKS
# ============================================================

alert_tracks = []
all_alert_events = []

skipped_tracks = 0
skipped_observations = 0


for track in tracks:

    if not isinstance(track, dict):
        skipped_tracks += 1
        continue

    track_id = track.get("track_id")
    segment_id = track.get("segment_id")

    history = track.get("history", [])

    if not isinstance(history, list) or not history:
        skipped_tracks += 1
        continue

    # --------------------------------------------------------
    # Keep only valid observation dictionaries with a frame.
    # --------------------------------------------------------

    valid_history = []

    for obs in history:

        if not isinstance(obs, dict):
            skipped_observations += 1
            continue

        frame = safe_int(obs.get("frame"))

        if frame is None:
            skipped_observations += 1
            continue

        valid_history.append(obs)

    if not valid_history:
        skipped_tracks += 1
        continue

    # --------------------------------------------------------
    # Sort chronologically.
    # --------------------------------------------------------

    history = sorted(
        valid_history,
        key=lambda x: safe_int(
            x.get("frame"),
            default=0,
        ),
    )

    # --------------------------------------------------------
    # Alert state machine.
    #
    # NO_ALERT
    #     ↓ eligible for persistence
    # ALERT_ACTIVE
    #     ↓ non-eligible for resolution persistence
    # ALERT_RESOLVED
    #     ↓ new eligible event
    # ALERT_ACTIVE
    # --------------------------------------------------------

    alert_state = "NO_ALERT"

    active_alert_id = None
    alert_counter = 0
    resolution_counter = 0

    alert_events = []

    # ========================================================
    # PROCESS OBSERVATIONS
    # ========================================================

    for obs in history:

        frame = safe_int(obs.get("frame"))

        if frame is None:
            skipped_observations += 1
            continue

        timestamp = safe_float(
            obs.get("timestamp"),
            default=0.0,
        )

        intrusion_state = normalize_intrusion_state(
            obs.get(
                "intrusion_state",
                "NO_INTRUSION",
            )
        )

        intrusion_confirmed = safe_bool(
            obs.get(
                "intrusion_confirmed",
                False,
            ),
            default=False,
        )

        risk_level = normalize_risk_level(
            obs.get(
                "risk_level",
                "LOW",
            )
        )

        risk_score = safe_float(
            obs.get(
                "risk_score",
                0.0,
            ),
            default=0.0,
        )

        persistence = safe_int(
            obs.get(
                "state_persistence",
                0,
            ),
            default=0,
        )

        # ----------------------------------------------------
        # FINAL STEP 5.9 IS PIXEL-SPACE.
        #
        # Do NOT interpret these as metres or metres/second.
        # ----------------------------------------------------

        distance_to_boundary_px = safe_float(
            obs.get("distance_to_boundary_px")
        )

        velocity_px_per_s = safe_float(
            obs.get("velocity_px_per_s")
        )

        # Compatibility with alternate pixel-space field names.
        if distance_to_boundary_px is None:
            distance_to_boundary_px = safe_float(
                obs.get("distance_to_boundary")
            )

        if velocity_px_per_s is None:
            velocity_px_per_s = safe_float(
                obs.get("velocity_px_per_s")
            )

        reliability = normalize_reliability(
            obs.get(
                "reliability",
                "LOW",
            )
        )

        observation_type = obs.get(
            "observation_type"
        )

        # ====================================================
        # ALERT ELIGIBILITY
        # ====================================================

        eligible = (
            intrusion_confirmed
            and intrusion_state == "INTRUSION"
            and risk_level in ALERT_RISK_LEVELS
            and risk_score >= ALERT_MIN_RISK_SCORE
            and persistence >= ALERT_TRIGGER_PERSISTENCE
        )

        # ====================================================
        # ELIGIBLE OBSERVATION
        # ====================================================

        if eligible:

            resolution_counter = 0
            alert_counter += 1

            # ------------------------------------------------
            # Trigger a NEW alert.
            #
            # This permits:
            # NO_ALERT -> ALERT_ACTIVE
            # ALERT_RESOLVED -> ALERT_ACTIVE
            # ------------------------------------------------

            if alert_state in {
                "NO_ALERT",
                "ALERT_RESOLVED",
            }:

                active_alert_id = make_alert_id(
                    track_id,
                    segment_id,
                    frame,
                )

                alert_state = "ALERT_ACTIVE"

                alert_event = {
                    "alert_id": active_alert_id,
                    "track_id": track_id,
                    "segment_id": segment_id,
                    "trigger_frame": frame,
                    "trigger_timestamp": timestamp,
                    "intrusion_state": intrusion_state,
                    "risk_level": risk_level,
                    "risk_score": round(
                        risk_score,
                        2,
                    ),
                    "state_persistence": persistence,

                    # Pixel-space metadata only.
                    "distance_to_boundary_px": (
                        round(
                            distance_to_boundary_px,
                            3,
                        )
                        if distance_to_boundary_px is not None
                        else None
                    ),
                    "velocity_px_per_s": (
                        round(
                            velocity_px_per_s,
                            3,
                        )
                        if velocity_px_per_s is not None
                        else None
                    ),

                    # TTI is intentionally unavailable because
                    # Step 5.8 has no physical calibration.
                    "tti_s": None,

                    "reliability": reliability,
                    "observation_type": observation_type,
                    "alert_status": "ACTIVE",
                }

                alert_events.append(alert_event)
                all_alert_events.append(alert_event)

            # ------------------------------------------------
            # Existing alert remains active.
            # ------------------------------------------------

            elif alert_state == "ALERT_ACTIVE":
                alert_state = "ALERT_ACTIVE"

        # ====================================================
        # NON-ELIGIBLE OBSERVATION
        # ====================================================

        else:

            alert_counter = 0

            if alert_state == "ALERT_ACTIVE":

                resolution_counter += 1

                if (
                    resolution_counter
                    >= ALERT_RESOLUTION_PERSISTENCE
                ):

                    alert_state = "ALERT_RESOLVED"

                    # Attach resolution information to the
                    # currently active alert.
                    if alert_events:

                        alert_events[-1][
                            "resolved_frame"
                        ] = frame

                        alert_events[-1][
                            "resolved_timestamp"
                        ] = timestamp

                        alert_events[-1][
                            "alert_status"
                        ] = "RESOLVED"

                    active_alert_id = None
                    resolution_counter = 0

            elif alert_state == "ALERT_RESOLVED":

                # Remain resolved until a new eligible
                # intrusion occurs.
                pass

    # ========================================================
    # TRACK SUMMARY
    # ========================================================

    active_alerts = [
        event
        for event in alert_events
        if event.get("alert_status") == "ACTIVE"
    ]

    resolved_alerts = [
        event
        for event in alert_events
        if event.get("alert_status") == "RESOLVED"
    ]

    alert_tracks.append(
        {
            "track_id": track_id,
            "segment_id": segment_id,
            "final_alert_state": alert_state,
            "alert_count": len(alert_events),
            "active_alert_count": len(active_alerts),
            "resolved_alert_count": len(resolved_alerts),
            "alerts": alert_events,
        }
    )


# ============================================================
# 6. OUTPUT OBJECT
# ============================================================

output = {
    "step": "5.10",
    "step_name": "Alert Generation",
    "version": "persistent_alert_state_machine_v3_pixel_space",

    "input_file": str(INPUT_PATH),

    "calibration_status": calibration_status,
    "coordinate_system": coordinate_system,
    "spatial_units": spatial_units,

    "calibration_warning": (
        "Step 5.8 uses application-defined pixel-space "
        "geometry without physical calibration. Distances "
        "are pixels, velocities are pixels/second, and "
        "physical time-to-intercept (TTI) is unavailable."
    ),

    "parameters": {
        "alert_trigger_persistence":
            ALERT_TRIGGER_PERSISTENCE,

        "alert_min_risk_score":
            ALERT_MIN_RISK_SCORE,

        "alert_risk_levels":
            sorted(ALERT_RISK_LEVELS),

        "alert_resolution_persistence":
            ALERT_RESOLUTION_PERSISTENCE,
    },

    "alert_state_machine": [
        "NO_ALERT",
        "ALERT_ACTIVE",
        "ALERT_RESOLVED",
    ],

    "track_count": len(alert_tracks),

    "total_alert_events": len(all_alert_events),

    "tracks": alert_tracks,

    "alert_events": all_alert_events,

    "processing_notes": {
        "skipped_tracks": skipped_tracks,
        "skipped_observations": skipped_observations,
        "external_notifications_sent": False,
    },
}


# ============================================================
# 7. WRITE OUTPUT
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

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
# 8. CONSOLE SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("STEP 5.10 — ALERT GENERATION")
print("=" * 70)

print(f"\nInput tracks       : {len(tracks)}")
print(f"Processed tracks   : {len(alert_tracks)}")
print(f"Calibration        : {calibration_status}")
print(f"Coordinate system  : {coordinate_system}")
print(f"Spatial units      : {spatial_units}")
print(
    f"Trigger persistence: "
    f"{ALERT_TRIGGER_PERSISTENCE}"
)
print(
    f"Minimum risk score : "
    f"{ALERT_MIN_RISK_SCORE}"
)
print(
    f"Alert risk levels  : "
    f"{sorted(ALERT_RISK_LEVELS)}"
)

print("\n")
print("TRACK ALERT SUMMARY")
print("-" * 70)

for track in alert_tracks:

    track_id = track.get("track_id")
    segment_id = track.get("segment_id")

    final_alert_state = (
        track.get("final_alert_state")
        or "N/A"
    )

    alert_count = track.get(
        "alert_count",
        0,
    )

    print(
        f"ID {str(track_id):>2} "
        f"| Seg {str(segment_id):>2} "
        f"| State: "
        f"{str(final_alert_state):<15} "
        f"| Alerts: "
        f"{alert_count}"
    )

    for event in track.get(
        "alerts",
        [],
    ):

        trigger_timestamp = safe_float(
            event.get("trigger_timestamp"),
            default=0.0,
        )

        risk_score = safe_float(
            event.get("risk_score"),
            default=0.0,
        )

        print(
            f"     Alert "
            f"{event.get('alert_id')} "
            f"| Frame "
            f"{event.get('trigger_frame')} "
            f"| Time "
            f"{trigger_timestamp:.2f}s "
            f"| Risk "
            f"{event.get('risk_level')} "
            f"| Score "
            f"{risk_score:.2f} "
            f"| Status "
            f"{event.get('alert_status')}"
        )

        distance_px = event.get(
            "distance_to_boundary_px"
        )

        velocity_px_s = event.get(
            "velocity_px_per_s"
        )

        if distance_px is not None:
            print(
                f"       Boundary distance: "
                f"{distance_px:.2f} px"
            )

        if velocity_px_s is not None:
            print(
                f"       Velocity: "
                f"{velocity_px_s:.2f} px/s"
            )

        if (
            event.get("alert_status")
            == "RESOLVED"
        ):

            resolved_timestamp = safe_float(
                event.get("resolved_timestamp"),
                default=0.0,
            )

            print(
                f"       Resolved at frame "
                f"{event.get('resolved_frame')} "
                f"| Time "
                f"{resolved_timestamp:.2f}s"
            )


print("\n")
print("TOTAL ALERT EVENTS")
print("-" * 70)
print(len(all_alert_events))

print("\n")
print("OUTPUT")
print("-" * 70)
print(OUTPUT_PATH)

print("\n")
print("=" * 70)
print("STEP 5.10 COMPLETE")
print("=" * 70)

print("\nIMPORTANT:")
print(
    "This step generates alert EVENTS only."
)
print(
    "No SMS, email, Telegram, app notification, "
    "or external notification has been sent."
)
print(
    "Pixel-space mode is active: distances are pixels "
    "and velocities are pixels/second."
)
print(
    "Physical TTI is disabled because Step 5.8 has "
    "no real-world metric calibration."
)














# # ============================================================
# # STEP 5.10 — ALERT GENERATION
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
#     / "track_risk_5_9.json"
# )

# OUTPUT_PATH = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
#     / "24638-344119107_medium"
#     / "track_alerts_5_10.json"
# )


# # ============================================================
# # 2. ALERT PARAMETERS
# # ============================================================

# # Number of consecutive observations required before
# # generating an alert.
# ALERT_TRIGGER_PERSISTENCE = 3

# # Risk levels that are eligible to generate an alert.
# ALERT_RISK_LEVELS = {
#     "HIGH",
#     "CRITICAL"
# }

# # Minimum risk score for an alert.
# ALERT_MIN_RISK_SCORE = 60.0

# # Number of observations after which an active alert
# # can be considered resolved if the intrusion state
# # is no longer present.
# ALERT_RESOLUTION_PERSISTENCE = 3


# # ============================================================
# # 3. LOAD STEP 5.9
# # ============================================================

# if not INPUT_PATH.exists():
#     raise FileNotFoundError(
#         f"Step 5.9 output not found:\n{INPUT_PATH}"
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


# # ============================================================
# # 4. HELPER FUNCTIONS
# # ============================================================

# def safe_float(value, default=None):
#     """
#     Safely convert a value to a finite float.

#     Returns default for:
#     - None
#     - invalid strings
#     - NaN
#     - positive/negative infinity
#     """
#     if value is None:
#         return default

#     try:
#         value = float(value)
#     except (TypeError, ValueError):
#         return default

#     if math.isfinite(value):
#         return value

#     return default


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
#             "true",
#             "1",
#             "yes",
#             "y"
#         }:
#             return True

#         if value_lower in {
#             "false",
#             "0",
#             "no",
#             "n"
#         }:
#             return False

#     if value is None:
#         return default

#     return bool(value)


# def normalize_intrusion_state(value):
#     """Normalize intrusion state."""
#     if value is None:
#         return "NO_INTRUSION"

#     state = str(value).strip().upper()

#     valid_states = {
#         "NO_INTRUSION",
#         "APPROACHING",
#         "BOUNDARY_RISK",
#         "INTRUSION"
#     }

#     if state in valid_states:
#         return state

#     return "NO_INTRUSION"


# def normalize_risk_level(value):
#     """Normalize risk level."""
#     if value is None:
#         return "LOW"

#     level = str(value).strip().upper()

#     valid_levels = {
#         "LOW",
#         "MEDIUM",
#         "HIGH",
#         "CRITICAL"
#     }

#     if level in valid_levels:
#         return level

#     return "LOW"


# def normalize_reliability(value):
#     """Normalize reliability label."""
#     if value is None:
#         return "LOW"

#     reliability = str(value).strip().upper()

#     if reliability in {
#         "HIGH",
#         "MEDIUM",
#         "LOW"
#     }:
#         return reliability

#     return "LOW"


# def make_alert_id(
#     track_id,
#     segment_id,
#     frame
# ):
#     """
#     Generate a unique alert identifier.

#     Format:
#         T<track>_S<segment>_F<frame>
#     """
#     return (
#         f"T{track_id}"
#         f"_S{segment_id}"
#         f"_F{frame}"
#     )


# # ============================================================
# # 5. PROCESS TRACKS
# # ============================================================

# alert_tracks = []

# all_alert_events = []

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
#     # Keep only valid observation dictionaries.
#     # --------------------------------------------------------

#     valid_history = []

#     for obs in history:

#         if not isinstance(obs, dict):
#             skipped_observations += 1
#             continue

#         frame = safe_int(
#             obs.get("frame")
#         )

#         if frame is None:
#             skipped_observations += 1
#             continue

#         valid_history.append(obs)

#     if not valid_history:
#         skipped_tracks += 1
#         continue

#     # --------------------------------------------------------
#     # Sort chronologically.
#     # --------------------------------------------------------

#     history = sorted(
#         valid_history,
#         key=lambda x: safe_int(
#             x.get("frame"),
#             default=0
#         )
#     )

#     # --------------------------------------------------------
#     # Alert state
#     # --------------------------------------------------------

#     alert_state = "NO_ALERT"

#     active_alert_id = None

#     alert_counter = 0

#     resolution_counter = 0

#     alert_events = []

#     # ========================================================
#     # PROCESS OBSERVATIONS
#     # ========================================================

#     for obs in history:

#         frame = safe_int(
#             obs.get("frame")
#         )

#         if frame is None:
#             skipped_observations += 1
#             continue

#         timestamp = safe_float(
#             obs.get("timestamp"),
#             default=0.0
#         )

#         intrusion_state = normalize_intrusion_state(
#             obs.get(
#                 "intrusion_state",
#                 "NO_INTRUSION"
#             )
#         )

#         intrusion_confirmed = safe_bool(
#             obs.get(
#                 "intrusion_confirmed",
#                 False
#             ),
#             default=False
#         )

#         risk_level = normalize_risk_level(
#             obs.get(
#                 "risk_level",
#                 "LOW"
#             )
#         )

#         risk_score = safe_float(
#             obs.get(
#                 "risk_score",
#                 0.0
#             ),
#             default=0.0
#         )

#         persistence = safe_int(
#             obs.get(
#                 "state_persistence",
#                 0
#             ),
#             default=0
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

#         tti = safe_float(
#             obs.get(
#                 "tti_s"
#             )
#         )

#         reliability = normalize_reliability(
#             obs.get(
#                 "reliability",
#                 "LOW"
#             )
#         )

#         observation_type = obs.get(
#             "observation_type"
#         )

#         # ====================================================
#         # ALERT ELIGIBILITY
#         # ====================================================

#         eligible = (
#             intrusion_confirmed
#             and intrusion_state == "INTRUSION"
#             and risk_level
#                 in ALERT_RISK_LEVELS
#             and risk_score
#                 >= ALERT_MIN_RISK_SCORE
#             and persistence
#                 >= ALERT_TRIGGER_PERSISTENCE
#         )

#         # ====================================================
#         # ACTIVE ALERT / NEW ALERT
#         # ====================================================

#         if eligible:

#             resolution_counter = 0

#             alert_counter += 1

#             # ------------------------------------------------
#             # Trigger a NEW alert.
#             #
#             # IMPORTANT:
#             # A previously resolved alert may trigger a NEW
#             # alert if another valid intrusion occurs.
#             # ------------------------------------------------

#             if alert_state in {
#                 "NO_ALERT",
#                 "ALERT_RESOLVED"
#             }:

#                 active_alert_id = make_alert_id(
#                     track_id,
#                     segment_id,
#                     frame
#                 )

#                 alert_state = (
#                     "ALERT_ACTIVE"
#                 )

#                 alert_event = {

#                     "alert_id":
#                         active_alert_id,

#                     "track_id":
#                         track_id,

#                     "segment_id":
#                         segment_id,

#                     "trigger_frame":
#                         frame,

#                     "trigger_timestamp":
#                         timestamp,

#                     "intrusion_state":
#                         intrusion_state,

#                     "risk_level":
#                         risk_level,

#                     "risk_score":
#                         round(
#                             risk_score,
#                             2
#                         ),

#                     "state_persistence":
#                         persistence,

#                     "distance_to_boundary_m":
#                         safe_float(
#                             distance_to_boundary
#                         ),

#                     "velocity_m_per_s":
#                         safe_float(
#                             velocity
#                         ),

#                     "tti_s":
#                         safe_float(
#                             tti
#                         ),

#                     "reliability":
#                         reliability,

#                     "observation_type":
#                         observation_type,

#                     "alert_status":
#                         "ACTIVE"
#                 }

#                 alert_events.append(
#                     alert_event
#                 )

#                 all_alert_events.append(
#                     alert_event
#                 )

#             # ------------------------------------------------
#             # Existing alert remains active.
#             # ------------------------------------------------

#             elif alert_state == "ALERT_ACTIVE":

#                 alert_state = (
#                     "ALERT_ACTIVE"
#                 )

#         # ====================================================
#         # POSSIBLE ALERT RESOLUTION
#         # ====================================================

#         else:

#             alert_counter = 0

#             if alert_state == "ALERT_ACTIVE":

#                 resolution_counter += 1

#                 if (
#                     resolution_counter
#                     >= ALERT_RESOLUTION_PERSISTENCE
#                 ):

#                     alert_state = (
#                         "ALERT_RESOLVED"
#                     )

#                     # Attach resolution information
#                     # to the currently active alert.

#                     if alert_events:

#                         alert_events[-1][
#                             "resolved_frame"
#                         ] = frame

#                         alert_events[-1][
#                             "resolved_timestamp"
#                         ] = timestamp

#                         alert_events[-1][
#                             "alert_status"
#                         ] = "RESOLVED"

#                     active_alert_id = None

#                     resolution_counter = 0

#             elif alert_state == "ALERT_RESOLVED":

#                 # Remain resolved until a new eligible
#                 # intrusion is detected.
#                 pass

#     # ========================================================
#     # TRACK SUMMARY
#     # ========================================================

#     active_alerts = [
#         event
#         for event in alert_events
#         if event.get(
#             "alert_status"
#         ) == "ACTIVE"
#     ]

#     resolved_alerts = [
#         event
#         for event in alert_events
#         if event.get(
#             "alert_status"
#         ) == "RESOLVED"
#     ]

#     alert_tracks.append({

#         "track_id":
#             track_id,

#         "segment_id":
#             segment_id,

#         "final_alert_state":
#             alert_state,

#         "alert_count":
#             len(alert_events),

#         "active_alert_count":
#             len(active_alerts),

#         "resolved_alert_count":
#             len(resolved_alerts),

#         "alerts":
#             alert_events
#     })


# # ============================================================
# # 6. OUTPUT OBJECT
# # ============================================================

# output = {

#     "step":
#         "5.10",

#     "step_name":
#         "Alert Generation",

#     "version":
#         "persistent_alert_state_machine_v2",

#     "input_file":
#         str(INPUT_PATH),

#     "calibration_status":
#         calibration_status,

#     "calibration_warning":
#         (
#             "Alert decisions depend on Step 5.9 risk "
#             "calculations. Physical distance, velocity "
#             "and TTI remain provisional while Step 5.8 "
#             "uses placeholder calibration."
#         ),

#     "parameters":
#         {

#             "alert_trigger_persistence":
#                 ALERT_TRIGGER_PERSISTENCE,

#             "alert_min_risk_score":
#                 ALERT_MIN_RISK_SCORE,

#             "alert_risk_levels":
#                 sorted(
#                     ALERT_RISK_LEVELS
#                 ),

#             "alert_resolution_persistence":
#                 ALERT_RESOLUTION_PERSISTENCE
#         },

#     "alert_state_machine":
#         [
#             "NO_ALERT",
#             "ALERT_ACTIVE",
#             "ALERT_RESOLVED"
#         ],

#     "track_count":
#         len(alert_tracks),

#     "total_alert_events":
#         len(all_alert_events),

#     "tracks":
#         alert_tracks,

#     "alert_events":
#         all_alert_events,

#     "processing_notes":
#         {
#             "skipped_tracks":
#                 skipped_tracks,

#             "skipped_observations":
#                 skipped_observations,

#             "external_notifications_sent":
#                 False
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
#     "STEP 5.10 — ALERT GENERATION"
# )
# print("=" * 70)

# print(
#     f"\nInput tracks : {len(tracks)}"
# )

# print(
#     f"Processed tracks : "
#     f"{len(alert_tracks)}"
# )

# print(
#     f"Calibration : "
#     f"{calibration_status}"
# )

# print(
#     f"Trigger persistence : "
#     f"{ALERT_TRIGGER_PERSISTENCE}"
# )

# print(
#     f"Minimum risk score : "
#     f"{ALERT_MIN_RISK_SCORE}"
# )

# print(
#     f"Alert risk levels : "
#     f"{sorted(ALERT_RISK_LEVELS)}"
# )

# print("\n")
# print("TRACK ALERT SUMMARY")
# print("-" * 70)


# for track in alert_tracks:

#     track_id = track.get(
#         "track_id"
#     )

#     segment_id = track.get(
#         "segment_id"
#     )

#     final_alert_state = (
#         track.get(
#             "final_alert_state"
#         )
#         or "N/A"
#     )

#     alert_count = track.get(
#         "alert_count",
#         0
#     )

#     print(
#         f"ID {str(track_id):>2} "
#         f"| Seg {str(segment_id):>2} "
#         f"| State: "
#         f"{str(final_alert_state):<15} "
#         f"| Alerts: "
#         f"{alert_count}"
#     )

#     for event in track.get(
#         "alerts",
#         []
#     ):

#         trigger_timestamp = safe_float(
#             event.get(
#                 "trigger_timestamp"
#             ),
#             default=0.0
#         )

#         risk_score = safe_float(
#             event.get(
#                 "risk_score"
#             ),
#             default=0.0
#         )

#         print(
#             f"     Alert "
#             f"{event.get('alert_id')} "
#             f"| Frame "
#             f"{event.get('trigger_frame')} "
#             f"| Time "
#             f"{trigger_timestamp:.2f}s "
#             f"| Risk "
#             f"{event.get('risk_level')} "
#             f"| Score "
#             f"{risk_score:.2f} "
#             f"| Status "
#             f"{event.get('alert_status')}"
#         )

#         if (
#             event.get(
#                 "alert_status"
#             ) == "RESOLVED"
#         ):

#             resolved_timestamp = safe_float(
#                 event.get(
#                     "resolved_timestamp"
#                 ),
#                 default=0.0
#             )

#             print(
#                 f"     Resolved at frame "
#                 f"{event.get('resolved_frame')} "
#                 f"| Time "
#                 f"{resolved_timestamp:.2f}s"
#             )


# print("\n")
# print("TOTAL ALERT EVENTS")
# print("-" * 70)
# print(
#     len(all_alert_events)
# )

# print("\n")
# print("OUTPUT")
# print("-" * 70)
# print(OUTPUT_PATH)

# print("\n")
# print("=" * 70)
# print("STEP 5.10 COMPLETE")
# print("=" * 70)

# print("\nIMPORTANT:")

# print(
#     "This step generates alert EVENTS only."
# )

# print(
#     "No SMS, email, Telegram, app notification, "
#     "or external notification has been sent."
# )

# print(
#     "Physical risk values remain provisional until "
#     "Step 5.8 receives real calibration."
# )












# # ============================================================
# # STEP 5.10 — ALERT GENERATION
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
#     / "track_risk_5_9.json"
# )

# OUTPUT_PATH = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
#     / "track_alerts_5_10.json"
# )


# # ============================================================
# # 2. ALERT PARAMETERS
# # ============================================================

# # Number of consecutive observations required before
# # generating an alert.
# ALERT_TRIGGER_PERSISTENCE = 3

# # Risk levels that are eligible to generate an alert.
# ALERT_RISK_LEVELS = {
#     "HIGH",
#     "CRITICAL"
# }

# # Minimum risk score for an alert.
# ALERT_MIN_RISK_SCORE = 60.0

# # Number of observations after which an active alert
# # can be considered resolved if the intrusion state
# # is no longer present.
# ALERT_RESOLUTION_PERSISTENCE = 3


# # ============================================================
# # 3. LOAD STEP 5.9
# # ============================================================

# if not INPUT_PATH.exists():

#     raise FileNotFoundError(
#         f"Step 5.9 output not found:\n{INPUT_PATH}"
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
# # 4. HELPER
# # ============================================================

# def safe_float(value):

#     if value is None:
#         return None

#     value = float(value)

#     if math.isfinite(value):
#         return value

#     return None


# # ============================================================
# # 5. PROCESS TRACKS
# # ============================================================

# alert_tracks = []

# all_alert_events = []


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


#     # --------------------------------------------------------
#     # Alert state
#     # --------------------------------------------------------

#     alert_state = "NO_ALERT"

#     active_alert_id = None

#     alert_counter = 0

#     resolution_counter = 0

#     alert_events = []


#     # ========================================================
#     # PROCESS OBSERVATIONS
#     # ========================================================

#     for obs in history:

#         frame = int(
#             obs["frame"]
#         )

#         timestamp = float(
#             obs["timestamp"]
#         )

#         intrusion_state = obs.get(
#             "intrusion_state",
#             "NO_INTRUSION"
#         )

#         intrusion_confirmed = bool(
#             obs.get(
#                 "intrusion_confirmed",
#                 False
#             )
#         )

#         risk_level = obs.get(
#             "risk_level",
#             "LOW"
#         )

#         risk_score = float(
#             obs.get(
#                 "risk_score",
#                 0.0
#             )
#         )

#         persistence = int(
#             obs.get(
#                 "state_persistence",
#                 0
#             )
#         )

#         distance_to_boundary = obs.get(
#             "distance_to_boundary_m"
#         )

#         velocity = obs.get(
#             "velocity_m_per_s"
#         )

#         tti = obs.get(
#             "tti_s"
#         )

#         reliability = obs.get(
#             "reliability",
#             "LOW"
#         )

#         observation_type = obs.get(
#             "observation_type"
#         )


#         # ====================================================
#         # ALERT ELIGIBILITY
#         # ====================================================

#         eligible = (

#             intrusion_confirmed

#             and risk_level
#                 in ALERT_RISK_LEVELS

#             and risk_score
#                 >= ALERT_MIN_RISK_SCORE

#             and persistence
#                 >= ALERT_TRIGGER_PERSISTENCE
#         )


#         # ====================================================
#         # ACTIVE ALERT
#         # ====================================================

#         if eligible:

#             resolution_counter = 0

#             alert_counter += 1


#             # ------------------------------------------------
#             # Trigger a NEW alert
#             # ------------------------------------------------

#             if alert_state == "NO_ALERT":

#                 active_alert_id = (
#                     f"T{track_id}"
#                     f"_S{segment_id}"
#                     f"_F{frame}"
#                 )

#                 alert_state = (
#                     "ALERT_ACTIVE"
#                 )


#                 alert_event = {

#                     "alert_id":
#                         active_alert_id,

#                     "track_id":
#                         track_id,

#                     "segment_id":
#                         segment_id,

#                     "trigger_frame":
#                         frame,

#                     "trigger_timestamp":
#                         timestamp,

#                     "intrusion_state":
#                         intrusion_state,

#                     "risk_level":
#                         risk_level,

#                     "risk_score":
#                         round(
#                             risk_score,
#                             2
#                         ),

#                     "state_persistence":
#                         persistence,

#                     "distance_to_boundary_m":
#                         safe_float(
#                             distance_to_boundary
#                         ),

#                     "velocity_m_per_s":
#                         safe_float(
#                             velocity
#                         ),

#                     "tti_s":
#                         safe_float(
#                             tti
#                         ),

#                     "reliability":
#                         reliability,

#                     "observation_type":
#                         observation_type,

#                     "alert_status":
#                         "ACTIVE"
#                 }


#                 alert_events.append(
#                     alert_event
#                 )

#                 all_alert_events.append(
#                     alert_event
#                 )


#             # ------------------------------------------------
#             # Existing alert remains active
#             # ------------------------------------------------

#             elif alert_state == "ALERT_ACTIVE":

#                 alert_state = (
#                     "ALERT_ACTIVE"
#                 )


#         # ====================================================
#         # POSSIBLE ALERT RESOLUTION
#         # ====================================================

#         else:

#             alert_counter = 0

#             if alert_state == "ALERT_ACTIVE":

#                 resolution_counter += 1


#                 if (
#                     resolution_counter
#                     >= ALERT_RESOLUTION_PERSISTENCE
#                 ):

#                     alert_state = (
#                         "ALERT_RESOLVED"
#                     )

#                     # Attach resolution information
#                     # to the active alert.

#                     if alert_events:

#                         alert_events[-1][
#                             "resolved_frame"
#                         ] = frame

#                         alert_events[-1][
#                             "resolved_timestamp"
#                         ] = timestamp

#                         alert_events[-1][
#                             "alert_status"
#                         ] = "RESOLVED"


#                     active_alert_id = None

#                     resolution_counter = 0


#             elif alert_state == "ALERT_RESOLVED":

#                 # Remain resolved until a new eligible
#                 # intrusion is detected.

#                 pass


#     # ========================================================
#     # TRACK SUMMARY
#     # ========================================================

#     active_alerts = [
#         event
#         for event in alert_events
#         if event.get(
#             "alert_status"
#         ) == "ACTIVE"
#     ]

#     resolved_alerts = [
#         event
#         for event in alert_events
#         if event.get(
#             "alert_status"
#         ) == "RESOLVED"
#     ]


#     alert_tracks.append({

#         "track_id":
#             track_id,

#         "segment_id":
#             segment_id,

#         "final_alert_state":
#             alert_state,

#         "alert_count":
#             len(alert_events),

#         "active_alert_count":
#             len(active_alerts),

#         "resolved_alert_count":
#             len(resolved_alerts),

#         "alerts":
#             alert_events
#     })


# # ============================================================
# # 6. OUTPUT OBJECT
# # ============================================================

# output = {

#     "step":
#         "5.10",

#     "step_name":
#         "Alert Generation",

#     "version":
#         "persistent_alert_state_machine_v1",

#     "input_file":
#         str(INPUT_PATH),

#     "video_fps":
#         video_fps,

#     "calibration_status":
#         calibration_status,

#     "calibration_warning":
#         (
#             "Alert decisions depend on Step 5.9 risk "
#             "calculations. Physical distance, velocity "
#             "and TTI remain provisional while Step 5.8 "
#             "uses placeholder calibration."
#         ),

#     "parameters":
#         {

#             "alert_trigger_persistence":
#                 ALERT_TRIGGER_PERSISTENCE,

#             "alert_min_risk_score":
#                 ALERT_MIN_RISK_SCORE,

#             "alert_risk_levels":
#                 sorted(
#                     ALERT_RISK_LEVELS
#                 ),

#             "alert_resolution_persistence":
#                 ALERT_RESOLUTION_PERSISTENCE
#         },

#     "alert_state_machine":
#         [
#             "NO_ALERT",
#             "ALERT_ACTIVE",
#             "ALERT_RESOLVED"
#         ],

#     "track_count":
#         len(alert_tracks),

#     "total_alert_events":
#         len(all_alert_events),

#     "tracks":
#         alert_tracks,

#     "alert_events":
#         all_alert_events
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
#     "STEP 5.10 — ALERT GENERATION"
# )
# print("=" * 70)

# print(
#     f"\nInput tracks : {len(tracks)}"
# )

# print(
#     f"Calibration  : "
#     f"{calibration_status}"
# )

# print(
#     f"Trigger persistence : "
#     f"{ALERT_TRIGGER_PERSISTENCE}"
# )

# print(
#     f"Minimum risk score : "
#     f"{ALERT_MIN_RISK_SCORE}"
# )

# print(
#     f"Alert risk levels : "
#     f"{sorted(ALERT_RISK_LEVELS)}"
# )


# print("\n")
# print("TRACK ALERT SUMMARY")
# print("-" * 70)


# for track in alert_tracks:

#     print(
#         f"ID {track['track_id']:>2} "
#         f"| Seg {track['segment_id']:>2} "
#         f"| State: "
#         f"{track['final_alert_state']:<15} "
#         f"| Alerts: "
#         f"{track['alert_count']}"
#     )

#     for event in track["alerts"]:

#         print(
#             f"     Alert {event['alert_id']} "
#             f"| Frame "
#             f"{event['trigger_frame']} "
#             f"| Time "
#             f"{event['trigger_timestamp']:.2f}s "
#             f"| Risk "
#             f"{event['risk_level']} "
#             f"| Score "
#             f"{event['risk_score']:.2f} "
#             f"| Status "
#             f"{event['alert_status']}"
#         )


# print("\n")
# print("TOTAL ALERT EVENTS")
# print("-" * 70)
# print(
#     len(all_alert_events)
# )


# print("\n")
# print("OUTPUT")
# print("-" * 70)
# print(OUTPUT_PATH)


# print("\n")
# print("=" * 70)
# print("STEP 5.10 COMPLETE")
# print("=" * 70)

# print("\nIMPORTANT:")

# print(
#     "This step generates alert EVENTS only."
# )

# print(
#     "No SMS, email, Telegram, app notification, "
#     "or external notification has been sent."
# )

# print(
#     "Physical risk values remain provisional until "
#     "Step 5.8 receives real calibration."
# )