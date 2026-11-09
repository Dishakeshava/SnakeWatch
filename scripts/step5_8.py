
# ============================================================
# STEP 5.8 — SPATIAL MAPPING & GEOFENCING
# PIXEL-SPACE VERSION
# ============================================================
"""
Purpose
-------
Convert the movement history from Step 5.7 into spatial/geofence
information without pretending that the online video has metric
camera calibration.

Why pixel space?
----------------
The source video was found online and provides no reliable:
    - ground measurements,
    - camera calibration,
    - known-size landmarks,
    - measured world coordinates.

Therefore a metre-based homography would be artificial.

This version works entirely in IMAGE PIXELS and reports:
    - pixel position
    - distance to geofence boundary in pixels
    - displacement in pixels
    - velocity in pixels/second
    - approach to boundary
    - zone transitions
    - spatial summaries

IMPORTANT
---------
The protected polygon below is an APPLICATION ROI in pixel
coordinates. It is NOT claimed to be a physical farm boundary.
Change PROTECTED_POLYGON_PX when the intended protected region
is known.

This output is designed to be consumed by a corresponding
pixel-space version of Step 5.9.
"""

from pathlib import Path
import json
import math

import cv2
import numpy as np


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
    / "track_movement_5_7_revised.json"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "dataset"
    / "tracking_output"
    / "24638-344119107_medium"
    / "track_spatial_5_8.json"
)


# ============================================================
# 2. VIDEO / IMAGE GEOMETRY
# ============================================================

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720


# ============================================================
# 3. PIXEL-SPACE CONFIGURATION
# ============================================================

COORDINATE_SYSTEM = "image_pixel_space"

PIXEL_UNITS = "pixels"

CALIBRATION_STATUS = "NOT_APPLICABLE_PIXEL_SPACE"

CALIBRATION_WARNING = (
    "Metric camera calibration is not available for this "
    "online source video. Spatial quantities are reported "
    "in image pixels, not metres. Pixel displacement and "
    "pixel velocity are not physical-world measurements."
)


# ============================================================
# 4. GEOFENCE
# ============================================================
"""
The protected polygon is defined directly in IMAGE PIXELS.

IMPORTANT:
This is an application-defined ROI, not a measured physical
boundary from the source video.

The default below uses an inset rectangular ROI so that the
geofence has a real boundary inside the image.

Change these four points if your intended protected region
is different.

Point order:
    top-left
    top-right
    bottom-right
    bottom-left
"""

PROTECTED_POLYGON_PX = np.float32([
    [100.0, 60.0],
    [1180.0, 60.0],
    [1180.0, 660.0],
    [100.0, 660.0]
])

# Distance from the geofence boundary at which an observation
# is considered to be in the BOUNDARY zone.
#
# This is a PIXEL threshold, not metres.
BOUNDARY_BUFFER_PX = 50.0


# ============================================================
# 5. LOAD STEP 5.7
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Step 5.7 output not found:\n{INPUT_PATH}\n\n"
        "Run Step 5.7 first."
    )

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

tracks = data.get("tracks", [])

video_fps = float(
    data.get("video_fps", 29.97002997)
)


# ============================================================
# 6. BASIC VALIDATION
# ============================================================

if len(PROTECTED_POLYGON_PX) < 3:
    raise ValueError(
        "Protected polygon requires at least 3 points."
    )

if video_fps <= 0:
    raise ValueError(
        f"Invalid video FPS: {video_fps}"
    )

for point in PROTECTED_POLYGON_PX:
    x = float(point[0])
    y = float(point[1])

    if not (
        0.0 <= x <= VIDEO_WIDTH
        and 0.0 <= y <= VIDEO_HEIGHT
    ):
        raise ValueError(
            "A protected polygon point lies outside the "
            f"{VIDEO_WIDTH}x{VIDEO_HEIGHT} image."
        )

if BOUNDARY_BUFFER_PX < 0:
    raise ValueError(
        "BOUNDARY_BUFFER_PX cannot be negative."
    )


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def safe_float(value):
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if math.isfinite(value):
        return value

    return None


def safe_int(value):
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def valid_point(point):
    if point is None:
        return False

    if not isinstance(point, (list, tuple, np.ndarray)):
        return False

    if len(point) < 2:
        return False

    x = safe_float(point[0])
    y = safe_float(point[1])

    return (
        x is not None
        and y is not None
        and math.isfinite(x)
        and math.isfinite(y)
    )


def euclidean_distance_px(p1, p2):
    if not valid_point(p1) or not valid_point(p2):
        return None

    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])

    return math.sqrt(
        dx * dx + dy * dy
    )


def point_to_polygon_boundary_distance_px(
    point,
    polygon
):
    """
    Minimum Euclidean distance from a pixel point to the
    pixel-space geofence boundary.
    """

    if not valid_point(point):
        return None

    px = float(point[0])
    py = float(point[1])

    min_distance = float("inf")

    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]

        x1 = float(p1[0])
        y1 = float(p1[1])

        x2 = float(p2[0])
        y2 = float(p2[1])

        dx = x2 - x1
        dy = y2 - y1

        segment_length_squared = (
            dx * dx + dy * dy
        )

        if segment_length_squared == 0:
            distance = math.sqrt(
                (px - x1) ** 2
                + (py - y1) ** 2
            )

        else:
            t = (
                (px - x1) * dx
                + (py - y1) * dy
            ) / segment_length_squared

            t = max(
                0.0,
                min(1.0, t)
            )

            closest_x = x1 + t * dx
            closest_y = y1 + t * dy

            distance = math.sqrt(
                (px - closest_x) ** 2
                + (py - closest_y) ** 2
            )

        min_distance = min(
            min_distance,
            distance
        )

    return min_distance


def classify_zone(
    point,
    polygon,
    boundary_buffer_px
):
    """
    Classify a pixel-space point as:

        INSIDE
        BOUNDARY
        OUTSIDE
        UNKNOWN
    """

    if not valid_point(point):
        return "UNKNOWN"

    polygon_array = np.asarray(
        polygon,
        dtype=np.float32
    )

    point_tuple = (
        float(point[0]),
        float(point[1])
    )

    signed_distance = cv2.pointPolygonTest(
        polygon_array,
        point_tuple,
        True
    )

    # Inside polygon
    if signed_distance > 0:

        if signed_distance <= boundary_buffer_px:
            return "BOUNDARY"

        return "INSIDE"

    # Exactly on polygon boundary
    if signed_distance == 0:
        return "BOUNDARY"

    # Outside polygon
    boundary_distance = (
        point_to_polygon_boundary_distance_px(
            point,
            polygon
        )
    )

    if (
        boundary_distance is not None
        and boundary_distance <= boundary_buffer_px
    ):
        return "BOUNDARY"

    return "OUTSIDE"


def get_observation_center(obs):
    """
    Prefer the smoothed center generated by Step 5.7.
    Fall back to the raw center if needed.
    """

    point = obs.get("smoothed_center")

    if valid_point(point):
        return [
            float(point[0]),
            float(point[1])
        ]

    point = obs.get("center")

    if valid_point(point):
        return [
            float(point[0]),
            float(point[1])
        ]

    return None


def normalize_timestamp(obs, fps):
    """
    Step 5.7 normally provides timestamp.

    If timestamp is unavailable, derive it from frame/FPS.
    """

    timestamp = safe_float(
        obs.get("timestamp")
    )

    if timestamp is not None:
        return timestamp

    frame_id = safe_int(
        obs.get("frame")
    )

    if frame_id is None:
        return None

    return frame_id / fps


# ============================================================
# 8. PROCESS TRACKS
# ============================================================

spatial_tracks = []

for track in tracks:

    track_id = track.get("track_id")
    segment_id = track.get("segment_id")

    history = track.get(
        "history",
        []
    )

    if not isinstance(history, list):
        continue

    if not history:
        continue

    # Keep only dictionary observations with a usable frame.
    cleaned_history = []

    for obs in history:

        if not isinstance(obs, dict):
            continue

        frame_id = safe_int(
            obs.get("frame")
        )

        if frame_id is None:
            continue

        cleaned_history.append(
            (frame_id, obs)
        )

    cleaned_history.sort(
        key=lambda item: item[0]
    )

    if not cleaned_history:
        continue

    spatial_history = []

    previous_pixel_position = None
    previous_timestamp = None
    previous_zone = None

    minimum_boundary_distance = None

    inside_count = 0
    boundary_count = 0
    outside_count = 0
    unknown_count = 0

    # --------------------------------------------------------
    # PER-OBSERVATION SPATIAL ANALYSIS
    # --------------------------------------------------------

    for frame_id, obs in cleaned_history:

        timestamp = normalize_timestamp(
            obs,
            video_fps
        )

        pixel_center = get_observation_center(
            obs
        )

        zone = classify_zone(
            pixel_center,
            PROTECTED_POLYGON_PX,
            BOUNDARY_BUFFER_PX
        )

        boundary_distance = (
            point_to_polygon_boundary_distance_px(
                pixel_center,
                PROTECTED_POLYGON_PX
            )
            if pixel_center is not None
            else None
        )

        if boundary_distance is not None:

            if (
                minimum_boundary_distance is None
                or boundary_distance
                < minimum_boundary_distance
            ):
                minimum_boundary_distance = (
                    boundary_distance
                )

        # ----------------------------------------------------
        # Zone counts
        # ----------------------------------------------------

        if zone == "INSIDE":
            inside_count += 1

        elif zone == "BOUNDARY":
            boundary_count += 1

        elif zone == "OUTSIDE":
            outside_count += 1

        else:
            unknown_count += 1

        # ----------------------------------------------------
        # Pixel displacement / velocity
        # ----------------------------------------------------

        displacement_px = None
        velocity_px_per_s = None

        if (
            pixel_center is not None
            and previous_pixel_position is not None
            and timestamp is not None
            and previous_timestamp is not None
        ):

            dt = (
                timestamp
                - previous_timestamp
            )

            if dt > 0:

                displacement_px = (
                    euclidean_distance_px(
                        previous_pixel_position,
                        pixel_center
                    )
                )

                if displacement_px is not None:

                    velocity_px_per_s = (
                        displacement_px / dt
                    )

        # ----------------------------------------------------
        # Boundary approach
        # ----------------------------------------------------

        approaching_boundary = None

        if (
            boundary_distance is not None
            and previous_pixel_position is not None
        ):

            previous_boundary_distance = (
                point_to_polygon_boundary_distance_px(
                    previous_pixel_position,
                    PROTECTED_POLYGON_PX
                )
            )

            if previous_boundary_distance is not None:

                approaching_boundary = (
                    boundary_distance
                    < previous_boundary_distance
                )

        # ----------------------------------------------------
        # Zone transition
        # ----------------------------------------------------

        zone_transition = None

        if (
            previous_zone is not None
            and zone != previous_zone
        ):
            zone_transition = (
                f"{previous_zone}_TO_{zone}"
            )

        # ----------------------------------------------------
        # Save observation
        # ----------------------------------------------------

        spatial_history.append({

            "frame": frame_id,

            "timestamp": (
                safe_float(timestamp)
            ),

            "pixel_center": (
                pixel_center
            ),

            "zone": zone,

            "distance_to_boundary_px": (
                safe_float(
                    boundary_distance
                )
            ),

            "displacement_px": (
                safe_float(
                    displacement_px
                )
            ),

            "velocity_px_per_s": (
                safe_float(
                    velocity_px_per_s
                )
            ),

            "approaching_boundary": (
                approaching_boundary
            ),

            "zone_transition": (
                zone_transition
            ),

            "observation_type": (
                obs.get(
                    "observation_type"
                )
            ),

            "confidence": (
                safe_float(
                    obs.get("confidence")
                )
            ),

            "reliability": (
                obs.get("reliability")
            )
        })

        previous_pixel_position = (
            pixel_center
        )

        previous_timestamp = (
            timestamp
        )

        previous_zone = zone

    # --------------------------------------------------------
    # TRACK SUMMARY
    # --------------------------------------------------------

    initial_zone = (
        spatial_history[0]["zone"]
        if spatial_history
        else None
    )

    final_zone = (
        spatial_history[-1]["zone"]
        if spatial_history
        else None
    )

    boundary_crossed = any(
        item.get("zone_transition")
        in [
            "OUTSIDE_TO_INSIDE",
            "BOUNDARY_TO_INSIDE",
            "INSIDE_TO_OUTSIDE",
            "INSIDE_TO_BOUNDARY"
        ]
        for item in spatial_history
        if item.get("zone_transition") is not None
    )

    valid_velocities = [
        item["velocity_px_per_s"]
        for item in spatial_history
        if item["velocity_px_per_s"] is not None
    ]

    maximum_velocity = (
        max(valid_velocities)
        if valid_velocities
        else None
    )

    average_velocity = (
        sum(valid_velocities)
        / len(valid_velocities)
        if valid_velocities
        else None
    )

    spatial_tracks.append({

        "track_id": track_id,

        "segment_id": segment_id,

        "history_status": (
            track.get("history_status")
        ),

        "initial_zone": initial_zone,

        "final_zone": final_zone,

        "boundary_crossed": (
            boundary_crossed
        ),

        "inside_observations": (
            inside_count
        ),

        "boundary_observations": (
            boundary_count
        ),

        "outside_observations": (
            outside_count
        ),

        "unknown_observations": (
            unknown_count
        ),

        "minimum_boundary_distance_px": (
            safe_float(
                minimum_boundary_distance
            )
        ),

        "average_velocity_px_per_s": (
            safe_float(
                average_velocity
            )
        ),

        "maximum_velocity_px_per_s": (
            safe_float(
                maximum_velocity
            )
        ),

        "history": spatial_history
    })


# ============================================================
# 9. OUTPUT OBJECT
# ============================================================

output = {

    "step": "5.8",

    "step_name":
        "Spatial Mapping & Geofencing",

    "version":
        "pixel_space_geofence_v1",

    "input_file":
        str(INPUT_PATH),

    "video_width":
        VIDEO_WIDTH,

    "video_height":
        VIDEO_HEIGHT,

    "video_fps":
        video_fps,

    "coordinate_system":
        COORDINATE_SYSTEM,

    "spatial_units":
        PIXEL_UNITS,

    "calibration_status":
        CALIBRATION_STATUS,

    "calibration_warning":
        CALIBRATION_WARNING,

    "metric_calibration_available":
        False,

    "protected_polygon_px":
        PROTECTED_POLYGON_PX.tolist(),

    "boundary_buffer_px":
        BOUNDARY_BUFFER_PX,

    "track_count":
        len(spatial_tracks),

    "tracks":
        spatial_tracks
}


# ============================================================
# 10. WRITE OUTPUT
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
# 11. CONSOLE SUMMARY
# ============================================================

print()
print("=" * 70)
print("STEP 5.8 — SPATIAL MAPPING & GEOFENCING")
print("=" * 70)

print(
    f"\nInput tracks : {len(tracks)}"
)

print(
    f"Processed tracks : {len(spatial_tracks)}"
)

print(
    f"Video size   : "
    f"{VIDEO_WIDTH} x {VIDEO_HEIGHT}"
)

print(
    f"FPS          : {video_fps:.6f}"
)

print(
    f"Coordinate system : {COORDINATE_SYSTEM}"
)

print(
    f"Spatial units : {PIXEL_UNITS}"
)

print(
    f"Calibration  : {CALIBRATION_STATUS}"
)

print(
    f"Boundary buf.: "
    f"{BOUNDARY_BUFFER_PX:.1f} px"
)

print()
print("TRACK SPATIAL SUMMARY")
print("-" * 70)

for track in spatial_tracks:

    track_id = track.get(
        "track_id"
    )

    segment_id = track.get(
        "segment_id"
    )

    initial_zone = track.get(
        "initial_zone"
    )

    final_zone = track.get(
        "final_zone"
    )

    print(
        f"ID {str(track_id):>2} "
        f"| Seg {str(segment_id):>2} "
        f"| Initial: "
        f"{str(initial_zone):<8} "
        f"| Final: "
        f"{str(final_zone):<8}"
    )

    print(
        f"     Outside: "
        f"{track.get('outside_observations', 0)} | "
        f"Boundary: "
        f"{track.get('boundary_observations', 0)} | "
        f"Inside: "
        f"{track.get('inside_observations', 0)} | "
        f"Unknown: "
        f"{track.get('unknown_observations', 0)}"
    )

    print(
        f"     Min boundary distance: "
        f"{track.get('minimum_boundary_distance_px')} px"
    )

    print(
        f"     Avg velocity: "
        f"{track.get('average_velocity_px_per_s')} px/s"
    )

    print(
        f"     Max velocity: "
        f"{track.get('maximum_velocity_px_per_s')} px/s"
    )

    print(
        f"     Boundary crossed: "
        f"{track.get('boundary_crossed')}"
    )


print()
print("OUTPUT")
print("-" * 70)
print(OUTPUT_PATH)

print()
print("=" * 70)
print("STEP 5.8 COMPLETE")
print("=" * 70)

print()
print("IMPORTANT:")
print(
    "Metric calibration is NOT available for this online video."
)
print(
    "All spatial distances are pixels and all spatial velocities "
    "are pixels/second."
)
print(
    "The protected polygon is an application-defined pixel ROI; "
    "adjust PROTECTED_POLYGON_PX to match the intended protected area."
)
print(
    "Do not interpret pixel values as metres or m/s."
)




















# # ============================================================
# # STEP 5.8 — SPATIAL MAPPING & GEOFENCING
# # ============================================================

# from pathlib import Path
# import json
# import math

# import cv2
# import numpy as np


# # ============================================================
# # 1. PATHS
# # ============================================================

# PROJECT_DIR = Path(
#     "/content/drive/MyDrive/snake_detection/Snake_detection"
# )

# # Step 5.7 output for the 24638 video.
# INPUT_PATH = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
#     / "24638-344119107_medium"
#     / "track_movement_5_7_revised.json"
# )

# # Step 5.8 output.
# OUTPUT_PATH = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
#     / "24638-344119107_medium"
#     / "track_spatial_5_8.json"
# )


# # ============================================================
# # 2. VIDEO / IMAGE GEOMETRY
# # ============================================================

# # 24638-344119107_medium.mp4
# VIDEO_WIDTH = 1280
# VIDEO_HEIGHT = 720


# # ============================================================
# # 3. CALIBRATION
# # ============================================================
# #
# # IMPORTANT:
# #
# # The four IMAGE_POINTS below MUST be replaced with four
# # physically identifiable points on the actual ground plane
# # visible in the 24638 video.
# #
# # WORLD_POINTS must contain the REAL measured coordinates
# # of those same four points, in metres.
# #
# # The order MUST correspond:
# #
# # IMAGE_POINTS[0] <-> WORLD_POINTS[0]
# # IMAGE_POINTS[1] <-> WORLD_POINTS[1]
# # IMAGE_POINTS[2] <-> WORLD_POINTS[2]
# # IMAGE_POINTS[3] <-> WORLD_POINTS[3]
# #
# # DO NOT use arbitrary pixel corners and arbitrary metre
# # dimensions and call that a real calibration.
# #
# # Until measured correspondences are supplied, the code
# # deliberately remains UNVALIDATED_PLACEHOLDER.
# # ============================================================


# # ------------------------------------------------------------
# # PLACEHOLDER POINTS
# #
# # These are ONLY temporary values so the pipeline can be
# # executed and tested.
# # ------------------------------------------------------------

# IMAGE_POINTS = np.float32([
#     [100, 100],       # P1
#     [1180, 100],      # P2
#     [1180, 620],      # P3
#     [100, 620]        # P4
# ])

# WORLD_POINTS = np.float32([
#     [0.0, 0.0],       # P1
#     [10.0, 0.0],      # P2
#     [10.0, 6.0],      # P3
#     [0.0, 6.0]        # P4
# ])


# # Automatically detect whether the placeholder values are
# # still being used.
# PLACEHOLDER_IMAGE_POINTS = np.float32([
#     [100, 100],
#     [1180, 100],
#     [1180, 620],
#     [100, 620]
# ])

# PLACEHOLDER_WORLD_POINTS = np.float32([
#     [0.0, 0.0],
#     [10.0, 0.0],
#     [10.0, 6.0],
#     [0.0, 6.0]
# ])


# def points_match(a, b):
#     return np.allclose(
#         np.asarray(a, dtype=np.float32),
#         np.asarray(b, dtype=np.float32)
#     )


# USING_PLACEHOLDER_CALIBRATION = (
#     points_match(
#         IMAGE_POINTS,
#         PLACEHOLDER_IMAGE_POINTS
#     )
#     and points_match(
#         WORLD_POINTS,
#         PLACEHOLDER_WORLD_POINTS
#     )
# )

# if USING_PLACEHOLDER_CALIBRATION:
#     CALIBRATION_STATUS = "UNVALIDATED_PLACEHOLDER"
# else:
#     CALIBRATION_STATUS = "MEASURED_CALIBRATION"


# # ============================================================
# # 4. GEOFENCE
# # ============================================================
# #
# # Protected region in WORLD coordinates.
# #
# # IMPORTANT:
# # This polygon must represent the actual protected region
# # after calibration.
# #
# # For now it follows the supplied calibration rectangle.
# # ============================================================

# PROTECTED_POLYGON = np.float32([
#     [0.0, 0.0],
#     [10.0, 0.0],
#     [10.0, 6.0],
#     [0.0, 6.0]
# ])


# # Distance from the protected boundary at which the point
# # is considered to be in the BOUNDARY zone.
# #
# # This is a development parameter and should be tuned after
# # real calibration.
# BOUNDARY_BUFFER_METERS = 0.50


# # ============================================================
# # 5. LOAD STEP 5.7
# # ============================================================

# if not INPUT_PATH.exists():
#     raise FileNotFoundError(
#         f"Step 5.7 output not found:\n{INPUT_PATH}"
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


# def safe_float(value, default=None):
#     if value is None:
#         return default

#     try:
#         value = float(value)
#     except (TypeError, ValueError):
#         return default

#     if math.isfinite(value):
#         return value

#     return default


# video_fps = safe_float(
#     data.get(
#         "video_fps",
#         29.97002997
#     ),
#     default=29.97002997
# )


# # ============================================================
# # 6. BASIC VALIDATION
# # ============================================================

# if len(IMAGE_POINTS) < 4:
#     raise ValueError(
#         "At least 4 point correspondences are required."
#     )

# if len(IMAGE_POINTS) != len(WORLD_POINTS):
#     raise ValueError(
#         "IMAGE_POINTS and WORLD_POINTS must contain "
#         "the same number of points."
#     )

# if len(PROTECTED_POLYGON) < 3:
#     raise ValueError(
#         "Protected polygon requires at least 3 points."
#     )

# # Image points must lie inside the video frame.
# for i, point in enumerate(IMAGE_POINTS):
#     x = float(point[0])
#     y = float(point[1])

#     if not (
#         0 <= x < VIDEO_WIDTH
#         and 0 <= y < VIDEO_HEIGHT
#     ):
#         raise ValueError(
#             f"IMAGE_POINTS[{i}] = ({x}, {y}) "
#             f"is outside the {VIDEO_WIDTH}x{VIDEO_HEIGHT} "
#             "video frame."
#         )

# # World points must be finite.
# if not np.isfinite(WORLD_POINTS).all():
#     raise ValueError(
#         "WORLD_POINTS contains invalid values."
#     )

# # A physically calibrated run must use measured points.
# # The placeholder run is allowed only for development/testing.
# if USING_PLACEHOLDER_CALIBRATION:
#     print(
#         "\nWARNING: Placeholder calibration is still active."
#     )
#     print(
#         "Physical distance, velocity and TTI values are "
#         "NOT physically validated."
#     )


# # ============================================================
# # 7. COMPUTE HOMOGRAPHY
# # ============================================================

# H, homography_mask = cv2.findHomography(
#     IMAGE_POINTS,
#     WORLD_POINTS,
#     method=cv2.RANSAC,
#     ransacReprojThreshold=3.0
# )

# if H is None:
#     raise RuntimeError(
#         "Homography calculation failed."
#     )


# # ============================================================
# # 8. HOMOGRAPHY QUALITY CHECK
# # ============================================================

# # Reproject calibration points to the world plane and calculate
# # the reprojection error.
# projected_world = cv2.perspectiveTransform(
#     IMAGE_POINTS.reshape(-1, 1, 2),
#     H
# ).reshape(-1, 2)

# calibration_errors = []

# for actual, predicted in zip(
#     WORLD_POINTS,
#     projected_world
# ):
#     error = math.sqrt(
#         (
#             float(actual[0])
#             - float(predicted[0])
#         ) ** 2
#         +
#         (
#             float(actual[1])
#             - float(predicted[1])
#         ) ** 2
#     )

#     calibration_errors.append(
#         error
#     )

# mean_reprojection_error = (
#     sum(calibration_errors)
#     / len(calibration_errors)
# )

# max_reprojection_error = max(
#     calibration_errors
# )


# # ============================================================
# # 9. HELPER FUNCTIONS
# # ============================================================

# def pixel_to_world(point):
#     """
#     Convert one image point (x, y) into world coordinates
#     (X, Y) using the homography.
#     """

#     if point is None:
#         return None

#     if not isinstance(point, (list, tuple, np.ndarray)):
#         return None

#     if len(point) < 2:
#         return None

#     x = safe_float(point[0])
#     y = safe_float(point[1])

#     if x is None or y is None:
#         return None

#     point_array = np.array(
#         [[[x, y]]],
#         dtype=np.float32
#     )

#     transformed = cv2.perspectiveTransform(
#         point_array,
#         H
#     )

#     X = safe_float(
#         transformed[0, 0, 0]
#     )

#     Y = safe_float(
#         transformed[0, 0, 1]
#     )

#     if X is None or Y is None:
#         return None

#     return [X, Y]


# def euclidean_distance(p1, p2):
#     if p1 is None or p2 is None:
#         return None

#     try:
#         dx = float(p2[0]) - float(p1[0])
#         dy = float(p2[1]) - float(p1[1])
#     except (TypeError, ValueError, IndexError):
#         return None

#     distance = math.sqrt(
#         dx * dx + dy * dy
#     )

#     return safe_float(distance)


# def point_to_polygon_boundary_distance(
#     point,
#     polygon
# ):
#     """
#     Minimum Euclidean distance from a point to the
#     polygon boundary.

#     Distance is calculated in world coordinates (metres).
#     """

#     if point is None:
#         return None

#     min_distance = float("inf")

#     for i in range(len(polygon)):

#         p1 = polygon[i]

#         p2 = polygon[
#             (i + 1) % len(polygon)
#         ]

#         px = float(point[0])
#         py = float(point[1])

#         x1 = float(p1[0])
#         y1 = float(p1[1])

#         x2 = float(p2[0])
#         y2 = float(p2[1])

#         dx = x2 - x1
#         dy = y2 - y1

#         segment_length_squared = (
#             dx * dx + dy * dy
#         )

#         if segment_length_squared == 0:

#             distance = math.sqrt(
#                 (px - x1) ** 2
#                 +
#                 (py - y1) ** 2
#             )

#         else:

#             t = (
#                 (px - x1) * dx
#                 +
#                 (py - y1) * dy
#             ) / segment_length_squared

#             t = max(
#                 0.0,
#                 min(1.0, t)
#             )

#             closest_x = x1 + t * dx
#             closest_y = y1 + t * dy

#             distance = math.sqrt(
#                 (px - closest_x) ** 2
#                 +
#                 (py - closest_y) ** 2
#             )

#         min_distance = min(
#             min_distance,
#             distance
#         )

#     return safe_float(
#         min_distance
#     )


# def classify_zone(
#     point,
#     polygon,
#     boundary_buffer
# ):
#     """
#     Classify a world-coordinate point as:

#         INSIDE
#         BOUNDARY
#         OUTSIDE
#     """

#     if point is None:
#         return "UNKNOWN"

#     polygon_array = np.asarray(
#         polygon,
#         dtype=np.float32
#     )

#     point_array = (
#         float(point[0]),
#         float(point[1])
#     )

#     signed_distance = cv2.pointPolygonTest(
#         polygon_array,
#         point_array,
#         True
#     )

#     # Inside polygon.
#     if signed_distance > 0:

#         if abs(signed_distance) <= boundary_buffer:
#             return "BOUNDARY"

#         return "INSIDE"

#     # Exactly on boundary.
#     if signed_distance == 0:
#         return "BOUNDARY"

#     # Outside polygon.
#     boundary_distance = (
#         point_to_polygon_boundary_distance(
#             point,
#             polygon
#         )
#     )

#     if (
#         boundary_distance is not None
#         and boundary_distance <= boundary_buffer
#     ):
#         return "BOUNDARY"

#     return "OUTSIDE"


# def get_ground_point_from_observation(obs):
#     """
#     Get the point used for ground-plane mapping.

#     Preferred order:
#         1. smoothed_center
#         2. center

#     This is the current tracking representation available
#     from Step 5.7.
#     """

#     point = obs.get(
#         "smoothed_center"
#     )

#     if point is not None:
#         return point

#     return obs.get(
#         "center"
#     )


# # ============================================================
# # 10. PROCESS TRACKS
# # ============================================================

# spatial_tracks = []

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

#     if not isinstance(
#         history,
#         list
#     ) or not history:

#         skipped_tracks += 1
#         continue

#     valid_history = []

#     for obs in history:

#         if not isinstance(obs, dict):
#             skipped_observations += 1
#             continue

#         frame_id = safe_float(
#             obs.get("frame")
#         )

#         if frame_id is None:
#             skipped_observations += 1
#             continue

#         valid_history.append(
#             obs
#         )

#     if not valid_history:
#         skipped_tracks += 1
#         continue

#     history = sorted(
#         valid_history,
#         key=lambda x: int(
#             safe_float(
#                 x.get("frame"),
#                 0
#             )
#         )
#     )

#     spatial_history = []

#     previous_world_position = None
#     previous_timestamp = None
#     previous_zone = None

#     minimum_boundary_distance = None

#     inside_count = 0
#     boundary_count = 0
#     outside_count = 0
#     unknown_count = 0


#     # ========================================================
#     # PER-FRAME SPATIAL ANALYSIS
#     # ========================================================

#     for obs in history:

#         frame_value = safe_float(
#             obs.get("frame")
#         )

#         if frame_value is None:
#             skipped_observations += 1
#             continue

#         frame_id = int(
#             frame_value
#         )

#         timestamp = safe_float(
#             obs.get("timestamp")
#         )

#         if timestamp is None:

#             timestamp = (
#                 frame_id
#                 / video_fps
#                 if video_fps > 0
#                 else 0.0
#             )

#         pixel_center = (
#             get_ground_point_from_observation(
#                 obs
#             )
#         )

#         # ----------------------------------------------------
#         # Pixel → world coordinate
#         # ----------------------------------------------------

#         world_position = pixel_to_world(
#             pixel_center
#         )

#         # ----------------------------------------------------
#         # Zone classification
#         # ----------------------------------------------------

#         zone = classify_zone(
#             world_position,
#             PROTECTED_POLYGON,
#             BOUNDARY_BUFFER_METERS
#         )

#         # ----------------------------------------------------
#         # Boundary distance
#         # ----------------------------------------------------

#         boundary_distance = (
#             point_to_polygon_boundary_distance(
#                 world_position,
#                 PROTECTED_POLYGON
#             )
#         )

#         if boundary_distance is not None:

#             if (
#                 minimum_boundary_distance is None
#                 or
#                 boundary_distance
#                 < minimum_boundary_distance
#             ):

#                 minimum_boundary_distance = (
#                     boundary_distance
#                 )

#         # ----------------------------------------------------
#         # Count spatial states
#         # ----------------------------------------------------

#         if zone == "INSIDE":
#             inside_count += 1

#         elif zone == "BOUNDARY":
#             boundary_count += 1

#         elif zone == "OUTSIDE":
#             outside_count += 1

#         else:
#             unknown_count += 1

#         # ----------------------------------------------------
#         # Real-world displacement / velocity
#         # ----------------------------------------------------

#         displacement_m = None
#         velocity_m_per_s = None

#         if (
#             world_position is not None
#             and previous_world_position is not None
#             and previous_timestamp is not None
#         ):

#             dt = (
#                 timestamp
#                 - previous_timestamp
#             )

#             if dt > 0:

#                 displacement_m = (
#                     euclidean_distance(
#                         previous_world_position,
#                         world_position
#                     )
#                 )

#                 if displacement_m is not None:

#                     velocity_m_per_s = (
#                         displacement_m / dt
#                     )

#         # ----------------------------------------------------
#         # Boundary approach
#         # ----------------------------------------------------

#         approaching_boundary = None

#         if (
#             boundary_distance is not None
#             and previous_world_position is not None
#         ):

#             previous_boundary_distance = (
#                 point_to_polygon_boundary_distance(
#                     previous_world_position,
#                     PROTECTED_POLYGON
#                 )
#             )

#             if previous_boundary_distance is not None:

#                 approaching_boundary = (
#                     boundary_distance
#                     < previous_boundary_distance
#                 )

#         # ----------------------------------------------------
#         # Zone transition
#         # ----------------------------------------------------

#         zone_transition = None

#         if (
#             previous_zone is not None
#             and zone != previous_zone
#         ):

#             zone_transition = (
#                 f"{previous_zone}_TO_{zone}"
#             )

#         # ----------------------------------------------------
#         # Save spatial observation
#         # ----------------------------------------------------

#         spatial_history.append({

#             "frame":
#                 frame_id,

#             "timestamp":
#                 timestamp,

#             "pixel_center":
#                 pixel_center,

#             "world_position_m":
#                 world_position,

#             "zone":
#                 zone,

#             "distance_to_boundary_m":
#                 safe_float(
#                     boundary_distance
#                 ),

#             "displacement_m":
#                 safe_float(
#                     displacement_m
#                 ),

#             "velocity_m_per_s":
#                 safe_float(
#                     velocity_m_per_s
#                 ),

#             "approaching_boundary":
#                 approaching_boundary,

#             "zone_transition":
#                 zone_transition,

#             "observation_type":
#                 obs.get(
#                     "observation_type"
#                 ),

#             "confidence":
#                 safe_float(
#                     obs.get(
#                         "confidence"
#                     )
#                 ),

#             "reliability":
#                 obs.get(
#                     "reliability"
#                 )
#         })

#         previous_world_position = (
#             world_position
#         )

#         previous_timestamp = (
#             timestamp
#         )

#         previous_zone = zone


#     # ========================================================
#     # TRACK SUMMARY
#     # ========================================================

#     initial_zone = (
#         spatial_history[0]["zone"]
#         if spatial_history
#         else None
#     )

#     final_zone = (
#         spatial_history[-1]["zone"]
#         if spatial_history
#         else None
#     )

#     boundary_crossed = any(
#         item["zone_transition"]
#         in [
#             "OUTSIDE_TO_INSIDE",
#             "BOUNDARY_TO_INSIDE",
#             "INSIDE_TO_OUTSIDE",
#             "INSIDE_TO_BOUNDARY"
#         ]
#         for item in spatial_history
#         if item["zone_transition"]
#         is not None
#     )

#     valid_velocities = [
#         item["velocity_m_per_s"]
#         for item in spatial_history
#         if item["velocity_m_per_s"]
#         is not None
#     ]

#     maximum_velocity = (
#         max(valid_velocities)
#         if valid_velocities
#         else None
#     )

#     average_velocity = (
#         sum(valid_velocities)
#         / len(valid_velocities)
#         if valid_velocities
#         else None
#     )

#     spatial_tracks.append({

#         "track_id":
#             track_id,

#         "segment_id":
#             segment_id,

#         "history_status":
#             track.get(
#                 "history_status"
#             ),

#         "initial_zone":
#             initial_zone,

#         "final_zone":
#             final_zone,

#         "boundary_crossed":
#             boundary_crossed,

#         "inside_observations":
#             inside_count,

#         "boundary_observations":
#             boundary_count,

#         "outside_observations":
#             outside_count,

#         "unknown_observations":
#             unknown_count,

#         "minimum_boundary_distance_m":
#             safe_float(
#                 minimum_boundary_distance
#             ),

#         "average_velocity_m_per_s":
#             safe_float(
#                 average_velocity
#             ),

#         "maximum_velocity_m_per_s":
#             safe_float(
#                 maximum_velocity
#             ),

#         "history":
#             spatial_history
#     })


# # ============================================================
# # 11. OUTPUT OBJECT
# # ============================================================

# if USING_PLACEHOLDER_CALIBRATION:

#     calibration_warning = (
#         "UNVALIDATED_PLACEHOLDER calibration. "
#         "Physical distances, velocities and TTI are "
#         "NOT physically validated. Replace IMAGE_POINTS "
#         "and WORLD_POINTS with measured correspondences "
#         "from the actual camera scene."
#     )

# else:

#     calibration_warning = (
#         "Measured calibration supplied. "
#         "Physical values depend on the accuracy of the "
#         "measured image/world correspondences."
#     )


# output = {

#     "step":
#         "5.8",

#     "step_name":
#         "Spatial Mapping & Geofencing",

#     "version":
#         "homography_geofence_v2",

#     "input_file":
#         str(INPUT_PATH),

#     "video_width":
#         VIDEO_WIDTH,

#     "video_height":
#         VIDEO_HEIGHT,

#     "video_fps":
#         video_fps,

#     "coordinate_system":
#         "image_pixels_to_world_ground_plane",

#     "world_units":
#         "meters",

#     "calibration_status":
#         CALIBRATION_STATUS,

#     "calibration_warning":
#         calibration_warning,

#     "homography":
#         H.tolist(),

#     "homography_mask":
#         (
#             homography_mask.tolist()
#             if homography_mask is not None
#             else None
#         ),

#     "calibration_image_points":
#         IMAGE_POINTS.tolist(),

#     "calibration_world_points_m":
#         WORLD_POINTS.tolist(),

#     "calibration_reprojection_error_mean_m":
#         safe_float(
#             mean_reprojection_error
#         ),

#     "calibration_reprojection_error_max_m":
#         safe_float(
#             max_reprojection_error
#         ),

#     "protected_polygon_world_m":
#         PROTECTED_POLYGON.tolist(),

#     "boundary_buffer_m":
#         BOUNDARY_BUFFER_METERS,

#     "track_count":
#         len(spatial_tracks),

#     "tracks":
#         spatial_tracks,

#     "processing_notes":
#         {
#             "using_placeholder_calibration":
#                 USING_PLACEHOLDER_CALIBRATION,

#             "skipped_tracks":
#                 skipped_tracks,

#             "skipped_observations":
#                 skipped_observations
#         }
# }


# # ============================================================
# # 12. WRITE OUTPUT
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
# # 13. CONSOLE SUMMARY
# # ============================================================

# print("\n")
# print("=" * 70)
# print(
#     "STEP 5.8 — SPATIAL MAPPING & GEOFENCING"
# )
# print("=" * 70)

# print(
#     f"\nInput tracks : {len(tracks)}"
# )

# print(
#     f"Processed tracks : "
#     f"{len(spatial_tracks)}"
# )

# print(
#     f"Video size   : "
#     f"{VIDEO_WIDTH} x {VIDEO_HEIGHT}"
# )

# print(
#     f"FPS          : "
#     f"{video_fps:.6f}"
# )

# print(
#     f"Calibration  : "
#     f"{CALIBRATION_STATUS}"
# )

# print(
#     f"World units  : meters"
# )

# print(
#     f"Boundary buf.: "
#     f"{BOUNDARY_BUFFER_METERS:.2f} m"
# )

# print(
#     f"Mean reprojection error : "
#     f"{mean_reprojection_error:.6f} m"
# )

# print(
#     f"Max reprojection error  : "
#     f"{max_reprojection_error:.6f} m"
# )


# print("\n")
# print("TRACK SPATIAL SUMMARY")
# print("-" * 70)


# for track in spatial_tracks:

#     track_id = track.get(
#         "track_id"
#     )

#     segment_id = track.get(
#         "segment_id"
#     )

#     initial_zone = (
#         track.get(
#             "initial_zone"
#         )
#         or "N/A"
#     )

#     final_zone = (
#         track.get(
#             "final_zone"
#         )
#         or "N/A"
#     )

#     min_distance = track.get(
#         "minimum_boundary_distance_m"
#     )

#     avg_velocity = track.get(
#         "average_velocity_m_per_s"
#     )

#     max_velocity = track.get(
#         "maximum_velocity_m_per_s"
#     )

#     print(
#         f"ID {str(track_id):>2} "
#         f"| Seg {str(segment_id):>2} "
#         f"| Initial: "
#         f"{str(initial_zone):<8} "
#         f"| Final: "
#         f"{str(final_zone):<8}"
#     )

#     print(
#         f"     Outside: "
#         f"{track.get('outside_observations', 0)} | "
#         f"Boundary: "
#         f"{track.get('boundary_observations', 0)} | "
#         f"Inside: "
#         f"{track.get('inside_observations', 0)} | "
#         f"Unknown: "
#         f"{track.get('unknown_observations', 0)}"
#     )

#     print(
#         f"     Min boundary distance: "
#         f"{min_distance} m"
#     )

#     print(
#         f"     Avg velocity: "
#         f"{avg_velocity} m/s"
#     )

#     print(
#         f"     Max velocity: "
#         f"{max_velocity} m/s"
#     )

#     print(
#         f"     Boundary crossed: "
#         f"{track.get('boundary_crossed')}"
#     )


# print("\n")
# print("OUTPUT")
# print("-" * 70)
# print(OUTPUT_PATH)

# print("\n")
# print("=" * 70)
# print("STEP 5.8 COMPLETE")
# print("=" * 70)

# print("\nIMPORTANT:")

# if USING_PLACEHOLDER_CALIBRATION:

#     print(
#         "Calibration is currently "
#         "UNVALIDATED_PLACEHOLDER."
#     )

#     print(
#         "Do NOT interpret world distances or velocities "
#         "as physically measured values."
#     )

#     print(
#         "Replace IMAGE_POINTS and WORLD_POINTS with "
#         "real measured camera-scene correspondences."
#     )

# else:

#     print(
#         "Measured calibration is active."
#     )

#     print(
#         "Physical accuracy depends on the quality of "
#         "the measured correspondences."
#     )

# print(
#     "\nStep 5.9 and Step 5.10 should consume this "
#     "Step 5.8 output after calibration is finalized."
# )














# # ============================================================

# # STEP 5.8 — SPATIAL MAPPING & GEOFENCING

# # ============================================================



# from pathlib import Path

# import json

# import math



# import cv2

# import numpy as np





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

#     / "24638-344119107_medium"
#     / "track_movement_5_7_revised.json"

# )



# OUTPUT_PATH = (

#     PROJECT_DIR

#     / "dataset"

#     / "tracking_output"

#     / "24638-344119107_medium"
#     / "track_spatial_5_8.json"

# )





# # ============================================================

# # 2. VIDEO / IMAGE GEOMETRY

# # ============================================================



# VIDEO_WIDTH = 1280

# VIDEO_HEIGHT = 720





# # ============================================================

# # 3. CALIBRATION

# # ============================================================

# #

# # IMPORTANT:

# # These are PLACEHOLDER calibration points.

# #

# # Replace them with REAL measured correspondences before

# # reporting physical distances or velocities.

# #

# # Image points:

# #       pixel coordinates (x, y)

# #

# # World points:

# #       ground-plane coordinates (X, Y) in metres

# #

# # IMPORTANT:

# # The order of IMAGE_POINTS and WORLD_POINTS must correspond.

# #

# # ------------------------------------------------------------

# # PLACEHOLDER RECTANGLE (1280 x 720 compatible)

# #

# # Image:

# #       top-left      top-right

# #          P1 -------- P2

# #          |            |

# #          |            |

# #          P4 -------- P3

# #       bottom-left   bottom-right

# #

# # World:

# #       (0,0) -------- (10,0)

# #          |             |

# #          |             |

# #       (0,6) -------- (10,6)

# #

# # ------------------------------------------------------------



# CALIBRATION_STATUS = "UNVALIDATED_PLACEHOLDER"



# IMAGE_POINTS = np.float32([

#     [100, 100],       # P1

#     [1180, 100],      # P2

#     [1180, 620],      # P3

#     [100, 620]        # P4

# ])



# WORLD_POINTS = np.float32([

#     [0.0, 0.0],       # P1

#     [10.0, 0.0],      # P2

#     [10.0, 6.0],      # P3

#     [0.0, 6.0]        # P4

# ])





# # ============================================================

# # 4. GEOFENCE

# # ============================================================

# #

# # Protected region in WORLD coordinates.

# #

# # This example defines a 10 m x 6 m protected rectangle.

# #

# # Replace this with the actual protected/farm region after

# # calibration.

# #

# # Coordinates are metres.

# # ============================================================



# PROTECTED_POLYGON = np.float32([

#     [0.0, 0.0],

#     [10.0, 0.0],

#     [10.0, 6.0],

#     [0.0, 6.0]

# ])





# # Distance from the boundary at which the point is considered

# # to be in the BOUNDARY zone.

# #

# # This is a PLACEHOLDER value.

# BOUNDARY_BUFFER_METERS = 0.50





# # ============================================================

# # 5. LOAD STEP 5.7

# # ============================================================



# if not INPUT_PATH.exists():

#     raise FileNotFoundError(

#         f"Step 5.7 output not found:\n{INPUT_PATH}"

#     )



# with open(

#     INPUT_PATH,

#     "r",

#     encoding="utf-8"

# ) as f:

#     data = json.load(f)





# tracks = data.get("tracks", [])



# video_fps = float(

#     data.get("video_fps", 29.97002997)

# )





# # ============================================================

# # 6. BASIC VALIDATION

# # ============================================================



# if len(IMAGE_POINTS) < 4:

#     raise ValueError(

#         "At least 4 point correspondences are required."

#     )



# if len(IMAGE_POINTS) != len(WORLD_POINTS):

#     raise ValueError(

#         "IMAGE_POINTS and WORLD_POINTS must contain "

#         "the same number of points."

#     )



# if len(PROTECTED_POLYGON) < 3:

#     raise ValueError(

#         "Protected polygon requires at least 3 points."

#     )





# # ============================================================

# # 7. COMPUTE HOMOGRAPHY

# # ============================================================



# H, homography_mask = cv2.findHomography(

#     IMAGE_POINTS,

#     WORLD_POINTS,

#     method=cv2.RANSAC,

#     ransacReprojThreshold=3.0

# )



# if H is None or not np.all(np.isfinite(H)):

#     raise RuntimeError(

#         "Homography calculation failed or produced invalid values."

#     )


# _test_points = IMAGE_POINTS.reshape(-1, 1, 2)

# _test_world = cv2.perspectiveTransform(

#     _test_points, H

# )

# if not np.all(np.isfinite(_test_world)):

#     raise RuntimeError(

#         "Homography produced invalid transformed calibration points."

#     )





# # ============================================================

# # 8. HELPER FUNCTIONS

# # ============================================================



# def pixel_to_world(point):

#     """

#     Convert one image point (x, y) into world coordinates

#     (X, Y) using the homography.

#     """



#     if point is None:

#         return None



#     point = safe_point(point)

#     if point is None:
#         return None

#     point_array = np.array(

#         [[[point[0], point[1]]]],

#         dtype=np.float32

#     )



#     transformed = cv2.perspectiveTransform(

#         point_array,

#         H

#     )



#     X = float(transformed[0, 0, 0])

#     Y = float(transformed[0, 0, 1])



#     return [X, Y]





# def euclidean_distance(p1, p2):

#     if p1 is None or p2 is None:

#         return None



#     dx = p2[0] - p1[0]

#     dy = p2[1] - p1[1]



#     return math.sqrt(

#         dx * dx + dy * dy

#     )





# def point_to_polygon_boundary_distance(

#     point,

#     polygon

# ):

#     """

#     Minimum Euclidean distance from a point to the

#     polygon boundary.



#     Distance is calculated in world coordinates (metres).

#     """



#     if point is None:

#         return None



#     min_distance = float("inf")



#     for i in range(len(polygon)):



#         p1 = polygon[i]

#         p2 = polygon[

#             (i + 1) % len(polygon)

#         ]



#         px = float(point[0])

#         py = float(point[1])



#         x1 = float(p1[0])

#         y1 = float(p1[1])



#         x2 = float(p2[0])

#         y2 = float(p2[1])



#         dx = x2 - x1

#         dy = y2 - y1



#         segment_length_squared = (

#             dx * dx + dy * dy

#         )



#         if segment_length_squared == 0:



#             distance = math.sqrt(

#                 (px - x1) ** 2

#                 + (py - y1) ** 2

#             )



#         else:



#             t = (

#                 (px - x1) * dx

#                 + (py - y1) * dy

#             ) / segment_length_squared



#             t = max(

#                 0.0,

#                 min(1.0, t)

#             )



#             closest_x = x1 + t * dx

#             closest_y = y1 + t * dy



#             distance = math.sqrt(

#                 (px - closest_x) ** 2

#                 + (py - closest_y) ** 2

#             )



#         min_distance = min(

#             min_distance,

#             distance

#         )



#     return min_distance





# def classify_zone(

#     point,

#     polygon,

#     boundary_buffer

# ):

#     """

#     Classify a world-coordinate point as:



#         INSIDE

#         BOUNDARY

#         OUTSIDE

#     """



#     if point is None:

#         return "UNKNOWN"



#     polygon_array = np.asarray(

#         polygon,

#         dtype=np.float32

#     )



#     point_array = (

#         float(point[0]),

#         float(point[1])

#     )



#     signed_distance = cv2.pointPolygonTest(

#         polygon_array,

#         point_array,

#         True

#     )



#     # Inside polygon

#     if signed_distance > 0:



#         # Close to boundary

#         if abs(signed_distance) <= boundary_buffer:

#             return "BOUNDARY"



#         return "INSIDE"



#     # Exactly on boundary

#     elif signed_distance == 0:



#         return "BOUNDARY"



#     # Outside polygon

#     else:



#         # Distance to boundary

#         boundary_distance = (

#             point_to_polygon_boundary_distance(

#                 point,

#                 polygon

#             )

#         )



#         if (

#             boundary_distance is not None

#             and boundary_distance <= boundary_buffer

#         ):

#             return "BOUNDARY"



#         return "OUTSIDE"





# def safe_float(value):

#     """Return a finite float or None without raising on bad input."""

#     if value is None:
#         return None

#     try:
#         value = float(value)
#     except (TypeError, ValueError):
#         return None

#     if math.isfinite(value):
#         return value

#     return None


# def safe_point(value):

#     """Return a valid finite [x, y] point or None."""

#     if value is None:
#         return None

#     try:
#         if len(value) < 2:
#             return None

#         x = safe_float(value[0])
#         y = safe_float(value[1])

#     except (TypeError, ValueError, IndexError):
#         return None

#     if x is None or y is None:
#         return None

#     return [x, y]


# # ============================================================

# # 9. PROCESS TRACKS

# # ============================================================



# spatial_tracks = []





# for track in tracks:



#     track_id = track.get("track_id")

#     segment_id = track.get("segment_id")



#     history = track.get(

#         "history",

#         []

#     )



#     if not history:

#         continue



#     valid_history = []


#     for obs in history:

#         if not isinstance(obs, dict):
#             continue

#         frame_value = safe_float(obs.get("frame"))
#         timestamp_value = safe_float(obs.get("timestamp"))

#         if frame_value is None or timestamp_value is None:
#             continue

#         obs_copy = dict(obs)
#         obs_copy["frame"] = int(frame_value)
#         obs_copy["timestamp"] = timestamp_value
#         valid_history.append(obs_copy)


#     history = sorted(

#         valid_history,

#         key=lambda x: x["frame"]

#     )


#     if not history:
#         continue


#     spatial_history = []



#     previous_world_position = None

#     previous_timestamp = None



#     previous_zone = None



#     minimum_boundary_distance = None



#     inside_count = 0

#     boundary_count = 0

#     outside_count = 0





#     # ========================================================

#     # PER-FRAME SPATIAL ANALYSIS

#     # ========================================================



#     for obs in history:



#         frame_id = int(

#             obs["frame"]

#         )



#         timestamp = float(

#             obs["timestamp"]

#         )



#         pixel_center = safe_point(

#             obs.get("smoothed_center")

#         )



#         # ----------------------------------------------------

#         # Pixel → world coordinate

#         # ----------------------------------------------------



#         world_position = pixel_to_world(

#             pixel_center

#         )





#         # ----------------------------------------------------

#         # Zone classification

#         # ----------------------------------------------------



#         zone = classify_zone(

#             world_position,

#             PROTECTED_POLYGON,

#             BOUNDARY_BUFFER_METERS

#         )





#         # ----------------------------------------------------

#         # Boundary distance

#         # ----------------------------------------------------



#         boundary_distance = (

#             point_to_polygon_boundary_distance(

#                 world_position,

#                 PROTECTED_POLYGON

#             )

#         )





#         if boundary_distance is not None:



#             if (

#                 minimum_boundary_distance is None

#                 or boundary_distance

#                 < minimum_boundary_distance

#             ):



#                 minimum_boundary_distance = (

#                     boundary_distance

#                 )





#         # ----------------------------------------------------

#         # Count spatial states

#         # ----------------------------------------------------



#         if zone == "INSIDE":

#             inside_count += 1



#         elif zone == "BOUNDARY":

#             boundary_count += 1



#         elif zone == "OUTSIDE":

#             outside_count += 1





#         # ----------------------------------------------------

#         # Real-world displacement / velocity

#         # ----------------------------------------------------



#         displacement_m = None

#         velocity_m_per_s = None



#         if (

#             world_position is not None

#             and previous_world_position is not None

#             and previous_timestamp is not None

#         ):



#             dt = (

#                 timestamp

#                 - previous_timestamp

#             )



#             if dt > 0:



#                 displacement_m = (

#                     euclidean_distance(

#                         previous_world_position,

#                         world_position

#                     )

#                 )



#                 if displacement_m is not None:



#                     velocity_m_per_s = (

#                         displacement_m / dt

#                     )





#         # ----------------------------------------------------

#         # Boundary approach

#         # ----------------------------------------------------



#         approaching_boundary = None



#         if (

#             boundary_distance is not None

#             and previous_world_position is not None

#         ):



#             previous_boundary_distance = (

#                 point_to_polygon_boundary_distance(

#                     previous_world_position,

#                     PROTECTED_POLYGON

#                 )

#             )



#             if previous_boundary_distance is not None:



#                 approaching_boundary = (

#                     boundary_distance

#                     < previous_boundary_distance

#                 )





#         # ----------------------------------------------------

#         # Zone transition

#         # ----------------------------------------------------



#         zone_transition = None



#         if (

#             previous_zone is not None

#             and zone != previous_zone

#         ):



#             zone_transition = (

#                 f"{previous_zone}_TO_{zone}"

#             )





#         # ----------------------------------------------------

#         # Save spatial observation

#         # ----------------------------------------------------



#         spatial_history.append({



#             "frame": frame_id,



#             "timestamp": timestamp,



#             "pixel_center": pixel_center,



#             "world_position_m": (

#                 world_position

#                 if world_position is not None

#                 else None

#             ),



#             "zone": zone,



#             "distance_to_boundary_m":

#                 safe_float(

#                     boundary_distance

#                 ),



#             "displacement_m":

#                 safe_float(

#                     displacement_m

#                 ),



#             "velocity_m_per_s":

#                 safe_float(

#                     velocity_m_per_s

#                 ),



#             "approaching_boundary":

#                 approaching_boundary,



#             "zone_transition":

#                 zone_transition,



#             "observation_type":

#                 obs.get(

#                     "observation_type"

#                 ),



#             "confidence":

#                 obs.get(

#                     "confidence"

#                 ),



#             "reliability":

#                 obs.get(

#                     "reliability"

#                 )

#         })





#         previous_world_position = (

#             world_position

#         )



#         previous_timestamp = (

#             timestamp

#         )



#         previous_zone = zone





#     # ========================================================

#     # TRACK SUMMARY

#     # ========================================================



#     initial_zone = (

#         spatial_history[0]["zone"]

#         if spatial_history

#         else None

#     )



#     final_zone = (

#         spatial_history[-1]["zone"]

#         if spatial_history

#         else None

#     )



#     boundary_crossed = any(

#         item["zone_transition"]

#         in [

#             "OUTSIDE_TO_INSIDE",

#             "BOUNDARY_TO_INSIDE",

#             "INSIDE_TO_OUTSIDE",

#             "INSIDE_TO_BOUNDARY"

#         ]

#         for item in spatial_history

#         if item["zone_transition"]

#         is not None

#     )



#     valid_velocities = [

#         item["velocity_m_per_s"]

#         for item in spatial_history

#         if item["velocity_m_per_s"]

#         is not None

#     ]



#     maximum_velocity = (

#         max(valid_velocities)

#         if valid_velocities

#         else None

#     )



#     average_velocity = (

#         sum(valid_velocities)

#         / len(valid_velocities)

#         if valid_velocities

#         else None

#     )





#     spatial_tracks.append({



#         "track_id":

#             track_id,



#         "segment_id":

#             segment_id,



#         "history_status":

#             track.get(

#                 "history_status"

#             ),



#         "initial_zone":

#             initial_zone,



#         "final_zone":

#             final_zone,



#         "boundary_crossed":

#             boundary_crossed,



#         "inside_observations":

#             inside_count,



#         "boundary_observations":

#             boundary_count,



#         "outside_observations":

#             outside_count,



#         "minimum_boundary_distance_m":

#             safe_float(

#                 minimum_boundary_distance

#             ),



#         "average_velocity_m_per_s":

#             safe_float(

#                 average_velocity

#             ),



#         "maximum_velocity_m_per_s":

#             safe_float(

#                 maximum_velocity

#             ),



#         "history":

#             spatial_history

#     })





# # ============================================================

# # 10. OUTPUT OBJECT

# # ============================================================



# output = {



#     "step":

#         "5.8",



#     "step_name":

#         "Spatial Mapping & Geofencing",



#     "version":

#         "homography_geofence_v1",



#     "input_file":

#         str(INPUT_PATH),



#     "video_width":

#         VIDEO_WIDTH,



#     "video_height":

#         VIDEO_HEIGHT,



#     "video_fps":

#         video_fps,



#     "coordinate_system":

#         "image_pixels_to_world_ground_plane",



#     "world_units":

#         "meters",



#     "calibration_status":

#         CALIBRATION_STATUS,



#     "calibration_warning":

#         (

#             "PLACEHOLDER calibration. "

#             "Physical distances and velocities are "

#             "NOT validated until IMAGE_POINTS and "

#             "WORLD_POINTS are replaced with measured "

#             "correspondences from the actual camera scene."

#         ),



#     "homography":

#         H.tolist(),



#     "calibration_image_points":

#         IMAGE_POINTS.tolist(),



#     "calibration_world_points_m":

#         WORLD_POINTS.tolist(),



#     "protected_polygon_world_m":

#         PROTECTED_POLYGON.tolist(),



#     "boundary_buffer_m":

#         BOUNDARY_BUFFER_METERS,



#     "track_count":

#         len(spatial_tracks),



#     "tracks":

#         spatial_tracks

# }





# # ============================================================

# # 11. WRITE OUTPUT

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

# # 12. CONSOLE SUMMARY

# # ============================================================



# print("\n")

# print("=" * 70)

# print(

#     "STEP 5.8 — SPATIAL MAPPING & GEOFENCING"

# )

# print("=" * 70)



# print(

#     f"\nInput tracks : {len(tracks)}"

# )



# print(

#     f"FPS          : {video_fps:.6f}"

# )



# print(

#     f"Calibration  : {CALIBRATION_STATUS}"

# )



# print(

#     f"World units  : meters"

# )



# print(

#     f"Boundary buf.: "

#     f"{BOUNDARY_BUFFER_METERS:.2f} m"

# )





# print("\n")

# print("TRACK SPATIAL SUMMARY")

# print("-" * 70)





# for track in spatial_tracks:



#     print(

#         f"ID {track['track_id']:>2} "

#         f"| Seg {track['segment_id']:>2} "

#         f"| Initial: "

#         f"{track['initial_zone']:<8} "

#         f"| Final: "

#         f"{track['final_zone']:<8}"

#     )



#     print(

#         f"     Outside: "

#         f"{track['outside_observations']} | "

#         f"Boundary: "

#         f"{track['boundary_observations']} | "

#         f"Inside: "

#         f"{track['inside_observations']}"

#     )



#     min_boundary = track.get("minimum_boundary_distance_m")

#     avg_velocity = track.get("average_velocity_m_per_s")

#     max_velocity = track.get("maximum_velocity_m_per_s")


#     print(

#         "     Min boundary distance: "
#         + ("N/A" if min_boundary is None else f"{min_boundary:.4f}")
#         + " m"

#     )


#     print(

#         "     Avg velocity: "
#         + ("N/A" if avg_velocity is None else f"{avg_velocity:.4f}")
#         + " m/s"

#     )


#     print(

#         "     Max velocity: "
#         + ("N/A" if max_velocity is None else f"{max_velocity:.4f}")
#         + " m/s"

#     )



#     print(

#         f"     Boundary crossed: "

#         f"{track['boundary_crossed']}"

#     )


#     if track.get("history_status") != "sufficient":

#         print(

#             "     NOTE: Insufficient movement history; "

#             "spatial values are provisional."

#         )





# print("\n")

# print("OUTPUT")

# print("-" * 70)

# print(OUTPUT_PATH)



# print("\n")

# print("=" * 70)

# print("STEP 5.8 COMPLETE")

# print("=" * 70)



# print("\nIMPORTANT:")

# print(

#     "Calibration is currently UNVALIDATED_PLACEHOLDER."

# )



# print(

#     "Do NOT interpret world distances or velocities "

#     "as physically measured values until real "

#     "camera-scene correspondences are supplied."

# )







# # ============================================================
# # STEP 5.8 — SPATIAL MAPPING & GEOFENCING
# # ============================================================

# from pathlib import Path
# import json
# import math

# import cv2
# import numpy as np


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
#     / "track_movement_5_7_revised.json"
# )

# OUTPUT_PATH = (
#     PROJECT_DIR
#     / "dataset"
#     / "tracking_output"
#     / "track_spatial_5_8.json"
# )


# # ============================================================
# # 2. VIDEO / IMAGE GEOMETRY
# # ============================================================

# VIDEO_WIDTH = 1080
# VIDEO_HEIGHT = 1920


# # ============================================================
# # 3. CALIBRATION
# # ============================================================
# #
# # IMPORTANT:
# # These are PLACEHOLDER calibration points.
# #
# # Replace them with REAL measured correspondences before
# # reporting physical distances or velocities.
# #
# # Image points:
# #       pixel coordinates (x, y)
# #
# # World points:
# #       ground-plane coordinates (X, Y) in metres
# #
# # IMPORTANT:
# # The order of IMAGE_POINTS and WORLD_POINTS must correspond.
# #
# # ------------------------------------------------------------
# # PLACEHOLDER RECTANGLE
# #
# # Image:
# #       top-left      top-right
# #          P1 -------- P2
# #          |            |
# #          |            |
# #          P4 -------- P3
# #       bottom-left   bottom-right
# #
# # World:
# #       (0,0) -------- (10,0)
# #          |             |
# #          |             |
# #       (0,6) -------- (10,6)
# #
# # ------------------------------------------------------------

# CALIBRATION_STATUS = "UNVALIDATED_PLACEHOLDER"

# IMAGE_POINTS = np.float32([
#     [100, 100],       # P1
#     [980, 100],       # P2
#     [980, 1800],      # P3
#     [100, 1800]       # P4
# ])

# WORLD_POINTS = np.float32([
#     [0.0, 0.0],       # P1
#     [10.0, 0.0],      # P2
#     [10.0, 6.0],      # P3
#     [0.0, 6.0]        # P4
# ])


# # ============================================================
# # 4. GEOFENCE
# # ============================================================
# #
# # Protected region in WORLD coordinates.
# #
# # This example defines a 10 m x 6 m protected rectangle.
# #
# # Replace this with the actual protected/farm region after
# # calibration.
# #
# # Coordinates are metres.
# # ============================================================

# PROTECTED_POLYGON = np.float32([
#     [0.0, 0.0],
#     [10.0, 0.0],
#     [10.0, 6.0],
#     [0.0, 6.0]
# ])


# # Distance from the boundary at which the point is considered
# # to be in the BOUNDARY zone.
# #
# # This is a PLACEHOLDER value.
# BOUNDARY_BUFFER_METERS = 0.50


# # ============================================================
# # 5. LOAD STEP 5.7
# # ============================================================

# if not INPUT_PATH.exists():
#     raise FileNotFoundError(
#         f"Step 5.7 output not found:\n{INPUT_PATH}"
#     )

# with open(
#     INPUT_PATH,
#     "r",
#     encoding="utf-8"
# ) as f:
#     data = json.load(f)


# tracks = data.get("tracks", [])

# video_fps = float(
#     data.get("video_fps", 29.97002997)
# )


# # ============================================================
# # 6. BASIC VALIDATION
# # ============================================================

# if len(IMAGE_POINTS) < 4:
#     raise ValueError(
#         "At least 4 point correspondences are required."
#     )

# if len(IMAGE_POINTS) != len(WORLD_POINTS):
#     raise ValueError(
#         "IMAGE_POINTS and WORLD_POINTS must contain "
#         "the same number of points."
#     )

# if len(PROTECTED_POLYGON) < 3:
#     raise ValueError(
#         "Protected polygon requires at least 3 points."
#     )


# # ============================================================
# # 7. COMPUTE HOMOGRAPHY
# # ============================================================

# H, homography_mask = cv2.findHomography(
#     IMAGE_POINTS,
#     WORLD_POINTS,
#     method=cv2.RANSAC,
#     ransacReprojThreshold=3.0
# )

# if H is None:
#     raise RuntimeError(
#         "Homography calculation failed."
#     )


# # ============================================================
# # 8. HELPER FUNCTIONS
# # ============================================================

# def pixel_to_world(point):
#     """
#     Convert one image point (x, y) into world coordinates
#     (X, Y) using the homography.
#     """

#     if point is None:
#         return None

#     point_array = np.array(
#         [[[float(point[0]), float(point[1])]]],
#         dtype=np.float32
#     )

#     transformed = cv2.perspectiveTransform(
#         point_array,
#         H
#     )

#     X = float(transformed[0, 0, 0])
#     Y = float(transformed[0, 0, 1])

#     return [X, Y]


# def euclidean_distance(p1, p2):
#     if p1 is None or p2 is None:
#         return None

#     dx = p2[0] - p1[0]
#     dy = p2[1] - p1[1]

#     return math.sqrt(
#         dx * dx + dy * dy
#     )


# def point_to_polygon_boundary_distance(
#     point,
#     polygon
# ):
#     """
#     Minimum Euclidean distance from a point to the
#     polygon boundary.

#     Distance is calculated in world coordinates (metres).
#     """

#     if point is None:
#         return None

#     min_distance = float("inf")

#     for i in range(len(polygon)):

#         p1 = polygon[i]
#         p2 = polygon[
#             (i + 1) % len(polygon)
#         ]

#         px = float(point[0])
#         py = float(point[1])

#         x1 = float(p1[0])
#         y1 = float(p1[1])

#         x2 = float(p2[0])
#         y2 = float(p2[1])

#         dx = x2 - x1
#         dy = y2 - y1

#         segment_length_squared = (
#             dx * dx + dy * dy
#         )

#         if segment_length_squared == 0:

#             distance = math.sqrt(
#                 (px - x1) ** 2
#                 + (py - y1) ** 2
#             )

#         else:

#             t = (
#                 (px - x1) * dx
#                 + (py - y1) * dy
#             ) / segment_length_squared

#             t = max(
#                 0.0,
#                 min(1.0, t)
#             )

#             closest_x = x1 + t * dx
#             closest_y = y1 + t * dy

#             distance = math.sqrt(
#                 (px - closest_x) ** 2
#                 + (py - closest_y) ** 2
#             )

#         min_distance = min(
#             min_distance,
#             distance
#         )

#     return min_distance


# def classify_zone(
#     point,
#     polygon,
#     boundary_buffer
# ):
#     """
#     Classify a world-coordinate point as:

#         INSIDE
#         BOUNDARY
#         OUTSIDE
#     """

#     if point is None:
#         return "UNKNOWN"

#     polygon_array = np.asarray(
#         polygon,
#         dtype=np.float32
#     )

#     point_array = (
#         float(point[0]),
#         float(point[1])
#     )

#     signed_distance = cv2.pointPolygonTest(
#         polygon_array,
#         point_array,
#         True
#     )

#     # Inside polygon
#     if signed_distance > 0:

#         # Close to boundary
#         if abs(signed_distance) <= boundary_buffer:
#             return "BOUNDARY"

#         return "INSIDE"

#     # Exactly on boundary
#     elif signed_distance == 0:

#         return "BOUNDARY"

#     # Outside polygon
#     else:

#         # Distance to boundary
#         boundary_distance = (
#             point_to_polygon_boundary_distance(
#                 point,
#                 polygon
#             )
#         )

#         if (
#             boundary_distance is not None
#             and boundary_distance <= boundary_buffer
#         ):
#             return "BOUNDARY"

#         return "OUTSIDE"


# def safe_float(value):

#     if value is None:
#         return None

#     value = float(value)

#     if math.isfinite(value):
#         return value

#     return None


# # ============================================================
# # 9. PROCESS TRACKS
# # ============================================================

# spatial_tracks = []


# for track in tracks:

#     track_id = track.get("track_id")
#     segment_id = track.get("segment_id")

#     history = track.get(
#         "history",
#         []
#     )

#     if not history:
#         continue

#     history = sorted(
#         history,
#         key=lambda x: int(x["frame"])
#     )

#     spatial_history = []

#     previous_world_position = None
#     previous_timestamp = None

#     previous_zone = None

#     minimum_boundary_distance = None

#     inside_count = 0
#     boundary_count = 0
#     outside_count = 0


#     # ========================================================
#     # PER-FRAME SPATIAL ANALYSIS
#     # ========================================================

#     for obs in history:

#         frame_id = int(
#             obs["frame"]
#         )

#         timestamp = float(
#             obs["timestamp"]
#         )

#         pixel_center = obs.get(
#             "smoothed_center"
#         )

#         # ----------------------------------------------------
#         # Pixel → world coordinate
#         # ----------------------------------------------------

#         world_position = pixel_to_world(
#             pixel_center
#         )


#         # ----------------------------------------------------
#         # Zone classification
#         # ----------------------------------------------------

#         zone = classify_zone(
#             world_position,
#             PROTECTED_POLYGON,
#             BOUNDARY_BUFFER_METERS
#         )


#         # ----------------------------------------------------
#         # Boundary distance
#         # ----------------------------------------------------

#         boundary_distance = (
#             point_to_polygon_boundary_distance(
#                 world_position,
#                 PROTECTED_POLYGON
#             )
#         )


#         if boundary_distance is not None:

#             if (
#                 minimum_boundary_distance is None
#                 or boundary_distance
#                 < minimum_boundary_distance
#             ):

#                 minimum_boundary_distance = (
#                     boundary_distance
#                 )


#         # ----------------------------------------------------
#         # Count spatial states
#         # ----------------------------------------------------

#         if zone == "INSIDE":
#             inside_count += 1

#         elif zone == "BOUNDARY":
#             boundary_count += 1

#         elif zone == "OUTSIDE":
#             outside_count += 1


#         # ----------------------------------------------------
#         # Real-world displacement / velocity
#         # ----------------------------------------------------

#         displacement_m = None
#         velocity_m_per_s = None

#         if (
#             world_position is not None
#             and previous_world_position is not None
#             and previous_timestamp is not None
#         ):

#             dt = (
#                 timestamp
#                 - previous_timestamp
#             )

#             if dt > 0:

#                 displacement_m = (
#                     euclidean_distance(
#                         previous_world_position,
#                         world_position
#                     )
#                 )

#                 if displacement_m is not None:

#                     velocity_m_per_s = (
#                         displacement_m / dt
#                     )


#         # ----------------------------------------------------
#         # Boundary approach
#         # ----------------------------------------------------

#         approaching_boundary = None

#         if (
#             boundary_distance is not None
#             and previous_world_position is not None
#         ):

#             previous_boundary_distance = (
#                 point_to_polygon_boundary_distance(
#                     previous_world_position,
#                     PROTECTED_POLYGON
#                 )
#             )

#             if previous_boundary_distance is not None:

#                 approaching_boundary = (
#                     boundary_distance
#                     < previous_boundary_distance
#                 )


#         # ----------------------------------------------------
#         # Zone transition
#         # ----------------------------------------------------

#         zone_transition = None

#         if (
#             previous_zone is not None
#             and zone != previous_zone
#         ):

#             zone_transition = (
#                 f"{previous_zone}_TO_{zone}"
#             )


#         # ----------------------------------------------------
#         # Save spatial observation
#         # ----------------------------------------------------

#         spatial_history.append({

#             "frame": frame_id,

#             "timestamp": timestamp,

#             "pixel_center": pixel_center,

#             "world_position_m": (
#                 world_position
#                 if world_position is not None
#                 else None
#             ),

#             "zone": zone,

#             "distance_to_boundary_m":
#                 safe_float(
#                     boundary_distance
#                 ),

#             "displacement_m":
#                 safe_float(
#                     displacement_m
#                 ),

#             "velocity_m_per_s":
#                 safe_float(
#                     velocity_m_per_s
#                 ),

#             "approaching_boundary":
#                 approaching_boundary,

#             "zone_transition":
#                 zone_transition,

#             "observation_type":
#                 obs.get(
#                     "observation_type"
#                 ),

#             "confidence":
#                 obs.get(
#                     "confidence"
#                 ),

#             "reliability":
#                 obs.get(
#                     "reliability"
#                 )
#         })


#         previous_world_position = (
#             world_position
#         )

#         previous_timestamp = (
#             timestamp
#         )

#         previous_zone = zone


#     # ========================================================
#     # TRACK SUMMARY
#     # ========================================================

#     initial_zone = (
#         spatial_history[0]["zone"]
#         if spatial_history
#         else None
#     )

#     final_zone = (
#         spatial_history[-1]["zone"]
#         if spatial_history
#         else None
#     )

#     boundary_crossed = any(
#         item["zone_transition"]
#         in [
#             "OUTSIDE_TO_INSIDE",
#             "BOUNDARY_TO_INSIDE",
#             "INSIDE_TO_OUTSIDE",
#             "INSIDE_TO_BOUNDARY"
#         ]
#         for item in spatial_history
#         if item["zone_transition"]
#         is not None
#     )

#     valid_velocities = [
#         item["velocity_m_per_s"]
#         for item in spatial_history
#         if item["velocity_m_per_s"]
#         is not None
#     ]

#     maximum_velocity = (
#         max(valid_velocities)
#         if valid_velocities
#         else None
#     )

#     average_velocity = (
#         sum(valid_velocities)
#         / len(valid_velocities)
#         if valid_velocities
#         else None
#     )


#     spatial_tracks.append({

#         "track_id":
#             track_id,

#         "segment_id":
#             segment_id,

#         "history_status":
#             track.get(
#                 "history_status"
#             ),

#         "initial_zone":
#             initial_zone,

#         "final_zone":
#             final_zone,

#         "boundary_crossed":
#             boundary_crossed,

#         "inside_observations":
#             inside_count,

#         "boundary_observations":
#             boundary_count,

#         "outside_observations":
#             outside_count,

#         "minimum_boundary_distance_m":
#             safe_float(
#                 minimum_boundary_distance
#             ),

#         "average_velocity_m_per_s":
#             safe_float(
#                 average_velocity
#             ),

#         "maximum_velocity_m_per_s":
#             safe_float(
#                 maximum_velocity
#             ),

#         "history":
#             spatial_history
#     })


# # ============================================================
# # 10. OUTPUT OBJECT
# # ============================================================

# output = {

#     "step":
#         "5.8",

#     "step_name":
#         "Spatial Mapping & Geofencing",

#     "version":
#         "homography_geofence_v1",

#     "input_file":
#         str(INPUT_PATH),

#     "video_width":
#         VIDEO_WIDTH,

#     "video_height":
#         VIDEO_HEIGHT,

#     "video_fps":
#         video_fps,

#     "coordinate_system":
#         "image_pixels_to_world_ground_plane",

#     "world_units":
#         "meters",

#     "calibration_status":
#         CALIBRATION_STATUS,

#     "calibration_warning":
#         (
#             "PLACEHOLDER calibration. "
#             "Physical distances and velocities are "
#             "NOT validated until IMAGE_POINTS and "
#             "WORLD_POINTS are replaced with measured "
#             "correspondences from the actual camera scene."
#         ),

#     "homography":
#         H.tolist(),

#     "calibration_image_points":
#         IMAGE_POINTS.tolist(),

#     "calibration_world_points_m":
#         WORLD_POINTS.tolist(),

#     "protected_polygon_world_m":
#         PROTECTED_POLYGON.tolist(),

#     "boundary_buffer_m":
#         BOUNDARY_BUFFER_METERS,

#     "track_count":
#         len(spatial_tracks),

#     "tracks":
#         spatial_tracks
# }


# # ============================================================
# # 11. WRITE OUTPUT
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
# # 12. CONSOLE SUMMARY
# # ============================================================

# print("\n")
# print("=" * 70)
# print(
#     "STEP 5.8 — SPATIAL MAPPING & GEOFENCING"
# )
# print("=" * 70)

# print(
#     f"\nInput tracks : {len(tracks)}"
# )

# print(
#     f"FPS          : {video_fps:.6f}"
# )

# print(
#     f"Calibration  : {CALIBRATION_STATUS}"
# )

# print(
#     f"World units  : meters"
# )

# print(
#     f"Boundary buf.: "
#     f"{BOUNDARY_BUFFER_METERS:.2f} m"
# )


# print("\n")
# print("TRACK SPATIAL SUMMARY")
# print("-" * 70)


# for track in spatial_tracks:

#     print(
#         f"ID {track['track_id']:>2} "
#         f"| Seg {track['segment_id']:>2} "
#         f"| Initial: "
#         f"{track['initial_zone']:<8} "
#         f"| Final: "
#         f"{track['final_zone']:<8}"
#     )

#     print(
#         f"     Outside: "
#         f"{track['outside_observations']} | "
#         f"Boundary: "
#         f"{track['boundary_observations']} | "
#         f"Inside: "
#         f"{track['inside_observations']}"
#     )

#     print(
#         f"     Min boundary distance: "
#         f"{track['minimum_boundary_distance_m']} m"
#     )

#     print(
#         f"     Avg velocity: "
#         f"{track['average_velocity_m_per_s']} m/s"
#     )

#     print(
#         f"     Max velocity: "
#         f"{track['maximum_velocity_m_per_s']} m/s"
#     )

#     print(
#         f"     Boundary crossed: "
#         f"{track['boundary_crossed']}"
#     )


# print("\n")
# print("OUTPUT")
# print("-" * 70)
# print(OUTPUT_PATH)

# print("\n")
# print("=" * 70)
# print("STEP 5.8 COMPLETE")
# print("=" * 70)

# print("\nIMPORTANT:")
# print(
#     "Calibration is currently UNVALIDATED_PLACEHOLDER."
# )

# print(
#     "Do NOT interpret world distances or velocities "
#     "as physically measured values until real "
#     "camera-scene correspondences are supplied."
# )