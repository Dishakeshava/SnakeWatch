# ============================================================
# STEP 5.2 — RT-DETRv2 → BYTETrack DETECTION ADAPTER
# ============================================================

"""
Purpose:
    Run the frozen RT-DETRv2 detector on video frames and
    convert detections into a format that ByteTrack can consume.

Pipeline:

    VideoFrameReader
         ↓
    RT-DETRv2
         ↓
    Tracker-input confidence filtering
         ↓
    Snake detection candidates
         ↓
    ByteTrack format

Important:
    Two confidence thresholds are used in the video pipeline.

    TRACKER_INPUT_THRESHOLD = 0.03
        Minimum confidence required for a detector candidate
        to reach ByteTrack.

    FINAL_DETECTION_THRESHOLD = 0.055
        Frozen detector operating threshold selected during
        validation.

    The 0.055 threshold is NOT removed from the system.
    It remains the threshold for strong/final detector evidence.

    This step only exposes weaker candidates to the tracker
    so that temporal information can be used.
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import sys
import time

import cv2
import torch

from transformers import (
    RTDetrImageProcessor,
    RTDetrV2ForObjectDetection,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/content/drive/MyDrive/snake_detection/Snake_detection"
)


# ------------------------------------------------------------
# Original pretrained checkpoint
# Used for the processor
# ------------------------------------------------------------

BASE_CHECKPOINT = (
    "PekingU/rtdetr_v2_r18vd"
)


# ------------------------------------------------------------
# Frozen detector checkpoint
# Selected during evaluation
# ------------------------------------------------------------

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "rtdetrv2_snake_finetuned_v3_earlystop"
    / "checkpoint-8072"
)


# ------------------------------------------------------------
# Test video
# ------------------------------------------------------------

VIDEO_PATH = (
    PROJECT_DIR
    / "snake_videos"
    / "10231672-hd_1080_1920_30fps.mp4"
)


# ------------------------------------------------------------
# TWO-THRESHOLD DESIGN
# ------------------------------------------------------------

# Minimum confidence required for a detector candidate
# to reach ByteTrack.
#
# This allows weaker detections to contribute to
# temporal association.
TRACKER_INPUT_THRESHOLD = 0.03


# Frozen operating threshold selected during validation.
#
# This remains the threshold for strong/final detector
# evidence.
FINAL_DETECTION_THRESHOLD = 0.055


# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------

# Process all frames when set to None.
#
# Keep None here because Step 5.2 is the detector adapter.
# Step 5.3 can impose its own testing limit.
MAX_FRAMES = None


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMPORT STEP 5.1
# ============================================================

sys.path.append(
    str(PROJECT_DIR)
)

from step5_1_video_input import (
    VideoFrameReader
)


# ============================================================
# LOAD DETECTOR
# ============================================================

def load_detector():

    print(
        "\nLoading RT-DETRv2 processor..."
    )

    # IMPORTANT:
    # checkpoint-8072 does not contain the
    # processor config, so the processor is
    # loaded from the original checkpoint.

    processor = (
        RTDetrImageProcessor.from_pretrained(
            BASE_CHECKPOINT
        )
    )

    print(
        "Processor loaded."
    )

    print(
        "\nLoading fine-tuned RT-DETRv2 model..."
    )

    model = (
        RTDetrV2ForObjectDetection
        .from_pretrained(
            CHECKPOINT_PATH
        )
    )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        "Model loaded."
    )

    print(
        "Device:",
        DEVICE
    )

    return processor, model


# ============================================================
# DETECTION FUNCTION
# ============================================================

def detect_snakes(
    frame,
    processor,
    model,
):
    """
    Run RT-DETRv2 on one frame.

    Returns:
        List of dictionaries containing:

        {
            "bbox": [x1, y1, x2, y2],
            "confidence": float,
            "class_id": 0,
            "class_name": "snake"
        }

    IMPORTANT:
        Detections are filtered at
        TRACKER_INPUT_THRESHOLD = 0.03.

        This means detections between
        0.03 and 0.055 are intentionally
        preserved for ByteTrack.
    """

    # --------------------------------------------------------
    # OpenCV BGR → RGB
    # --------------------------------------------------------

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    inputs = processor(
        images=frame_rgb,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs
        )


    # --------------------------------------------------------
    # Original frame dimensions
    # --------------------------------------------------------

    height, width = (
        frame.shape[:2]
    )


    target_sizes = torch.tensor(
        [[height, width]],
        device=DEVICE
    )


    # --------------------------------------------------------
    # Post-processing
    # --------------------------------------------------------

    results = (
        processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=TRACKER_INPUT_THRESHOLD
        )
    )

    result = results[0]


    scores = result["scores"]
    boxes = result["boxes"]
    labels = result["labels"]


    detections = []


    # --------------------------------------------------------
    # Convert detections
    # --------------------------------------------------------

    for box, score, label in zip(
        boxes,
        scores,
        labels
    ):

        # This project has one class:
        # 0 = snake

        class_id = int(
            label.item()
        )

        if class_id != 0:
            continue


        confidence = float(
            score.item()
        )


        x1, y1, x2, y2 = (
            box
            .detach()
            .cpu()
            .tolist()
        )


        # ----------------------------------------------------
        # Clamp coordinates to image boundaries
        # ----------------------------------------------------

        x1 = max(
            0.0,
            min(
                x1,
                width - 1
            )
        )

        y1 = max(
            0.0,
            min(
                y1,
                height - 1
            )
        )

        x2 = max(
            0.0,
            min(
                x2,
                width - 1
            )
        )

        y2 = max(
            0.0,
            min(
                y2,
                height - 1
            )
        )


        # ----------------------------------------------------
        # Make sure box is valid
        # ----------------------------------------------------

        if x2 <= x1 or y2 <= y1:
            continue


        detections.append(
            {
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],

                "confidence": confidence,

                "class_id": 0,

                "class_name": "snake",
            }
        )


    return detections


# ============================================================
# CONVERT TO BYTETrack FORMAT
# ============================================================

def convert_to_bytetrack_format(
    detections
):
    """
    Convert detector output to the format expected by
    ByteTrack / Supervision.

    Format:

        [x1, y1, x2, y2, confidence, class_id]

    Both strong and weak detector candidates are passed
    through. ByteTrack receives every candidate with
    confidence >= TRACKER_INPUT_THRESHOLD.
    """

    bytetrack_detections = []


    for detection in detections:

        x1, y1, x2, y2 = (
            detection["bbox"]
        )

        confidence = (
            detection["confidence"]
        )

        class_id = (
            detection["class_id"]
        )


        bytetrack_detections.append(
            [
                x1,
                y1,
                x2,
                y2,
                confidence,
                class_id,
            ]
        )


    return bytetrack_detections


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print(
        "=" * 65
    )

    print(
        "STEP 5.2 — RT-DETRv2 → BYTETrack DETECTION ADAPTER"
    )

    print(
        "=" * 65
    )


    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )


    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"Video not found:\n"
            f"{VIDEO_PATH}"
        )


    # --------------------------------------------------------
    # Load detector
    # --------------------------------------------------------

    processor, model = (
        load_detector()
    )


    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    reader = VideoFrameReader(
        video_path=VIDEO_PATH,
        frame_skip=1,
        resize_width=None,
        resize_height=None,
    )


    print(
        "\nVIDEO INFORMATION"
    )

    print(
        "-" * 65
    )

    print(
        f"Video       : "
        f"{VIDEO_PATH}"
    )

    print(
        f"FPS         : "
        f"{reader.fps:.2f}"
    )

    print(
        f"Total frames: "
        f"{reader.total_frames}"
    )

    print(
        f"Resolution  : "
        f"{reader.width} x "
        f"{reader.height}"
    )

    print(
        f"Tracker input threshold : "
        f"{TRACKER_INPUT_THRESHOLD}"
    )

    print(
        f"Final detection threshold: "
        f"{FINAL_DETECTION_THRESHOLD}"
    )

    print(
        f"Device      : "
        f"{DEVICE}"
    )


    # --------------------------------------------------------
    # Process video
    # --------------------------------------------------------

    total_detections = 0

    frames_processed = 0

    frames_with_detections = 0

    strong_detections = 0

    weak_detections = 0

    total_time = 0.0


    print(
        "\nRUNNING DETECTOR"
    )

    print(
        "-" * 65
    )


    for frame_data in reader.frames():

        if (
            MAX_FRAMES is not None
            and frames_processed
            >= MAX_FRAMES
        ):
            break


        frame_id = (
            frame_data["frame_id"]
        )

        timestamp = (
            frame_data["timestamp"]
        )

        frame = (
            frame_data["frame"]
        )


        start_time = time.time()


        detections = (
            detect_snakes(
                frame,
                processor,
                model
            )
        )


        elapsed = (
            time.time()
            - start_time
        )


        total_time += elapsed

        frames_processed += 1


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        num_detections = (
            len(detections)
        )


        total_detections += (
            num_detections
        )


        if num_detections > 0:

            frames_with_detections += 1


        for detection in detections:

            confidence = (
                detection["confidence"]
            )

            if (
                confidence
                >= FINAL_DETECTION_THRESHOLD
            ):

                strong_detections += 1

            else:

                weak_detections += 1


        # ----------------------------------------------------
        # Print useful frames
        # ----------------------------------------------------

        if num_detections > 0:

            best_detection = max(
                detections,
                key=lambda x:
                x["confidence"]
            )


            best_confidence = (
                best_detection["confidence"]
            )


            if (
                best_confidence
                >= FINAL_DETECTION_THRESHOLD
            ):

                detection_type = (
                    "STRONG"
                )

            else:

                detection_type = (
                    "WEAK/TRACKER"
                )


            print(
                f"Frame {frame_id:4d} | "
                f"time={timestamp:6.2f}s | "
                f"detections={num_detections:2d} | "
                f"best_conf="
                f"{best_confidence:.4f} | "
                f"{detection_type}"
            )


        # ----------------------------------------------------
        # Progress every 100 frames
        # ----------------------------------------------------

        elif (
            frames_processed % 100 == 0
        ):

            print(
                f"Processed "
                f"{frames_processed} "
                f"frames..."
            )


    # --------------------------------------------------------
    # Release video
    # --------------------------------------------------------

    reader.release()


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    average_time = (
        total_time
        / frames_processed
        if frames_processed > 0
        else 0
    )


    effective_fps = (
        1.0 / average_time
        if average_time > 0
        else 0
    )


    average_detections = (
        total_detections
        / frames_processed
        if frames_processed > 0
        else 0
    )


    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print(
        "\n" + "=" * 65
    )

    print(
        "STEP 5.2 COMPLETE"
    )

    print(
        "=" * 65
    )


    print(
        f"Frames processed        : "
        f"{frames_processed}"
    )

    print(
        f"Frames with candidates  : "
        f"{frames_with_detections}"
    )

    print(
        f"Total detector candidates: "
        f"{total_detections}"
    )

    print(
        f"Strong detections >= "
        f"{FINAL_DETECTION_THRESHOLD}: "
        f"{strong_detections}"
    )

    print(
        f"Weak candidates "
        f"({TRACKER_INPUT_THRESHOLD}–"
        f"{FINAL_DETECTION_THRESHOLD}): "
        f"{weak_detections}"
    )

    print(
        f"Average candidates/frame: "
        f"{average_detections:.3f}"
    )

    print(
        f"Average inference time  : "
        f"{average_time * 1000:.2f} ms"
    )

    print(
        f"Effective FPS           : "
        f"{effective_fps:.2f}"
    )

    print(
        f"Video FPS               : "
        f"{reader.fps:.2f}"
    )

    print(
        f"Tracker input threshold: "
        f"{TRACKER_INPUT_THRESHOLD}"
    )

    print(
        f"Final detection threshold: "
        f"{FINAL_DETECTION_THRESHOLD}"
    )

    print(
        "=" * 65
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()















# # ============================================================
# # STEP 5.2 — RT-DETRv2 → BYTETrack DETECTION ADAPTER
# # ============================================================
# #
# # Purpose:
# #   Run the frozen RT-DETRv2 detector on video frames and
# #   convert detections into a format that ByteTrack can consume.
# #
# # Pipeline:
# #
# #   VideoFrameReader
# #        ↓
# #   RT-DETRv2
# #        ↓
# #   Confidence filtering
# #        ↓
# #   Snake detections
# #        ↓
# #   ByteTrack format
# #
# # ============================================================

# from pathlib import Path
# import sys
# import time

# import cv2
# import torch

# from transformers import (
#     RTDetrImageProcessor,
#     RTDetrV2ForObjectDetection,
# )


# # ============================================================
# # CONFIGURATION
# # ============================================================

# PROJECT_DIR = Path(
#     "/content/drive/MyDrive/snake_detection/Snake_detection"
# )

# # Original pretrained checkpoint.
# # Used for the processor.
# BASE_CHECKPOINT = "PekingU/rtdetr_v2_r18vd"

# # Frozen detector checkpoint selected during evaluation.
# CHECKPOINT_PATH = (
#     PROJECT_DIR
#     / "rtdetrv2_snake_finetuned_v3_earlystop"
#     / "checkpoint-8072"
# )

# # Test video
# VIDEO_PATH = (
#     PROJECT_DIR
#     / "snake_videos"
#     / "10231672-hd_1080_1920_30fps.mp4"
# )

# # ------------------------------------------------------------
# # IMPORTANT
# # ------------------------------------------------------------
# # This is the frozen operating threshold selected during
# # validation.
# DETECTION_THRESHOLD = 0.055

# # Process all frames when set to None.
# MAX_FRAMES = None

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )


# # ============================================================
# # IMPORT STEP 5.1
# # ============================================================

# sys.path.append(str(PROJECT_DIR))

# from step5_1_video_input import VideoFrameReader


# # ============================================================
# # LOAD DETECTOR
# # ============================================================

# def load_detector():

#     print("\nLoading RT-DETRv2 processor...")

#     # IMPORTANT:
#     # checkpoint-8072 does not contain the processor config,
#     # so the processor is loaded from the original checkpoint.
#     processor = RTDetrImageProcessor.from_pretrained(
#         BASE_CHECKPOINT
#     )

#     print("Processor loaded.")

#     print("\nLoading fine-tuned RT-DETRv2 model...")

#     model = RTDetrV2ForObjectDetection.from_pretrained(
#         CHECKPOINT_PATH
#     )

#     model.to(DEVICE)
#     model.eval()

#     print("Model loaded.")
#     print("Device:", DEVICE)

#     return processor, model


# # ============================================================
# # DETECTION FUNCTION
# # ============================================================

# def detect_snakes(
#     frame,
#     processor,
#     model,
# ):
#     """
#     Run RT-DETRv2 on one frame.

#     Returns:
#         List of dictionaries containing:

#         {
#             "bbox": [x1, y1, x2, y2],
#             "confidence": float,
#             "class_id": 0,
#             "class_name": "snake"
#         }
#     """

#     # --------------------------------------------------------
#     # OpenCV BGR → RGB
#     # --------------------------------------------------------

#     frame_rgb = cv2.cvtColor(
#         frame,
#         cv2.COLOR_BGR2RGB
#     )

#     # --------------------------------------------------------
#     # Processor
#     # --------------------------------------------------------

#     inputs = processor(
#         images=frame_rgb,
#         return_tensors="pt"
#     )

#     inputs = {
#         key: value.to(DEVICE)
#         for key, value in inputs.items()
#     }

#     # --------------------------------------------------------
#     # Inference
#     # --------------------------------------------------------

#     with torch.no_grad():

#         outputs = model(**inputs)

#     # --------------------------------------------------------
#     # Original frame dimensions
#     # --------------------------------------------------------

#     height, width = frame.shape[:2]

#     target_sizes = torch.tensor(
#         [[height, width]],
#         device=DEVICE
#     )

#     # --------------------------------------------------------
#     # Post-processing
#     # --------------------------------------------------------

#     results = processor.post_process_object_detection(
#         outputs,
#         target_sizes=target_sizes,
#         threshold=DETECTION_THRESHOLD
#     )

#     result = results[0]

#     scores = result["scores"]
#     boxes = result["boxes"]
#     labels = result["labels"]

#     detections = []

#     # --------------------------------------------------------
#     # Convert detections
#     # --------------------------------------------------------

#     for box, score, label in zip(
#         boxes,
#         scores,
#         labels
#     ):

#         # This project has one class:
#         # 0 = snake
#         class_id = int(label.item())

#         if class_id != 0:
#             continue

#         confidence = float(
#             score.item()
#         )

#         x1, y1, x2, y2 = (
#             box.detach()
#             .cpu()
#             .tolist()
#         )

#         # ----------------------------------------------------
#         # Clamp coordinates to image boundaries
#         # ----------------------------------------------------

#         x1 = max(0.0, min(x1, width - 1))
#         y1 = max(0.0, min(y1, height - 1))
#         x2 = max(0.0, min(x2, width - 1))
#         y2 = max(0.0, min(y2, height - 1))

#         # ----------------------------------------------------
#         # Make sure box is valid
#         # ----------------------------------------------------

#         if x2 <= x1 or y2 <= y1:
#             continue

#         detections.append(
#             {
#                 "bbox": [
#                     x1,
#                     y1,
#                     x2,
#                     y2,
#                 ],
#                 "confidence": confidence,
#                 "class_id": 0,
#                 "class_name": "snake",
#             }
#         )

#     return detections


# # ============================================================
# # CONVERT TO BYTETrack FORMAT
# # ============================================================

# def convert_to_bytetrack_format(
#     detections
# ):
#     """
#     Convert detector output to the format expected by
#     ByteTrack / Supervision.

#     Format:

#         [x1, y1, x2, y2, confidence, class_id]
#     """

#     bytetrack_detections = []

#     for detection in detections:

#         x1, y1, x2, y2 = (
#             detection["bbox"]
#         )

#         confidence = detection["confidence"]

#         class_id = detection["class_id"]

#         bytetrack_detections.append(
#             [
#                 x1,
#                 y1,
#                 x2,
#                 y2,
#                 confidence,
#                 class_id,
#             ]
#         )

#     return bytetrack_detections


# # ============================================================
# # MAIN TEST
# # ============================================================

# def main():

#     print("=" * 65)
#     print("STEP 5.2 — RT-DETRv2 → BYTETrack DETECTION ADAPTER")
#     print("=" * 65)

#     # --------------------------------------------------------
#     # Validate paths
#     # --------------------------------------------------------

#     if not CHECKPOINT_PATH.exists():

#         raise FileNotFoundError(
#             f"Checkpoint not found:\n{CHECKPOINT_PATH}"
#         )

#     if not VIDEO_PATH.exists():

#         raise FileNotFoundError(
#             f"Video not found:\n{VIDEO_PATH}"
#         )

#     # --------------------------------------------------------
#     # Load detector
#     # --------------------------------------------------------

#     processor, model = load_detector()

#     # --------------------------------------------------------
#     # Open video
#     # --------------------------------------------------------

#     reader = VideoFrameReader(
#         video_path=VIDEO_PATH,
#         frame_skip=1,
#         resize_width=None,
#         resize_height=None,
#     )

#     print("\nVIDEO INFORMATION")
#     print("-" * 65)

#     print(
#         f"Video       : {VIDEO_PATH}"
#     )

#     print(
#         f"FPS         : {reader.fps:.2f}"
#     )

#     print(
#         f"Total frames: {reader.total_frames}"
#     )

#     print(
#         f"Resolution  : "
#         f"{reader.width} x {reader.height}"
#     )

#     print(
#         f"Threshold   : {DETECTION_THRESHOLD}"
#     )

#     print(
#         f"Device      : {DEVICE}"
#     )

#     # --------------------------------------------------------
#     # Process video
#     # --------------------------------------------------------

#     total_detections = 0
#     frames_processed = 0
#     frames_with_detections = 0

#     total_time = 0.0

#     print("\nRUNNING DETECTOR")
#     print("-" * 65)

#     for frame_data in reader.frames():

#         if (
#             MAX_FRAMES is not None
#             and frames_processed >= MAX_FRAMES
#         ):
#             break

#         frame_id = frame_data["frame_id"]
#         timestamp = frame_data["timestamp"]
#         frame = frame_data["frame"]

#         start_time = time.time()

#         detections = detect_snakes(
#             frame,
#             processor,
#             model
#         )

#         elapsed = time.time() - start_time

#         total_time += elapsed
#         frames_processed += 1

#         # ----------------------------------------------------
#         # Statistics
#         # ----------------------------------------------------

#         num_detections = len(detections)

#         total_detections += num_detections

#         if num_detections > 0:
#             frames_with_detections += 1

#         # ----------------------------------------------------
#         # Print useful frames
#         # ----------------------------------------------------

#         if num_detections > 0:

#             best_detection = max(
#                 detections,
#                 key=lambda x: x["confidence"]
#             )

#             print(
#                 f"Frame {frame_id:4d} | "
#                 f"time={timestamp:6.2f}s | "
#                 f"detections={num_detections:2d} | "
#                 f"best_conf="
#                 f"{best_detection['confidence']:.4f}"
#             )

#         # Progress every 100 frames
#         elif frames_processed % 100 == 0:

#             print(
#                 f"Processed {frames_processed} frames..."
#             )

#     # --------------------------------------------------------
#     # Release video
#     # --------------------------------------------------------

#     reader.release()

#     # --------------------------------------------------------
#     # Statistics
#     # --------------------------------------------------------

#     average_time = (
#         total_time / frames_processed
#         if frames_processed > 0
#         else 0
#     )

#     effective_fps = (
#         1.0 / average_time
#         if average_time > 0
#         else 0
#     )

#     average_detections = (
#         total_detections / frames_processed
#         if frames_processed > 0
#         else 0
#     )

#     # --------------------------------------------------------
#     # Final report
#     # --------------------------------------------------------

#     print("\n" + "=" * 65)
#     print("STEP 5.2 COMPLETE")
#     print("=" * 65)

#     print(
#         f"Frames processed        : "
#         f"{frames_processed}"
#     )

#     print(
#         f"Frames with detections  : "
#         f"{frames_with_detections}"
#     )

#     print(
#         f"Total detections        : "
#         f"{total_detections}"
#     )

#     print(
#         f"Average detections/frame: "
#         f"{average_detections:.3f}"
#     )

#     print(
#         f"Average inference time  : "
#         f"{average_time * 1000:.2f} ms"
#     )

#     print(
#         f"Effective FPS           : "
#         f"{effective_fps:.2f}"
#     )

#     print(
#         f"Video FPS               : "
#         f"{reader.fps:.2f}"
#     )

#     print(
#         f"Detection threshold     : "
#         f"{DETECTION_THRESHOLD}"
#     )

#     print("=" * 65)


# # ============================================================
# # RUN
# # ============================================================

# if __name__ == "__main__":
#     main()

