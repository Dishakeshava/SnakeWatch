# ============================================================
# STEP 5.1 — VIDEO INPUT & FRAME PROCESSING
# ============================================================
# Purpose:
#   Read surveillance video frame-by-frame and provide frames
#   to the RT-DETRv2 detection pipeline.
#
# Responsibilities:
#   1. Open input video
#   2. Read video properties
#   3. Process frames sequentially
#   4. Maintain frame number and timestamp
#   5. Optionally resize frames
#   6. Optionally save sampled frames for inspection
#   7. Provide clean frame information for later modules
#
# This module DOES NOT:
#   - Detect snakes
#   - Run RT-DETRv2
#   - Run ByteTrack
#   - Calculate velocity
#   - Calculate trajectory
#
# ============================================================

from pathlib import Path
import cv2


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Google Drive project directory
# ------------------------------------------------------------

PROJECT_DIR = Path(
    "/content/drive/MyDrive/snake_detection/Snake_detection"
)

# ------------------------------------------------------------
# Test video
# ------------------------------------------------------------
# Start with ONE video only.
# We can process all 53 videos later.

VIDEO_PATH = (
    PROJECT_DIR
    / "snake_videos"
    / "10231672-hd_1080_1920_30fps.mp4"
)

# ------------------------------------------------------------
# Directory where sample frames will be saved
# ------------------------------------------------------------

FRAME_OUTPUT_DIR = (
    PROJECT_DIR
    / "dataset"
    / "video_frames"
)

# ------------------------------------------------------------
# Process every Nth frame
#
# 1 = every frame
# 2 = every second frame
# 5 = every fifth frame
#
# For tracking, keep this at 1.
# ------------------------------------------------------------

FRAME_SKIP = 1

# ------------------------------------------------------------
# Maximum number of frames to save for inspection
# ------------------------------------------------------------

MAX_SAVED_FRAMES = 30

# ------------------------------------------------------------
# Whether to save sampled frames
# ------------------------------------------------------------

SAVE_FRAMES = True

# ------------------------------------------------------------
# Optional resize
#
# None = preserve original video resolution
# ------------------------------------------------------------

RESIZE_WIDTH = None
RESIZE_HEIGHT = None


# ============================================================
# VIDEO FRAME READER
# ============================================================

class VideoFrameReader:
    """
    Reads a video sequentially and provides frames along with
    frame number and timestamp information.
    """

    def __init__(
        self,
        video_path,
        frame_skip=1,
        resize_width=None,
        resize_height=None,
    ):

        self.video_path = Path(video_path)

        self.frame_skip = max(1, int(frame_skip))

        self.resize_width = resize_width
        self.resize_height = resize_height

        # ----------------------------------------------------
        # Open video
        # ----------------------------------------------------

        self.cap = cv2.VideoCapture(
            str(self.video_path)
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open video:\n"
                f"{self.video_path}"
            )

        # ----------------------------------------------------
        # VIDEO PROPERTIES
        # ----------------------------------------------------

        self.fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        if self.fps <= 0:
            self.fps = 30.0

        self.total_frames = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        self.width = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        self.height = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        self.duration = (
            self.total_frames / self.fps
            if self.fps > 0
            else 0
        )

    # --------------------------------------------------------
    # VIDEO INFORMATION
    # --------------------------------------------------------

    def get_video_info(self):
        """
        Return basic video properties.
        """

        return {
            "video_path": str(
                self.video_path
            ),

            "fps": self.fps,

            "total_frames":
                self.total_frames,

            "width":
                self.width,

            "height":
                self.height,

            "duration_seconds":
                self.duration,
        }

    # --------------------------------------------------------
    # FRAME ITERATOR
    # --------------------------------------------------------

    def frames(self):
        """
        Yield video frames sequentially.

        Each yielded item contains:

            {
                "frame_id": int,
                "timestamp": float,
                "frame": numpy array,
                "width": int,
                "height": int,
            }
        """

        frame_id = 0

        while True:

            success, frame = (
                self.cap.read()
            )

            if not success:
                break

            # -----------------------------------------------
            # FRAME SKIPPING
            # -----------------------------------------------

            if (
                frame_id % self.frame_skip
                != 0
            ):

                frame_id += 1
                continue

            # -----------------------------------------------
            # TIMESTAMP
            # -----------------------------------------------

            timestamp = (
                frame_id / self.fps
            )

            # -----------------------------------------------
            # OPTIONAL RESIZE
            # -----------------------------------------------

            if (
                self.resize_width is not None
                and self.resize_height is not None
            ):

                frame = cv2.resize(
                    frame,
                    (
                        self.resize_width,
                        self.resize_height,
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )

            frame_height, frame_width = (
                frame.shape[:2]
            )

            yield {
                "frame_id": frame_id,

                "timestamp": timestamp,

                "frame": frame,

                "width": frame_width,

                "height": frame_height,
            }

            frame_id += 1

    # --------------------------------------------------------
    # RELEASE VIDEO
    # --------------------------------------------------------

    def release(self):

        if self.cap is not None:
            self.cap.release()

    # --------------------------------------------------------
    # CONTEXT MANAGER SUPPORT
    # --------------------------------------------------------

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.release()


# ============================================================
# SAVE SAMPLE FRAMES
# ============================================================

def save_sample_frames(
    reader,
    output_dir,
    max_frames=30,
):
    """
    Read the video and save a limited number of frames.

    This is only for verifying that the video input pipeline
    is working correctly.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    saved_count = 0

    for item in reader.frames():

        frame = item["frame"]
        frame_id = item["frame_id"]

        output_path = (
            output_dir /
            f"frame_{frame_id:06d}.jpg"
        )

        success = cv2.imwrite(
            str(output_path),
            frame
        )

        if success:
            saved_count += 1

        if (
            max_frames > 0
            and saved_count >= max_frames
        ):
            break

    return saved_count


# ============================================================
# STANDALONE TEST
# ============================================================

def main():

    print("=" * 65)
    print("STEP 5.1 — VIDEO INPUT & FRAME PROCESSING")
    print("=" * 65)

    # --------------------------------------------------------
    # CHECK VIDEO
    # --------------------------------------------------------

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"\nVideo not found:\n"
            f"{VIDEO_PATH}\n\n"
            f"Check the VIDEO_PATH."
        )

    # --------------------------------------------------------
    # CREATE READER
    # --------------------------------------------------------

    reader = VideoFrameReader(
        video_path=VIDEO_PATH,
        frame_skip=FRAME_SKIP,
        resize_width=RESIZE_WIDTH,
        resize_height=RESIZE_HEIGHT,
    )

    # --------------------------------------------------------
    # DISPLAY VIDEO INFORMATION
    # --------------------------------------------------------

    info = reader.get_video_info()

    print("\nVIDEO INFORMATION")
    print("-" * 65)

    print(
        f"Video           : "
        f"{info['video_path']}"
    )

    print(
        f"FPS             : "
        f"{info['fps']:.2f}"
    )

    print(
        f"Total frames    : "
        f"{info['total_frames']}"
    )

    print(
        f"Resolution      : "
        f"{info['width']} x "
        f"{info['height']}"
    )

    print(
        f"Duration        : "
        f"{info['duration_seconds']:.2f} seconds"
    )

    print(
        f"Frame skip      : "
        f"{FRAME_SKIP}"
    )

    # --------------------------------------------------------
    # READ FRAMES
    # --------------------------------------------------------

    print("\nREADING VIDEO")
    print("-" * 65)

    frames_read = 0

    if SAVE_FRAMES:

        reader.release()

        reader = VideoFrameReader(
            video_path=VIDEO_PATH,
            frame_skip=FRAME_SKIP,
            resize_width=RESIZE_WIDTH,
            resize_height=RESIZE_HEIGHT,
        )

        frames_read = save_sample_frames(
            reader,
            FRAME_OUTPUT_DIR,
            MAX_SAVED_FRAMES,
        )

    else:

        for item in reader.frames():

            frames_read += 1

            if frames_read >= MAX_SAVED_FRAMES:
                break

    # --------------------------------------------------------
    # RELEASE
    # --------------------------------------------------------

    reader.release()

    print(
        f"Frames processed : "
        f"{frames_read}"
    )

    if SAVE_FRAMES:

        print(
            "Sample frames saved to:"
        )

        print(
            FRAME_OUTPUT_DIR
        )

    print("\n" + "=" * 65)
    print("STEP 5.1 COMPLETE")
    print("=" * 65)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()