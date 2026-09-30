import cv2
from collections import Counter

from ultralytics import YOLO

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


class VideoAnalyzer:

    def __init__(self):

        print("Loading YOLO model...")

        self.yolo = YOLO("yolo26n.pt")

        print("YOLO model loaded.")

        print("Loading image captioning model...")

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        print("Image captioning model loaded.")

    # =====================================================
    # IMAGE CAPTION
    # =====================================================

    def generate_caption(self, frame):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb_frame
        )

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        output = self.caption_model.generate(
            **inputs,
            max_new_tokens=40
        )

        caption = self.processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return caption

    # =====================================================
    # VIDEO ANALYSIS
    # =====================================================

    def analyze_video(self, video_path):

        print()
        print("======================================")
        print("STARTING VIDEO ANALYSIS")
        print("======================================")

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            return {
                "success": False,
                "error": "Could not open video."
            }

        # -------------------------------------------------
        # VIDEO INFORMATION
        # -------------------------------------------------

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

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

        if fps <= 0:
            fps = 30

        duration = (
            total_frames / fps
        )

        print("FPS:", fps)
        print("Total frames:", total_frames)
        print("Duration:", duration)

        # -------------------------------------------------
        # SAMPLE EVERY 2 SECONDS
        # -------------------------------------------------

        sample_seconds = 2

        frame_interval = max(
            1,
            int(fps * sample_seconds)
        )

        # -------------------------------------------------
        # STORAGE
        # -------------------------------------------------

        frame_number = 0

        analyzed_frames = 0

        previous_gray = None

        all_objects = []

        timeline = []

        motion_values = []

        # -------------------------------------------------
        # READ VIDEO
        # -------------------------------------------------

        while True:

            success, frame = cap.read()

            if not success:
                break

            # Only analyze selected frames
            if frame_number % frame_interval == 0:

                analyzed_frames += 1

                current_time = (
                    frame_number / fps
                )

                print(
                    f"Analyzing frame "
                    f"{analyzed_frames} "
                    f"at {current_time:.1f}s"
                )

                # =========================================
                # YOLO OBJECT DETECTION
                # =========================================

                results = self.yolo.predict(
                    frame,
                    conf=0.35,
                    verbose=False
                )

                detected_objects = []

                for result in results:

                    if result.boxes is None:
                        continue

                    for box in result.boxes:

                        class_id = int(
                            box.cls[0]
                        )

                        object_name = (
                            self.yolo.names[
                                class_id
                            ]
                        )

                        confidence = float(
                            box.conf[0]
                        )

                        if confidence >= 0.35:

                            detected_objects.append(
                                object_name
                            )

                            all_objects.append(
                                object_name
                            )

                # =========================================
                # MOTION DETECTION
                # =========================================

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

                gray = cv2.resize(
                    gray,
                    (320, 240)
                )

                motion_ratio = 0

                if previous_gray is not None:

                    difference = cv2.absdiff(
                        previous_gray,
                        gray
                    )

                    _, threshold = cv2.threshold(
                        difference,
                        25,
                        255,
                        cv2.THRESH_BINARY
                    )

                    moving_pixels = (
                        cv2.countNonZero(
                            threshold
                        )
                    )

                    total_pixels = (
                        threshold.shape[0]
                        *
                        threshold.shape[1]
                    )

                    motion_ratio = (
                        moving_pixels /
                        total_pixels
                    )

                previous_gray = gray

                motion_values.append(
                    motion_ratio
                )

                # =========================================
                # ACTIVITY LEVEL
                # =========================================

                if motion_ratio < 0.03:

                    activity = "Very Low"

                elif motion_ratio < 0.08:

                    activity = "Low"

                elif motion_ratio < 0.18:

                    activity = "Moderate"

                else:

                    activity = "High"

                # =========================================
                # IMAGE CAPTION
                # =========================================

                caption = self.generate_caption(
                    frame
                )

                print(
                    "Caption:",
                    caption
                )

                # =========================================
                # UNIQUE OBJECTS
                # =========================================

                unique_objects = sorted(
                    set(
                        detected_objects
                    )
                )

                # =========================================
                # TIMELINE EVENT
                # =========================================

                event = {

                    "time":
                        round(
                            current_time,
                            2
                        ),

                    "objects":
                        unique_objects,

                    "activity":
                        activity,

                    "motion":
                        round(
                            motion_ratio,
                            4
                        ),

                    "caption":
                        caption
                }

                timeline.append(
                    event
                )

            frame_number += 1

        cap.release()

        # =================================================
        # OBJECT COUNTS
        # =================================================

        object_counts = Counter(
            all_objects
        )

        object_counts = dict(
            object_counts
        )

        # =================================================
        # AVERAGE MOTION
        # =================================================

        if motion_values:

            average_motion = (
                sum(motion_values)
                /
                len(motion_values)
            )

        else:

            average_motion = 0

        # =================================================
        # ACTIVITY
        # =================================================

        if average_motion < 0.03:

            activity_level = "Very Low"

        elif average_motion < 0.08:

            activity_level = "Low"

        elif average_motion < 0.18:

            activity_level = "Moderate"

        else:

            activity_level = "High"

        # =================================================
        # DEBUG TIMELINE
        # =================================================

        print()
        print("======================================")
        print("TIMELINE CREATED")
        print("======================================")

        print(
            "Timeline events:",
            len(timeline)
        )

        for event in timeline:

            print(event)

        print("======================================")

        # =================================================
        # DESCRIPTION
        # =================================================

        description = (
            self.generate_description(
                duration,
                object_counts,
                timeline,
                activity_level
            )
        )

        # =================================================
        # FINAL RESULT
        # =================================================

        return {

            "success":
                True,

            "duration":
                round(
                    duration,
                    2
                ),

            "fps":
                round(
                    fps,
                    2
                ),

            "width":
                width,

            "height":
                height,

            "total_frames":
                total_frames,

            "analyzed_frames":
                analyzed_frames,

            "objects":
                object_counts,

            "activity": {

                "average":
                    round(
                        average_motion,
                        4
                    ),

                "level":
                    activity_level
            },

            "timeline":
                timeline,

            "events":
                timeline,

            "description":
                description
        }

    # =====================================================
    # DESCRIPTION GENERATOR
    # =====================================================

    def generate_description(
        self,
        duration,
        object_counts,
        timeline,
        activity_level
    ):

        paragraphs = []

        # -------------------------------------------------
        # INTRODUCTION
        # -------------------------------------------------

        paragraphs.append(
            f"The video is approximately "
            f"{duration:.1f} seconds long. "
            f"The AI analyzed "
            f"{len(timeline)} sampled frames "
            f"to understand the visual content "
            f"and changes throughout the video."
        )

        # -------------------------------------------------
        # OBJECTS
        # -------------------------------------------------

        if object_counts:

            objects = []

            for name, count in (
                object_counts.items()
            ):

                objects.append(
                    f"{name} "
                    f"({count} detections)"
                )

            object_text = ", ".join(
                objects
            )

            paragraphs.append(
                f"The system detected "
                f"{object_text} "
                f"throughout the sampled frames."
            )

        else:

            paragraphs.append(
                "The system did not detect "
                "any recognizable objects "
                "with sufficient confidence."
            )

        # -------------------------------------------------
        # ACTIVITY
        # -------------------------------------------------

        paragraphs.append(
            f"The overall visual activity "
            f"was {activity_level.lower()}."
        )

        # -------------------------------------------------
        # TIMELINE
        # -------------------------------------------------

        if timeline:

            paragraphs.append(
                "The visual timeline is as follows:"
            )

            for event in timeline:

                time = event["time"]

                caption = event["caption"]

                activity = event["activity"]

                objects = event["objects"]

                if objects:

                    object_text = ", ".join(
                        objects
                    )

                    text = (
                        f"At approximately "
                        f"{time:.1f} seconds, "
                        f"{caption}. "
                        f"Detected objects included "
                        f"{object_text}. "
                        f"Activity was "
                        f"{activity.lower()}."
                    )

                else:

                    text = (
                        f"At approximately "
                        f"{time:.1f} seconds, "
                        f"{caption}. "
                        f"Activity was "
                        f"{activity.lower()}."
                    )

                paragraphs.append(
                    text
                )

        # -------------------------------------------------
        # BEGINNING AND END
        # -------------------------------------------------

        if len(timeline) >= 2:

            first = timeline[0]

            last = timeline[-1]

            paragraphs.append(
                f"The video begins with "
                f"{first['caption']}. "
                f"Toward the end of the recording, "
                f"the scene is described as "
                f"{last['caption']}."
            )

        # -------------------------------------------------
        # LIMITATION
        # -------------------------------------------------

        paragraphs.append(
            "This report describes visible "
            "computer-vision observations. "
            "Object detection, image captioning, "
            "and motion analysis cannot by themselves "
            "establish the exact intention, cause, "
            "or meaning of an observed event."
        )

        return "\n\n".join(
            paragraphs
        )