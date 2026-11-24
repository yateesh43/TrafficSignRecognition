import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
import tensorflow as tf
from class_names import CLASS_NAMES


# ==========================================
# MODEL PATHS
# ==========================================

YOLO_MODEL_PATH = "runs/detect/traffic_sign_detector-2/weights/best.pt"
CNN_MODEL_PATH = "models/baseline_cnn.keras"

IMG_SIZE = (64, 64)

# YOLO detection threshold
DETECTION_CONFIDENCE = 0.25

# CNN threshold
CLASSIFICATION_CONFIDENCE = 0.25

# Run YOLO once every N frames
DETECTION_INTERVAL = 3

# Keep detection alive for this many frames
MAX_MISSED_FRAMES = 12

# CNN prediction smoothing
SMOOTHING_FRAMES = 8


# ==========================================
# LOAD MODELS
# ==========================================

print("Loading YOLO detector...")

detector = YOLO(YOLO_MODEL_PATH)

print("YOLO loaded.")

print("Loading CNN classifier...")

classifier = tf.keras.models.load_model(
    CNN_MODEL_PATH
)

print("CNN loaded.")

print("======================================")
print("YOLO + CNN SYSTEM READY")
print("======================================")


# ==========================================
# CAMERA
# ==========================================

# ==========================================
# VIDEO / WEBCAM INPUT
# ==========================================

print("Select input source:")
print("1. Webcam")
print("2. Video file")

choice = input("Enter your choice (1 or 2): ").strip()

if choice == "1":
    cap = cv2.VideoCapture(0)
    print("Starting webcam...")

elif choice == "2":
    video_path = input(
        "Enter the full path of your video file: "
    ).strip().strip('"')

    cap = cv2.VideoCapture(video_path)
    print("Loading video...")

else:
    print("Invalid choice.")
    exit()

if not cap.isOpened():
    print("Error: Could not open the selected input.")
    exit()


# ==========================================
# VARIABLES
# ==========================================

frame_count = 0

last_box = None
last_detection_confidence = 0

missed_frames = 0

prediction_history = deque(
    maxlen=SMOOTHING_FRAMES
)

last_class = None
last_class_confidence = 0


# ==========================================
# FPS
# ==========================================

prev_time = time.time()

fps = 0


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        if choice == "2":
            print("Video playback finished.")
        else:
            print("Error reading frame.")

        break


    frame_count += 1


    # ======================================
    # YOLO DETECTION
    # ======================================

    if frame_count % DETECTION_INTERVAL == 0:

        results = detector.predict(
            frame,
            imgsz=320,
            conf=DETECTION_CONFIDENCE,
            verbose=False
        )

        found_detection = False


        for result in results:

            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue


            # Highest confidence detection
            best_index = int(
                boxes.conf.argmax().item()
            )

            box = boxes[best_index]


            confidence = float(
                box.conf.item()
            )


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )


            # Keep coordinates inside frame
            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(
                frame.shape[1],
                x2
            )

            y2 = min(
                frame.shape[0],
                y2
            )


            if x2 <= x1 or y2 <= y1:
                continue


            # Save detection
            last_box = (
                x1,
                y1,
                x2,
                y2
            )

            last_detection_confidence = confidence

            missed_frames = 0

            found_detection = True

            break


        # No detection this cycle
        if not found_detection:

            missed_frames += DETECTION_INTERVAL


    # ======================================
    # HANDLE TEMPORARY MISSED DETECTION
    # ======================================

    if last_box is not None:

        if missed_frames <= MAX_MISSED_FRAMES:

            x1, y1, x2, y2 = last_box

            # Draw detection box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # YOLO label
            cv2.putText(
                frame,
                f"Traffic Sign "
                f"{last_detection_confidence * 100:.1f}%",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            # ==================================
            # CROP SIGN
            # ==================================

            crop = frame[
                y1:y2,
                x1:x2
            ]


            if crop.size > 0:

                # Resize
                crop_resized = cv2.resize(
                    crop,
                    IMG_SIZE
                )


                # BGR → RGB
                crop_rgb = cv2.cvtColor(
                    crop_resized,
                    cv2.COLOR_BGR2RGB
                )


                # Normalize
                crop_normalized = (
                    crop_rgb.astype(
                        np.float32
                    ) / 255.0
                )


                crop_input = np.expand_dims(
                    crop_normalized,
                    axis=0
                )


                # ==================================
                # CNN CLASSIFICATION
                # ==================================

                predictions = classifier.predict(
                    crop_input,
                    verbose=0
                )[0]


                predicted_class = int(
                    np.argmax(predictions)
                )


                class_confidence = float(
                    predictions[predicted_class]
                )


                # Save prediction
                prediction_history.append(
                    predicted_class
                )


                # ==================================
                # STABLE CLASS
                # ==================================

                values, counts = np.unique(
                    prediction_history,
                    return_counts=True
                )


                stable_class = int(
                    values[np.argmax(counts)]
                )


                # Confidence for current prediction
                last_class = stable_class

                last_class_confidence = (
                    class_confidence
                )


                # ==================================
                # DISPLAY CLASSIFICATION
                # ==================================

                if class_confidence >= CLASSIFICATION_CONFIDENCE:

                    sign_name = CLASS_NAMES[stable_class]

                    classification_text = (
                        f"{sign_name} "
                        f"{class_confidence * 100:.1f}%"
                    )

                else:

                    classification_text = (
                        f"{sign_name} "
                        f"{class_confidence * 100:.1f}% "
                        f"(low confidence)"
                    )


                cv2.putText(
                    frame,
                    classification_text,
                    (
                        x1,
                        min(
                            y2 + 28,
                            frame.shape[0] - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )


        else:

            # Detection has been missing too long
            last_box = None

            prediction_history.clear()

            last_class = None


    else:

        cv2.putText(
            frame,
            "Searching for traffic sign...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


    # ======================================
    # FPS
    # ======================================

    current_time = time.time()

    elapsed = current_time - prev_time

    if elapsed > 0:

        current_fps = 1 / elapsed

        # Smooth FPS
        fps = (
            0.9 * fps
            +
            0.1 * current_fps
        )

    prev_time = current_time


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (
            20,
            frame.shape[0] - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ======================================
    # TITLE
    # ======================================

    cv2.putText(
        frame,
        "YOLO + CNN Traffic Sign Recognition",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ======================================
    # SHOW FRAME
    # ======================================

    cv2.imshow(
        "Traffic Sign Detection + Recognition",
        frame
    )


    # ======================================
    # QUIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("Camera stopped.")