import cv2
import time
import numpy as np
import tensorflow as tf
from ultralytics import YOLO


# ==========================================
# MODEL PATHS
# ==========================================

YOLO_MODEL_PATH = "runs/detect/traffic_sign_detector-2/weights/best.pt"
CNN_MODEL_PATH = "models/baseline_cnn.keras"

IMG_SIZE = (64, 64)

DETECTION_CONFIDENCE = 0.40

# Run YOLO every 3 frames, same as our working system
DETECTION_INTERVAL = 3


# ==========================================
# LOAD MODELS
# ==========================================

print("Loading YOLO...")

detector = YOLO(YOLO_MODEL_PATH)

print("YOLO loaded.")

print("Loading CNN...")

classifier = tf.keras.models.load_model(
    CNN_MODEL_PATH
)

print("CNN loaded.")


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Could not open webcam.")
    exit()


# ==========================================
# VARIABLES
# ==========================================

frame_count = 0

yolo_times = []
cnn_times = []
total_times = []

last_box = None


print()
print("======================================")
print("REAL-TIME BENCHMARK STARTED")
print("Press Q to stop")
print("======================================")


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    frame_start = time.perf_counter()


    # ======================================
    # YOLO
    # ======================================

    if frame_count % DETECTION_INTERVAL == 0:

        yolo_start = time.perf_counter()

        results = detector.predict(
            frame,
            imgsz=320,
            conf=DETECTION_CONFIDENCE,
            verbose=False
        )

        yolo_end = time.perf_counter()

        yolo_time = (
            yolo_end - yolo_start
        ) * 1000

        yolo_times.append(yolo_time)


        # Find detection
        found = False

        for result in results:

            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            best_index = int(
                boxes.conf.argmax().item()
            )

            box = boxes[best_index]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

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

            last_box = (
                x1,
                y1,
                x2,
                y2
            )

            found = True

            break


    # ======================================
    # CNN
    # ======================================

    if last_box is not None:

        x1, y1, x2, y2 = last_box

        crop = frame[
            y1:y2,
            x1:x2
        ]

        if crop.size > 0:

            crop_resized = cv2.resize(
                crop,
                IMG_SIZE
            )

            crop_rgb = cv2.cvtColor(
                crop_resized,
                cv2.COLOR_BGR2RGB
            )

            crop_normalized = (
                crop_rgb.astype(
                    np.float32
                ) / 255.0
            )

            crop_input = np.expand_dims(
                crop_normalized,
                axis=0
            )


            cnn_start = time.perf_counter()

            classifier.predict(
                crop_input,
                verbose=0
            )

            cnn_end = time.perf_counter()

            cnn_time = (
                cnn_end - cnn_start
            ) * 1000

            cnn_times.append(
                cnn_time
            )


    # ======================================
    # TOTAL FRAME TIME
    # ======================================

    frame_end = time.perf_counter()

    total_time = (
        frame_end - frame_start
    ) * 1000

    total_times.append(
        total_time
    )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.putText(
        frame,
        "Benchmark Running - Press Q",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Real-Time Benchmark",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()


# ==========================================
# RESULTS
# ==========================================

print()
print("======================================")
print("REAL-TIME BENCHMARK RESULTS")
print("======================================")


if len(yolo_times) > 0:

    print(
        f"Average YOLO time: "
        f"{np.mean(yolo_times):.2f} ms"
    )


if len(cnn_times) > 0:

    print(
        f"Average CNN time: "
        f"{np.mean(cnn_times):.2f} ms"
    )


if len(total_times) > 0:

    print(
        f"Average frame time: "
        f"{np.mean(total_times):.2f} ms"
    )

    fps = 1000 / np.mean(total_times)

    print(
        f"Approximate FPS: "
        f"{fps:.2f}"
    )


print("======================================")