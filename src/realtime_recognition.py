import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque, Counter

# ======================================
# CONFIGURATION
# ======================================

MODEL_PATH = "models/baseline_cnn.keras"
IMG_SIZE = (64, 64)

CONFIDENCE_THRESHOLD = 50
SMOOTHING_FRAMES = 8

CLASS_NAMES = {
    0: "Speed limit 20",
    1: "Speed limit 30",
    2: "Speed limit 50",
    3: "Speed limit 60",
    4: "Speed limit 70",
    5: "Speed limit 80",
    6: "End of speed limit 80",
    7: "Speed limit 100",
    8: "Speed limit 120",
    9: "No passing",
    10: "No passing (>3.5 tons)",
    11: "Right-of-way",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles >3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of speed limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Straight or right",
    37: "Straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing (>3.5 tons)"
}


# ======================================
# LOAD MODEL
# ======================================

print("\nLoading CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ======================================
# START CAMERA
# ======================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

print("Webcam started.")
print("Place a traffic sign inside the green box.")
print("Press Q to quit.")


# ======================================
# PREDICTION HISTORY
# ======================================

prediction_history = deque(
    maxlen=SMOOTHING_FRAMES
)


# ======================================
# FPS
# ======================================

prev_time = time.time()


# ======================================
# MAIN LOOP
# ======================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    height, width = frame.shape[:2]

    # ==================================
    # DEFINE CENTER ROI
    # ==================================

    roi_width = int(width * 0.45)
    roi_height = int(height * 0.65)

    x1 = (width - roi_width) // 2
    y1 = (height - roi_height) // 2

    x2 = x1 + roi_width
    y2 = y1 + roi_height

    roi = frame[y1:y2, x1:x2]


    # ==================================
    # PREPROCESS ROI
    # ==================================

    roi_rgb = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB
    )

    roi_resized = cv2.resize(
        roi_rgb,
        IMG_SIZE
    )

    image = roi_resized.astype(
        np.float32
    ) / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )


    # ==================================
    # CNN PREDICTION
    # ==================================

    prediction = model.predict(
        image,
        verbose=0
    )

    predicted_class = int(
        np.argmax(prediction[0])
    )

    confidence = float(
        prediction[0][predicted_class]
    ) * 100


    # ==================================
    # PREDICTION SMOOTHING
    # ==================================

    if confidence >= CONFIDENCE_THRESHOLD:

        prediction_history.append(
            predicted_class
        )

    else:

        prediction_history.append(-1)


    valid_predictions = [
        p for p in prediction_history
        if p != -1
    ]


    if len(valid_predictions) >= 3:

        most_common = Counter(
            valid_predictions
        ).most_common(1)[0][0]

        stable_class = most_common

        stable_name = CLASS_NAMES[
            stable_class
        ]

        # Average confidence for stable class
        stable_confidences = []

        for p in valid_predictions:

            if p == stable_class:
                stable_confidences.append(
                    confidence
                )

        if stable_confidences:
            stable_confidence = np.mean(
                stable_confidences
            )
        else:
            stable_confidence = confidence

    else:

        stable_class = None
        stable_name = "Place sign inside ROI"
        stable_confidence = 0


    # ==================================
    # DRAW ROI
    # ==================================

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )


    # ==================================
    # INFORMATION PANEL
    # ==================================

    cv2.rectangle(
        frame,
        (10, 10),
        (width - 10, 125),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "EDGE AI TRAFFIC SIGN RECOGNITION",
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    if stable_class is not None:

        cv2.putText(
            frame,
            f"Prediction: {stable_name}",
            (25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {stable_confidence:.1f}%",
            (25, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "Place traffic sign inside ROI",
            (25, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    # ==================================
    # FPS
    # ==================================

    current_time = time.time()

    fps = 1 / max(
        current_time - prev_time,
        0.001
    )

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (width - 120, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ==================================
    # INSTRUCTIONS
    # ==================================

    cv2.putText(
        frame,
        "Place sign inside green box | Press Q to quit",
        (25, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ==================================
    # DISPLAY
    # ==================================

    cv2.imshow(
        "Edge AI Traffic Sign Recognition",
        frame
    )


    # ==================================
    # QUIT
    # ==================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ======================================
# CLEANUP
# ======================================

cap.release()
cv2.destroyAllWindows()

print("Webcam stopped.")