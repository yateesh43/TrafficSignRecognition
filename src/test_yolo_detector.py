from ultralytics import YOLO
import os

MODEL_PATH = "runs/detect/traffic_sign_detector-2/weights/best.pt"

SOURCE = "dataset/GTSRB/Test"

print("Loading YOLO detector...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")

print("\nRunning detection on test images...")

results = model.predict(
    source=SOURCE,
    imgsz=320,
    conf=0.5,
    save=True,
    project="results",
    name="yolo_test"
)

print("\n======================================")
print("YOLO DETECTION TEST COMPLETE")
print("======================================")

print("Results saved to:")
print("results/yolo_test")

print("======================================")