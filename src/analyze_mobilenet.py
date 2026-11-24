import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import os

from sklearn.metrics import confusion_matrix, classification_report

MODEL_PATH = "models/mobilenetv3_small.keras"
TEST_CSV = "dataset/GTSRB/Test.csv"
BASE_DIR = "dataset/GTSRB"

IMG_SIZE = (64, 64)
NUM_CLASSES = 43

print("\n======================================")
print("LOADING MODEL")
print("======================================")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ======================================
# LOAD TEST DATA
# ======================================

df = pd.read_csv(TEST_CSV)

images = []
labels = []

print("\n======================================")
print("LOADING TEST IMAGES")
print("======================================")

for index, row in df.iterrows():

    image_path = os.path.join(BASE_DIR, row["Path"])

    image = cv2.imread(image_path)

    if image is None:
        print("Could not load:", image_path)
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)
    image = image.astype(np.float32)

    images.append(image)
    labels.append(int(row["ClassId"]))

    if (index + 1) % 2000 == 0:
        print(f"Loaded {index + 1}/{len(df)} images")


X_test = np.array(images)
y_test = np.array(labels)

print("\nTest image shape:", X_test.shape)
print("Test labels:", y_test.shape)


# ======================================
# PREDICTIONS
# ======================================

print("\n======================================")
print("GENERATING PREDICTIONS")
print("======================================")

predictions = model.predict(
    X_test,
    batch_size=32,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)


# ======================================
# ACCURACY
# ======================================

accuracy = np.mean(y_pred == y_test)

print("\n======================================")
print("MOBILENETV3 TEST RESULT")
print("======================================")

print(f"Correct: {np.sum(y_pred == y_test)}")
print(f"Incorrect: {np.sum(y_pred != y_test)}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# ======================================
# CLASSIFICATION REPORT
# ======================================

report = classification_report(
    y_test,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    zero_division=0
)

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

print(report)

with open(
    "results/mobilenetv3_classification_report.txt",
    "w"
) as f:
    f.write(report)


# ======================================
# CONFUSION MATRIX
# ======================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(NUM_CLASSES))
)

np.savetxt(
    "results/mobilenetv3_confusion_matrix.csv",
    cm,
    fmt="%d",
    delimiter=","
)

print("\nConfusion matrix saved.")


# ======================================
# CLASS-WISE ACCURACY
# ======================================

print("\n======================================")
print("CLASS-WISE ACCURACY")
print("======================================")

for class_id in range(NUM_CLASSES):

    total = np.sum(y_test == class_id)
    correct = np.sum(
        (y_test == class_id) &
        (y_pred == class_id)
    )

    if total > 0:
        class_accuracy = correct / total * 100
    else:
        class_accuracy = 0

    print(
        f"Class {class_id:2d}: "
        f"{class_accuracy:6.2f}% "
        f"({correct}/{total})"
    )

print("\n======================================")
print("ANALYSIS COMPLETE")
print("======================================")