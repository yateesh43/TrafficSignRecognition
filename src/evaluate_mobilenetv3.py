import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "models/mobilenetv3_small.keras"
TEST_CSV = "dataset/GTSRB/Test.csv"
DATASET_DIR = "dataset/GTSRB"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
NUM_CLASSES = 43


# ==========================================
# Load Model
# ==========================================

print("\n======================================")
print("LOADING MOBILENETV3 MODEL")
print("======================================")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ==========================================
# Load Test CSV
# ==========================================

print("\n======================================")
print("LOADING TEST DATA")
print("======================================")

df = pd.read_csv(TEST_CSV)

print(
    "Total test images:",
    len(df)
)


# ==========================================
# Prepare Test Images
# ==========================================

print("\n======================================")
print("PREPARING TEST IMAGES")
print("======================================")

images = []
labels = []

for index, row in df.iterrows():

    image_path = os.path.join(
        DATASET_DIR,
        row["Path"]
    )

    image = cv2.imread(image_path)

    if image is None:
        print(
            "Warning: Could not load:",
            image_path
        )
        continue

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        IMG_SIZE
    )

    # IMPORTANT:
    # Do NOT divide by 255.
    # MobileNetV3 performs its own preprocessing.

    image = image.astype("float32")

    images.append(image)

    labels.append(
        int(row["ClassId"])
    )

    if (index + 1) % 2000 == 0:
        print(
            f"Processed {index + 1} images..."
        )


X_test = np.array(images)
y_test = np.array(labels)


print("\nTest image shape:")
print(X_test.shape)

print("\nTest label shape:")
print(y_test.shape)

print(
    "\nPixel range:",
    X_test.min(),
    "to",
    X_test.max()
)


# ==========================================
# Generate Predictions
# ==========================================

print("\n======================================")
print("GENERATING PREDICTIONS")
print("======================================")

predictions = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

print(
    "Predictions generated successfully."
)


# ==========================================
# Test Accuracy
# ==========================================

test_accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n======================================")
print("MOBILENETV3 TEST RESULTS")
print("======================================")

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    "Correct predictions:",
    np.sum(y_test == y_pred)
)

print(
    "Incorrect predictions:",
    np.sum(y_test != y_pred)
)


# ==========================================
# Classification Report
# ==========================================

print("\n======================================")
print("CLASSIFICATION REPORT")
print("======================================")

class_names = [
    str(i)
    for i in range(NUM_CLASSES)
]

report = classification_report(
    y_test,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)


# ==========================================
# Save Classification Report
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)

report_path = (
    "results/mobilenetv3_classification_report.txt"
)

with open(
    report_path,
    "w"
) as f:

    f.write(
        "GTSRB MOBILENETV3SMALL EVALUATION\n"
    )

    f.write(
        "====================================\n\n"
    )

    f.write(
        f"Test Images: {len(y_test)}\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        "Classification Report\n"
    )

    f.write(
        "------------------------------------\n"
    )

    f.write(report)


print(
    "\nClassification report saved to:"
)

print(report_path)


# ==========================================
# Confusion Matrix
# ==========================================

print("\n======================================")
print("GENERATING CONFUSION MATRIX")
print("======================================")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(NUM_CLASSES))
)

plt.figure(
    figsize=(14, 12)
)

plt.imshow(cm)

plt.title(
    "MobileNetV3Small - GTSRB Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.colorbar()

plt.xticks(
    range(NUM_CLASSES),
    class_names,
    rotation=90
)

plt.yticks(
    range(NUM_CLASSES),
    class_names
)

plt.tight_layout()

cm_path = (
    "results/mobilenetv3_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Confusion matrix saved to:"
)

print(cm_path)


# ==========================================
# Model Information
# ==========================================

print("\n======================================")
print("MODEL INFORMATION")
print("======================================")

total_params = model.count_params()

print(
    "Total Parameters:",
    f"{total_params:,}"
)

print(
    "Approximate Parameter Size:",
    f"{total_params * 4 / (1024 * 1024):.2f} MB"
)


# ==========================================
# Final Summary
# ==========================================

print("\n======================================")
print("MOBILENETV3 EVALUATION COMPLETED")
print("======================================")

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Images: {len(y_test)}"
)

print(
    f"Model Parameters: "
    f"{total_params:,}"
)

print("\nGenerated files:")

print(
    "- results/mobilenetv3_classification_report.txt"
)

print(
    "- results/mobilenetv3_confusion_matrix.png"
)