import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os

from sklearn.metrics import confusion_matrix


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "models/baseline_cnn.keras"
TEST_CSV = "dataset/GTSRB/Test.csv"
DATASET_DIR = "dataset/GTSRB"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
NUM_CLASSES = 43


# ==========================================
# Load Model
# ==========================================

print("\nLoading CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded.")


# ==========================================
# Load Test Data
# ==========================================

df = pd.read_csv(TEST_CSV)

images = []
labels = []


print("\nLoading test images...")

for index, row in df.iterrows():

    image_path = os.path.join(
        DATASET_DIR,
        row["Path"]
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        IMG_SIZE
    )

    image = image.astype("float32") / 255.0

    images.append(image)
    labels.append(int(row["ClassId"]))


X_test = np.array(images)
y_test = np.array(labels)


# ==========================================
# Predictions
# ==========================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(NUM_CLASSES))
)


# ==========================================
# Find Misclassification Pairs
# ==========================================

errors = []

for true_class in range(NUM_CLASSES):

    for predicted_class in range(NUM_CLASSES):

        if true_class != predicted_class:

            count = cm[true_class][predicted_class]

            if count > 0:

                errors.append(
                    (
                        count,
                        true_class,
                        predicted_class
                    )
                )


errors.sort(
    reverse=True
)


# ==========================================
# Display Top Errors
# ==========================================

print("\n======================================")
print("TOP MISCLASSIFICATION PAIRS")
print("======================================")

print(
    "\nActual Class → Predicted Class → Errors\n"
)

for count, true_class, predicted_class in errors[:20]:

    print(
        f"{true_class:2d} → {predicted_class:2d}"
        f"    {count} errors"
    )


# ==========================================
# Per-Class Accuracy
# ==========================================

print("\n======================================")
print("LOWEST CLASS ACCURACY")
print("======================================")

class_accuracy = []

for class_id in range(NUM_CLASSES):

    total = np.sum(
        y_test == class_id
    )

    correct = cm[class_id][class_id]

    accuracy = correct / total

    class_accuracy.append(
        (
            accuracy,
            class_id,
            total
        )
    )


class_accuracy.sort()


print(
    "\nClass → Accuracy → Test Samples\n"
)

for accuracy, class_id, total in class_accuracy[:10]:

    print(
        f"{class_id:2d} → "
        f"{accuracy * 100:6.2f}% "
        f"→ {total} samples"
    )


# ==========================================
# Save Analysis
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)

output_file = (
    "results/confusion_analysis.txt"
)

with open(output_file, "w") as f:

    f.write(
        "BASELINE CNN CONFUSION MATRIX ANALYSIS\n"
    )

    f.write(
        "======================================\n\n"
    )

    f.write(
        "Top Misclassification Pairs\n"
    )

    f.write(
        "--------------------------------------\n"
    )

    for count, true_class, predicted_class in errors[:20]:

        f.write(
            f"{true_class} -> {predicted_class}: "
            f"{count} errors\n"
        )

    f.write(
        "\nLowest Class Accuracy\n"
    )

    f.write(
        "--------------------------------------\n"
    )

    for accuracy, class_id, total in class_accuracy[:10]:

        f.write(
            f"Class {class_id}: "
            f"{accuracy * 100:.2f}% "
            f"({total} samples)\n"
        )


print(
    f"\nAnalysis saved to: {output_file}"
)

print("\nAnalysis completed successfully.")