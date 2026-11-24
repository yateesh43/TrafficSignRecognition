import tensorflow as tf
import pandas as pd
import cv2
import os
import numpy as np

MODEL_PATH = "models/mobilenetv3_small.keras"
TRAIN_DIR = "dataset/GTSRB/Train"
TEST_CSV = "dataset/GTSRB/Test.csv"
BASE_DIR = "dataset/GTSRB"

IMG_SIZE = (64, 64)
NUM_CLASSES = 43
CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]

print("\n======================================")
print("LOADING MODEL")
print("======================================")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------
# VALIDATION DATA
# --------------------------------------

print("\n======================================")
print("VALIDATION SAMPLE")
print("======================================")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=10,
    validation_split=0.2,
    subset="validation",
    seed=42,
    shuffle=False,
    class_names=CLASS_NAMES
)

for images, labels in validation_dataset.take(1):

    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    for i in range(10):
        confidence = predictions[i][predicted_classes[i]] * 100

        print(
            f"Image {i+1}: "
            f"Actual={labels[i].numpy()} | "
            f"Predicted={predicted_classes[i]} | "
            f"Confidence={confidence:.2f}%"
        )


# --------------------------------------
# TEST DATA
# --------------------------------------

print("\n======================================")
print("TEST SAMPLE")
print("======================================")

df = pd.read_csv(TEST_CSV)

for i in range(10):

    relative_path = df.iloc[i]["Path"]
    actual_class = int(df.iloc[i]["ClassId"])

    image_path = os.path.join(BASE_DIR, relative_path)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)

    image = image.astype(np.float32)

    image_batch = np.expand_dims(image, axis=0)

    prediction = model.predict(image_batch, verbose=0)

    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class] * 100

    print(
        f"Image {i+1}: "
        f"Actual={actual_class} | "
        f"Predicted={predicted_class} | "
        f"Confidence={confidence:.2f}%"
    )


print("\n======================================")
print("CHECK COMPLETE")
print("======================================")