import tensorflow as tf
import pandas as pd
import cv2
import os
import numpy as np

MODEL_PATH = "models/mobilenetv3_small.keras"
TRAIN_DIR = "dataset/GTSRB/Train"

IMG_SIZE = (64, 64)
NUM_CLASSES = 43
SEED = 42

print("\n======================================")
print("LOADING MODEL")
print("======================================")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ======================================
# GET VALIDATION FILES
# ======================================

print("\n======================================")
print("PREPARING VALIDATION FILES")
print("======================================")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=None,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    shuffle=False,
    class_names=[str(i) for i in range(NUM_CLASSES)]
)

validation_paths = []

# Get file paths from the dataset directory
class_to_index = {str(i): i for i in range(NUM_CLASSES)}

for class_name in sorted(os.listdir(TRAIN_DIR), key=lambda x: int(x)):
    class_dir = os.path.join(TRAIN_DIR, class_name)

    if not os.path.isdir(class_dir):
        continue

    files = sorted(os.listdir(class_dir))

    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            validation_paths.append(
                (os.path.join(class_dir, file), int(class_name))
            )


# Reproduce TensorFlow's validation split
total_images = len(validation_paths)
split_index = int(total_images * (1 - 0.2))

validation_paths = validation_paths[split_index:]

print("Total images:", total_images)
print("Validation images:", len(validation_paths))


# ======================================
# LOAD USING OPENCV
# ======================================

print("\n======================================")
print("EVALUATING WITH OPENCV")
print("======================================")

correct = 0
total = len(validation_paths)

for index, (image_path, actual_class) in enumerate(validation_paths):

    image = cv2.imread(image_path)

    if image is None:
        print("Could not load:", image_path)
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        IMG_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    image = image.astype(np.float32)

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)

    predicted_class = np.argmax(prediction[0])

    if predicted_class == actual_class:
        correct += 1

    if (index + 1) % 500 == 0:
        print(
            f"Processed {index + 1}/{total} images"
        )


accuracy = correct / total

print("\n======================================")
print("OPENCV VALIDATION RESULT")
print("======================================")

print(f"Correct: {correct}")
print(f"Incorrect: {total - correct}")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

print("\n======================================")
print("CHECK COMPLETE")
print("======================================")