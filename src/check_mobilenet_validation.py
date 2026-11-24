import tensorflow as tf


# ==========================================
# Configuration
# ==========================================

DATASET_DIR = "dataset/GTSRB/Train"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
NUM_CLASSES = 43

SEED = 42
VALIDATION_SPLIT = 0.2

MODEL_PATH = "models/mobilenetv3_small.keras"

CLASS_NAMES = [
    str(i) for i in range(NUM_CLASSES)
]


# ==========================================
# Load Validation Dataset
# ==========================================

print("\n======================================")
print("LOADING VALIDATION DATA")
print("======================================")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    shuffle=False,
    class_names=CLASS_NAMES
)


# ==========================================
# IMPORTANT:
# MobileNetV3 expects 0-255 input
# ==========================================

def preprocess(images, labels):

    images = tf.cast(
        images,
        tf.float32
    )

    return images, labels


validation_dataset = validation_dataset.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ==========================================
# Load Saved Model
# ==========================================

print("\n======================================")
print("LOADING SAVED MODEL")
print("======================================")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ==========================================
# Evaluate Saved Model
# ==========================================

print("\n======================================")
print("EVALUATING SAVED MODEL")
print("======================================")

loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)

print("\n======================================")
print("SAVED MODEL VALIDATION RESULT")
print("======================================")

print(
    f"Validation Loss: {loss:.4f}"
)

print(
    f"Validation Accuracy: {accuracy * 100:.2f}%"
)