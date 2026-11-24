import tensorflow as tf
import pandas as pd
import numpy as np
import os

from sklearn.utils.class_weight import compute_class_weight


# ==========================================
# Configuration
# ==========================================

DATASET_DIR = "dataset/GTSRB/Train"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
NUM_CLASSES = 43

SEED = 42
VALIDATION_SPLIT = 0.2

EPOCHS = 10


# ==========================================
# Define Correct Class Order
# ==========================================

# GTSRB classes must be in numeric order:
# 0, 1, 2, 3, ..., 42

CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]


# ==========================================
# Load Dataset
# ==========================================

print("\nLoading dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    shuffle=True,
    class_names=CLASS_NAMES
)

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

class_names = train_dataset.class_names

print("\nNumber of classes:", len(class_names))
print("Class order:", class_names)


# ==========================================
# Data Augmentation
# ==========================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08),

    tf.keras.layers.RandomTranslation(
        height_factor=0.1,
        width_factor=0.1
    ),

    tf.keras.layers.RandomZoom(0.1),

    tf.keras.layers.RandomContrast(0.1)
])


# ==========================================
# Preprocessing
# ==========================================

def preprocess_train(images, labels):

    images = tf.cast(images, tf.float32) / 255.0

    return images, labels


def preprocess_validation(images, labels):

    images = tf.cast(images, tf.float32) / 255.0

    return images, labels


train_dataset = train_dataset.map(
    preprocess_train,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    preprocess_validation,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ==========================================
# Dataset Performance
# ==========================================

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ==========================================
# Calculate Class Weights
# ==========================================

print("\nCalculating class weights...")

df = pd.read_csv("dataset/GTSRB/Train.csv")

classes = np.sort(df["ClassId"].unique())

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=df["ClassId"]
)

class_weights = dict(
    zip(classes, weights)
)

print("Class weights calculated.")

print("\nClass Weights:")

for class_id, weight in class_weights.items():
    print(f"Class {class_id}: {weight:.4f}")


# ==========================================
# Build Baseline CNN
# ==========================================

print("\nBuilding CNN model...")

model = tf.keras.Sequential([

    # Input
    tf.keras.layers.Input(
        shape=(64, 64, 3)
    ),

    # Data Augmentation
    data_augmentation,


    # ======================================
    # Convolution Block 1
    # ======================================

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # ======================================
    # Convolution Block 2
    # ======================================

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # ======================================
    # Convolution Block 3
    # ======================================

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),


    # ======================================
    # Classification Layers
    # ======================================

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.5),


    # ======================================
    # Output - 43 Classes
    # ======================================

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ==========================================
# Compile Model
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ==========================================
# Display Model
# ==========================================

model.summary()


# ==========================================
# Callbacks
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    "models/baseline_cnn.keras",

    monitor="val_accuracy",

    save_best_only=True
)


# ==========================================
# Train Model
# ==========================================

print("\n======================================")
print("STARTING CNN TRAINING")
print("======================================")

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ==========================================
# Final Results
# ==========================================

print("\n======================================")
print("CNN TRAINING COMPLETED")
print("======================================")

print(
    "Best Validation Accuracy:",
    max(history.history["val_accuracy"])
)

print("\nBest model saved to:")
print("models/baseline_cnn.keras")