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

EPOCHS = 15

MODEL_PATH = "models/mobilenetv3_small.keras"


# ==========================================
# Correct Class Order
# ==========================================

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

print("\nNumber of classes:", len(train_dataset.class_names))
print("Class order:", train_dataset.class_names)


# ==========================================
# Preprocessing
# ==========================================

def preprocess(images, labels):

    images = tf.cast(images, tf.float32)

    return images, labels


train_dataset = train_dataset.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    preprocess,
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
# Calculate Class Weights
# ==========================================

print("\nCalculating class weights...")

df = pd.read_csv(
    "dataset/GTSRB/Train.csv"
)

classes = np.sort(
    df["ClassId"].unique()
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=df["ClassId"]
)

class_weights = dict(
    zip(classes, weights)
)

print("Class weights calculated.")


# ==========================================
# Build MobileNetV3Small
# ==========================================

print("\nBuilding MobileNetV3Small...")

base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(64, 64, 3),
    include_top=False,
    weights="imagenet",
    include_preprocessing=True
)


base_model.trainable = True

# Freeze the early layers
# and fine-tune the deeper layers

for layer in base_model.layers[:-30]:
    layer.trainable = False


# ==========================================
# Build Final Model
# ==========================================

inputs = tf.keras.Input(
    shape=(64, 64, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs,
    outputs
)


# ==========================================
# Compile
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
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
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True
)


# ==========================================
# Train
# ==========================================

print("\n======================================")
print("STARTING MOBILENETV3SMALL TRAINING")
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
print("MOBILENETV3SMALL TRAINING COMPLETED")
print("======================================")

print(
    "Best Validation Accuracy:",
    max(history.history["val_accuracy"])
)

print("\nBest model saved to:")
print(MODEL_PATH)

print("\nTotal Parameters:")
print(f"{model.count_params():,}")