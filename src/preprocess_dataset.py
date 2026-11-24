import tensorflow as tf

# ==========================================
# Configuration
# ==========================================

DATASET_DIR = "dataset/GTSRB/Train"

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
SEED = 42
VALIDATION_SPLIT = 0.2


# ==========================================
# Load Training Dataset
# ==========================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    shuffle=True
)

class_names = train_dataset.class_names


# ==========================================
# Load Validation Dataset
# ==========================================

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    shuffle=False
)


# ==========================================
# Normalize Images
# ==========================================

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255
)

train_dataset = train_dataset.map(
    lambda images, labels:
    (normalization_layer(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    lambda images, labels:
    (normalization_layer(images), labels),
    num_parallel_calls=tf.data.AUTOTUNE
)


# ==========================================
# Optimize Dataset Pipeline
# ==========================================

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ==========================================
# Display Information
# ==========================================

print("\n======================================")
print("GTSRB DATASET PREPROCESSING")
print("======================================")

print("\nNumber of Classes:", len(class_names))
print("Classes:", class_names)

print("\nImage Size:", IMG_SIZE)
print("Batch Size:", BATCH_SIZE)
print("Validation Split:", VALIDATION_SPLIT)


# ==========================================
# Test One Batch
# ==========================================

for images, labels in train_dataset.take(1):

    print("\nBatch Image Shape:", images.shape)
    print("Batch Label Shape:", labels.shape)

    print(
        "Pixel Range:",
        float(tf.reduce_min(images).numpy()),
        "to",
        float(tf.reduce_max(images).numpy())
    )


print("\nTraining and validation datasets are ready!")