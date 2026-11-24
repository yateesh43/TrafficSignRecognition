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
# Training Preprocessing
# ==========================================

def preprocess_train(images, labels):

    # Apply augmentation
    images = data_augmentation(images, training=True)

    # Normalize
    images = tf.cast(images, tf.float32) / 255.0

    return images, labels


# ==========================================
# Validation Preprocessing
# ==========================================

def preprocess_validation(images, labels):

    # Only normalize
    images = tf.cast(images, tf.float32) / 255.0

    return images, labels


# ==========================================
# Apply preprocessing
# ==========================================

train_dataset = train_dataset.map(
    preprocess_train,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    preprocess_validation,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ==========================================
# Optimize pipeline
# ==========================================

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ==========================================
# Verify Pipeline
# ==========================================

print("\n======================================")
print("TRAINING PIPELINE")
print("======================================")

print("Number of classes:", len(class_names))
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)

for images, labels in train_dataset.take(1):

    print("Training batch shape:", images.shape)
    print("Training label shape:", labels.shape)

    print(
        "Training pixel range:",
        float(tf.reduce_min(images).numpy()),
        "to",
        float(tf.reduce_max(images).numpy())
    )

for images, labels in validation_dataset.take(1):

    print("Validation batch shape:", images.shape)
    print("Validation label shape:", labels.shape)

    print(
        "Validation pixel range:",
        float(tf.reduce_min(images).numpy()),
        "to",
        float(tf.reduce_max(images).numpy())
    )

print("\nTraining pipeline is ready!")