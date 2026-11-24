import tensorflow as tf
import matplotlib.pyplot as plt

# ==========================================
# Load one batch of images
# ==========================================

DATASET_DIR = "dataset/GTSRB/Train"

dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels="inferred",
    label_mode="int",
    image_size=(64, 64),
    batch_size=12,
    shuffle=True,
    seed=42
)

# Get one batch
images, labels = next(iter(dataset))

# ==========================================
# Define data augmentation
# ==========================================

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomTranslation(
        height_factor=0.1,
        width_factor=0.1
    ),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])


# ==========================================
# Apply augmentation
# ==========================================

augmented_images = augmentation(images)


# ==========================================
# Display original vs augmented
# ==========================================

plt.figure(figsize=(12, 8))

for i in range(6):

    # Original
    plt.subplot(3, 4, 2 * i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title("Original")
    plt.axis("off")

    # Augmented
    plt.subplot(3, 4, 2 * i + 2)
    plt.imshow(
        tf.clip_by_value(
            augmented_images[i],
            0,
            255
        ).numpy().astype("uint8")
    )
    plt.title("Augmented")
    plt.axis("off")

plt.tight_layout()

# Save result
plt.savefig("results/augmentation_test.png")

plt.show()

print("Data augmentation test completed successfully!")