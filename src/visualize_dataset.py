import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# -----------------------------------
# 1. Load dataset
# -----------------------------------

csv_path = "dataset/GTSRB/Train.csv"

df = pd.read_csv(csv_path)

print("Dataset loaded successfully!")
print("Total images:", len(df))
print("Total classes:", df["ClassId"].nunique())


# -----------------------------------
# 2. Display sample traffic signs
# -----------------------------------

sample = df.sample(12, random_state=42)

plt.figure(figsize=(12, 8))

for i, (_, row) in enumerate(sample.iterrows()):

    # Construct image path
    image_path = os.path.join("dataset/GTSRB", row["Path"])

    # Read image
    image = cv2.imread(image_path)

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create subplot
    plt.subplot(3, 4, i + 1)

    plt.imshow(image)
    plt.title(f"Class: {row['ClassId']}")
    plt.axis("off")

plt.tight_layout()

# Save the figure
plt.savefig("results/sample_traffic_signs.png")

plt.show()


# -----------------------------------
# 3. Class distribution
# -----------------------------------

class_counts = df["ClassId"].value_counts().sort_index()

plt.figure(figsize=(14, 6))

plt.bar(class_counts.index, class_counts.values)

plt.xlabel("Traffic Sign Class")
plt.ylabel("Number of Images")
plt.title("GTSRB Training Dataset - Class Distribution")

plt.xticks(range(43))

plt.tight_layout()

# Save graph
plt.savefig("results/class_distribution.png")

plt.show()