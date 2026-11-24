import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

# Load training CSV
df = pd.read_csv("dataset/GTSRB/Train.csv")

# Select one image
row = df.iloc[0]

# Get image path
image_path = os.path.join("dataset/GTSRB", row["Path"])

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Error: Image could not be loaded.")
    exit()

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get ROI coordinates
x1 = int(row["Roi.X1"])
y1 = int(row["Roi.Y1"])
x2 = int(row["Roi.X2"])
y2 = int(row["Roi.Y2"])

# Crop traffic sign
cropped = image_rgb[y1:y2, x1:x2]

# Resize to 64 × 64
resized = cv2.resize(cropped, (64, 64))

# Normalize pixel values
normalized = resized.astype("float32") / 255.0

# Display the images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(resized)
plt.title("Resized 64 × 64")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(normalized)
plt.title("Normalized")
plt.axis("off")

plt.tight_layout()
plt.show()

# Print information
print("===================================")
print("IMAGE PREPROCESSING TEST")
print("===================================")

print("Original shape :", image.shape)
print("Cropped shape  :", cropped.shape)
print("Resized shape  :", resized.shape)
print("Normalized shape:", normalized.shape)
print("Class ID       :", row["ClassId"])
print("Pixel range    :", normalized.min(), "to", normalized.max())