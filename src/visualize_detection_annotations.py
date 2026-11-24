import pandas as pd
import cv2
import os

CSV_PATH = "dataset/GTSRB/Train.csv"
BASE_DIR = "dataset/GTSRB"

OUTPUT_PATH = "results/detection_annotations.png"

df = pd.read_csv(CSV_PATH)

# Select a few images from different classes
selected = []

classes_seen = set()

for _, row in df.iterrows():

    class_id = int(row["ClassId"])

    if class_id not in classes_seen:
        selected.append(row)
        classes_seen.add(class_id)

    if len(selected) >= 12:
        break


images = []

for row in selected:

    image_path = os.path.join(
        BASE_DIR,
        row["Path"]
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    x1 = int(row["Roi.X1"])
    y1 = int(row["Roi.Y1"])
    x2 = int(row["Roi.X2"])
    y2 = int(row["Roi.Y2"])

    # Draw bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Add class label
    label = f"Class {int(row['ClassId'])}"

    cv2.putText(
        image,
        label,
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1
    )

    # Resize for visualization
    image = cv2.resize(
        image,
        (180, 180)
    )

    images.append(image)


# Create 3 x 4 grid
while len(images) < 12:
    images.append(
        255 * __import__("numpy").ones(
            (180, 180, 3),
            dtype="uint8"
        )
    )


rows = []

for i in range(0, 12, 4):
    row_images = images[i:i + 4]
    rows.append(cv2.hconcat(row_images))

grid = cv2.vconcat(rows)

cv2.imwrite(
    OUTPUT_PATH,
    grid
)

print("======================================")
print("DETECTION ANNOTATION VISUALIZATION")
print("======================================")

print("Saved to:")
print(OUTPUT_PATH)

print("======================================")