import cv2
import os
import random

IMAGE_DIR = "dataset/yolo/images/train"
LABEL_DIR = "dataset/yolo/labels/train"

OUTPUT_PATH = "results/yolo_label_check.png"

# Get available images
images = os.listdir(IMAGE_DIR)

# Select 12 random images
random.seed(42)
selected_images = random.sample(images, 12)

visualized = []

for image_name in selected_images:

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    height, width = image.shape[:2]

    # Label file
    label_name = os.path.splitext(image_name)[0] + ".txt"

    label_path = os.path.join(
        LABEL_DIR,
        label_name
    )

    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as file:
        lines = file.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # Convert normalized YOLO coordinates
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Draw bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Class label
        cv2.putText(
            image,
            f"Class {class_id}",
            (x1, max(y1 - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1
        )

    # Resize for visualization
    image = cv2.resize(
        image,
        (180, 180)
    )

    visualized.append(image)


# Create 3 x 4 grid
rows = []

for i in range(0, len(visualized), 4):

    row = visualized[i:i + 4]

    while len(row) < 4:

        blank = 255 * __import__("numpy").ones(
            (180, 180, 3),
            dtype="uint8"
        )

        row.append(blank)

    rows.append(
        cv2.hconcat(row)
    )

grid = cv2.vconcat(rows)

cv2.imwrite(
    OUTPUT_PATH,
    grid
)

print("======================================")
print("YOLO LABEL VERIFICATION")
print("======================================")

print("Images checked:", len(visualized))

print("\nSaved visualization to:")
print(OUTPUT_PATH)

print("======================================")