import pandas as pd
import cv2
import os

CSV_PATH = "dataset/GTSRB/Train.csv"
BASE_DIR = "dataset/GTSRB"

print("\n======================================")
print("CHECKING DETECTION ANNOTATIONS")
print("======================================")

df = pd.read_csv(CSV_PATH)

print("Total annotations:", len(df))

print("\nCSV columns:")
print(df.columns.tolist())

print("\nFirst 10 annotations:\n")

for i in range(10):

    row = df.iloc[i]

    image_path = os.path.join(
        BASE_DIR,
        row["Path"]
    )

    image = cv2.imread(image_path)

    if image is None:
        print("Could not load:", image_path)
        continue

    height, width = image.shape[:2]

    x1 = int(row["Roi.X1"])
    y1 = int(row["Roi.Y1"])
    x2 = int(row["Roi.X2"])
    y2 = int(row["Roi.Y2"])

    print(f"\nImage {i + 1}")
    print("Path:", row["Path"])
    print("Class:", row["ClassId"])
    print("Image size:", width, "x", height)
    print("ROI:", x1, y1, x2, y2)

    # Validate coordinates
    valid = (
        0 <= x1 < x2 <= width
        and
        0 <= y1 < y2 <= height
    )

    if valid:
        print("ROI status: VALID")
    else:
        print("ROI status: INVALID")


print("\n======================================")
print("ANNOTATION CHECK COMPLETE")
print("======================================")