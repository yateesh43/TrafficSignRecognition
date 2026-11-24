import pandas as pd
import cv2
import os
import numpy as np

CSV_PATH = "dataset/GTSRB/Test.csv"
BASE_DIR = "dataset/GTSRB"

print("\n======================================")
print("CHECKING TEST DATA")
print("======================================")

df = pd.read_csv(CSV_PATH)

print("\nTest CSV columns:")
print(df.columns.tolist())

print("\nTotal test images:", len(df))

print("\nFirst 5 rows:")
print(df.head())


print("\n======================================")
print("CHECKING IMAGE PATHS")
print("======================================")

for i in range(5):

    relative_path = df.iloc[i]["Path"]

    image_path = os.path.join(BASE_DIR, relative_path)

    print(f"\nImage {i + 1}")
    print("CSV Path :", relative_path)
    print("Full Path:", image_path)

    image = cv2.imread(image_path)

    if image is None:
        print("❌ Could not load image")
    else:
        print("✅ Image loaded")
        print("Original Shape:", image.shape)
        print("Pixel Min:", image.min())
        print("Pixel Max:", image.max())
        print("Pixel Mean:", image.mean())


print("\n======================================")
print("TEST DATA CHECK COMPLETE")
print("======================================")