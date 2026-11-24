import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split


# ==============================
# PATHS
# ==============================

CSV_PATH = "dataset/GTSRB/Train.csv"
BASE_DIR = "dataset/GTSRB"

OUTPUT_DIR = "dataset/yolo"

TRAIN_IMG_DIR = os.path.join(OUTPUT_DIR, "images", "train")
VAL_IMG_DIR = os.path.join(OUTPUT_DIR, "images", "val")

TRAIN_LABEL_DIR = os.path.join(OUTPUT_DIR, "labels", "train")
VAL_LABEL_DIR = os.path.join(OUTPUT_DIR, "labels", "val")


# ==============================
# CREATE DIRECTORIES
# ==============================

for directory in [
    TRAIN_IMG_DIR,
    VAL_IMG_DIR,
    TRAIN_LABEL_DIR,
    VAL_LABEL_DIR
]:
    os.makedirs(directory, exist_ok=True)


# ==============================
# LOAD CSV
# ==============================

print("Loading annotations...")

df = pd.read_csv(CSV_PATH)

print("Total images:", len(df))


# ==============================
# TRAIN / VALIDATION SPLIT
# ==============================

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["ClassId"]
)

print("Training images:", len(train_df))
print("Validation images:", len(val_df))


# ==============================
# FUNCTION TO CONVERT DATA
# ==============================

def process_dataframe(dataframe, image_dir, label_dir):

    for index, row in dataframe.iterrows():

        source_image = os.path.join(
            BASE_DIR,
            row["Path"]
        )

        if not os.path.exists(source_image):
            print("Missing image:", source_image)
            continue

        # Create unique filename
        filename = os.path.basename(row["Path"])

        image_name = os.path.splitext(filename)[0]

        destination_image = os.path.join(
            image_dir,
            filename
        )

        label_file = os.path.join(
            label_dir,
            image_name + ".txt"
        )

        # Copy image
        shutil.copy2(
            source_image,
            destination_image
        )

        # Original image dimensions
        image_width = float(row["Width"])
        image_height = float(row["Height"])

        # Bounding box
        x1 = float(row["Roi.X1"])
        y1 = float(row["Roi.Y1"])
        x2 = float(row["Roi.X2"])
        y2 = float(row["Roi.Y2"])

        # Convert to YOLO format
        box_width = x2 - x1
        box_height = y2 - y1

        x_center = x1 + (box_width / 2)
        y_center = y1 + (box_height / 2)

        # Normalize
        x_center /= image_width
        y_center /= image_height
        box_width /= image_width
        box_height /= image_height

        class_id = int(row["ClassId"])

        # Write YOLO label
        with open(label_file, "w") as file:

            file.write(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}\n"
            )


# ==============================
# PROCESS DATA
# ==============================

print("\nCreating training dataset...")

process_dataframe(
    train_df,
    TRAIN_IMG_DIR,
    TRAIN_LABEL_DIR
)

print("Training dataset created.")


print("\nCreating validation dataset...")

process_dataframe(
    val_df,
    VAL_IMG_DIR,
    VAL_LABEL_DIR
)

print("Validation dataset created.")


# ==============================
# SUMMARY
# ==============================

print("\n======================================")
print("YOLO DATASET PREPARATION COMPLETE")
print("======================================")

print("Training images :", len(train_df))
print("Validation images:", len(val_df))

print("\nDataset location:")
print(OUTPUT_DIR)

print("\n======================================")