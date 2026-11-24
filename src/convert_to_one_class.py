import os

LABEL_ROOT = "dataset/yolo/labels"

folders = [
    os.path.join(LABEL_ROOT, "train"),
    os.path.join(LABEL_ROOT, "val")
]

print("Converting 43-class labels to 1-class detection labels...")

total_files = 0

for folder in folders:

    for filename in os.listdir(folder):

        if not filename.endswith(".txt"):
            continue

        path = os.path.join(folder, filename)

        with open(path, "r") as file:
            lines = file.readlines()

        new_lines = []

        for line in lines:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            # Keep bounding-box coordinates.
            # Change class ID to 0 = traffic sign.
            new_line = (
                f"0 "
                f"{parts[1]} "
                f"{parts[2]} "
                f"{parts[3]} "
                f"{parts[4]}\n"
            )

            new_lines.append(new_line)

        with open(path, "w") as file:
            file.writelines(new_lines)

        total_files += 1

print()
print("======================================")
print("CONVERSION COMPLETE")
print("======================================")
print("Label files converted:", total_files)
print("Class 0 = Traffic Sign")
print("======================================")