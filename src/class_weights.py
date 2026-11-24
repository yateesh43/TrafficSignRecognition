import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Load training labels
df = pd.read_csv("dataset/GTSRB/Train.csv")

# Get class labels
classes = np.sort(df["ClassId"].unique())

# Calculate balanced class weights
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=df["ClassId"]
)

# Convert to dictionary
class_weights = dict(zip(classes, weights))

print("======================================")
print("CLASS WEIGHTS")
print("======================================")

for class_id, weight in class_weights.items():
    print(f"Class {class_id}: {weight:.4f}")

print("\nClass weights calculated successfully!")