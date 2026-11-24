import pandas as pd
import os

# Path to our training CSV
csv_path = "dataset/GTSRB/Train.csv"

# Load the dataset
df = pd.read_csv(csv_path)

print("=" * 50)
print("GTSRB DATASET EXPLORATION")
print("=" * 50)

# Number of images
print("\nTotal Images:", len(df))

# Number of classes
print("Number of Classes:", df["ClassId"].nunique())

# List of classes
print("Class IDs:", sorted(df["ClassId"].unique()))

# Images per class
print("\nImages per Class:")
print(df["ClassId"].value_counts().sort_index())

# Display column names
print("\nCSV Columns:")
print(df.columns.tolist())

# Display first 5 records
print("\nFirst 5 Records:")
print(df.head())