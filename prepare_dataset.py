
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer

# Load the dataset
print("Loading dataset...")

cancer_data = load_breast_cancer(as_frame=True)

# Convert into a pandas DataFrame
df = cancer_data.frame.copy()

# Convert target values into readable class names
df["diagnosis"] = df["target"].map({
    0: "malignant",
    1: "benign"
})

# Remove the original numerical target
df = df.drop(columns=["target"])

# Create data folder if it does not exist
Path("data").mkdir(exist_ok=True)

# Save dataset as CSV
df.to_csv("data/breast_cancer.csv", index=False)

# Display basic information
print("\nDataset prepared successfully!")
print("Dataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nClass distribution:")
print(df["diagnosis"].value_counts())

print("\nMissing values:", df.isnull().sum().sum())
print("Duplicate records:", df.duplicated().sum())
