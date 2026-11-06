
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer

# Load the dataset
print("Loading Breast Cancer Wisconsin dataset...")

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

# -------------------------------
# Dataset Validation
# -------------------------------

print("\nValidating dataset...")

# Check dataset dimensions
if df.shape != (569, 31):
    raise ValueError(
        f"Unexpected dataset shape: {df.shape}. "
        "Expected 569 rows and 31 columns."
    )

# Check for missing values
missing_values = df.isnull().sum().sum()
if missing_values > 0:
    raise ValueError(
        f"Dataset contains {missing_values} missing values."
    )

# Check for duplicate records
duplicate_records = df.duplicated().sum()
if duplicate_records > 0:
    raise ValueError(
        f"Dataset contains {duplicate_records} duplicate records."
    )

# Check target classes
expected_classes = {"malignant", "benign"}
actual_classes = set(df["diagnosis"].unique())

if actual_classes != expected_classes:
    raise ValueError(
        f"Unexpected diagnosis classes: {actual_classes}"
    )

# Check feature count
feature_count = df.drop(columns=["diagnosis"]).shape[1]
if feature_count != 30:
    raise ValueError(
        f"Expected 30 features, but found {feature_count}."
    )

print("All dataset validation checks passed!")

# -------------------------------
# Save Dataset
# -------------------------------

Path("data").mkdir(exist_ok=True)

df.to_csv("data/breast_cancer.csv", index=False)

# Display dataset information
print("\nDataset prepared successfully!")
print("Dataset shape:", df.shape)
print("Number of features:", feature_count)

print("\nFirst five records:")
print(df.head())

print("\nClass distribution:")
print(df["diagnosis"].value_counts())

print("\nMissing values:", missing_values)
print("Duplicate records:", duplicate_records)
