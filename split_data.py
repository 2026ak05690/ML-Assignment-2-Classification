
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# -----------------------------------------------
# 1. Load the prepared dataset
# -----------------------------------------------

df = pd.read_csv("data/breast_cancer.csv")

# Separate input features and target variable
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# -----------------------------------------------
# 2. Split the dataset into training and testing
# -----------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------------------------
# 3. Combine features and target for saving
# -----------------------------------------------

train_data = X_train.copy()
train_data["diagnosis"] = y_train

test_data = X_test.copy()
test_data["diagnosis"] = y_test

# -----------------------------------------------
# 4. Save the split datasets
# -----------------------------------------------

Path("data").mkdir(exist_ok=True)

train_data.to_csv("data/train_data.csv", index=False)
test_data.to_csv("data/test_data.csv", index=False)

# -----------------------------------------------
# 5. Display split information
# -----------------------------------------------

print("Dataset split completed successfully!")

print("\nOriginal dataset:", df.shape)
print("Training dataset:", train_data.shape)
print("Testing dataset:", test_data.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nTraining percentage:",
      round(len(X_train) / len(df) * 100, 2), "%")

print("Testing percentage:",
      round(len(X_test) / len(df) * 100, 2), "%")
