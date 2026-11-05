
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load dataset
df = pd.read_csv("data/breast_cancer.csv")

# Create folder for saving graphs
Path("plots").mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# Display dataset information
print("Dataset shape:", df.shape)

print("\nStatistical summary:")
print(df.describe().T)

print("\nMissing values:")
print(df.isnull().sum().sum())

# 1. Class distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="diagnosis",
    hue="diagnosis",
    legend=False
)

plt.title("Distribution of Diagnosis Classes")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig("plots/class_distribution.png", dpi=300)
plt.show()

# 2. Distribution of selected features
selected_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness"
]

df[selected_features].hist(figsize=(12, 8), bins=20)

plt.suptitle("Distribution of Selected Features")
plt.tight_layout()
plt.savefig("plots/feature_distribution.png", dpi=300)
plt.show()

# 3. Correlation heatmap
plt.figure(figsize=(15, 11))

correlation = df.drop(columns=["diagnosis"]).corr()

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=300)
plt.show()

# 4. Compare mean radius by diagnosis
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="diagnosis",
    y="mean radius",
    hue="diagnosis",
    legend=False
)

plt.title("Mean Radius by Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Mean Radius")
plt.tight_layout()
plt.savefig("plots/mean_radius_boxplot.png", dpi=300)
plt.show()

print("\nEDA completed successfully!")
