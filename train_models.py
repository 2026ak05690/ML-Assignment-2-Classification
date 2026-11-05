
import pandas as pd
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report
)

# ------------------------------------------------
# 1. Load training and testing datasets
# ------------------------------------------------

print("Loading training and testing data...")

train_data = pd.read_csv("data/train_data.csv")
test_data = pd.read_csv("data/test_data.csv")

X_train = train_data.drop(columns=["diagnosis"])
y_train = train_data["diagnosis"]

X_test = test_data.drop(columns=["diagnosis"])
y_test = test_data["diagnosis"]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ------------------------------------------------
# 2. Define the five classification models
# ------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Naive Bayes": GaussianNB(),

    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ])
}

# ------------------------------------------------
# 3. Create folder to save trained models
# ------------------------------------------------

Path("models").mkdir(exist_ok=True)

# ------------------------------------------------
# 4. Train and evaluate all five models
# ------------------------------------------------

results = []

for model_name, model in models.items():

    print("\n" + "=" * 55)
    print("Training:", model_name)
    print("=" * 55)

    # Train model using training data
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Get probability of malignant class
    malignant_index = list(model.classes_).index("malignant")
    y_prob = model.predict_proba(X_test)[:, malignant_index]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        (y_test == "malignant").astype(int),
        y_prob
    )

    precision = precision_score(
        y_test, y_pred,
        pos_label="malignant",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        pos_label="malignant",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        pos_label="malignant",
        zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    # Store results
    results.append({
        "ML Model Name": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

    # Display model results
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"MCC       : {mcc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save complete trained model
    model_filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".pkl"
    )

    joblib.dump(
        model,
        f"models/{model_filename}"
    )

    print("Model saved:", model_filename)

# ------------------------------------------------
# 5. Create model comparison table
# ------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\n" + "=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ------------------------------------------------
# 6. Identify the model with highest accuracy
# ------------------------------------------------

best_model = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\nBest model based on accuracy:")
print(best_model["ML Model Name"])

print("\nModel training and evaluation completed!")
