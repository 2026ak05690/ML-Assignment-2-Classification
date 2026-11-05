import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------
# 1. Page configuration
# ------------------------------------------------

st.set_page_config(
    page_title="ML Classifier Evaluation",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------
# 2. Application heading
# ------------------------------------------------

st.title("Machine Learning Classifier Evaluation")

st.write(
    "This application evaluates five classification "
    "models using an uploaded test dataset."
)

st.divider()

# ------------------------------------------------
# 3. Model file configuration
# ------------------------------------------------

MODEL_PATH = Path("models")

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}


# ------------------------------------------------
# 4. Load saved trained models
# ------------------------------------------------

@st.cache_resource
def load_models():

    loaded_models = {}

    for model_name, filename in model_files.items():

        file_path = MODEL_PATH / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {file_path}"
            )

        loaded_models[model_name] = joblib.load(file_path)

    return loaded_models


try:
    models = load_models()

except Exception as error:

    st.error(
        f"Unable to load trained models: {error}"
    )

    st.stop()


# ------------------------------------------------
# 5. Upload test dataset
# ------------------------------------------------

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        # Read uploaded CSV
        test_data = pd.read_csv(uploaded_file)

        st.success(
            "Test dataset uploaded successfully!"
        )

        st.write("Dataset Preview")

        st.dataframe(
            test_data.head(),
            use_container_width=True
        )

        # ----------------------------------------
        # 6. Validate uploaded dataset
        # ----------------------------------------

        if "diagnosis" not in test_data.columns:

            st.error(
                "The uploaded CSV must contain a diagnosis column."
            )

            st.stop()

        # Separate features and target
        X_test = test_data.drop(
            columns=["diagnosis"]
        )

        y_test = test_data["diagnosis"]

        # Get expected feature names
        expected_features = list(
            models["Logistic Regression"].feature_names_in_
        )

        # Check missing features
        missing_features = [
            feature for feature in expected_features
            if feature not in X_test.columns
        ]

        # Check unexpected features
        extra_features = [
            feature for feature in X_test.columns
            if feature not in expected_features
        ]

        if missing_features or extra_features:

            if missing_features:

                st.error(
                    f"Missing features: {missing_features}"
                )

            if extra_features:

                st.error(
                    f"Unexpected features: {extra_features}"
                )

            st.stop()

        # Arrange features in training order
        X_test = X_test[expected_features]

        # Convert features into numeric values
        X_test = X_test.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Check missing values
        if X_test.isnull().any().any():

            st.error(
                "The test dataset contains missing or non-numeric values."
            )

            st.stop()

        # Check infinite values
        if not np.isfinite(
            X_test.to_numpy(dtype=float)
        ).all():

            st.error(
                "The dataset contains infinite values."
            )

            st.stop()

        # Check target values
        if y_test.isnull().any():

            st.error(
                "The diagnosis column contains missing values."
            )

            st.stop()

        if not set(y_test.unique()).issubset(
            {"benign", "malignant"}
        ):

            st.error(
                "Diagnosis must contain only benign and malignant values."
            )

            st.stop()

        # Display dataset information
        st.write(
            f"Number of test records: **{len(X_test)}**"
        )

        st.divider()

        # ----------------------------------------
        # 7. Select classification model
        # ----------------------------------------

        st.subheader(
            "Select Classification Model"
        )

        selected_model_name = st.selectbox(
            "Choose a model",
            list(models.keys())
        )

        selected_model = models[
            selected_model_name
        ]

        # ----------------------------------------
        # 8. Generate predictions
        # ----------------------------------------

        y_pred = selected_model.predict(
            X_test
        )

        # Get malignant class probability
        malignant_index = list(
            selected_model.classes_
        ).index("malignant")

        y_prob = selected_model.predict_proba(
            X_test
        )[:, malignant_index]

        # ----------------------------------------
        # 9. Calculate evaluation metrics
        # ----------------------------------------

        y_true_binary = (
            y_test == "malignant"
        ).astype(int)

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        auc = roc_auc_score(
            y_true_binary,
            y_prob
        )

        precision = precision_score(
            y_test,
            y_pred,
            pos_label="malignant",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            pos_label="malignant",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            pos_label="malignant",
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test,
            y_pred
        )

        # ----------------------------------------
        # 10. Display evaluation metrics
        # ----------------------------------------

        st.divider()

        st.subheader(
            "Model Evaluation Results"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        col2.metric(
            "AUC Score",
            f"{auc:.4f}"
        )

        col3.metric(
            "Precision",
            f"{precision:.4f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{recall:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        col6.metric(
            "MCC Score",
            f"{mcc:.4f}"
        )

        # ----------------------------------------
        # 11. Display confusion matrix
        # ----------------------------------------

        st.divider()

        st.subheader(
            "Confusion Matrix"
        )

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=["benign", "malignant"]
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual Benign",
                "Actual Malignant"
            ],
            columns=[
                "Predicted Benign",
                "Predicted Malignant"
            ]
        )

        st.dataframe(
            cm_df,
            use_container_width=True
        )

        # ----------------------------------------
        # 12. Display classification report
        # ----------------------------------------

        st.subheader(
            "Classification Report"
        )

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df.style.format("{:.4f}"),
            use_container_width=True
        )

        # ----------------------------------------
        # 13. Display prediction results
        # ----------------------------------------

        st.divider()

        st.subheader(
            "Prediction Results"
        )

        prediction_results = X_test.copy()

        prediction_results[
            "Actual Diagnosis"
        ] = y_test.values

        prediction_results[
            "Predicted Diagnosis"
        ] = y_pred

        st.dataframe(
            prediction_results,
            use_container_width=True
        )

        # ----------------------------------------
        # 14. Download prediction results
        # ----------------------------------------

        csv_data = prediction_results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Prediction Results (CSV)",
            data=csv_data,
            file_name="prediction_results.csv",
            mime="text/csv"
        )

        # ----------------------------------------
        # 15. Completion message
        # ----------------------------------------

        st.success(
            f"Evaluation completed using {selected_model_name}."
        )

    except Exception as error:

        st.error(
            f"An error occurred: {error}"
        )

else:

    st.info(
        "Please upload a test CSV file containing "
        "all 30 input features and the diagnosis column."
    )


# ------------------------------------------------
# 16. Application footer
# ------------------------------------------------

st.divider()

st.caption(
    "Machine Learning Assignment 2 | M.Tech AIML"
)
