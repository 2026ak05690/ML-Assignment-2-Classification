# Machine Learning Classification – Assignment 2

## 1. Introduction

This project is part of Machine Learning Assignment 2 for the M.Tech AIML course.

The main objective is to implement and compare five different classification algorithms using the Breast Cancer Wisconsin (Diagnostic) dataset.

The models are trained and evaluated using Python and Scikit-learn. A Streamlit application is also developed to allow users to upload test data, select a trained model and view its performance.

**GitHub Repository:**  
https://github.com/2026ak05690/ML-Assignment-2-Classification

**Live Streamlit Application:**  
https://ml-assignment-2-classification-2026.streamlit.app/

## 2. Problem Statement

The objective of this assignment is to develop and evaluate five machine learning classification algorithms using a common dataset.

The models are compared using six evaluation metrics: Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

The aim is to identify the best-performing classification model and make the trained models available through a Streamlit application for evaluating test data.

## 3. Dataset Description

For this assignment, I have used the Breast Cancer Wisconsin (Diagnostic) dataset.

The dataset contains measurements of cell nuclei obtained from digitized images of breast mass samples. The objective is to classify the diagnosis as either malignant or benign.

### Dataset Details

| Description | Details |
|---|---|
| Dataset name | Breast Cancer Wisconsin (Diagnostic) |
| Total records | 569 |
| Input features | 30 |
| Target column | diagnosis |
| Classification type | Binary classification |
| Classes | Malignant and Benign |
| Missing values | 0 |

### Class Distribution

| Diagnosis | Number of records |
|---|---:|
| Benign | 357 |
| Malignant | 212 |
| **Total** | **569** |

**Dataset Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

The dataset is loaded using the Scikit-learn dataset utility.

## 4. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the dataset before training the models.

The following activities were performed:

- Checked the dataset shape and data types.
- Calculated descriptive statistics.
- Checked missing values and duplicate records.
- Analyzed the distribution of diagnosis classes.
- Examined numerical feature distributions.
- Created a correlation heatmap.
- Used boxplots to compare feature distributions across diagnosis classes.

The generated plots are available in the `plots` folder.

## 5. Data Preprocessing

The dataset was divided into training and testing sets using an 80:20 split.

| Dataset | Records |
|---|---:|
| Training data | 455 |
| Testing data | 114 |
| **Total** | **569** |

The following settings were used:

- Train-test split: 80:20
- Random state: 42
- Stratified sampling: Enabled

StandardScaler was used for Logistic Regression, KNN and SVM because these algorithms are affected by differences in feature scales.

Decision Tree and Gaussian Naive Bayes were used without feature scaling.

The preprocessing steps are included in the model pipelines where scaling is required.

## 6. Classification Algorithms

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

All five models were trained using the same training dataset and evaluated using the same testing dataset.

The trained models were saved using Joblib for later use in the Streamlit application.

## 7. Evaluation Metrics

The performance of each classifier was evaluated using the following six measures:

- **Accuracy:** Measures the overall proportion of correctly classified records.
- **AUC Score:** Measures the model's ability to distinguish between the two classes.
- **Precision:** Measures the proportion of predicted malignant cases that were actually malignant.
- **Recall:** Measures the proportion of actual malignant cases correctly identified.
- **F1 Score:** Represents the harmonic mean of precision and recall.
- **Matthews Correlation Coefficient (MCC):** Measures the overall quality of binary classification using all four confusion matrix outcomes.

Malignant was considered the positive class for the evaluation metrics.

## 8. Model Performance Comparison

The following table shows the actual results obtained after executing the training script on the test dataset.

| Classification Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| **Support Vector Machine** | **0.9737** | **0.9947** | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

## 9. Observations

Based on the evaluation results, the following observations were made:

- **Logistic Regression:** Achieved an accuracy of 96.49% and the highest AUC score of 0.9960.
- **Decision Tree:** Achieved an accuracy of 92.98%, which was the lowest among the five models in this experiment.
- **KNN:** Performed well with an accuracy of 95.61% and an AUC score of 0.9823.
- **Naive Bayes:** Achieved 100% precision but had a recall of 83.33%, meaning some malignant cases were not correctly identified.
- **Support Vector Machine:** Achieved the highest accuracy of 97.37%, F1 score of 0.9630 and MCC of 0.9442.

Overall, SVM showed the best performance based on accuracy, F1 score and MCC. However, Logistic Regression achieved a slightly higher AUC score.

## 10. Streamlit Application

A Streamlit application was developed using the five saved classification models.

The application provides the following features:

- Upload a test dataset in CSV format.
- Select any one of the five trained classification models.
- View all six evaluation metrics.
- Display the confusion matrix.
- View the classification report.
- Display actual and predicted diagnosis values.
- Download prediction results as a CSV file.

The application loads the saved models and does not retrain them when test data is uploaded.

**Live Application:**  
https://ml-assignment-2-classification-2026.streamlit.app/

## 11. Project Structure

```text
ML-Assignment-2-Classification/
│
├── data/
│   ├── breast_cancer.csv
│   ├── train_data.csv
│   └── test_data.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   └── support_vector_machine.pkl
│
├── plots/
│
├── app.py
├── prepare_dataset.py
├── eda.py
├── split_data.py
├── train_models.py
├── model_comparison.csv
├── requirements.txt
└── README.md
```

## 12. Installation and Execution

### Prerequisites

- Python 3.10 or a compatible version
- pip
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/2026ak05690/ML-Assignment-2-Classification.git

cd ML-Assignment-2-Classification
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
venv\Scripts\activate
```

On Linux:

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset

```bash
python prepare_dataset.py
```

### Step 5: Perform EDA

```bash
python eda.py
```

### Step 6: Split the Dataset

```bash
python split_data.py
```

### Step 7: Train and Evaluate the Models

```bash
python train_models.py
```

This script trains all five classifiers, calculates the evaluation metrics and saves the trained models in the `models` folder.

### Step 8: Run the Streamlit Application

```bash
streamlit run app.py
```

Open the local application in your browser:

```text
http://localhost:8501
```

Upload `data/test_data.csv` to evaluate the trained models.

## 13. Conclusion

Through this assignment, I implemented five classification algorithms and compared their performance using six evaluation metrics.

The results showed that different classifiers performed differently on the same dataset. SVM achieved the highest accuracy, while Logistic Regression achieved the highest AUC score.

This assignment helped me understand the implementation of classification algorithms, data preprocessing, model evaluation and the use of trained models in a Streamlit application.