# Breast Cancer Wisconsin (Diagnostic) Dataset Models

This repository contains machine learning models for classifying breast cancer as malignant or benign using the Breast Cancer Wisconsin (Diagnostic) dataset.

## Dataset
The Breast Cancer Wisconsin (Diagnostic) dataset is obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)). The dataset contains features computed from digitized images of fine needle aspirates (FNAs) of breast masses, which describe characteristics of the cell nuclei present in the images.

## Project Structure
- `data/`: Contains the dataset file.
- `notebooks/`: Jupyter notebooks for data analysis and model training.
- `Raymond_model_v1.py`: Python script implementing the Random Forest Classifier.
- `Raymond_model_v2.py`: Python script implementing the Support Vector Machine.
- `README.md`: Project description and instructions.
- `requirements.txt`: List of Python packages required to run the scripts.

## Models
### Random Forest Classifier (`Raymond_model_v1.py`)
This model uses a Random Forest Classifier to predict whether the cancer is malignant or benign based on the provided features.

#### Key Steps:
1. Data Preprocessing: Scaling the features.
2. Model Training: Using the `RandomForestClassifier` from `scikit-learn`.
3. Evaluation: Confusion matrix, classification report, and accuracy score.

### Support Vector Machine (`Raymond_model_v2.py`)
This model uses a Support Vector Machine (SVM) with a linear kernel to predict the diagnosis.

#### Key Steps:
1. Data Preprocessing: Scaling the features.
2. Model Training: Using the `SVC` from `scikit-learn`.
3. Evaluation: Confusion matrix, classification report, and accuracy score.

