
# Diabetes Prediction Model

## Objective
This project aims to develop a model for predicting diabetes using medical and demographic data.

## About the Dataset
The dataset comprises patient data, including medical history and demographic information, alongside their diabetes status. Key features are age, gender, BMI, hypertension, heart disease, smoking history, HbA1c level, and blood glucose level. This data is instrumental for healthcare professionals and researchers in understanding diabetes risk factors.

Dataset link: [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data)

## Steps
1. **Import and Install Dependencies:** 
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import OneHotEncoder
   ```

2. **Read the Dataset:**
   ```python
   dataset = pd.read_csv('diabetes_prediction_dataset.csv')
   ```

3. **Data Processing and Quick EDA:**
   - Check for duplicates and nulls.
   - Remove duplicates.
   - Data encoding for categorical variables.
   - Sample data for Exploratory Data Analysis (EDA).

4. **Model Building using Artificial Neural Network (ANN):**
   - Define and compile the TensorFlow model.
   - Train the model with early stopping.

5. **Model Evaluation:**
   - Visualize training and validation accuracy and loss.
   - Evaluate the model on test data.
   - Generate a classification report and confusion matrix.

## Results

The dataset has about 100,000 rows. After removing 3,854 duplicates, 96,146 rows were left and split 60% train, 20% validation, 20% test (19,230 test rows).

| Metric (test set) | Value |
|:--|:--|
| Accuracy | 92.98% |
| Diabetes class precision | 1.00 |
| Diabetes class recall | 0.19 |
| Diabetes class F1 | 0.33 |

The accuracy looks good, but it is misleading. Only about 8.7% of the test rows (1,677 of 19,230) are diabetic, so a model that always predicts "no diabetes" would score about 91.3%. This model catches only 19% of the diabetic cases, so it is not useful for screening yet.

## Limitations and Next Steps

- Handle the class imbalance (class weights or resampling) and tune the decision threshold to raise recall.
- Report recall, AUROC and PR-AUC, not just accuracy.
- Scale the numeric features and one-hot encode `smoking_history` (it is currently label encoded, which implies an order that does not exist).
- This is a practice project on a public Kaggle dataset, not a medical tool.

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- TensorFlow

## Dataset Reading
Read the dataset using pandas:
```python
dataset = pd.read_csv('diabetes_prediction_dataset.csv')
```

## Data Processing
Includes checking and removing duplicates, handling null values, and data encoding.

## Model Building
Utilize TensorFlow to build an ANN model. Include layers, activation functions, dropout for overfitting prevention, and compile the model.

## Model Evaluation
Evaluate the model's performance on test data, visualize accuracy and loss, and generate predictions.

## Conclusion
This model serves as a tool for predicting diabetes using patient data, aiding in early detection and healthcare planning. 

---

*Note: Ensure that you have installed all the necessary libraries and dependencies to run this project successfully.*
