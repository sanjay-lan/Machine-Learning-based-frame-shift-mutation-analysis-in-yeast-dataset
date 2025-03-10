# Machine-Learning-based-frame-shift-mutation-analysis-in-yeast-dataset
## Overview

This project implements multiple machine learning classifiers to analyze gene expression data. It evaluates the performance of various models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

## Dependencies

  Ensure you have the following libraries installed before running the code:
  
  pandas, matplotlib, scikit-learn, numpy, imbalanced-learn, xgboost, openpyxl

## Dataset

The script reads a dataset from an Excel file stored in Google Drive. The dataset should contain a column named Gene_expression (target variable) and multiple features. The file_path variable should be updated to point to the correct dataset file.

## Machine Learning Models Used

  The script evaluates the following classifiers:
  
  Logistic Regression
  
  Naïve Bayes (GaussianNB)
  
  Support Vector Classifier (SVC)
  
  Random Forest Classifier
  
  AdaBoost Classifier
  
  Gradient Boosting Classifier
  
  XGBoost Classifier

## Workflow

  Load the dataset and remove missing values.
  
  Encode categorical variables using one-hot encoding.
  
  Define multiple classifiers.
  
  Split the dataset into training and testing sets (70% training, 30% testing).
  
  Train each classifier and evaluate performance over 10 iterations.
  
  Compute ROC curves and AUC scores for model comparison.
  
  Plot the averaged ROC curves for each classifier.

## Output

  Performance metrics:
  
  Accuracy
  
  Precision
  
  Recall
  
  F1-score
  
  A plot displaying the average ROC curves for each classifier.
