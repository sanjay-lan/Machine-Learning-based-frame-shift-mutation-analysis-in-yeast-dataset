# import the required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

# Define the path to the dataset (update as needed)

#file_path = "fs+1_final_dataset.xlsx"
#file_path = "fs+1_best.xlsx"
# file_path = "fs+1_worst1.xlsx"
# file_path = "fs-1_yeast_ratio_all_features.xlsx"
# file_path = "fs-1_yeasst_top20_features.xlsx"
file_path = "fs-1_yeast_worst20_features_.xlsx"

# Load the dataset into a pandas DataFrame
df=pd.read_excel(file_path ,index_col= 0)
df.dropna(inplace=True)  # Remove missing values

# excel_data = pd.ExcelFile(file_path)

# Define a list of classifiers to evaluate
classifiers = [
    LogisticRegression(random_state=1234, max_iter=1000),
    GaussianNB(),
    SVC(probability=True),
    RandomForestClassifier(random_state=1234),
    AdaBoostClassifier(random_state=1234),
    GradientBoostingClassifier(random_state=1234),
    XGBClassifier(random_state=1234)
]

# Initialize variables for ROC curve analysis
mean_fpr = np.linspace(0, 1, 100)
mean_tprs = {cls.__class__.__name__: [] for cls in classifiers}
metrics = {cls.__class__.__name__: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for cls in classifiers}

# Prepare feature matrix X and target variable y
X = pd.get_dummies(df.drop(columns=['Gene_expression']), drop_first=True)
y = df['Gene_expression']

# Define the number of iterations for model evaluation
num_iterations = 10

# Train and evaluate each classifier
for cls in classifiers:
   for _ in range(num_iterations):
            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1234)

            # Train the model
            model = cls.fit(X_train, y_train)

            # Predict probabilities and labels
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            # calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            mean_tprs[cls.__class__.__name__].append(interp_tpr)

           # calculate and store classification metrics
            metrics[cls.__class__.__name__]['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics[cls.__class__.__name__]['precision'].append(precision_score(y_test, y_pred))
            metrics[cls.__class__.__name__]['recall'].append(recall_score(y_test, y_pred))
            metrics[cls.__class__.__name__]['f1'].append(f1_score(y_test, y_pred))

# calculate mean true positive rates for ROC curves
mean_tpr_dict = {}
for cls_name, tpr_list in mean_tprs.items():
    mean_tpr = np.mean(tpr_list, axis=0)
    mean_tpr_dict[cls_name] = mean_tpr
    
# Plot the average ROC curves
plt.figure(figsize=(10, 6))

# calculate the mean of all the matrices
for cls_name, mean_tpr in mean_tpr_dict.items():
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, lw=2, alpha=0.8,
             label=f'{cls_name} (AUC = {mean_auc * 100:.2f}%)')
    
# Plot the diagonal line representing random guessing
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

# Add labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Average ROC curve across sets')
plt.legend(loc='lower right')

# Display the plot
plt.show()
