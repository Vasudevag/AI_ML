# Task 7: Support Vector Machines (SVM) – Breast Cancer Classification

# 📦 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 📥 Step 2: Load Dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Step 3: Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🔁 Step 4: Train SVM (Linear Kernel)
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)

y_pred_linear = svm_linear.predict(X_test)
print("Linear Kernel Accuracy:", svm_linear.score(X_test, y_test))
print(classification_report(y_test, y_pred_linear))

# 🔁 Step 5: Train SVM (RBF Kernel)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)

y_pred_rbf = svm_rbf.predict(X_test)
print("RBF Kernel Accuracy:", svm_rbf.score(X_test, y_test))
print(classification_report(y_test, y_pred_rbf))

# 🔍 Step 6: Hyperparameter Tuning (Grid Search)
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Grid Search Accuracy:", grid.best_estimator_.score(X_test, y_test))

# 🔁 Step 7: Cross-Validation Accuracy
cv_scores = cross_val_score(SVC(kernel='rbf', C=1, gamma=0.1), X_scaled, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))
