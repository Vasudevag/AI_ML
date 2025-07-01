# Task 5: Decision Trees and Random Forests – Heart Disease Prediction

# 📦 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📥 Step 2: Load Dataset
df = pd.read_csv("/Users/vasu/Documents/python/AI-ml/heart.csv")  # Rename if needed
df.head()

# 🔍 Step 3: Preprocessing
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌳 Step 4: Decision Tree Model
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

# Predictions and Evaluation
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# 🧾 Visualize the Tree
plt.figure(figsize=(15, 8))
plot_tree(dt, filled=True, feature_names=df.drop('target', axis=1).columns, class_names=['No Disease', 'Disease'])
plt.title("Decision Tree Visualization")
plt.show()

# 🌲 Step 5: Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# 🔍 Feature Importances
importances = pd.Series(rf.feature_importances_, index=df.drop('target', axis=1).columns)
importances.sort_values().plot(kind='barh', figsize=(10, 6), title="Feature Importances")
plt.show()

# 🔁 Step 6: Cross-validation
cv_scores = cross_val_score(rf, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Average CV accuracy:", np.mean(cv_scores))
