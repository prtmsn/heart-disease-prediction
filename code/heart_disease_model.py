# =========================
# Heart Disease Prediction
# =========================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv("heart_disease_uci.csv")

# -------------------------
# Data Cleaning
# -------------------------
print(df.columns)
# Drop unnecessary columns
df = df.drop(columns=['id', 'dataset'])

# Fill numeric columns with median
numeric_cols = ['trestbps', 'chol', 'thalch', 'oldpeak']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill boolean columns with mode
bool_cols = ['fbs', 'exang']
for col in bool_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
    df[col] = df[col].astype(int)

# Fill categorical columns with mode
cat_cols = ['restecg', 'slope', 'thal']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Drop column with excessive missing values
df = df.drop(columns=['ca'])

# Convert target to binary
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# -------------------------
# Encoding
# -------------------------
categorical_cols = ['sex', 'cp', 'restecg', 'slope', 'thal']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -------------------------
# Train-Test Split
# -------------------------
X = df.drop(columns=['num'])
y = df['num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Logistic Regression
# =========================

log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))
print("\nLogistic Regression Classification Report:\n")
print(classification_report(y_test, y_pred_log))

# =========================
# Decision Tree
# =========================

dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))
print("\nDecision Tree Classification Report:\n")
print(classification_report(y_test, y_pred_dt))

# =========================
# Visualizations
# =========================

# Target distribution
df['num'].value_counts().plot(kind='bar')
plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Age vs Heart Disease
df.boxplot(column='age', by='num')
plt.title("Age vs Heart Disease")
plt.suptitle("")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.show()

# Confusion Matrix (Logistic Regression)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_log)
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
