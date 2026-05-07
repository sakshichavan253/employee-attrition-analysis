import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- PAGE TITLE ----------------

st.set_page_config(page_title="Employee Attrition Analysis", layout="wide")

st.title("Employee Attrition Analysis Dashboard")

# ---------------- LOAD DATA ----------------

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

st.subheader("Employee Dataset")
st.dataframe(df.head())

# ---------------- SELECT IMPORTANT COLUMNS ----------------

df = df[['Age', 'Department', 'DistanceFromHome', 'EducationField',
         'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
         'YearsAtCompany', 'OverTime', 'BusinessTravel', 'Attrition']]

# ---------------- TARGET ENCODING ----------------

df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# ---------------- ONE HOT ENCODING ----------------

df = pd.get_dummies(
    df,
    columns=['Department', 'EducationField', 'OverTime', 'BusinessTravel'],
    drop_first=True
)

# ---------------- SCALING ----------------

scaler = StandardScaler()

num_cols = [
    'Age',
    'DistanceFromHome',
    'MonthlyIncome',
    'NumCompaniesWorked',
    'YearsAtCompany'
]

df[num_cols] = scaler.fit_transform(df[num_cols])

# ---------------- METRICS ----------------

total_emp = len(df)
employees_left = df['Attrition'].sum()
attrition_rate = (employees_left / total_emp) * 100

col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", total_emp)
col2.metric("Employees Left", employees_left)
col3.metric("Attrition Rate", f"{attrition_rate:.2f}%")

# ---------------- CHART 1 ----------------

st.subheader("Attrition Distribution")

fig1, ax1 = plt.subplots()

sns.countplot(x='Attrition', data=df, ax=ax1)

st.pyplot(fig1)

# ---------------- CHART 2 ----------------

st.subheader("Attrition vs Monthly Income")

fig2, ax2 = plt.subplots()

sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, ax=ax2)

st.pyplot(fig2)

# ---------------- CHART 3 ----------------

st.subheader("Attrition vs Age")

fig3, ax3 = plt.subplots()

sns.boxplot(x='Attrition', y='Age', data=df, ax=ax3)

st.pyplot(fig3)

# ---------------- CHART 4 ----------------

st.subheader("Attrition vs Overtime")

fig4, ax4 = plt.subplots()

sns.countplot(x='OverTime_Yes', hue='Attrition', data=df, ax=ax4)

st.pyplot(fig4)

# ---------------- HEATMAP ----------------

st.subheader("Correlation Heatmap")

fig5, ax5 = plt.subplots(figsize=(12,8))

sns.heatmap(
    df.select_dtypes(include=['number']).corr(),
    cmap='coolwarm',
    ax=ax5
)

st.pyplot(fig5)

# ---------------- FEATURES & TARGET ----------------

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# ---------------- TRAIN TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- RANDOM FOREST ----------------

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)

# ---------------- LOGISTIC REGRESSION ----------------

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, y_pred_lr)

# ---------------- MODEL ACCURACY ----------------

st.subheader("Model Accuracy")

col4, col5 = st.columns(2)

col4.metric("Random Forest Accuracy", f"{rf_accuracy:.2f}")

col5.metric("Logistic Regression Accuracy", f"{lr_accuracy:.2f}")

# ---------------- CONFUSION MATRIX ----------------

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred_rf)

fig6, ax6 = plt.subplots()

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6)

ax6.set_xlabel("Predicted")
ax6.set_ylabel("Actual")

st.pyplot(fig6)

# ---------------- CLASSIFICATION REPORT ----------------

st.subheader("Classification Report")

report = classification_report(y_test, y_pred_rf)

st.text(report)

# ---------------- FEATURE IMPORTANCE ----------------

st.subheader("Top 10 Important Features")

importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
)

top_features = importance.sort_values(ascending=False).head(10)

fig7, ax7 = plt.subplots(figsize=(10,5))

top_features.plot(kind='bar', ax=ax7)

st.pyplot(fig7)

# ---------------- HIGH RISK EMPLOYEES ----------------

df['RiskScore'] = rf_model.predict_proba(X)[:, 1]

at_risk = df[df['RiskScore'] > 0.7]

st.subheader("High Risk Employees")

st.dataframe(at_risk.head())

# ---------------- DOWNLOAD CSV ----------------

at_risk.to_csv("at_risk_employees.csv", index=False)

st.success("Project Completed Successfully")
