import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score



st.title("Employee Attrition Analysis Dashboard")



df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")



st.subheader("Employee Dataset")
st.dataframe(df.head())



df_encoded = df.copy()

label_encoders = {}

for column in df_encoded.columns:
    if df_encoded[column].dtype == 'object':
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le


X = df_encoded.drop("Attrition", axis=1)
y = df_encoded["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_predictions)

st.subheader("Logistic Regression Accuracy")

st.success(f"Accuracy: {lr_accuracy:.2f}")



rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

st.subheader("Random Forest Accuracy")

st.success(f"Accuracy: {rf_accuracy:.2f}")



st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, rf_predictions)

fig, ax = plt.subplots(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig)



st.subheader("Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(12,8))

sns.heatmap(
    df.corr(),
    cmap='coolwarm',
    annot=False,
    ax=ax2
)

st.pyplot(fig2)



st.subheader("Attrition by Department")

dept_attrition = pd.crosstab(
    df['Department'],
    df['Attrition']
)

fig3, ax3 = plt.subplots()

dept_attrition.plot(kind='bar', ax=ax3)

plt.xticks(rotation=20)

st.pyplot(fig3)


st.subheader("Overtime vs Attrition")

overtime = pd.crosstab(
    df['OverTime'],
    df['Attrition']
)

fig4, ax4 = plt.subplots()

overtime.plot(kind='bar', ax=ax4)

st.pyplot(fig4)



st.subheader("High Risk Employees")

risk_probabilities = rf_model.predict_proba(X)[:,1]

df_encoded["Risk Score"] = risk_probabilities

high_risk = df_encoded[df_encoded["Risk Score"] > 0.7]

st.dataframe(
    high_risk[
        ["Age", "MonthlyIncome", "JobSatisfaction", "Risk Score"]
    ].head(10)
)


csv = df.to_csv(index=False)

st.download_button(
    label="Download Employee Dataset",
    data=csv,
    file_name="employee_attrition.csv",
    mime="text/csv"
)
