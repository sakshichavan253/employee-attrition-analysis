import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Title
st.title("Employee Attrition Analysis Dashboard")

# Load Dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Show Dataset
st.subheader("Employee Dataset")
st.dataframe(df.head())

# KPI Cards
total_employees = df.shape[0]
attrition_count = df[df['Attrition'] == 'Yes'].shape[0]
attrition_rate = (attrition_count / total_employees) * 100

col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", total_employees)
col2.metric("Employees Left", attrition_count)
col3.metric("Attrition Rate", f"{attrition_rate:.2f}%")

# Attrition by Department
st.subheader("Attrition by Department")

dept_attrition = df.groupby("Department")["Attrition"].value_counts().unstack()

fig, ax = plt.subplots()
dept_attrition.plot(kind='bar', ax=ax)

plt.xticks(rotation=20)
plt.ylabel("Count")

st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=['int64', 'float64'])

fig2, ax2 = plt.subplots(figsize=(10,6))

sns.heatmap(
    numeric_df.corr(),
    cmap='coolwarm',
    annot=False,
    ax=ax2
)

st.pyplot(fig2)

# Overtime Analysis
st.subheader("Overtime vs Attrition")

overtime = pd.crosstab(df['OverTime'], df['Attrition'])

fig3, ax3 = plt.subplots()

overtime.plot(kind='bar', ax=ax3)

st.pyplot(fig3)
