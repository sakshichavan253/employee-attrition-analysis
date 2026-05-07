


import os
print("CURRENT PATH:", os.getcwd())


# In[49]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# In[85]:


import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(df.head())
print("Data loaded successfully")


# In[51]:


import os
print(os.getcwd())


# In[52]:


df = df[['Age', 'Department', 'DistanceFromHome', 'EducationField',
         'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
         'YearsAtCompany', 'OverTime', 'BusinessTravel', 'Attrition']]


# In[53]:


df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})


# In[54]:


df = pd.get_dummies(df,
                   columns=['Department', 'EducationField', 'OverTime', 'BusinessTravel'],
                   drop_first=True)
print("Data preprocessing completed")


# In[55]:


print(df.columns)


# In[56]:


print(df.head())


# In[57]:


print(df.dtypes)


# In[58]:


print('Attrition' in df.columns)


# In[59]:


print("Final columns:", df.columns)
print("Attrition exists after encoding:", 'Attrition' in df.columns)


# In[60]:


scaler = StandardScaler()

num_cols = ['Age', 'DistanceFromHome', 'MonthlyIncome',
            'NumCompaniesWorked', 'YearsAtCompany']

df[num_cols] = scaler.fit_transform(df[num_cols])


# In[61]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[62]:


sns.countplot(x='Attrition', data=df)
plt.title("Attrition Distribution")
plt.show()


# In[63]:


sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
plt.title("Attrition vs Monthly Income")
plt.show()


# In[64]:


sns.boxplot(x='Attrition', y='Age', data=df)
plt.title("Attrition vs Age")
plt.show()


# In[65]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='OverTime_Yes', hue='Attrition', data=df)
plt.title("Attrition vs Overtime")
plt.show()


# In[66]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,8))

sns.heatmap(df.select_dtypes(include=['number']).corr(), cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()


# In[67]:


X = df.drop('Attrition', axis=1)
y = df['Attrition']


# In[68]:


from sklearn.model_selection import train_test_split

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Model training completed")


# In[69]:


print(y_train.shape)
print(type(y_train))


# In[70]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# In[71]:


y_pred_rf = model.predict(X_test)   
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


# In[72]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)


# In[74]:

y_pred = model.predict(X_test)
print("Prediction completed")

# In[75]:


from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))


# In[76]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[77]:


y_pred_rf = model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))# In[78]:


from sklearn.metrics import accuracy_score, classification_report

y_pred = lr.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[79]:


importance = pd.Series(model.feature_importances_, index=X.columns)

print("\nTop 10 Important Features:")
print(importance.sort_values(ascending=False).head(10))
# In[80]:


importance.sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top 10 Important Features")
plt.show()

# In[81]:


df['RiskScore'] = model.predict_proba(X)[:, 1]

at_risk = df[df['RiskScore'] > 0.7]

print("\n===== HIGH RISK EMPLOYEES =====")
print(at_risk.head())
# In[82]:


at_risk.to_csv("outputs/at_risk_employees.csv", index=False)

print("PROJECT COMPLETED SUCCESSFULLY")
