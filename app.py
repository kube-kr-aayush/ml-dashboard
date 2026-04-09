import streamlit as st
import pandas as pd

st.title("🍷 Wine Quality ML Dashboard")

# Load dataset
df = pd.read_csv("winequality-red.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Info")
st.write("Shape:", df.shape)
st.write("Columns:", df.columns)

import matplotlib.pyplot as plt
import seaborn as sns

st.subheader("📊 Data Visualization")

# Histogram
st.write("### Distribution of Features")
df.hist(figsize=(10,8))
st.pyplot(plt)

# Correlation Heatmap
st.write("### Correlation Heatmap")
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
st.pyplot(plt)

st.subheader("🧹 Data Cleaning & Feature Selection")

# Missing values check
st.write("Missing Values:")
st.write(df.isnull().sum())

# Drop missing values (if any)
df = df.dropna()

# Select target
target = st.selectbox("Select Target Column", df.columns)

# Features & target split
X = df.drop(columns=[target])
y = df[target].astype(int)

st.write("Selected Target:", target)
st.write("Feature Columns:", X.columns)

from sklearn.model_selection import train_test_split

st.subheader("🔀 Train-Test Split")

# Split ratio slider
test_size = st.slider("Select Test Size", 0.1, 0.5, 0.2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

st.write("Training Size:", X_train.shape)
st.write("Testing Size:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

st.subheader("🤖 Model Selection")

model_name = st.selectbox(
    "Choose Model",
    ["Logistic Regression", "Decision Tree"]
)

if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
else:
    model = DecisionTreeClassifier()

from sklearn.metrics import accuracy_score, confusion_matrix

st.subheader("🏋️ Model Training & Evaluation")

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
st.write("✅ Accuracy:", acc)

# Confusion Matrix
st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)


