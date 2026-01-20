# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# ==============================
# STEP 0: Page Config
# ==============================
st.set_page_config(page_title="COVID KNN Predictor", layout="wide")

# ==============================
# STEP 1: Load & Clean Data
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\HAJI LAPTOP G55\Downloads\archive\patient.csv")
    df = df[["sex", "age", "pneumonia", "diabetes", "asthma", "outcome"]]
    # Binary conversion
    for col in df.columns:
        if col != "age":
            df[col] = df[col].apply(lambda x: 1 if x==1 else 0)
    return df

df = load_data()

# ==============================
# STEP 2: Train KNN Model
# ==============================
X = df.drop(columns="outcome")
y = df["outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale age
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled["age"] = scaler.fit_transform(X_train[["age"]])
X_test_scaled["age"] = scaler.transform(X_test[["age"]])

# ==============================
# STEP 2.1: Handle Class Imbalance
# ==============================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# Weighted KNN
knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
knn.fit(X_train_res, y_train_res)
y_pred = knn.predict(X_test_scaled)
accuracy = round(accuracy_score(y_test, y_pred)*100,2)
cm = confusion_matrix(y_test, y_pred)

# ==============================
# STEP 3: Streamlit Tabs
# ==============================
tabs = st.tabs(["📊 EDA", "🩺 Model Info", "📝 Prediction"])

# ==============================
# TAB 1: EDA
# ==============================
with tabs[0]:
    st.header("Exploratory Data Analysis (EDA)")
    st.subheader("Raw Data Head")
    st.dataframe(df.head())
    st.subheader("Raw Data Tail")
    st.dataframe(df.tail())

    st.markdown("**COVID Outcome Distribution**")
    fig1, ax1 = plt.subplots(figsize=(6,4))
    bars = ax1.bar(["Negative","Positive"], df["outcome"].value_counts().sort_index(),
                   color=["#2ECC71","#C73E1D"], edgecolor="black", linewidth=1.5)
    ax1.set_ylabel("Number of Patients")
    ax1.set_title("Outcome Distribution")
    for bar in bars:
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                 f"{bar.get_height():,}", ha="center", va="bottom", weight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.hist(df["age"], bins=20, color="#F18F01", edgecolor="black", linewidth=1.2)
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Number of Patients")
    ax2.set_title("Age Distribution")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    st.pyplot(fig2)

# ==============================
# TAB 2: Model Info
# ==============================
with tabs[1]:
    st.header("KNN Model Information & Performance")
    st.subheader("Model Accuracy")
    st.info(f"KNN Test Accuracy: {accuracy}%")

    correct = (y_test == y_pred).sum()
    incorrect = (y_test != y_pred).sum()
    st.markdown("**Prediction Results (Correct vs Incorrect)**")
    fig3, ax3 = plt.subplots(figsize=(6,4))
    bars = ax3.bar(["Correct","Incorrect"], [correct, incorrect], color=["#2ECC71","#C73E1D"], edgecolor="black", linewidth=1.5)
    for bar in bars:
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                 f"{bar.get_height():,}", ha="center", va="bottom", weight="bold")
    ax3.set_ylabel("Number of Samples")
    ax3.set_title("Prediction Results")
    ax3.grid(axis="y", linestyle="--", alpha=0.35)
    st.pyplot(fig3)

    st.markdown("**Confusion Matrix**")
    fig4, ax4 = plt.subplots(figsize=(6,4))
    im = ax4.imshow(cm, cmap="Blues")
    ax4.set_xticks([0,1]); ax4.set_xticklabels(["Negative","Positive"])
    ax4.set_yticks([0,1]); ax4.set_yticklabels(["Negative","Positive"])
    ax4.set_xlabel("Predicted"); ax4.set_ylabel("Actual")
    ax4.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax4.text(j,i, cm[i,j], ha="center", va="center", weight="bold", fontsize=14)
    fig4.colorbar(im)
    st.pyplot(fig4)

# ==============================
# TAB 3: Prediction Form
# ==============================
with tabs[2]:
    st.header("Predict Your COVID Risk")
    st.write("Fill your details to get predicted COVID outcome probability.")

    sex = st.selectbox("Sex", ["Male","Female"])
    age = st.slider("Age", 0, 100, 30)
    pneumonia = st.selectbox("Pneumonia", ["Yes","No"])
    diabetes = st.selectbox("Diabetes", ["Yes","No"])
    asthma = st.selectbox("Asthma", ["Yes","No"])

    if st.button("Predict"):
        input_df = pd.DataFrame({
            "sex":[1 if sex=="Male" else 0],
            "age":[age],
            "pneumonia":[1 if pneumonia=="Yes" else 0],
            "diabetes":[1 if diabetes=="Yes" else 0],
            "asthma":[1 if asthma=="Yes" else 0]
        })
        input_df["age"] = scaler.transform(input_df[["age"]])
        pred = knn.predict(input_df)[0]
        proba = knn.predict_proba(input_df)[0]

        st.subheader("Prediction Result")
        if pred == 1:
            st.error(f"⚠️ Positive COVID Outcome\nEstimated Risk: {proba[1]*100:.2f}%")
        else:
            st.success(f"✅ Negative COVID Outcome\nEstimated Safety: {proba[0]*100:.2f}%")
