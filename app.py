import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np
import shap
import matplotlib.pyplot as plt

# Step 0: Download the model from Google Drive
model_url = "https://drive.google.com/uc?id=1x4Vmmr6Ip-msXGQpeIa-WFkpyD5aECOo"
model_path = "credit_default_model.pkl"

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(model_url, model_path, quiet=False)

# Step 1: Load the trained model
@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# Step 2: Define Streamlit app
st.title("💳 Credit Card Default Prediction with Explainability")

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Mode", ["🏠 Home", "📊 Feature Importance"])

# Expected input features
expected_columns = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# Data privacy assurance
st.sidebar.markdown("### Data Privacy")
st.sidebar.write("Your data is not stored or shared. All computations are done locally on your device.")

# Model transparency
st.sidebar.markdown("### About the Model")
st.sidebar.write("This model predicts the likelihood of credit card default based on user-provided data. It was trained on a dataset of credit card users and uses SHAP for explainability.")

if app_mode == "🏠 Home":
    st.write("### Predict Credit Card Default")

    # Manual input form
    st.write("#### Enter Your Details")
    with st.form("user_input_form"):
        # Collect all 23 features
        limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=0)
        age = st.number_input("Age (AGE)", min_value=18, max_value=100)
        sex = st.selectbox("Sex (SEX)", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        education = st.selectbox("Education (EDUCATION)", options=[1, 2, 3, 4], format_func=lambda x: {1: "Graduate", 2: "University", 3: "High School", 4: "Others"}[x])
        marriage = st.selectbox("Marriage (MARRIAGE)", options=[1, 2, 3], format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])
        
        # Add inputs for PAY_0 to PAY_6
        pay_0 = st.number_input("Repayment Status (PAY_0)", min_value=-2, max_value=8)
        pay_2 = st.number_input("Repayment Status (PAY_2)", min_value=-2, max_value=8)
        pay_3 = st.number_input("Repayment Status (PAY_3)", min_value=-2, max_value=8)
        pay_4 = st.number_input("Repayment Status (PAY_4)", min_value=-2, max_value=8)
        pay_5 = st.number_input("Repayment Status (PAY_5)", min_value=-2, max_value=8)
        pay_6 = st.number_input("Repayment Status (PAY_6)", min_value=-2, max_value=8)
        
        # Add inputs for BILL_AMT1 to BILL_AMT6
        bill_amt1 = st.number_input("Bill Amount 1 (BILL_AMT1)", min_value=0)
        bill_amt2 = st.number_input("Bill Amount 2 (BILL_AMT2)", min_value=0)
        bill_amt3 = st.number_input("Bill Amount 3 (BILL_AMT3)", min_value=0)
        bill_amt4 = st.number_input("Bill Amount 4 (BILL_AMT4)", min_value=0)
        bill_amt5 = st.number_input("Bill Amount 5 (BILL_AMT5)", min_value=0)
        bill_amt6 = st.number_input("Bill Amount 6 (BILL_AMT6)", min_value=0)
        
        # Add inputs for PAY_AMT1 to PAY_AMT6
        pay_amt1 = st.number_input("Payment Amount 1 (PAY_AMT1)", min_value=0)
        pay_amt2 = st.number_input("Payment Amount 2 (PAY_AMT2)", min_value=0)
        pay_amt3 = st.number_input("Payment Amount 3 (PAY_AMT3)", min_value=0)
        pay_amt4 = st.number_input("Payment Amount 4 (PAY_AMT4)", min_value=0)
        pay_amt5 = st.number_input("Payment Amount 5 (PAY_AMT5)", min_value=0)
        pay_amt6 = st.number_input("Payment Amount 6 (PAY_AMT6)", min_value=0)
        
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Create a DataFrame with all 23 features
        user_data = pd.DataFrame([[limit_bal, sex, education, marriage, age,
                                   pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
                                   bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
                                   pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6]],
                                 columns=expected_columns)
        
        # Check if the model is a Pipeline
        if isinstance(model, Pipeline):
            preprocessor = model.named_steps['preprocessor']
            classifier = model.named_steps['classifier']
            X_transformed = preprocessor.transform(user_data)
            prediction = classifier.predict(X_transformed)
            probability = classifier.predict_proba(X_transformed)[:, 1]
        else:
            # Preprocess manually if not a Pipeline
            def preprocess_input_data(df):
                df['SEX'] = df['SEX'].astype(int)
                df['EDUCATION'] = df['EDUCATION'].astype(int)
                df['MARRIAGE'] = df['MARRIAGE'].astype(int)
                return df

            user_data = preprocess_input_data(user_data)
            prediction = model.predict(user_data)
            probability = model.predict_proba(user_data)[:, 1]

        st.write("### Prediction Result")
        st.write(f"Default Risk: {'High' if prediction[0] == 1 else 'Low'}")
        st.write(f"Probability of Default: {probability[0]:.2f}")

    # CSV upload functionality
    st.write("#### Upload a CSV File for Predictions")
    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=None, names=expected_columns)

        if df.shape[1] != len(expected_columns):
            st.error("Uploaded CSV format is incorrect! Check the column count.")
        else:
            # Check if the model is a Pipeline
            if isinstance(model, Pipeline):
                preprocessor = model.named_steps['preprocessor']
                classifier = model.named_steps['classifier']
                X_transformed = preprocessor.transform(df)
                predictions = classifier.predict(X_transformed)
                probabilities = classifier.predict_proba(X_transformed)[:, 1]
            else:
                # Preprocess manually if not a Pipeline
                def preprocess_input_data(df):
                    df['SEX'] = df['SEX'].astype(int)
                    df['EDUCATION'] = df['EDUCATION'].astype(int)
                    df['MARRIAGE'] = df['MARRIAGE'].astype(int)
                    return df

                df = preprocess_input_data(df)
                predictions = model.predict(df)
                probabilities = model.predict_proba(df)[:, 1]

            df['Default_Risk'] = predictions
            df['Probability'] = probabilities

            st.write("### Prediction Results")
            st.dataframe(df[['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 'Default_Risk', 'Probability']])

elif app_mode == "📊 Feature Importance":
    st.write("### 🔍 Feature Importance & Explainability")

    uploaded_file = st.file_uploader("📂 Upload CSV for SHAP Analysis", type=["csv"])
