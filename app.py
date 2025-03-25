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
pipeline = joblib.load(model_path)

# Check if the model is a Pipeline
if hasattr(pipeline, 'named_steps'):
    preprocessor = pipeline.named_steps.get('preprocessor', None)
    model = pipeline.named_steps['classifier']
else:
    preprocessor = None
    model = pipeline

# Step 2: Define expected columns (based on training data)
expected_columns = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# Step 3: Streamlit App
st.title("💳 Credit Card Default Prediction with Explainability")

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Mode", ["🏠 Home", "📊 Feature Importance"])

# Data privacy assurance
st.sidebar.markdown("### Data Privacy")
st.sidebar.write("Your data is not stored or shared. All computations are done locally on your device.")

# Model transparency
st.sidebar.markdown("### About the Model")
st.sidebar.write("This model predicts the likelihood of credit card default based on user-provided data. It uses SHAP for explainability.")

if app_mode == "🏠 Home":
    st.write("### Predict Credit Card Default")

    # Manual Input Form
    st.write("#### Enter Your Details")
    with st.form("user_input_form"):
        # Collect all 23 features
        limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=0)
        age = st.number_input("Age (AGE)", min_value=18, max_value=100)
        sex = st.selectbox("Sex (SEX)", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        education = st.selectbox("Education (EDUCATION)", options=[1, 2, 3, 4], format_func=lambda x: {1: "Graduate", 2: "University", 3: "High School", 4: "Others"}[x])
        marriage = st.selectbox("Marriage (MARRIAGE)", options=[1, 2, 3], format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])

        pay_1 = st.number_input("Repayment Status (PAY_1)", min_value=-2, max_value=8)
        pay_2 = st.number_input("Repayment Status (PAY_2)", min_value=-2, max_value=8)
        pay_3 = st.number_input("Repayment Status (PAY_3)", min_value=-2, max_value=8)
        pay_4 = st.number_input("Repayment Status (PAY_4)", min_value=-2, max_value=8)
        pay_5 = st.number_input("Repayment Status (PAY_5)", min_value=-2, max_value=8)
        pay_6 = st.number_input("Repayment Status (PAY_6)", min_value=-2, max_value=8)

        bill_amt1 = st.number_input("Bill Amount 1 (BILL_AMT1)", min_value=0)
        bill_amt2 = st.number_input("Bill Amount 2 (BILL_AMT2)", min_value=0)
        bill_amt3 = st.number_input("Bill Amount 3 (BILL_AMT3)", min_value=0)
        bill_amt4 = st.number_input("Bill Amount 4 (BILL_AMT4)", min_value=0)
        bill_amt5 = st.number_input("Bill Amount 5 (BILL_AMT5)", min_value=0)
        bill_amt6 = st.number_input("Bill Amount 6 (BILL_AMT6)", min_value=0)

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
                                   pay_1, pay_2, pay_3, pay_4, pay_5, pay_6,
                                   bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
                                   pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6]],
                                 columns=expected_columns)

        # Preprocess the data if necessary
        if preprocessor:
            user_data = preprocessor.transform(user_data)

        # Make predictions
        prediction = model.predict(user_data)[0]
        probability = model.predict_proba(user_data)[0][1]

        st.write("### Prediction Result")
        st.write(f"Default Risk: {'High' if prediction == 1 else 'Low'}")
        st.write(f"Probability of Default: {probability:.2f}")

        # Local SHAP Explanation
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(user_data)

        # Ensure correct shape for SHAP values
        if isinstance(shap_values, list):  # For binary classification
            shap_values = shap_values[1]  # Use SHAP values for the positive class

        st.write("### Individual Prediction Explanation")
        fig, ax = plt.subplots()
        shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value[1], data=user_data.iloc[0]), max_display=10, show=False)
        st.pyplot(fig)

elif app_mode == "📊 Feature Importance":
    st.write("### 🔍 Feature Importance & Explainability")

    uploaded_file = st.file_uploader("📂 Upload CSV for SHAP Analysis", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Ensure only required columns are used
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in the uploaded file: {missing_cols}")
            st.stop()

        # Add missing columns with default values
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns to match the expected order
        df = df[expected_columns]

        # Preprocess the data if necessary
        if preprocessor:
            df = preprocessor.transform(df)

        # Compute SHAP values
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(df)

        # Ensure correct shape for SHAP values
        if isinstance(shap_values, list):  # For binary classification
            shap_values = shap_values[1]  # Use SHAP values for the positive class

        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_features = pd.DataFrame({
            'Feature': expected_columns,
            'Importance': feature_importance
        }).sort_values(by="Importance", ascending=False).head(10)

        st.write("### Top 10 Most Important Features")
        st.dataframe(top_features)

        # Select a specific row for local explanations
        index = st.number_input("Select a row index for individual explanation", min_value=0, max_value=len(df)-1, value=0)
        st.write("### Individual Prediction Explanation")

        # Waterfall Plot
        fig, ax = plt.subplots()
        shap.waterfall_plot(shap.Explanation(values=shap_values[index], base_values=explainer.expected_value[1], data=df.iloc[index]), max_display=10, show=False)
        st.pyplot(fig)
