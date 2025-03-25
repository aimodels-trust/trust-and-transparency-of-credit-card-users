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

if hasattr(pipeline, 'named_steps'):
    preprocessor = pipeline.named_steps.get('preprocessor', None)
    model = pipeline.named_steps['classifier']
else:
    preprocessor = None
    model = pipeline

expected_columns = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

st.title("💳 Credit Card Default Prediction with Explainability")
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Mode", ["🏠 Home", "📂 Batch Prediction", "📊 Feature Importance"])

if app_mode == "🏠 Home":
    st.write("### Predict Credit Card Default")
    with st.form("user_input_form"):
        inputs = []
        inputs.append(st.number_input("Credit Limit (LIMIT_BAL)", min_value=0))
        inputs.append(st.selectbox("Sex (SEX)", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female"))
        inputs.append(st.selectbox("Education (EDUCATION)", options=[1, 2, 3, 4], format_func=lambda x: {1: "Graduate", 2: "University", 3: "High School", 4: "Others"}[x]))
        inputs.append(st.selectbox("Marriage (MARRIAGE)", options=[1, 2, 3], format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x]))
        inputs.append(st.number_input("Age (AGE)", min_value=18, max_value=100))
        
        for i in range(1, 7):
            inputs.append(st.number_input(f"Repayment Status (PAY_{i})", min_value=-2, max_value=8))
        for i in range(1, 7):
            inputs.append(st.number_input(f"Bill Amount {i} (BILL_AMT{i})", min_value=0))
        for i in range(1, 7):
            inputs.append(st.number_input(f"Payment Amount {i} (PAY_AMT{i})", min_value=0))
        
        submitted = st.form_submit_button("Predict")

    if submitted:
        user_data = pd.DataFrame([inputs], columns=expected_columns)
        if preprocessor:
            user_data = preprocessor.transform(user_data)
        
        prediction = model.predict(user_data)[0]
        probability = model.predict_proba(user_data)[0][1]
        risk_level = "High Risk" if prediction == 1 else "Low Risk"

        st.write(f"### Prediction: {risk_level}")
        st.write(f"### Probability of Default: {probability:.2f}")

elif app_mode == "📂 Batch Prediction":
    st.write("### Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in uploaded file: {missing_cols}")
            st.stop()
        
        df = df[expected_columns]
        if preprocessor:
            df = preprocessor.transform(df)
        
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        df["Risk Level"] = ["High Risk" if p == 1 else "Low Risk" for p in predictions]
        df["Probability of Default"] = probabilities
        st.write(df)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

elif app_mode == "📊 Feature Importance":
    st.write("### Feature Importance & Explainability")
    uploaded_file = st.file_uploader("📂 Upload CSV for SHAP Analysis", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in uploaded file: {missing_cols}")
            st.stop()
        
        df = df[expected_columns]
        if preprocessor:
            df = preprocessor.transform(df)
        
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_features = pd.DataFrame({'Feature': expected_columns, 'Importance': feature_importance})
        top_features = top_features.sort_values(by="Importance", ascending=False).head(10)

        st.write("### Top 10 Most Important Features")
        st.dataframe(top_features)

        index = st.number_input("Select row index for individual explanation", min_value=0, max_value=len(df)-1, value=0)
        fig, ax = plt.subplots()
        shap.waterfall_plot(shap.Explanation(values=shap_values[index], base_values=explainer.expected_value[1], data=df.iloc[index]), max_display=10, show=False)
        st.pyplot(fig)
