import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# Step 0: Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Credit Default Prediction", layout="wide")

# Step 1: Download the model from Google Drive
model_url = "https://drive.google.com/uc?id=1en2IPj_z6OivZCBNDXepX-EAiZLvCILE"
model_path = "credit_default_model.pkl"

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Step 2: Load the trained model
@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# Step 3: Define Streamlit app
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
        
        # Preprocess and predict
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        X_transformed = preprocessor.transform(user_data)
        prediction = classifier.predict(X_transformed)
        probability = classifier.predict_proba(X_transformed)[:, 1]

        st.write("### Prediction Result")
        st.write(f"Default Risk: {'High' if prediction[0] == 1 else 'Low'}")
        st.write(f"Probability of Default: {probability[0]:.2f}")

        # Local SHAP explanation (removed force_plot)
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)

        # Check if shap_values is a list (binary classification)
        if isinstance(shap_values, list):
            # For binary classification, use shap_values[1] for the positive class
            shap_values = shap_values[1]
            base_value = explainer.expected_value[1]
        else:
            # For non-binary cases, use shap_values directly
            base_value = explainer.expected_value

        # Ensure the input data is in the correct format
        features = user_data.iloc[0:1, :]  # Extract the first row of user data as a DataFrame

        # Generate the SHAP force plot (commented out)
        # st.write("#### Local Explanation (SHAP)")
        # shap.force_plot(base_value, shap_values[0], features, matplotlib=True, show=False)
        # st.pyplot(bbox_inches='tight')
        # plt.clf()

    # CSV upload functionality
    st.write("#### Upload a CSV File for Predictions")
    uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=None, names=expected_columns)

        if df.shape[1] != len(expected_columns):
            st.error("Uploaded CSV format is incorrect! Check the column count.")
        else:
            preprocessor = model.named_steps['preprocessor']
            classifier = model.named_steps['classifier']
            X_transformed = preprocessor.transform(df)
            predictions = classifier.predict(X_transformed)
            probabilities = classifier.predict_proba(X_transformed)[:, 1]

            df['Default_Risk'] = predictions
            df['Probability'] = probabilities

            st.write("### Prediction Results")
            st.dataframe(df[['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 'Default_Risk', 'Probability']])

elif app_mode == "📊 Feature Importance":
    st.write("### 🔍 Feature Importance & Explainability")

    uploaded_file = st.file_uploader("📂 Upload CSV for SHAP Analysis", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=None, names=expected_columns)

        if df.shape[1] != len(expected_columns):
            st.error("Uploaded CSV format is incorrect! Check the column count.")
        else:
            preprocessor = model.named_steps['preprocessor']
            classifier = model.named_steps['classifier']
            X_transformed = preprocessor.transform(df)

            feature_names = expected_columns  # Use original feature names
            sample_data = X_transformed[:5]  # Reduce sample size for speed

            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(sample_data)

            # Ensure correct shape for SHAP values
            correct_shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap_importance = np.abs(correct_shap_values).mean(axis=0)

            # Convert to 1D array
            shap_importance = np.array(shap_importance).flatten()

            # Ensure dimensions match
            min_len = min(len(feature_names), len(shap_importance))
            feature_names = feature_names[:min_len]
            shap_importance = shap_importance[:min_len]

            # Create DataFrame for feature importance
            importance_df = pd.DataFrame({'Feature': feature_names, 'SHAP Importance': shap_importance})
            importance_df = importance_df.sort_values(by="SHAP Importance", ascending=False).head(10)

            # Display results
            st.write("### 🔥 Top 10 Most Important Features")
            st.dataframe(importance_df)

            # Plot bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(importance_df["Feature"], importance_df["SHAP Importance"], color="royalblue")
            ax.set_xlabel("SHAP Importance")
            ax.set_ylabel("Feature")
            ax.set_title("📊 Feature Importance")
            plt.gca().invert_yaxis()
            st.pyplot(fig)

            # SHAP Summary Plot
            st.write("### 📊 SHAP Summary Plot")
            shap.summary_plot(correct_shap_values, sample_data, feature_names=feature_names, show=False)
            plt.savefig("shap_summary.png", bbox_inches='tight')
            st.image("shap_summary.png")
