import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np
import shap
import matplotlib.pyplot as plt

# Step 1: Download the model from Google Drive
model_url = "https://drive.google.com/uc?id=1en2IPj_z6OivZCBNDXepX-EAiZLvCILE"
model_path = "credit_default_model_fixed.pkl"

# Check if the model file already exists; if not, download it
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(model_url, model_path, quiet=False)

# Step 2: Load the trained model using old scikit-learn and re-save with updated version
model = joblib.load(model_path)
joblib.dump(model, model_path)

# Step 3: Define the app
st.title("Credit Card Default Prediction with Explainability")

# Define expected columns
expected_columns = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# Explanation of variables
st.write("### Feature Explanation")
st.write("This study used the following 23 variables:")
st.write("""
- **LIMIT_BAL**: Amount of given credit (includes individual and family credit)
- **SEX**: Gender (1 = Male, 2 = Female)
- **EDUCATION**: (1 = Graduate School, 2 = University, 3 = High School, 4 = Others)
- **MARRIAGE**: Marital status (1 = Married, 2 = Single, 3 = Others)
- **AGE**: Age of the individual
- **PAY_0 to PAY_6**: Past monthly payment records (-1 = Pay duly, 1-9 = Months delayed)
- **BILL_AMT1 to BILL_AMT6**: Amount of bill statement from April to September 2005
- **PAY_AMT1 to PAY_AMT6**: Amount of previous payments from April to September 2005
""")

# Batch Upload Section
st.write("## Batch Upload (CSV)")
st.write("Upload a CSV file containing customer details in the expected format.")
st.write("**Expected CSV format:**")
st.write(pd.DataFrame(columns=expected_columns).head())

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)
    df.columns = expected_columns
    
    # Make batch predictions
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    df['Default_Risk'] = predictions
    df['Probability'] = probabilities
    
    # Improved Interface with XAI
    st.write("### Prediction Results")
    st.dataframe(df[['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 'Default_Risk', 'Probability']])
    
    st.download_button("Download Predictions", df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
    
    # Explainability using SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    
    st.write("### Feature Importance")
    shap.summary_plot(shap_values, df, show=False)
    plt.savefig("shap_summary.png", bbox_inches='tight')
    st.image("shap_summary.png")
    
    # Individual Prediction Explanations
    st.write("### Individual Prediction Explanation")
    for i in range(min(3, len(df))):  # Show SHAP force plot for first few records
        st.write(f"Explanation for Record {i+1}:")
        shap.force_plot(explainer.expected_value, shap_values[i, :], df.iloc[i, :], matplotlib=True)
        plt.savefig(f"shap_force_{i}.png", bbox_inches='tight')
        st.image(f"shap_force_{i}.png")
