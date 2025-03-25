import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np
import shap
import matplotlib.pyplot as plt

# Updated model URL from Google Drive
model_url = "https://drive.google.com/uc?id=1x4Vmmr6Ip-msXGQpeIa-WFkpyD5aECOo"
model_path = "credit_default_model.pkl"

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(model_url, model_path, quiet=False)

# Load the trained model
pipeline = joblib.load(model_path)
model = pipeline.named_steps["classifier"]  # Extract the classifier from the pipeline
preprocessor = pipeline.named_steps["preprocessor"]

# Streamlit app
st.title("Credit Card Default Prediction with Explainability")

expected_columns = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

st.write("### Feature Explanation")
st.write("This study used the following 23 variables:")
st.write("- **LIMIT_BAL**: Amount of given credit (includes individual and family credit)\n"
         "- **SEX**: Gender (1 = Male, 2 = Female)\n"
         "- **EDUCATION**: (1 = Graduate School, 2 = University, 3 = High School, 4 = Others)\n"
         "- **MARRIAGE**: Marital status (1 = Married, 2 = Single, 3 = Others)\n"
         "- **AGE**: Age of the individual\n"
         "- **PAY_0 to PAY_6**: Past payment records (-1 = Pay duly, 1-9 = Months delayed)\n"
         "- **BILL_AMT1 to BILL_AMT6**: Amount of bill statement (April to September 2005)\n"
         "- **PAY_AMT1 to PAY_AMT6**: Amount of previous payments (April to September 2005)")

st.write("## Batch Upload (CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df[expected_columns]  # Ensure only required columns are used
    
    # Predictions
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    df['Default_Risk'] = predictions
    df['Probability'] = probabilities
    
    st.write("### Prediction Results")
    st.dataframe(df[['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 'Default_Risk', 'Probability']])
    st.download_button("Download Predictions", df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
    
    # SHAP Explainability
    explainer = shap.Explainer(model)
    shap_values = explainer(df)
    
    st.write("### Feature Importance (Global View)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, df, show=False)
    st.pyplot(fig)
    
    # Select a specific row for local explanations
    index = st.number_input("Select a row index for individual explanation", min_value=0, max_value=len(df)-1, value=0)
    st.write("### Individual Prediction Explanation")
    
    # Force Plot
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value[1], shap_values[index].values, df.iloc[index])
    st.pyplot(force_plot)
    
    # Waterfall Plot
    st.write("### Waterfall Explanation")
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_values[index])
    st.pyplot(fig)
    
    st.write("### Decision Plot")
    fig, ax = plt.subplots()
    shap.decision_plot(explainer.expected_value[1], shap_values[index].values, df.iloc[index])
    st.pyplot(fig)
