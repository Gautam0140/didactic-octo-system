import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/churn_model.pkl")

st.title("Customer Churn Prediction")

st.write("Enter customer details:")

# Example inputs (modify based on your dataset)
tenure = st.number_input("Tenure", 0, 100)
monthly_charges = st.number_input("Monthly Charges", 0.0, 10000.0)

if st.button("Predict"):
    data = pd.DataFrame([[tenure, monthly_charges]], columns=["tenure", "MonthlyCharges"])
    
    prediction = model.predict(data)
    
    if prediction[0] == 1:
        st.error("Customer is likely to churn ❌")
    else:
        st.success("Customer will stay ✅")
