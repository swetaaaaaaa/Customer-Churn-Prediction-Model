import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model\churn_model\model_LR.pkl')

st.title('Customer Churn Prediction')
st.write('This app predicts whether a customer will churn or not based on their features.')

is_male = st.selectbox( "Gender",["Male", "Female"])

SeniorCitizen = st.selectbox("Senior Citizen",["No", "Yes"])

do_partner = st.selectbox("Partner",["No", "Yes"])

Dependents = st.selectbox("Dependents",["No", "Yes"])

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)
PhoneService = st.selectbox("Phone Service",["No", "Yes"])

MultipleLines = st.selectbox("Multiple Lines",["No", "Yes"])

OnlineSecurity = st.selectbox("Online Security",["No", "Yes"])

OnlineBackup = st.selectbox("Online Backup",["No", "Yes"])

DeviceProtection = st.selectbox("Device Protection",["No", "Yes"])

TechSupport = st.selectbox("Tech Support",["No", "Yes"])

StreamingTV = st.selectbox("Streaming TV",["No", "Yes"])

StreamingMovies = st.selectbox("Streaming Movies",["No", "Yes"])

PaperlessBilling = st.selectbox("Paperless Billing",["No", "Yes"])

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

InternetService = st.selectbox("Internet Service",["DSL", "Fiber optic", "No"])

InternetService_DSL = 1 if InternetService == "DSL" else 0

InternetService_Fiber_optic = 1 if InternetService == "Fiber optic" else 0

InternetService_No = 1 if InternetService == "No" else 0

Contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])

Contract_Month_to_month = 1 if Contract == "Month-to-month" else 0

Contract_One_year = 1 if Contract == "One year" else 0

Contract_Two_year= 1 if Contract == "Two year" else 0

PaymentMethod = st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])

PaymentMethod_Electronic_check = (
    1 if PaymentMethod == "Electronic check" else 0
)

PaymentMethod_Mailed_check = (
    1 if PaymentMethod == "Mailed check" else 0
)

PaymentMethod_Bank_transfer_automatic = (
    1 if PaymentMethod == "Bank transfer (automatic)" else 0
)

PaymentMethod_Credit_card_automatic = (
    1 if PaymentMethod == "Credit card (automatic)" else 0
)


input_data = pd.DataFrame({
    "is_male": [
        1 if is_male == "Male" else 0
    ],

    "SeniorCitizen": [
        1 if SeniorCitizen == "Yes" else 0
    ],

    "do_partner": [
        1 if do_partner == "Yes" else 0
    ],

    "Dependents": [
        1 if Dependents == "Yes" else 0
    ],

    "tenure": [tenure],

    "PhoneService": [
        1 if PhoneService == "Yes" else 0
    ],

    "MultipleLines": [
        1 if MultipleLines == "Yes" else 0
    ],

    "OnlineSecurity": [
        1 if OnlineSecurity == "Yes" else 0
    ],

    "OnlineBackup": [
        1 if OnlineBackup == "Yes" else 0
    ],

    "DeviceProtection": [
        1 if DeviceProtection == "Yes" else 0
    ],

    "TechSupport": [
        1 if TechSupport == "Yes" else 0
    ],

    "StreamingTV": [
        1 if StreamingTV == "Yes" else 0
    ],

    "StreamingMovies": [
        1 if StreamingMovies == "Yes" else 0
    ],

    "PaperlessBilling": [
        1 if PaperlessBilling == "Yes" else 0
    ],

    "MonthlyCharges": [MonthlyCharges],

    "TotalCharges": [TotalCharges],

    "InternetService_DSL": [InternetService_DSL],

    "InternetService_Fiber optic": [InternetService_Fiber_optic],

    "InternetService_No": [InternetService_No],

    "Contract_Month-to-month": [Contract_Month_to_month],

    "Contract_One year": [Contract_One_year],

    "Contract_Two year": [Contract_Two_year],

    "PaymentMethod_Electronic check": [PaymentMethod_Electronic_check],

    "PaymentMethod_Mailed check": [PaymentMethod_Mailed_check],

    "PaymentMethod_Bank transfer (automatic)": [PaymentMethod_Bank_transfer_automatic],

    "PaymentMethod_Credit card (automatic)": [PaymentMethod_Credit_card_automatic]
})

if st.button("Predict"):
    input_data = input_data[model.feature_names_in_]
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is not likely to churn.")

    st.write("Prediction Probability: ", model.predict_proba(input_data)[0][1])