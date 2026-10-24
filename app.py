import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺"
)

st.title("🩺 Diabetes Prediction")
st.write("Enter your details to get a model prediction.")

if not os.path.exists("diabetes_model.pkl"):
    st.error("First run: python train_model.py")
    st.stop()

model = joblib.load("diabetes_model.pkl")
columns = joblib.load("columns.pkl")

with st.form("diabetes_form"):
    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=30, value=0
    )
    glucose = st.number_input(
        "Glucose", min_value=0, max_value=300, value=120
    )
    blood_pressure = st.number_input(
        "Blood Pressure", min_value=0, max_value=200, value=70
    )
    skin_thickness = st.number_input(
        "Skin Thickness", min_value=0, max_value=100, value=20
    )
    insulin = st.number_input(
        "Insulin", min_value=0, max_value=900, value=80
    )
    bmi = st.number_input(
        "BMI", min_value=0.0, max_value=100.0, value=25.0
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0, max_value=3.0, value=0.5
    )
    age = st.number_input(
        "Age", min_value=1, max_value=120, value=25
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    values = [[
        pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi, dpf, age
    ]]

    input_df = pd.DataFrame(values, columns=columns)
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.warning("Model prediction: Diabetes indicated.")
    else:
        st.success("Model prediction: Diabetes not indicated.")

    st.caption(
        "This is an ML prediction, not a medical diagnosis. "
        "Consult a healthcare professional for medical advice."
    )