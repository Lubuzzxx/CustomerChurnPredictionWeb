from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# streamlit run chrun_pred.py

st.set_page_config(layout="centered")

with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

feature_names = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "EstimatedSalary", "Geography_France", "Geography_Germany", "Geography_Spain",
    "Gender_Female", "Gender_Male", "HasCrCard_0", "HasCrCard_1",
    "IsActiveMember_0", "IsActiveMember_1"
]

scale_vars = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]

default_values = [
    0, 0, 0, 0, 0, 0,
    True, True, True, True, True, True, True, True, True
]

left_col, right_col = st.columns(2)

with left_col:
    st.header("User Inputs")
    def collect_user_inputs(features, default_vals, scale_vars):
        user_inputs = {
            feature: (
                st.number_input(
                    feature, value=default_vals[i], step=1 if isinstance(default_vals[i], int) else 0.01
                )
                if feature in scale_vars else
                st.checkbox(feature, value=default_vals[i])
                if isinstance(default_vals[i], bool) else
                st.number_input(feature, value=default_vals[i], step=1)
            )
            for i, feature in enumerate(features)
        }
        return pd.DataFrame([user_inputs])

    input_data = collect_user_inputs(feature_names, default_values, scale_vars)
    input_data_scaled = input_data.copy()
    input_data_scaled[scale_vars] = scaler.transform(input_data[scale_vars])

with right_col:
    st.header("Prediction")
    if st.button("Predict"):
        probabilities = model.predict_proba(input_data_scaled)[0]
        prediction = model.predict(input_data_scaled)[0]
        prediction_label = "Churned" if prediction == 1 else "Retain"

        st.subheader(f"Predicted Value: {prediction_label}")
        st.write(f"Predicted Probability: {probabilities[1]:.2%} (Churn)")
        st.write(f"Predicted Probability: {probabilities[0]:.2%} (Retain)")
        st.markdown(f"### Output: **{prediction_label}**")

