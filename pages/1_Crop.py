import streamlit as st
import joblib
import pandas as pd
import shap
import numpy as np

# Load model and data
model = joblib.load("crop_model.pkl")
X_train = joblib.load("X_train.pkl")

st.title("🌱 Crop Recommendation")

st.write("Enter soil and environmental details:")

# Inputs
n = st.number_input("Nitrogen (N)", min_value=0.0)
p = st.number_input("Phosphorus (P)", min_value=0.0)
k = st.number_input("Potassium (K)", min_value=0.0)
temp = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
ph = st.number_input("Soil pH", min_value=0.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

if st.button("Predict Crop"):

    input_data = [[n, p, k, temp, humidity, ph, rainfall]]

    # Prediction
    result = model.predict(input_data)
    st.success(f"🌾 Recommended Crop: {result[0]}")

    # -------------------------------
    # 💧 Irrigation Advice
    # -------------------------------
    st.subheader("💧 Irrigation Advice")

    if rainfall < 50:
        st.warning("Low rainfall → Irrigation needed 💧")
    elif rainfall > 200:
        st.warning("Too much rain → No irrigation needed 🚫")
    else:
        st.info("Moderate rainfall → Monitor soil moisture 🌿")

    # -------------------------------
    # 📊 Input Graph
    # -------------------------------
    st.subheader("📊 Input Analysis")

    data = pd.DataFrame({
        "Parameters": ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"],
        "Values": [n, p, k, temp, humidity, ph, rainfall]
    })

    st.bar_chart(data.set_index("Parameters"))

    # -------------------------------
    # 🧠 SHAP Explanation
    # -------------------------------
    st.subheader("🧠 Why this prediction? (Explainable AI)")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(np.array(input_data))

    feature_names = ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"]

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_values[0][0]
    })

    st.bar_chart(shap_df.set_index("Feature"))

    st.info("Higher values show stronger influence on prediction")