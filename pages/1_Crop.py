import streamlit as st
import joblib
import pandas as pd
import shap
import numpy as np

# Load model
model = joblib.load("crop_model.pkl")

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

    # Prepare input
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
    # 🧠 SHAP Explanation (FINAL FIX)
    # -------------------------------
    st.subheader("🧠 Why this prediction? (Explainable AI)")

    try:
        explainer = shap.TreeExplainer(model)

        input_array = np.array([[n, p, k, temp, humidity, ph, rainfall]])

        shap_values = explainer.shap_values(input_array)

        feature_names = ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"]

        # Handle classification & regression safely
        if isinstance(shap_values, list):
            shap_values_single = shap_values[0][0]
        else:
            shap_values_single = shap_values[0]

        # Convert to 1D
        shap_values_single = np.array(shap_values_single).flatten()

        # Match length safely
        shap_values_single = shap_values_single[:len(feature_names)]

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": shap_values_single
        })

        st.bar_chart(shap_df.set_index("Feature"))

        st.info("Higher values = more influence on prediction")

    except Exception as e:
        st.warning("SHAP explanation could not be generated.")
        st.error(str(e))
