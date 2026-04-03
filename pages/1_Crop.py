import streamlit as st
import joblib
import pandas as pd

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

    # Prediction
    result = model.predict([[n, p, k, temp, humidity, ph, rainfall]])

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
    # 📊 Graph
    # -------------------------------
    st.subheader("📊 Input Analysis")

    data = pd.DataFrame({
        "Parameters": ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"],
        "Values": [n, p, k, temp, humidity, ph, rainfall]
    })

    st.bar_chart(data.set_index("Parameters"))

    # -------------------------------
    # 🧠 AI Insight
    # -------------------------------
    st.subheader("🧠 AI Insight")

    if result[0] == "rice":
        st.info("Rice grows well in high rainfall and humidity conditions.")
    elif result[0] == "wheat":
        st.info("Wheat prefers moderate temperature and low rainfall.")
    else:
        st.info("This crop matches your soil and climate conditions.")