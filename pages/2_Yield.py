import streamlit as st
import joblib

# Load model
model = joblib.load("yield_model.pkl")

st.title("🌾 Yield Prediction")

st.write("Enter farming details:")

# Inputs
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
fertilizer = st.number_input("Fertilizer Usage", min_value=0.0)
area = st.number_input("Area (hectares)", min_value=0.0)

if st.button("Predict Yield"):

    result = model.predict([[rainfall, fertilizer, area]])

    st.success(f"📈 Expected Yield: {result[0]:.2f} tons")

    # -------------------------------
    # 📊 Insight
    # -------------------------------
    st.subheader("📊 Yield Insight")

    if result[0] < 2:
        st.warning("Low yield expected. Improve fertilizer or irrigation.")
    elif result[0] > 5:
        st.success("High yield expected. Good conditions!")
    else:
        st.info("Moderate yield expected.")