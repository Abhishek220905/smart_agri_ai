import streamlit as st

st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Smart Agriculture AI System")

st.markdown("""
### 🚀 AI-powered farming assistant

This system helps you:
- 🌱 Select best crop
- 💧 Get irrigation advice
- 📊 Predict yield

👉 Use the sidebar to navigate between features
""")

st.sidebar.success("Select a page above 👆")