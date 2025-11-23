# app.py
import streamlit as st

st.set_page_config(page_title="Food Delivery Time Prediction", layout="wide")


st.title("🍽️ Welcome to Food Delivery Time Predictor")
st.write(
    """
    This app helps estimate **food delivery times** based on factors like distance, 
    traffic, weather, and more.  
    Click below to get started with the predictor.
    """
)


st.image("images/food_delivery.jpg")



st.markdown("---")

# -------------------
# Navigation
# -------------------
if st.button("➡️ Go to Predictor"):
    st.switch_page("pages/predictor.py")
