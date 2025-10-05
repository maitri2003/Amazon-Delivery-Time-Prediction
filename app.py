# app.py
"""
Streamlit App - Amazon Delivery Time Prediction
-------------------------------------------------
This app loads the trained RandomForest model
and predicts the delivery duration (in minutes)
based on user input.
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime

# ============================================================
# Load trained model
# ============================================================
@st.cache_resource
def load_model():
    model_path = "models/random_forest_model.pkl"
    model = joblib.load(model_path)
    return model

model = load_model()

# ============================================================
# Load training columns for consistency
# ============================================================
@st.cache_data
def load_training_columns():
    df = pd.read_csv("data/processed/amazon_delivery_clean.csv")
    X = df.drop(columns=["Delivery_Duration_min"])
    # Keep only numeric columns (same as train.py)
    X = X.select_dtypes(include=["int64", "float64"])
    return list(X.columns)

training_columns = load_training_columns()

# ============================================================
# Streamlit UI
# ============================================================
st.title("🚚 Amazon Delivery Time Prediction App")

st.markdown("""
Enter the details below to predict the **delivery time (in minutes)**.
""")

# Sidebar inputs
st.sidebar.header("Input Features")

Distance = st.sidebar.number_input("Distance (in km)", min_value=0.0, step=0.1)
Multiple_Deliveries = st.sidebar.number_input("Multiple Deliveries (count)", min_value=0, step=1)
Agent_Age = st.sidebar.number_input("Agent Age", min_value=18, max_value=60, step=1)
Agent_Rating = st.sidebar.number_input("Agent Rating (1–5)", min_value=1.0, max_value=5.0, step=0.1)
Pickup_Day = st.sidebar.slider("Pickup Day (1–31)", min_value=1, max_value=31)
Pickup_Hour = st.sidebar.slider("Pickup Hour (0–23)", min_value=0, max_value=23)
Traffic_Density = st.sidebar.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"])
Weather = st.sidebar.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Fog"])
Festival = st.sidebar.selectbox("Festival", ["No", "Yes"])

# ============================================================
# Create input DataFrame
# ============================================================
input_data = pd.DataFrame({
    "Distance": [Distance],
    "Multiple_Deliveries": [Multiple_Deliveries],
    "Agent_Age": [Agent_Age],
    "Agent_Rating": [Agent_Rating],
    "Pickup_Day": [Pickup_Day],
    "Pickup_Hour": [Pickup_Hour],
    "Traffic_Density_Low": [1 if Traffic_Density == "Low" else 0],
    "Traffic_Density_Medium": [1 if Traffic_Density == "Medium" else 0],
    "Traffic_Density_High": [1 if Traffic_Density == "High" else 0],
    "Traffic_Density_Jam": [1 if Traffic_Density == "Jam" else 0],
    "Weather_Sunny": [1 if Weather == "Sunny" else 0],
    "Weather_Cloudy": [1 if Weather == "Cloudy" else 0],
    "Weather_Rainy": [1 if Weather == "Rainy" else 0],
    "Weather_Fog": [1 if Weather == "Fog" else 0],
    "Festival_Yes": [1 if Festival == "Yes" else 0],
    "Festival_No": [1 if Festival == "No" else 0]
})

# ============================================================
# Align with training columns
# ============================================================
# Add any missing columns (fill 0), drop extras if any
for col in training_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[training_columns]  # Keep same order

# ============================================================
# Prediction
# ============================================================
if st.button("🚀 Predict Delivery Time"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"✅ Estimated Delivery Duration: **{prediction:.2f} minutes**")
    except Exception as e:
        st.error(f"⚠️ Error while predicting: {e}")
