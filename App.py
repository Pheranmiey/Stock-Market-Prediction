import streamlit as st
import pickle
import numpy as np
import random

# Load the model (cached)
@st.cache_resource
def load_model():
    with open('catboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Feature explanations
feature_info = {
    'EMA_20': '20-day Exponential Moving Average.',
    'EMA_200': '200-day Exponential Moving Average.',
    'EMA_10': '10-day Exponential Moving Average.',
    'EMA_50': '50-day Exponential Moving Average.',
    'Year': 'Year (e.g., 2023).',
    'DTB3': '3-Month Treasury Bill Rate.',
    'DTB6': '6-Month Treasury Bill Rate.',
    'DE2': 'Economic indicator DE2.',
    'DGS5': '5-Year Treasury Rate.',
    'Month': 'Month of the year (1–12).',
    'TE1': 'Economic indicator TE1.',
    'DE1': 'Economic indicator DE1.',
    'DTB4WK': '4-Week Treasury Bill Rate.',
    'DE4': 'Economic indicator DE4.',
    'DE6': 'Economic indicator DE6.',
    'ROC_15': 'Rate of Change over 15 days.',
    'ROC_10': 'Rate of Change over 10 days.',
    'ROC_20': 'Rate of Change over 20 days.',
    'DE5': 'Economic indicator DE5.',
    'DAAA': 'Moody’s Seasoned Aaa Corporate Bond Yield.'
}

# Define reasonable ranges for sample values
sample_ranges = {
    'EMA_20': (950, 1050),
    'EMA_200': (900, 1100),
    'EMA_10': (960, 1040),
    'EMA_50': (940, 1060),
    'Year': (2020, 2025),
    'DTB3': (1.0, 5.0),
    'DTB6': (1.0, 5.5),
    'DE2': (100, 200),
    'DGS5': (1.5, 4.0),
    'Month': (1, 12),
    'TE1': (100, 300),
    'DE1': (100, 300),
    'DTB4WK': (1.0, 4.5),
    'DE4': (50, 250),
    'DE6': (100, 300),
    'ROC_15': (-5, 5),
    'ROC_10': (-5, 5),
    'ROC_20': (-5, 5),
    'DE5': (50, 200),
    'DAAA': (3.0, 5.0)
}

st.title("Stock Market Prediction")

st.write("Please enter values below. Or click Autofill to test quickly.")

# Button to trigger autofill
autofill = st.button("Autofill with Sample Data")

# Collect inputs
features = []
for feature_name, explanation in feature_info.items():
    if autofill:
        # Randomly generate a sample value within range
        low, high = sample_ranges[feature_name]
        if isinstance(low, int) and isinstance(high, int):
            val = random.randint(low, high)
        else:
            val = round(random.uniform(low, high), 2)
    else:
        val = 0.0

    val = st.number_input(
        label=f"{feature_name} ℹ️",
        value=val,
        help=explanation
    )
    features.append(val)

input_array = np.array(features).reshape(1, -1)

if st.button("Predict"):
    prediction = model.predict(input_array)
    st.success(f"Predicted value: {prediction[0]:.4f}")
