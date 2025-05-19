import streamlit as st
import pickle
import numpy as np
import random

@st.cache_resource
def load_model():
    with open('catboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

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

# Use session state to manage inputs
if 'inputs' not in st.session_state:
    st.session_state.inputs = {key: 0.0 for key in feature_info}

# Button logic
col1, col2 = st.columns(2)
with col1:
    if st.button("Autofill with Sample Data"):
        for feature_name in feature_info:
            low, high = sample_ranges[feature_name]
            if isinstance(low, int) and isinstance(high, int):
                st.session_state.inputs[feature_name] = random.randint(low, high)
            else:
                st.session_state.inputs[feature_name] = round(random.uniform(low, high), 2)

with col2:
    if st.button("Reset"):
        st.session_state.inputs = {key: 0.0 for key in feature_info}

# Input fields
features = []
for feature_name, explanation in feature_info.items():
    val = st.number_input(
        label=f"{feature_name}",
        value=st.session_state.inputs.get(feature_name, 0.0),
        key=feature_name,
        help=explanation
    )
    features.append(val)
    st.session_state.inputs[feature_name] = val

input_array = np.array(features).reshape(1, -1)

if st.button("Predict"):
    prediction = model.predict(input_array)
    predicted_value = prediction[0]
    
    st.success(f"📈 Predicted Closing Price: ${predicted_value:.2f}")
    
    st.markdown("""
    **Note:** This is the estimated stock price at market close for the next trading day based on your inputs.
    """)
