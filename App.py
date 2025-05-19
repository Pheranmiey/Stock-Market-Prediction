import streamlit as st
import pickle
import numpy as np

# Your new selected features
selected_features = [
    'EMA_20', 'EMA_200', 'EMA_10', 'EMA_50', 'Year',
    'DTB3', 'DTB6', 'DE2', 'DGS5', 'Month',
    'TE1', 'DE1', 'DTB4WK', 'DE4', 'DE6',
    'ROC_15', 'ROC_10', 'ROC_20', 'DE5', 'DAAA'
]

# Feature explanations - update or add as you see fit
feature_info = {
    'EMA_20': 'Exponential Moving Average over 20 days.',
    'EMA_200': 'Exponential Moving Average over 200 days.',
    'EMA_10': 'Exponential Moving Average over 10 days.',
    'EMA_50': 'Exponential Moving Average over 50 days.',
    'Year': 'The calendar year of the data point (e.g., 2023).',
    'DTB3': '3-Month Treasury Bill Rate (percent).',
    'DTB6': '6-Month Treasury Bill Rate (percent).',
    'DE2': 'Economic indicator DE2 - specific to dataset domain.',
    'DGS5': '5-Year Treasury Constant Maturity Rate (percent).',
    'Month': 'Month of the year as a number (1-12).',
    'TE1': 'Economic indicator TE1 - specific to dataset domain.',
    'DE1': 'Economic indicator DE1 - specific to dataset domain.',
    'DTB4WK': '4-Week Treasury Bill Rate (percent).',
    'DE4': 'Economic indicator DE4 - specific to dataset domain.',
    'DE6': 'Economic indicator DE6 - specific to dataset domain.',
    'ROC_15': 'Rate of Change over 15 days.',
    'ROC_10': 'Rate of Change over 10 days.',
    'ROC_20': 'Rate of Change over 20 days.',
    'DE5': 'Economic indicator DE5 - specific to dataset domain.',
    'DAAA': 'Economic indicator DAAA - specific to dataset domain.'
}

# Cache the model loading to improve performance
@st.cache_resource
def load_model():
    with open('catboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("Stock Market Prediction")
st.write("Please enter values below. Hover over the ℹ️ icon for explanations.")

input_values = []
for feature in selected_features:
    explanation = feature_info.get(feature, "No explanation available.")
    val = st.number_input(
        label=f"{feature} ℹ️",
        value=0.0,
        help=explanation
    )
    input_values.append(val)

input_array = np.array(input_values).reshape(1, -1)

if st.button("Predict"):
    prediction = model.predict(input_array)
    st.success(f"Predicted value: {prediction[0]:.4f}")
