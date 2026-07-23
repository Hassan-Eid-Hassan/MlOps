"""
Minimal test app for the live demo.

Run it locally while `kubectl port-forward svc/diabetes-model-server 8080:8080`
is active (or point MODEL_ENDPOINT at any other reachable deployment):

    streamlit run app/streamlit_app.py
"""
import os

import requests
import streamlit as st

MODEL_ENDPOINT = os.environ.get("MODEL_ENDPOINT", "http://localhost:8080/invocations")
FEATURES = ["AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6"]

st.set_page_config(page_title="Diabetes Progression Predictor", page_icon="🩺")
st.title("🩺 Diabetes Progression Predictor")
st.caption(f"Calling model server at: {MODEL_ENDPOINT}")

st.write(
    "These 10 features are already standardized (mean-centered, unit-scaled), "
    "the same way the original scikit-learn diabetes dataset is provided. "
    "The default values are a real sample from the dataset - just click Predict, "
    "or drag the sliders to see the prediction change."
)

defaults = [0.038, 0.051, 0.062, 0.022, -0.044, -0.035, -0.043, -0.003, 0.020, -0.018]

cols = st.columns(2)
values = []
for i, (feature, default) in enumerate(zip(FEATURES, defaults)):
    with cols[i % 2]:
        values.append(
            st.slider(feature, min_value=-0.2, max_value=0.2, value=float(default), step=0.001)
        )

if st.button("Predict", type="primary"):
    payload = {"dataframe_split": {"columns": FEATURES, "data": [values]}}
    try:
        response = requests.post(MODEL_ENDPOINT, json=payload, timeout=10)
        response.raise_for_status()
        prediction = response.json()["predictions"][0]
        st.success(f"Predicted disease progression score: **{prediction:.1f}**")
    except Exception as exc:
        st.error(f"Could not reach the model server: {exc}")
