import streamlit as st
from src.Fertility_predictor import FertilityPredictor
from src.Visualizations import Visualization
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.set_page_config(page_title="Female Infertility Prediction", page_icon="🏥", layout="centered")

st.title("🏥 Female Infertility Prediction System")
st.caption("This is a prediction tool, not a medical diagnosis. Please consult a qualified healthcare professional.")

@st.cache_resource
def get_predictor():
    return FertilityPredictor()

predictor = get_predictor()
viz = Visualization(predictor)

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.header("Model")

if st.sidebar.button("Train Model"):
    with st.spinner("Training..."):
        predictor.train()
        predictor.save()
    st.sidebar.success(f"Trained! Accuracy: {predictor.accuracy * 100:.2f}%")

if st.sidebar.button("Load Model"):
    if os.path.exists("fertility_model.pkl"):
        predictor.load()
        st.sidebar.success("Model loaded!")
    else:
        st.sidebar.error("No saved model found. Train first.")

if predictor.is_trained:
    st.sidebar.info(f"Model ready | Accuracy: {predictor.accuracy * 100:.2f}%")
else:
    st.sidebar.warning("Model not trained yet.")

# ── Patient Input ─────────────────────────────────────────
st.header("Patient Data")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    ovulation_disorders = st.selectbox("Ovulation Disorders", [0, 1])
    blocked_fallopian = st.selectbox("Blocked Fallopian Tubes", [0, 1])
    endometriosis = st.selectbox("Endometriosis", [0, 1])
    uterine_abnormalities = st.selectbox("Uterine Abnormalities", [0, 1])
    pelvic_inflammatory = st.selectbox("Pelvic Inflammatory Disease", [0, 1])

with col2:
    hormonal_imbalances = st.selectbox("Hormonal Imbalances", [0, 1])
    premature_ovarian = st.selectbox("Premature Ovarian Insufficiency", [0, 1])
    autoimmune = st.selectbox("Autoimmune Disorders", [0, 1])
    previous_surgeries = st.selectbox("Previous Reproductive Surgeries", [0, 1])
    unexplained = st.selectbox("Unexplained Infertility", [0, 1])

    patient_data = {
        "Age": age,
        "Ovulation Disorders": ovulation_disorders,
        "Blocked Fallopian Tubes": blocked_fallopian,
        "Endometriosis": endometriosis,
        "Uterine Abnormalities": uterine_abnormalities,
        "Pelvic Inflammatory Disease": pelvic_inflammatory,
        "Hormonal Imbalances": hormonal_imbalances,
        "Premature Ovarian Insufficiency": premature_ovarian,
        "Autoimmune Disorders": autoimmune,
        "Previous Reproductive Surgeries": previous_surgeries,
        "Unexplained Infertility": unexplained
    }

# ── Prediction ────────────────────────────────────────────
st.header("Prediction")

if st.button("Predict", type="primary"):
    if not predictor.is_trained:
        st.error("Train or load a model first.")
    else:
        result = predictor.predict_patient(patient_data)

        if result["prediction"] == 1:
            st.error(f"Prediction: **Infertile** | Confidence: {result['confidence']:.2f}%")
        else:
            st.success(f"Prediction: **Fertile** | Confidence: {result['confidence']:.2f}%")

        st.subheader("Probability Breakdown")
        for cls, prob in result["probabilities"].items():
            label = "Fertile" if cls == "0" else "Infertile"
            st.progress(int(prob), text=f"{label}: {prob:.1f}%")

        # ── Plot 1: Prediction Probability ──
        st.subheader("Prediction Probability")
        fig1, ax1 = plt.subplots()
        probs = result["probabilities"]
        classes = ["Fertile" if c == "0" else "Infertile" for c in probs.keys()]
        values = list(probs.values())
        colors = ["#ffadbe" if str(result["prediction"]) == c else "#5bc0de" for c in probs.keys()]
        ax1.barh(classes, values, color=colors)
        ax1.set_xlabel("Probability (%)")
        ax1.set_title("Prediction Probability")
        st.pyplot(fig1)

        # ── Plot 2: Patient vs Average ──
        st.subheader("Your Values vs Dataset Average")
        df = predictor.loader.dataframe
        features = predictor.original_features[:8]
        features = [f for f in features if f in df.columns]

        dataset_means = df[features].mean()
        patient_vals = {f: float(result["patient_data"][f]) for f in features}
        patient_series = pd.Series(patient_vals)
        combined_max = pd.concat([dataset_means, patient_series]).groupby(level=0).max().replace(0, 1)
        norm_avg = dataset_means / combined_max
        norm_pat = patient_series / combined_max

        x = np.arange(len(features))
        width = 0.35
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.bar(x - width / 2, norm_avg, width, label="Dataset Average", color="#5bc0de")
        ax2.bar(x + width / 2, norm_pat, width, label="Your Values", color="#ffadbe")
        ax2.set_xticks(x)
        ax2.set_xticklabels(features, rotation=30, ha="right")
        ax2.set_ylabel("Normalised Value")
        ax2.set_title("Your Values vs Dataset Average")
        ax2.legend()
        st.pyplot(fig2)

        # ── Plot 3: Feature Importance ──
        st.subheader("Feature Importance")
        importance_df = predictor.feature_importance(top_n=11)
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.barh(importance_df["feature"], importance_df["importance"], color="#5bc0de")
        ax3.set_xlabel("Importance")
        ax3.set_title("Top Feature Importances")
        ax3.invert_yaxis()
        st.pyplot(fig3)