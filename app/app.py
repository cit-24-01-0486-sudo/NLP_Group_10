# =============================================================
# app.py
# CCS3356 - Natural Language Processing
# NLP_Group_10 - Fake News Detection
# Streamlit Web Application
# =============================================================

import streamlit as st
import pickle
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import preprocess_text

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    vectorizers = {}

    try:
        with open('models/m2_nb_model.pkl', 'rb') as f:
            models['Naïve Bayes (M2)'] = pickle.load(f)
        with open('models/m2_nb_vectorizer.pkl', 'rb') as f:
            vectorizers['Naïve Bayes (M2)'] = pickle.load(f)
        st.sidebar.success("✅ Naïve Bayes loaded")
    except:
        st.sidebar.warning("⚠️ Naïve Bayes model not found")

    return models, vectorizers

models, vectorizers = load_models()

# ── Header ────────────────────────────────────────────────────
st.title("🔍 Fake News Detection System")
st.markdown("**CCS3356 Natural Language Processing | NLP_Group_10**")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Detect", "📊 Model Comparison"])

# ── Tab 1 — Detection ─────────────────────────────────────────
with tab1:
    st.subheader("Paste a news article to check if it is real or fake")

    # Sample articles
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load sample FAKE article"):
            st.session_state['input_text'] = """
            SHOCKING: Secret government documents reveal massive conspiracy
            to control the population through water supply. Anonymous sources
            confirm that world leaders met last week to discuss plans that
            would affect billions of people. Share this before it gets deleted!
            """
    with col2:
        if st.button("Load sample REAL article"):
            st.session_state['input_text'] = """
            The Federal Reserve raised interest rates by 25 basis points on
            Wednesday, citing continued progress on inflation while acknowledging
            risks to economic growth. Fed Chair Jerome Powell said the committee
            remains committed to returning inflation to its 2 percent target.
            """

    # Text input
    input_text = st.text_area(
        "Enter news article text here:",
        value=st.session_state.get('input_text', ''),
        height=200,
        placeholder="Paste a news article or headline here..."
    )

    # Model selector
    model_choice = st.selectbox(
        "Select model:",
        list(models.keys()) if models else ["No models loaded"]
    )

    # Analyse button
    if st.button("🔍 Analyse", type="primary"):
        if not input_text.strip():
            st.warning("Please enter some text first!")
        elif model_choice not in models:
            st.error("Selected model not available!")
        else:
            with st.spinner("Analysing..."):
                # Preprocess
                cleaned = preprocess_text(input_text)

                # Vectorize and predict
                vectorizer = vectorizers[model_choice]
                model = models[model_choice]
                vec = vectorizer.transform([cleaned])
                prediction = model.predict(vec)[0]
                probability = model.predict_proba(vec)[0]
                confidence = probability[prediction] * 100

                # Display result
                st.markdown("---")
                if prediction == 1:
                    st.error(f"## 🚨 FAKE NEWS")
                    st.markdown(f"**Confidence: {confidence:.1f}%**")
                    st.progress(confidence / 100)
                else:
                    st.success(f"## ✅ REAL NEWS")
                    st.markdown(f"**Confidence: {confidence:.1f}%**")
                    st.progress(confidence / 100)

                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Fake Probability", f"{probability[1]*100:.1f}%")
                with col2:
                    st.metric("Real Probability", f"{probability[0]*100:.1f}%")

                st.caption("⚠️ This tool is for educational purposes only. Do not use as the sole basis for determining news credibility.")

# ── Tab 2 — Model Comparison ──────────────────────────────────
with tab2:
    st.subheader("Model Performance Comparison — NLP_Group_10")

    import pandas as pd
    comparison_data = {
        'Member': ['M1', 'M1', 'M2', 'M2', 'M3', 'M3'],
        'Model': ['Logistic Regression', 'CNN', 'Naïve Bayes', 'LSTM', 'SVM', 'DistilBERT'],
        'Type': ['ML', 'DL', 'ML', 'DL', 'ML', 'DL'],
        'Accuracy': ['TBD', 'TBD', '0.9700', '0.9969', 'TBD', 'TBD'],
        'F1-Score': ['TBD', 'TBD', '0.9671', '0.9970', 'TBD', 'TBD'],
        'ROC-AUC': ['TBD', 'TBD', '0.9850', '0.9993', 'TBD', 'TBD'],
    }

    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    ### Best Model Selection
    - **Current best (M2):** LSTM with GloVe embeddings — F1: 0.9970
    - Final best model will be updated once all members complete their models
    - Selection criteria: F1-Score and ROC-AUC
    """)

    st.caption("Results for M1 and M3 will be updated as members complete their models.")