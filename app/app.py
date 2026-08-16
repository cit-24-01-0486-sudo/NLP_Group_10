
import streamlit as st
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
import numpy as np

st.set_page_config(
    page_title="Fake News Detector - Member 3",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection")
st.caption("Member 3 – SVM + DistilBERT (Week 6 models)")

def simple_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def load_svm():
    model = joblib.load("models/m3_svm_model.pkl")
    vectorizer = joblib.load("models/m3_tfidf_vectorizer.pkl")
    return model, vectorizer

@st.cache_resource
def load_distilbert():
    model = DistilBertForSequenceClassification.from_pretrained("models/m3_bert_model")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model.eval()
    return model, tokenizer

def predict_svm(text, model, vectorizer):
    cleaned = simple_clean(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    decision = model.decision_function(vec)[0]
    confidence = 1 / (1 + np.exp(-abs(decision)))
    label = "FAKE" if pred == 1 else "REAL"
    return label, float(confidence)

def predict_distilbert(text, model, tokenizer):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
    pred = int(np.argmax(probs))
    label = "FAKE" if pred == 1 else "REAL"
    confidence = float(probs[pred])
    return label, confidence

model_choice = st.selectbox(
    "Choose model",
    ["SVM (LinearSVC)", "DistilBERT"]
)

user_text = st.text_area(
    "Paste a news article here",
    height=200,
    placeholder="Paste the full news text..."
)

if st.button("Analyse", type="primary"):
    if not user_text.strip():
        st.warning("Please paste some text first.")
    else:
        with st.spinner("Analysing..."):
            try:
                if model_choice == "SVM (LinearSVC)":
                    model, vectorizer = load_svm()
                    label, conf = predict_svm(user_text, model, vectorizer)
                else:
                    model, tokenizer = load_distilbert()
                    label, conf = predict_distilbert(user_text, model, tokenizer)

                if label == "FAKE":
                    st.error(f"**Prediction: {label}**")
                else:
                    st.success(f"**Prediction: {label}**")

                st.progress(conf)
                st.write(f"Confidence: **{conf:.1%}**")

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Check that models/m3_svm_model.pkl, models/m3_tfidf_vectorizer.pkl and models/m3_bert_model/ exist.")

st.markdown("---")
st.caption("NLP Group 10 | Member 3 (CIT-24-0598) – Week 6 models only")
