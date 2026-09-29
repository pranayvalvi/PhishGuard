import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# --- Setup & Caching ---
st.set_page_config(page_title="PhishGuard Detector", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_nltk():
    try:
        stopwords.words('english')
    except:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    return set(stopwords.words('english')), WordNetLemmatizer()

stop_words, lemmatizer = load_nltk()

@st.cache_resource
def load_classical_components():
    try:
        tfidf = joblib.load("models/tfidf_vectorizer.joblib")
        scaler = joblib.load("models/meta_scaler.joblib")
        lr = joblib.load("models/lr_model.joblib")
        svm = joblib.load("models/svm_model.joblib")
        rf = joblib.load("models/rf_model.joblib")
        return tfidf, scaler, lr, svm, rf
    except FileNotFoundError:
        st.error("Classical models not found. Please run '03_train_classical.py' first.")
        st.stop()

@st.cache_resource
def load_transformer():
    try:
        tokenizer = AutoTokenizer.from_pretrained("models/distilbert_model")
        model = AutoModelForSequenceClassification.from_pretrained("models/distilbert_model")
        return tokenizer, model
    except OSError:
        st.error("DistilBERT model not found. Please run '04_train_transformer.py' first.")
        st.stop()

# --- Helper Functions ---
def extract_metadata(text):
    text_length = len(text)
    upper_count = sum(1 for c in text if c.isupper())
    uppercase_ratio = upper_count / text_length if text_length > 0 else 0
    exclamation_count = text.count('!')
    url_count = len(re.findall(r'http[s]?://|www\.', text))
    email_address_count = len(re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text))
    html_presence = 1 if re.search(r'<\s*[a-z][\s\S]*>', text, re.IGNORECASE) else 0
    
    urgency_words = ['urgent', 'immediate', 'action required', 'account suspended', 'verify', 'update', 'important']
    urgency_pattern = '|'.join([r'\b{}\b'.format(w) for w in urgency_words])
    urgency_word_count = len(re.findall(urgency_pattern, text.lower()))
    
    return {
        'text_length': text_length,
        'uppercase_ratio': uppercase_ratio,
        'exclamation_count': exclamation_count,
        'url_count': url_count,
        'email_address_count': email_address_count,
        'html_presence': html_presence,
        'urgency_word_count': urgency_word_count
    }

def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    tokens = text.split()
    cleaned = " ".join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words])
    return cleaned

# --- UI ---
st.title("🛡️ PhishGuard: NLP Email Scam Detector")
st.markdown("Analyze an email to determine if it is **Phishing (Scam)** or **Legitimate (Safe)**.")

st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose the NLP Model:",
    ["Logistic Regression (TF-IDF)", "SVM (TF-IDF)", "Random Forest (TF-IDF + Metadata)", "DistilBERT (Contextual)"]
)

email_input = st.text_area("Paste the raw email content here:", height=250)

if st.button("Analyze Email", type="primary"):
    if not email_input.strip():
        st.warning("Please paste an email to analyze.")
    else:
        with st.spinner(f"Analyzing using {model_choice}..."):
            # 1. Feature Extraction
            meta = extract_metadata(email_input)
            cleaned = clean_text(email_input)
            
            prediction = 0
            confidence = 0.0
            
            # 2. Routing based on model choice
            if "DistilBERT" in model_choice:
                tokenizer, model = load_transformer()
                inputs = tokenizer(email_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    prediction = torch.argmax(probs).item()
                    confidence = probs[0][prediction].item() * 100
            else:
                tfidf, scaler, lr, svm, rf = load_classical_components()
                X_tfidf = tfidf.transform([cleaned])
                
                if "Logistic" in model_choice:
                    prediction = lr.predict(X_tfidf)[0]
                    confidence = np.max(lr.predict_proba(X_tfidf)) * 100
                elif "SVM" in model_choice:
                    prediction = svm.predict(X_tfidf)[0]
                    # Convert decision function to pseudo-probability
                    score = svm.decision_function(X_tfidf)[0]
                    prob = 1 / (1 + np.exp(-score))
                    confidence = (prob if prediction == 1 else (1 - prob)) * 100
                elif "Random Forest" in model_choice:
                    meta_df = pd.DataFrame([meta])
                    X_meta = scaler.transform(meta_df[['text_length', 'uppercase_ratio', 'exclamation_count', 'url_count', 'email_address_count', 'html_presence', 'urgency_word_count']])
                    X_rf = hstack([X_tfidf, X_meta])
                    prediction = rf.predict(X_rf)[0]
                    confidence = np.max(rf.predict_proba(X_rf)) * 100

            # 3. Display Results
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.error("🚨 **PHISHING DETECTED** 🚨")
                else:
                    st.success("✅ **LEGITIMATE EMAIL** ✅")
                    
                st.metric(label="Model Confidence", value=f"{confidence:.2f}%")

            with col2:
                st.subheader("Extracted Indicators")
                st.markdown(f"- **URLs Found:** {meta['url_count']}")
                st.markdown(f"- **HTML Detected:** {'Yes' if meta['html_presence'] else 'No'}")
                st.markdown(f"- **Urgency Words:** {meta['urgency_word_count']}")
                st.markdown(f"- **Exclamation Marks:** {meta['exclamation_count']}")
                st.markdown(f"- **Uppercase Ratio:** {meta['uppercase_ratio']*100:.1f}%")
                
            with st.expander("View Cleaned Text (For Classical Models)"):
                st.write(cleaned)
