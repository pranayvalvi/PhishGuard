import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import torch
import sqlite3
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from datetime import datetime

# --- Setup & Caching ---
st.set_page_config(page_title="PhishGuard 2.0", page_icon="🛡️", layout="wide")

def init_db():
    conn = sqlite3.connect("phishguard.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS scans
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME,
                  sender TEXT,
                  subject TEXT,
                  prediction TEXT,
                  confidence REAL,
                  risk_level TEXT,
                  model_used TEXT)''')
    conn.commit()
    return conn

conn = init_db()

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

# --- UI Layout ---
st.sidebar.title("🛡️ PhishGuard 2.0")
page = st.sidebar.radio("Navigation", ["📊 Security Dashboard", "🔍 Manual Scanner"])

if page == "📊 Security Dashboard":
    st.title("📊 Security Dashboard")
    
    # Fetch stats from DB
    df = pd.read_sql_query("SELECT * FROM scans ORDER BY timestamp DESC", conn)
    
    if df.empty:
        st.info("No emails scanned yet. Use the Manual Scanner to get started!")
    else:
        total = len(df)
        phishing = len(df[df['prediction'] == 'PHISHING'])
        legit = len(df[df['prediction'] == 'LEGITIMATE'])
        high_risk = len(df[df['risk_level'] == 'HIGH'])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Scanned", total)
        col2.metric("Phishing Detected", phishing)
        col3.metric("Legitimate", legit)
        col4.metric("High Risk Emails", high_risk)
        
        st.markdown("### 🕒 Recent Scans")
        
        # Formatting for UI
        display_df = df[['timestamp', 'sender', 'subject', 'prediction', 'confidence', 'risk_level']].copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2f}%")
        
        # Color coding
        def color_risk(val):
            color = 'red' if val == 'PHISHING' else 'green'
            return f'color: {color}'
            
        st.dataframe(display_df.style.map(color_risk, subset=['prediction']), use_container_width=True)

elif page == "🔍 Manual Scanner":
    st.title("🔍 Manual Email Scanner")
    st.markdown("Analyze an email and store the result in the local security database.")

    st.sidebar.header("Scanner Settings")
    model_choice = st.sidebar.selectbox(
        "Choose the NLP Model:",
        ["Logistic Regression (TF-IDF)", "SVM (TF-IDF)", "Random Forest (TF-IDF + Metadata)", "DistilBERT (Contextual)"]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        sender_input = st.text_input("Sender Email (Optional):", value="unknown@example.com")
    with col2:
        subject_input = st.text_input("Subject (Optional):", value="No Subject")

    email_input = st.text_area("Paste the raw email content here:", height=200)

    if st.button("Analyze & Log Email", type="primary"):
        if not email_input.strip():
            st.warning("Please paste an email to analyze.")
        else:
            with st.spinner(f"Analyzing using {model_choice}..."):
                # 1. Feature Extraction
                meta = extract_metadata(email_input)
                cleaned = clean_text(email_input)
                
                prediction_val = 0
                confidence = 0.0
                
                # 2. Routing based on model choice
                if "DistilBERT" in model_choice:
                    tokenizer, model = load_transformer()
                    inputs = tokenizer(email_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        prediction_val = torch.argmax(probs).item()
                        confidence = probs[0][prediction_val].item() * 100
                else:
                    tfidf, scaler, lr, svm, rf = load_classical_components()
                    X_tfidf = tfidf.transform([cleaned])
                    
                    if "Logistic" in model_choice:
                        prediction_val = lr.predict(X_tfidf)[0]
                        confidence = np.max(lr.predict_proba(X_tfidf)) * 100
                    elif "SVM" in model_choice:
                        prediction_val = svm.predict(X_tfidf)[0]
                        score = svm.decision_function(X_tfidf)[0]
                        prob = 1 / (1 + np.exp(-score))
                        confidence = (prob if prediction_val == 1 else (1 - prob)) * 100
                    elif "Random Forest" in model_choice:
                        meta_df = pd.DataFrame([meta])
                        X_meta = scaler.transform(meta_df[['text_length', 'uppercase_ratio', 'exclamation_count', 'url_count', 'email_address_count', 'html_presence', 'urgency_word_count']])
                        X_rf = hstack([X_tfidf, X_meta])
                        prediction_val = rf.predict(X_rf)[0]
                        confidence = np.max(rf.predict_proba(X_rf)) * 100

                # 3. Calculate Risk Score
                prediction_str = "PHISHING" if prediction_val == 1 else "LEGITIMATE"
                
                # Boost risk if URLs + Urgency are present
                risk_score = confidence if prediction_val == 1 else (100 - confidence)
                if meta['url_count'] > 0 and meta['urgency_word_count'] > 0:
                    risk_score = min(100, risk_score + 15)
                
                risk_level = "LOW"
                if risk_score > 85 and prediction_val == 1:
                    risk_level = "HIGH"
                elif risk_score > 60 and prediction_val == 1:
                    risk_level = "MEDIUM"

                # 4. Save to Database
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                c = conn.cursor()
                c.execute('''INSERT INTO scans (timestamp, sender, subject, prediction, confidence, risk_level, model_used)
                             VALUES (?, ?, ?, ?, ?, ?, ?)''', 
                          (timestamp, sender_input, subject_input, prediction_str, risk_score, risk_level, model_choice))
                conn.commit()

                # 5. Display Results
                st.markdown("---")
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.subheader("Classification Result")
                    if prediction_val == 1:
                        st.error(f"🚨 **{prediction_str}** 🚨")
                        st.warning(f"**Risk Level:** {risk_level}")
                    else:
                        st.success(f"✅ **{prediction_str}** ✅")
                        st.info(f"**Risk Level:** {risk_level}")
                        
                    st.metric(label="Calculated Risk Score", value=f"{risk_score:.2f}/100")

                with col_res2:
                    st.subheader("Detected Indicators")
                    st.markdown(f"- **URLs Found:** {meta['url_count']}")
                    st.markdown(f"- **HTML Detected:** {'Yes' if meta['html_presence'] else 'No'}")
                    st.markdown(f"- **Urgency Words:** {meta['urgency_word_count']}")
                    st.markdown(f"- **Exclamation Marks:** {meta['exclamation_count']}")
                    st.markdown(f"- **Uppercase Ratio:** {meta['uppercase_ratio']*100:.1f}%")
