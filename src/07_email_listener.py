import imaplib
import email
import sqlite3
import os
import re
import time
import joblib
import pandas as pd
import numpy as np
from email.header import decode_header
from dotenv import load_dotenv
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import datetime

# Load environment variables
load_dotenv()
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
EMAIL_ACCOUNT = os.getenv("EMAIL_ACCOUNT")
APP_PASSWORD = os.getenv("APP_PASSWORD")

if not EMAIL_ACCOUNT or not APP_PASSWORD:
    print("[!] Error: EMAIL_ACCOUNT and APP_PASSWORD must be set in the .env file.")
    exit(1)

# Ensure NLTK resources
try:
    stopwords.words('english')
except:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Database setup
conn = sqlite3.connect("phishguard.db")

print("[+] Loading the SVM Model for rapid scanning...")
try:
    tfidf = joblib.load("models/tfidf_vectorizer.joblib")
    svm = joblib.load("models/svm_model.joblib")
except Exception as e:
    print(f"[!] Error loading models: {e}. Please run the training scripts first.")
    exit(1)

def extract_metadata(text):
    text_length = len(text)
    upper_count = sum(1 for c in text if c.isupper())
    uppercase_ratio = upper_count / text_length if text_length > 0 else 0
    exclamation_count = text.count('!')
    url_count = len(re.findall(r'http[s]?://|www\.', text))
    html_presence = 1 if re.search(r'<\s*[a-z][\s\S]*>', text, re.IGNORECASE) else 0
    
    urgency_words = ['urgent', 'immediate', 'action required', 'account suspended', 'verify', 'update', 'important']
    urgency_pattern = '|'.join([r'\b{}\b'.format(w) for w in urgency_words])
    urgency_word_count = len(re.findall(urgency_pattern, text.lower()))
    
    return {
        'text_length': text_length,
        'uppercase_ratio': uppercase_ratio,
        'exclamation_count': exclamation_count,
        'url_count': url_count,
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

def process_email(sender, subject, body):
    meta = extract_metadata(body)
    cleaned = clean_text(body)
    
    # Vectorize and predict
    X_tfidf = tfidf.transform([cleaned])
    prediction_val = svm.predict(X_tfidf)[0]
    score = svm.decision_function(X_tfidf)[0]
    prob = 1 / (1 + np.exp(-score))
    confidence = (prob if prediction_val == 1 else (1 - prob)) * 100
    
    prediction_str = "PHISHING" if prediction_val == 1 else "LEGITIMATE"
    
    # Calculate Risk Score
    risk_score = confidence if prediction_val == 1 else (100 - confidence)
    if meta['url_count'] > 0 and meta['urgency_word_count'] > 0:
        risk_score = min(100, risk_score + 15)
    
    risk_level = "LOW"
    if risk_score > 85 and prediction_val == 1:
        risk_level = "HIGH"
    elif risk_score > 60 and prediction_val == 1:
        risk_level = "MEDIUM"
        
    # Save to Database
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c = conn.cursor()
    c.execute('''INSERT INTO scans (timestamp, sender, subject, prediction, confidence, risk_level, model_used)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''', 
              (timestamp, sender, subject, prediction_str, risk_score, risk_level, "SVM (Automated Listener)"))
    conn.commit()
    print(f"\n  [✔] Scanned: {subject} | Result: {prediction_str} (Risk: {risk_score:.2f})")

def listen_for_emails():
    print(f"[+] Connecting to {IMAP_SERVER}...")
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ACCOUNT, APP_PASSWORD)
    except Exception as e:
        print(f"[!] Login failed: {e}")
        return

    print(f"[+] Logged in as {EMAIL_ACCOUNT}. Listening for new emails...")

    while True:
        try:
            mail.select("inbox")
            status, messages = mail.search(None, "UNSEEN")
            
            email_ids = messages[0].split()
            if email_ids:
                print(f"\n[!] Found {len(email_ids)} new unread emails! Processing...")
                
                for e_id in email_ids:
                    _, msg_data = mail.fetch(e_id, "(RFC822)")
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            
                            # Decode Subject
                            subject, encoding = decode_header(msg["Subject"])[0]
                            if isinstance(subject, bytes):
                                subject = subject.decode(encoding if encoding else "utf-8", errors='replace')
                            
                            sender = msg.get("From")
                            
                            # Get Body
                            body = ""
                            if msg.is_multipart():
                                for part in msg.walk():
                                    if part.get_content_type() == "text/plain":
                                        body = part.get_payload(decode=True).decode(errors='replace')
                                        break
                            else:
                                body = msg.get_payload(decode=True).decode(errors='replace')
                            
                            process_email(sender, subject, body)
            else:
                # Print a small dot every 10 seconds to show it's alive
                print(".", end="", flush=True)
            
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n[+] Stopping Listener.")
            break
        except Exception as e:
            print(f"\n[!] Error: {e}")
            time.sleep(10)
            
    mail.logout()

if __name__ == "__main__":
    listen_for_emails()
