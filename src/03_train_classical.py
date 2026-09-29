import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
import joblib
import os

def train_models():
    print("[+] Loading cleaned datasets...")
    train_df = pd.read_csv("data/processed/train_cleaned.csv")
    test_df = pd.read_csv("data/processed/test_cleaned.csv")
    
    # Handle empty strings that became NaNs during cleaning
    train_df['cleaned_text'] = train_df['cleaned_text'].fillna('')
    test_df['cleaned_text'] = test_df['cleaned_text'].fillna('')
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    print("[+] Fitting TF-IDF Vectorizer (max_features=5000)...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(train_df['cleaned_text'])
    X_test_tfidf = tfidf.transform(test_df['cleaned_text'])
    
    # Save vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")
    print("    -> Saved TF-IDF vectorizer to models/tfidf_vectorizer.joblib")
    
    # 1. Logistic Regression
    print("\n[+] Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)
    joblib.dump(lr_model, "models/lr_model.joblib")
    
    # Quick validation (Provisional!)
    lr_preds = lr_model.predict(X_test_tfidf)
    print(f"    -> LR Provisional Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
    
    # 2. SVM
    print("\n[+] Training SVM (LinearSVC)...")
    # dual="auto" is recommended for newer sklearn to avoid warnings, but we'll use default LinearSVC for robustness.
    svm_model = LinearSVC(random_state=42, dual=False)
    svm_model.fit(X_train_tfidf, y_train)
    joblib.dump(svm_model, "models/svm_model.joblib")
    
    svm_preds = svm_model.predict(X_test_tfidf)
    print(f"    -> SVM Provisional Accuracy: {accuracy_score(y_test, svm_preds):.4f}")
    
    # 3. Random Forest (TF-IDF + Metadata)
    print("\n[+] Preparing Features for Random Forest...")
    meta_cols = ['text_length', 'uppercase_ratio', 'exclamation_count', 'url_count', 'email_address_count', 'html_presence', 'urgency_word_count']
    
    # Scale metadata
    scaler = StandardScaler()
    X_train_meta = scaler.fit_transform(train_df[meta_cols])
    X_test_meta = scaler.transform(test_df[meta_cols])
    joblib.dump(scaler, "models/meta_scaler.joblib")
    
    # Concatenate TF-IDF and metadata
    print("    -> Concatenating TF-IDF and Metadata matrices...")
    X_train_rf = hstack([X_train_tfidf, X_train_meta])
    X_test_rf = hstack([X_test_tfidf, X_test_meta])
    
    print("[+] Training Random Forest (n_estimators=100)...")
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf_model.fit(X_train_rf, y_train)
    joblib.dump(rf_model, "models/rf_model.joblib")
    
    rf_preds = rf_model.predict(X_test_rf)
    print(f"    -> RF Provisional Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    
    print("\n[+] Phase 3 Complete! All classical models and preprocessors saved to 'models/'")

if __name__ == "__main__":
    train_models()
