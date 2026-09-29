import pandas as pd
import numpy as np
import joblib
import torch
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.sparse import hstack
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os

warnings.filterwarnings("ignore")

def compute_metrics_dict(y_true, y_pred, y_prob):
    # Confusion matrix to get FNR
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob) if y_prob is not None else "N/A",
        "False Negative Rate": fnr
    }

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

def evaluate_models():
    print("[+] Loading test data (Sampling 2000 for fair, fast CPU comparison)...")
    test_df = pd.read_csv("data/processed/test_cleaned.csv").sample(min(2000, 16000), random_state=42)
    test_df['cleaned_text'] = test_df['cleaned_text'].fillna('')
    test_df['text'] = test_df['text'].fillna('')
    
    y_test = test_df['label'].values
    results = {}
    
    print("[+] Loading preprocessors for classical models...")
    tfidf = joblib.load("models/tfidf_vectorizer.joblib")
    meta_scaler = joblib.load("models/meta_scaler.joblib")
    
    X_test_tfidf = tfidf.transform(test_df['cleaned_text'])
    meta_cols = ['text_length', 'uppercase_ratio', 'exclamation_count', 'url_count', 'email_address_count', 'html_presence', 'urgency_word_count']
    X_test_meta = meta_scaler.transform(test_df[meta_cols])
    X_test_rf = hstack([X_test_tfidf, X_test_meta])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # 1. Logistic Regression
    print("[+] Evaluating Logistic Regression...")
    lr_model = joblib.load("models/lr_model.joblib")
    lr_preds = lr_model.predict(X_test_tfidf)
    lr_probs = lr_model.predict_proba(X_test_tfidf)[:, 1]
    results['Logistic Regression'] = compute_metrics_dict(y_test, lr_preds, lr_probs)
    sns.heatmap(confusion_matrix(y_test, lr_preds), annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Logistic Regression')
    
    # 2. SVM
    print("[+] Evaluating SVM (LinearSVC)...")
    svm_model = joblib.load("models/svm_model.joblib")
    svm_preds = svm_model.predict(X_test_tfidf)
    # LinearSVC doesn't have predict_proba by default, use decision_function
    svm_scores = svm_model.decision_function(X_test_tfidf)
    # Convert scores to pseudo-probabilities using sigmoid just for AUC
    svm_probs = 1 / (1 + np.exp(-svm_scores))
    results['SVM'] = compute_metrics_dict(y_test, svm_preds, svm_probs)
    sns.heatmap(confusion_matrix(y_test, svm_preds), annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('SVM')
    
    # 3. Random Forest
    print("[+] Evaluating Random Forest...")
    rf_model = joblib.load("models/rf_model.joblib")
    rf_preds = rf_model.predict(X_test_rf)
    rf_probs = rf_model.predict_proba(X_test_rf)[:, 1]
    results['Random Forest'] = compute_metrics_dict(y_test, rf_preds, rf_probs)
    sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_title('Random Forest')
    
    # 4. DistilBERT
    print("[+] Evaluating DistilBERT (This will take a moment)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("./models/distilbert_model")
        model = AutoModelForSequenceClassification.from_pretrained("./models/distilbert_model")
        
        test_texts = test_df['text'].tolist()
        encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)
        test_dataset = TestDataset(encodings)
        
        trainer = Trainer(model=model)
        predictions = trainer.predict(test_dataset)
        
        # Softmax for probabilities
        bert_probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()
        bert_preds = predictions.predictions.argmax(-1)
        
        results['DistilBERT'] = compute_metrics_dict(y_test, bert_preds, bert_probs)
        sns.heatmap(confusion_matrix(y_test, bert_preds), annot=True, fmt='d', cmap='Blues', ax=axes[3])
        axes[3].set_title('DistilBERT')
        
    except Exception as e:
        print(f"[-] DistilBERT evaluation failed: {e}")
        results['DistilBERT'] = {"Error": "Model not found or failed"}
    
    # Save Confusion Matrices Plot
    plt.tight_layout()
    plt.savefig("models/confusion_matrices.png")
    print("\n[+] Saved Confusion Matrices to 'models/confusion_matrices.png'")
    
    # Print Results Table
    results_df = pd.DataFrame(results).T
    print("\n--- FINAL EVALUATION METRICS ---\n")
    print(results_df.to_string())
    
    # Save to CSV
    results_df.to_csv("models/evaluation_metrics.csv")
    print("\n[+] Phase 5 Complete! Results saved.")

if __name__ == "__main__":
    evaluate_models()
