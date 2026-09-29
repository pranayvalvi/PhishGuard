# 🛡️ PhishGuard: NLP Email Scam Detector

Welcome to **PhishGuard**, an end-to-end Machine Learning pipeline and Streamlit application designed to classify emails as **Phishing** or **Legitimate**.

This project compares Classical NLP techniques (TF-IDF + ML Algorithms) against Modern Contextual NLP (Transformers) to find the optimal architecture for minimizing False Negative Rates (FNR) in scam detection.

## 📁 Project Structure

```text
PhishGuard/
│
├── data/
│   ├── raw/                  # Place your raw 'dataset.csv' here
│   └── processed/            # Cleaned and split datasets (train/test)
│
├── models/                   # Saved models, scalers, and vectorizers
│   ├── distilbert_model/     # Fine-tuned Hugging Face transformer
│   ├── lr_model.joblib       # Logistic Regression model
│   ├── svm_model.joblib      # Linear Support Vector Machine
│   ├── rf_model.joblib       # Random Forest (with Metadata Features)
│   └── confusion_matrices.png# Final evaluation visualizations
│
├── src/                      # Source code pipeline
│   ├── 01_data_prep.py           # Handles deduplication and stratified splits
│   ├── 02_nlp_preprocessing.py   # Extracts metadata & cleans text (Lemmatization)
│   ├── 03_train_classical.py     # Trains LR, SVM, and RF on TF-IDF
│   ├── 04_train_transformer.py   # Fine-tunes DistilBERT on raw text
│   ├── 05_evaluate.py            # Rigorous scientific testing & metric generation
│   └── 06_streamlit_app.py       # User Interface for real-time predictions
│
├── requirements.txt          # Project dependencies
├── run_phishguard.bat        # 1-Click launcher for the Streamlit App
└── PhishGuard_Project_Report.md  # Comprehensive findings and methodology report
```

## 🚀 How to Run the App

If all models have been trained, simply execute the included batch file to launch the web interface:

1. Double-click `run_phishguard.bat`
2. Your default web browser will open to `http://localhost:8501`
3. Paste an email, select a model from the sidebar, and click **Analyze Email**.

## 🛠️ Re-Running the Pipeline

If you wish to retrain the models on a new dataset:

1. **Environment Setup:** Ensure your virtual environment is active and dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
2. **Provide Data:** Place your CSV file containing `text` and `label` columns into `data/raw/dataset.csv`.
3. **Execute Pipeline in Order:**
   ```bash
   python src/01_data_prep.py
   python src/02_nlp_preprocessing.py
   python src/03_train_classical.py
   python src/04_train_transformer.py
   python src/05_evaluate.py
   ```

*(Note: Phase 4 Transformer training defaults to a 500-sample prototype for CPU efficiency. Open `04_train_transformer.py` and change `sample_size=None` to train on the full dataset).*

## 🔬 Model Comparisons & Features

* **Logistic Regression:** Serves as the rapid TF-IDF baseline.
* **SVM (LinearSVC):** Selected for high-dimensional text space, achieving the lowest False Negative Rate.
* **Random Forest:** Enhanced with 7 custom metadata heuristics (URL count, HTML presence, exclamation ratios) appended to the TF-IDF matrix.
* **DistilBERT:** Bypasses TF-IDF entirely, utilizing deep contextual embeddings and subword tokenization for semantic understanding.

## ⚖️ Evaluation Metrics
We measure success based on Accuracy, Precision, Recall, F1-Score, ROC-AUC, and specifically **False Negative Rate (FNR)**—the most critical metric in phishing detection. See `PhishGuard_Project_Report.md` for full baseline results.
