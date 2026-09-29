import pandas as pd
import numpy as np
import re
import nltk
import warnings
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Suppress pandas chained assignment warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def extract_metadata(df):
    """Vectorized metadata extraction for performance."""
    print("      * Calculating text lengths...")
    df['text_length'] = df['text'].astype(str).str.len()
    
    print("      * Calculating uppercase ratio...")
    upper_counts = df['text'].astype(str).apply(lambda x: sum(1 for c in x if c.isupper()))
    df['uppercase_ratio'] = np.where(df['text_length'] > 0, upper_counts / df['text_length'], 0)
    
    print("      * Counting exclamation marks...")
    df['exclamation_count'] = df['text'].astype(str).str.count('!')
    
    print("      * Counting URLs...")
    df['url_count'] = df['text'].astype(str).str.count(r'http[s]?://|www\.')
    
    print("      * Counting Email addresses...")
    df['email_address_count'] = df['text'].astype(str).str.count(r'[\w\.-]+@[\w\.-]+\.\w+')
    
    print("      * Detecting HTML tags...")
    df['html_presence'] = df['text'].astype(str).str.contains(r'<\s*[a-z][\s\S]*>', case=False, regex=True).astype(int)
    
    print("      * Counting urgency words...")
    urgency_words = ['urgent', 'immediate', 'action required', 'account suspended', 'verify', 'update', 'important']
    # Join urgency words into a regex pattern
    urgency_pattern = '|'.join([r'\b{}\b'.format(w) for w in urgency_words])
    df['urgency_word_count'] = df['text'].astype(str).str.lower().str.count(urgency_pattern)
    
    return df

def clean_text_series(text_series):
    """Cleans the text series optimally."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    print("      * Removing HTML tags...")
    text_series = text_series.str.replace(r'<[^>]+>', ' ', regex=True)
    
    print("      * Removing URLs...")
    text_series = text_series.str.replace(r'http\S+|www\.\S+', ' ', regex=True)
    
    print("      * Removing Email Addresses...")
    text_series = text_series.str.replace(r'\S+@\S+', ' ', regex=True)
    
    print("      * Removing punctuation and numbers...")
    text_series = text_series.str.replace(r'[^a-zA-Z\s]', ' ', regex=True)
    
    print("      * Lowercasing text...")
    text_series = text_series.str.lower()
    
    print("      * Tokenizing, removing stopwords, and lemmatizing (this may take a minute)...")
    # Doing this part via apply because lemmatization requires iteration
    def tokenize_and_lemmatize(text):
        tokens = str(text).split()
        return " ".join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words])
        
    return text_series.apply(tokenize_and_lemmatize)

def process_data():
    print("[+] Loading train and test sets...")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    # Fill NAs just in case
    train_df['text'] = train_df['text'].fillna('')
    test_df['text'] = test_df['text'].fillna('')
    
    print(f"\n[+] Processing Train Set ({len(train_df)} rows)...")
    print("    -> Extracting metadata features...")
    train_df = extract_metadata(train_df)
    print("    -> Cleaning text for classical NLP...")
    train_df['cleaned_text'] = clean_text_series(train_df['text'])
    
    print(f"\n[+] Processing Test Set ({len(test_df)} rows)...")
    print("    -> Extracting metadata features...")
    test_df = extract_metadata(test_df)
    print("    -> Cleaning text for classical NLP...")
    test_df['cleaned_text'] = clean_text_series(test_df['text'])
    
    # Save the cleaned datasets
    train_path = "data/processed/train_cleaned.csv"
    test_path = "data/processed/test_cleaned.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n[+] Phase 2 Complete! Saved to '{train_path}' and '{test_path}'")

if __name__ == "__main__":
    process_data()
