import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def prepare_data(raw_data_path="data/raw/dataset.csv", output_dir="data/processed"):
    if not os.path.exists(raw_data_path):
        print(f"[-] Error: Data file not found at {raw_data_path}")
        print("[*] Please download a dataset (e.g., from Kaggle) and place it there as 'dataset.csv'.")
        print("[*] Ensure it has at least two columns: one for the email text and one for the label (0/1).")
        return
    
    print(f"[+] Loading dataset from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    
    # Show columns so the user knows what was loaded
    print(f"[+] Columns found: {df.columns.tolist()}")
    
    # Try to auto-detect text and label columns if they aren't explicitly named 'text' and 'label'
    text_col = next((col for col in df.columns if col.lower() in ['text', 'email text', 'message', 'body', 'text_combined']), None)
    label_col = next((col for col in df.columns if col.lower() in ['label', 'spam', 'phishing', 'target', 'email type']), None)
    
    if not text_col or not label_col:
        print("[-] Could not automatically detect text and label columns. Please rename them to 'text' and 'label' in your CSV.")
        return
        
    print(f"[+] Using '{text_col}' for text and '{label_col}' for labels.")
    
    # Standardize names
    df = df.rename(columns={text_col: 'text', label_col: 'label'})
    
    # Convert labels to 0 and 1 if they are strings (e.g., 'Phishing Email', 'Safe Email')
    if df['label'].dtype == object:
        print("[*] Converting string labels to numeric (0 = Legitimate, 1 = Phishing)")
        # Simple heuristic: if 'phishing' or 'spam' is in the string, mark as 1, else 0
        df['label'] = df['label'].astype(str).str.lower().apply(lambda x: 1 if 'phishing' in x or 'spam' in x else 0)
    
    # Drop rows with missing text
    initial_shape = df.shape
    df = df.dropna(subset=['text', 'label'])
    print(f"[+] Dropped {initial_shape[0] - df.shape[0]} rows with missing values.")
    
    # Drop duplicates
    initial_shape = df.shape
    df = df.drop_duplicates(subset=['text'])
    print(f"[+] Dropped {initial_shape[0] - df.shape[0]} duplicate emails.")
    
    # Print class distribution
    dist = df['label'].value_counts(normalize=True) * 100
    print("\n--- Class Distribution ---")
    print(f"Legitimate (0): {df['label'].value_counts().get(0, 0)} ({dist.get(0, 0):.2f}%)")
    print(f"Phishing (1): {df['label'].value_counts().get(1, 0)} ({dist.get(1, 0):.2f}%)\n")
    
    # Train / Test split (80/20 stratified)
    print("[+] Splitting data into 80% Train and 20% Test (Stratified)...")
    train_df, test_df = train_test_split(df[['text', 'label']], test_size=0.2, random_state=42, stratify=df['label'])
    
    # Save the split
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"[+] Saved train set to: {train_path} ({train_df.shape[0]} rows)")
    print(f"[+] Saved test set to: {test_path} ({test_df.shape[0]} rows)")
    print("[+] Phase 1 Complete! We are ready for Phase 2.")

if __name__ == "__main__":
    prepare_data()
