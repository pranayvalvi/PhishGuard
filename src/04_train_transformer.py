import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import warnings

# Suppress HuggingFace warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_transformer(sample_size=10000):
    print(f"[+] Loading datasets (using raw text for BERT, up to {sample_size} samples)...")
    train_df = pd.read_csv("data/processed/train_cleaned.csv")
    test_df = pd.read_csv("data/processed/test_cleaned.csv")
    
    # Drop NAs in raw text just in case
    train_df = train_df.dropna(subset=['text'])
    test_df = test_df.dropna(subset=['text'])
    
    # Fast Prototype Sampling (change sample_size=None in __main__ for a full run)
    if sample_size and sample_size < len(train_df):
        print(f"    -> Sampling {sample_size} emails for faster fine-tuning prototype...")
        # Stratified sample
        train_df = train_df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), int(sample_size/2)), random_state=42))
    
    # Test set sample just for provisional evaluation (keep it fast)
    if sample_size and (sample_size * 0.2) < len(test_df):
        test_size = int(sample_size * 0.2)
        test_df = test_df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), int(test_size/2)), random_state=42))

    train_texts = train_df['text'].astype(str).tolist()
    train_labels = train_df['label'].tolist()
    
    test_texts = test_df['text'].astype(str).tolist()
    test_labels = test_df['label'].tolist()
    
    print(f"    -> Train set size: {len(train_texts)}")
    print(f"    -> Test set size: {len(test_texts)}")

    print("\n[+] Loading Tokenizer (distilbert-base-uncased)...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    print("[+] Tokenizing texts (truncating to 256 tokens for memory/speed)...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)
    
    train_dataset = EmailDataset(train_encodings, train_labels)
    test_dataset = EmailDataset(test_encodings, test_labels)
    
    print("\n[+] Loading DistilBERT Model...")
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    # Determine if we have CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    -> Device detected: {device}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./models/results',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./models/logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="none" # Disable wandb/etc.
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    print("\n[+] Starting Fine-Tuning (This will take time)...")
    trainer.train()
    
    print("\n[+] Saving model to 'models/distilbert_model/'...")
    model.save_pretrained('./models/distilbert_model')
    tokenizer.save_pretrained('./models/distilbert_model')
    print("[+] Phase 4 Complete!")

if __name__ == "__main__":
    # For a full run, set sample_size=None
    train_transformer(sample_size=500)
