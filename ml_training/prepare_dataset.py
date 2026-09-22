import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure ml_training is in python path to import feature_extractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_ml_features

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_CSV_PATH = os.path.join(DATA_DIR, "PhiUSIIL_Phishing_URL_Dataset.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_dataset.joblib")

def prepare_dataset():
    start_time = time.time()
    print("==================================================")
    print("   CYBER-GUARD DATASET PREPARATION PIPELINE       ")
    print("==================================================")
    
    if not os.path.exists(RAW_CSV_PATH):
        print(f"Error: Raw CSV dataset not found at {RAW_CSV_PATH}")
        sys.exit(1)
        
    print(f"\n[1/5] Loading raw dataset: {RAW_CSV_PATH}...")
    df = pd.read_csv(RAW_CSV_PATH)
    raw_count = len(df)
    print(f"      Loaded {raw_count:,} raw records.")
    
    # 2. Data Cleaning & Deduplication
    print("\n[2/5] Cleaning and deduplicating URLs...")
    # Find URL column
    url_col = 'URL' if 'URL' in df.columns else ('url' if 'url' in df.columns else None)
    label_col = 'label' if 'label' in df.columns else None
    
    if not url_col or not label_col:
        print("Error: Could not locate URL or label columns in dataset.")
        sys.exit(1)
        
    # Drop missing
    df = df.dropna(subset=[url_col, label_col]).copy()
    
    # Drop duplicates
    df = df.drop_duplicates(subset=[url_col]).copy()
    clean_count = len(df)
    print(f"      Removed {raw_count - clean_count:,} duplicate/missing entries.")
    print(f"      Clean unique URLs: {clean_count:,}")
    
    # 3. Label Mapping
    # PhiUSIIL convention: 1 = Legitimate, 0 = Phishing
    # Cyber-Guard convention: 0 = Safe/Legitimate, 1 = Phishing
    print("\n[3/5] Remapping labels to Cyber-Guard convention (0=Safe, 1=Phishing)...")
    y_raw = df[label_col].astype(int).values
    # Map 1 -> 0, 0 -> 1
    y = np.where(y_raw == 1, 0, 1)
    
    safe_count = int(np.sum(y == 0))
    phish_count = int(np.sum(y == 1))
    print(f"      Target Distribution -> Safe (0): {safe_count:,} ({safe_count/clean_count*100:.2f}%)")
    print(f"      Target Distribution -> Phishing (1): {phish_count:,} ({phish_count/clean_count*100:.2f}%)")
    
    # 4. Feature Extraction
    print("\n[4/5] Extracting 24 URL-only lexical features for all URLs...")
    urls = df[url_col].astype(str).tolist()
    
    features_list = []
    chunk_size = 50000
    for i, u in enumerate(urls):
        features_list.append(extract_ml_features(u))
        if (i + 1) % chunk_size == 0 or (i + 1) == clean_count:
            print(f"      Processed {i + 1:,} / {clean_count:,} URLs...")
            
    X_df = pd.DataFrame(features_list)
    feature_names = X_df.columns.tolist()
    print(f"      Extracted {len(feature_names)} features successfully.")
    
    # 5. Stratified Train / Test Split
    print("\n[5/5] Splitting dataset into Stratified Train (80%) and Test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"      Training set size: {len(X_train):,} samples")
    print(f"      Testing set size:  {len(X_test):,} samples")
    
    # Save prepared dataset artifact
    dataset_payload = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "label_mapping": {0: "Safe", 1: "Phishing"},
        "raw_count": raw_count,
        "clean_count": clean_count
    }
    
    print(f"\nSaving processed dataset artifact to: {PROCESSED_DATA_PATH}...")
    joblib.dump(dataset_payload, PROCESSED_DATA_PATH, compress=3)
    
    elapsed = time.time() - start_time
    print(f"\n==================================================")
    print(f" DATASET PREPARATION COMPLETE IN {elapsed:.2f} SECONDS ")
    print(f"==================================================")

if __name__ == "__main__":
    prepare_dataset()
