import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Ensure ml_training is in path for feature_extractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_ml_features

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "phishing_site_urls.csv")
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "new_phishing_model.pkl")

def extract_domain_group(url):
    try:
        url_str = str(url).strip()
        if not url_str.startswith(("http://", "https://", "ftp://")):
            url_str = "http://" + url_str
        parsed = urlparse(url_str)
        netloc = parsed.netloc.lower()
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        return netloc if netloc else "unknown_domain"
    except Exception:
        return "unknown_domain"

def train_new_model():
    start_time = time.time()
    print("==================================================")
    print("  TRAINING NEW CYBER-GUARD MODEL (phishing_site_urls) ")
    print("==================================================")
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: Dataset CSV not found at {CSV_PATH}")
        sys.exit(1)
        
    print(f"\n[1/5] Loading raw dataset: {CSV_PATH}...")
    df_raw = pd.read_csv(CSV_PATH)
    raw_count = len(df_raw)
    print(f"      Loaded {raw_count:,} raw records.")
    
    # Clean & Deduplicate
    url_col = 'URL' if 'URL' in df_raw.columns else 'url'
    label_col = 'Label' if 'Label' in df_raw.columns else 'label'
    
    print("\n[2/5] Cleaning, deduplicating URLs, and deriving domain groups...")
    df = df_raw.dropna(subset=[url_col, label_col]).drop_duplicates(subset=[url_col]).copy()
    clean_count = len(df)
    print(f"      Removed {raw_count - clean_count:,} duplicate/missing entries.")
    print(f"      Clean unique URLs: {clean_count:,}")
    
    # Label Mapping: good = 0 (Safe), bad = 1 (Phishing)
    df['y_target'] = np.where(df[label_col].astype(str).str.lower() == 'bad', 1, 0)
    df['domain_group'] = df[url_col].apply(extract_domain_group)
    
    safe_cnt = (df['y_target'] == 0).sum()
    phish_cnt = (df['y_target'] == 1).sum()
    print(f"      Target Distribution -> Safe (0): {safe_cnt:,} ({safe_cnt/clean_count*100:.2f}%)")
    print(f"      Target Distribution -> Phishing (1): {phish_cnt:,} ({phish_cnt/clean_count*100:.2f}%)")
    
    # Feature Extraction
    print("\n[3/5] Extracting 24 Cyber-Guard features...")
    urls = df[url_col].tolist()
    features_list = []
    chunk_size = 50000
    for i, u in enumerate(urls):
        features_list.append(extract_ml_features(u))
        if (i + 1) % chunk_size == 0 or (i + 1) == clean_count:
            print(f"      Processed {i + 1:,} / {clean_count:,} URLs...")
            
    X_df = pd.DataFrame(features_list)
    y = df['y_target'].values
    groups = df['domain_group'].values
    feature_names = X_df.columns.tolist()
    
    # Domain-based Train/Test Split
    print("\n[4/5] Performing Group-Based 80/20 Domain Split (GroupShuffleSplit)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X_df, y, groups=groups))
    
    X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    train_domains = set(groups[train_idx])
    test_domains = set(groups[test_idx])
    domain_overlap = train_domains.intersection(test_domains)
    
    print(f"      Training samples : {len(X_train):,}")
    print(f"      Testing samples  : {len(X_test):,}")
    print(f"      Training domains : {len(train_domains):,}")
    print(f"      Testing domains  : {len(test_domains):,}")
    print(f"      Domain Overlap   : {len(domain_overlap)} domains (0.00%)")
    assert len(domain_overlap) == 0, "CRITICAL ERROR: Domain overlap detected!"
    
    # Train RandomForestClassifier
    print("\n[5/5] Training RandomForestClassifier (100 trees, class_weight='balanced')...")
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    
    t0 = time.time()
    clf.fit(X_train, y_train)
    t_train = time.time() - t0
    print(f"      Model training completed in {t_train:.2f} seconds.")
    
    # Attach feature_names_in_ attribute to model for sklearn compatibility
    clf.feature_names_in_ = np.array(feature_names, dtype=object)
    
    # Evaluate
    print("\n==================================================")
    print("      UNSEEN DOMAIN EVALUATION METRICS            ")
    print("==================================================")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    
    print("\n--- CONFUSION MATRIX ---")
    print(f"True Negatives  (Safe correctly identified):     {cm[0][0]:,}")
    print(f"False Positives (Safe wrongly flagged):         {cm[0][1]:,}")
    print(f"False Negatives (Phishing missed):               {cm[1][0]:,}")
    print(f"True Positives  (Phishing correctly caught):    {cm[1][1]:,}")
    
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=["Safe (0)", "Phishing (1)"]))
    
    print("--- TOP 10 FEATURE IMPORTANCES ---")
    importances = clf.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    for r, row in enumerate(feat_imp_df.head(10).itertuples(), 1):
        print(f"  {r:2d}. {row.Feature:<28} : {row.Importance * 100:.2f}%")
        
    # Save Model
    joblib.dump(clf, MODEL_OUTPUT_PATH, compress=3)
    print(f"\nSaved new trained model to: {MODEL_OUTPUT_PATH}")
    
    total_time = time.time() - start_time
    print(f"==================================================")
    print(f" TRAINING PIPELINE COMPLETE IN {total_time:.2f} SECONDS ")
    print(f"==================================================")

if __name__ == "__main__":
    train_new_model()
