import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# Ensure ml_training is in path for feature_extractor and features
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_ml_features

OLD_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/phishing_model.pkl"))
NEW_MODEL_PATH = os.path.join(os.path.dirname(__file__), "new_phishing_model.pkl")
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "phishing_site_urls.csv")

def extract_5_features_old(url_str):
    import re
    url_raw = str(url_str).strip()
    url_len = len(url_raw)
    
    try:
        url_for_parse = url_raw
        if not url_for_parse.startswith(("http://", "https://", "ftp://")):
            url_for_parse = "http://" + url_for_parse
        parsed = urlparse(url_for_parse)
        netloc = parsed.netloc if parsed.netloc else ""
    except Exception:
        netloc = ""
        
    keywords_list = ["login", "verify", "secure", "account", "update"]
    keyword_count = sum(word in url_raw.lower() for word in keywords_list)
    
    subdomains = netloc.count(".") - 1 if netloc else 0
    if subdomains < 0:
        subdomains = 0
        
    has_ip = 1 if re.search(r"\d+\.\d+\.\d+\.\d+", netloc) else 0
    special_chars = sum(url_raw.count(c) for c in ["@", "?", "-", "=", "_"])
    
    return [url_len, keyword_count, subdomains, has_ip, special_chars]

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

def compare_models():
    start_time = time.time()
    print("==================================================")
    print("   CYBER-GUARD MODEL COMPARISON (OLD vs NEW)      ")
    print("==================================================")
    
    if not os.path.exists(OLD_MODEL_PATH) or not os.path.exists(NEW_MODEL_PATH):
        print("Error: Missing old or new model artifact.")
        sys.exit(1)
        
    print("\n[1/4] Loading models...")
    old_model = joblib.load(OLD_MODEL_PATH)
    new_model = joblib.load(NEW_MODEL_PATH)
    print("      Old Model Loaded (backend/phishing_model.pkl)")
    print("      New Model Loaded (ml_training/new_phishing_model.pkl)")
    
    print("\n[2/4] Loading and deduplicating dataset (phishing_site_urls.csv)...")
    df_raw = pd.read_csv(CSV_PATH)
    df = df_raw.dropna(subset=['URL', 'Label']).drop_duplicates(subset=['URL']).copy()
    
    df['y_target'] = np.where(df['Label'].astype(str).str.lower() == 'bad', 1, 0)
    df['domain_group'] = df['URL'].apply(extract_domain_group)
    
    # 80/20 Domain Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(df, df['y_target'], groups=df['domain_group']))
    
    test_df = df.iloc[test_idx].copy()
    y_test = test_df['y_target'].values
    test_urls = test_df['URL'].tolist()
    
    print(f"      Test Set Size (Unseen Domains): {len(test_df):,} URLs")
    
    # Extract features for both models
    print("\n[3/4] Extracting features for test set...")
    old_feats = [extract_5_features_old(u) for u in test_urls]
    new_feats = [extract_ml_features(u) for u in test_urls]
    
    old_cols = ["url_length", "keywords", "subdomains", "ip", "special_chars"]
    X_test_old = pd.DataFrame(old_feats, columns=old_cols)
    X_test_new = pd.DataFrame(new_feats)
    
    # Generate Predictions
    print("\n[4/4] Generating predictions for Old vs New models...")
    old_pred = old_model.predict(X_test_old)
    new_pred = new_model.predict(X_test_new)
    
    # Calculate Metrics
    acc_old = accuracy_score(y_test, old_pred)
    prec_old = precision_score(y_test, old_pred, zero_division=0)
    rec_old = recall_score(y_test, old_pred, zero_division=0)
    f1_old = f1_score(y_test, old_pred, zero_division=0)
    cm_old = confusion_matrix(y_test, old_pred)
    
    acc_new = accuracy_score(y_test, new_pred)
    prec_new = precision_score(y_test, new_pred, zero_division=0)
    rec_new = recall_score(y_test, new_pred, zero_division=0)
    f1_new = f1_score(y_test, new_pred, zero_division=0)
    cm_new = confusion_matrix(y_test, new_pred)
    
    print("\n==================================================")
    print("      SIDE-BY-SIDE METRICS COMPARISON             ")
    print("==================================================")
    print(f"{'Metric':<18} | {'Old Model (5 Feats)':<20} | {'New Model (24 Feats)':<20} | {'Delta (New - Old)':<15}")
    print("-" * 80)
    print(f"{'Accuracy':<18} | {acc_old*100:<19.2f}% | {acc_new*100:<19.2f}% | {(acc_new - acc_old)*100:<+14.2f}%")
    print(f"{'Precision':<18} | {prec_old*100:<19.2f}% | {prec_new*100:<19.2f}% | {(prec_new - prec_old)*100:<+14.2f}%")
    print(f"{'Recall':<18} | {rec_old*100:<19.2f}% | {rec_new*100:<19.2f}% | {(rec_new - rec_old)*100:<+14.2f}%")
    print(f"{'F1-Score':<18} | {f1_old*100:<19.2f}% | {f1_new*100:<19.2f}% | {(f1_new - f1_old)*100:<+14.2f}%")
    print("-" * 80)
    print(f"{'True Negatives':<18} | {cm_old[0][0]:<20,} | {cm_new[0][0]:<20,} | {cm_new[0][0] - cm_old[0][0]:<+15,}")
    print(f"{'False Positives':<18} | {cm_old[0][1]:<20,} | {cm_new[0][1]:<20,} | {cm_new[0][1] - cm_old[0][1]:<+15,}")
    print(f"{'False Negatives':<18} | {cm_old[1][0]:<20,} | {cm_new[1][0]:<20,} | {cm_new[1][0] - cm_old[1][0]:<+15,}")
    print(f"{'True Positives':<18} | {cm_old[1][1]:<20,} | {cm_new[1][1]:<20,} | {cm_new[1][1] - cm_old[1][1]:<+15,}")
    print("==================================================")
    
    # Key Conclusions
    print("\n--- PERFORMANCE VERDICT ---")
    if cm_new[1][1] > cm_old[1][1]:
        print(f"1. Phishing Detection: NEW Model catches {cm_new[1][1] - cm_old[1][1]:,} MORE phishing URLs (+{(rec_new - rec_old)*100:.2f}% Recall).")
    else:
        print(f"1. Phishing Detection: OLD Model catches {cm_old[1][1] - cm_new[1][1]:,} MORE phishing URLs.")
        
    if cm_new[0][1] < cm_old[0][1]:
        print(f"2. False Positives   : NEW Model produces {cm_old[0][1] - cm_new[0][1]:,} FEWER false positives (Higher Precision).")
    else:
        print(f"2. False Positives   : OLD Model produces {cm_old[0][1] - cm_new[0][1]:,} FEWER false positives.")
        
    # Disagreement Analysis
    disagree_mask = (old_pred != new_pred)
    total_disagreements = np.sum(disagree_mask)
    print(f"\nTotal Disagreement Cases: {total_disagreements:,} / {len(test_df):,} ({total_disagreements/len(test_df)*100:.2f}%)")
    
    test_df['old_pred'] = old_pred
    test_df['new_pred'] = new_pred
    
    dis_df = test_df[disagree_mask].copy()
    
    print("\n--- SAMPLE DISAGREEMENTS: OLD = SAFE (0), NEW = PHISHING (1) ---")
    sub1 = dis_df[(dis_df['old_pred'] == 0) & (dis_df['new_pred'] == 1)]
    for r, row in enumerate(sub1.head(5).itertuples(), 1):
        actual = "Phishing (1)" if row.y_target == 1 else "Safe (0)"
        print(f" {r}. Domain: {row.domain_group} | Actual: {actual}")
        print(f"    URL String: {row.URL}")
        
    print("\n--- SAMPLE DISAGREEMENTS: OLD = PHISHING (1), NEW = SAFE (0) ---")
    sub2 = dis_df[(dis_df['old_pred'] == 1) & (dis_df['new_pred'] == 0)]
    for r, row in enumerate(sub2.head(5).itertuples(), 1):
        actual = "Phishing (1)" if row.y_target == 1 else "Safe (0)"
        print(f" {r}. Domain: {row.domain_group} | Actual: {actual}")
        print(f"    URL String: {row.URL}")

    elapsed = time.time() - start_time
    print(f"\nMODEL COMPARISON COMPLETE IN {elapsed:.2f} SECONDS.")

if __name__ == "__main__":
    compare_models()
