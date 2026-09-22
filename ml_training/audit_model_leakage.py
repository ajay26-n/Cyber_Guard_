import os
import sys
import joblib
import pandas as pd
import numpy as np
from urllib.parse import urlparse

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_dataset.joblib")
RAW_CSV_PATH = os.path.join(DATA_DIR, "PhiUSIIL_Phishing_URL_Dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "new_phishing_model.pkl")

def run_audit():
    print("==================================================")
    print("  CYBER-GUARD MODEL & DATASET LEAKAGE AUDIT       ")
    print("==================================================")
    
    if not os.path.exists(PROCESSED_DATA_PATH) or not os.path.exists(MODEL_PATH):
        print("Error: Missing processed dataset or trained model artifact.")
        sys.exit(1)
        
    dataset = joblib.load(PROCESSED_DATA_PATH)
    clf = joblib.load(MODEL_PATH)
    
    X_train = dataset["X_train"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]
    feature_names = dataset["feature_names"]
    
    # Re-read CSV to map URLs back to indices for domain & false positive/negative analysis
    df_raw = pd.read_csv(RAW_CSV_PATH)
    url_col = 'URL' if 'URL' in df_raw.columns else 'url'
    label_col = 'label' if 'label' in df_raw.columns else 'label'
    
    df_clean = df_raw.dropna(subset=[url_col, label_col]).drop_duplicates(subset=[url_col]).copy()
    df_clean['y_clean'] = np.where(df_clean[label_col].astype(int) == 1, 0, 1)
    
    urls = df_clean[url_col].astype(str).values
    labels = df_clean['y_clean'].values
    
    # 1. Feature Distribution by Class
    print("\n--- 1. FEATURE MEAN / MEDIAN BY CLASS (SAFE vs PHISHING) ---")
    X_all = pd.concat([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    
    X_safe = X_all[y_all == 0]
    X_phish = X_all[y_all == 1]
    
    print(f"{'Feature':<28} | {'Safe Mean':<12} | {'Phish Mean':<12} | {'Diff (Phish - Safe)':<20}")
    print("-" * 80)
    for col in feature_names:
        s_m = X_safe[col].mean()
        p_m = X_phish[col].mean()
        diff = p_m - s_m
        print(f"{col:<28} | {s_m:<12.4f} | {p_m:<12.4f} | {diff:<+20.4f}")

    # 2. IS_HTTPS Breakdown
    print("\n--- 2. IS_HTTPS SPECIFIC BREAKDOWN ---")
    safe_https = X_safe['is_https'].value_counts(normalize=True) * 100
    phish_https = X_phish['is_https'].value_counts(normalize=True) * 100
    
    print(f"Safe URLs (Label 0):")
    print(f"  - is_https = 1 (HTTPS): {safe_https.get(1, 0):.2f}%")
    print(f"  - is_https = 0 (HTTP) : {safe_https.get(0, 0):.2f}%")
    print(f"Phishing URLs (Label 1):")
    print(f"  - is_https = 1 (HTTPS): {phish_https.get(1, 0):.2f}%")
    print(f"  - is_https = 0 (HTTP) : {phish_https.get(0, 0):.2f}%")

    # 3. Domain Overlap Analysis
    print("\n--- 3. DOMAIN OVERLAP & LEAKAGE ANALYSIS ---")
    # Extract domain for every clean URL
    domains = [urlparse(u if u.startswith(('http://', 'https://')) else 'http://' + u).netloc.lower() for u in urls]
    df_clean['domain'] = domains
    
    total_unique_domains = len(set(domains))
    print(f"Total Unique URLs    : {len(urls):,}")
    print(f"Total Unique Domains : {total_unique_domains:,}")
    
    # Split indices matching train_test_split (random_state=42)
    from sklearn.model_selection import train_test_split
    df_tr, df_te = train_test_split(df_clean, test_size=0.20, random_state=42, stratify=df_clean['y_clean'])
    
    train_domains = set(df_tr['domain'])
    test_domains = set(df_te['domain'])
    overlapping_domains = train_domains.intersection(test_domains)
    
    test_urls_in_train_domain = df_te['domain'].isin(train_domains).sum()
    pct_test_overlap = (test_urls_in_train_domain / len(df_te)) * 100
    
    print(f"Unique Domains in Train set : {len(train_domains):,}")
    print(f"Unique Domains in Test set  : {len(test_domains):,}")
    print(f"Overlapping Domains         : {len(overlapping_domains):,}")
    print(f"Test URLs with Train Domain : {test_urls_in_train_domain:,} / {len(df_te):,} ({pct_test_overlap:.2f}%)")

    # 4. Exact Duplicate Check Across Train and Test
    print("\n--- 4. EXACT DUPLICATE URL CHECK ---")
    train_url_set = set(df_tr[url_col])
    test_url_set = set(df_te[url_col])
    dup_urls = train_url_set.intersection(test_url_set)
    print(f"Exact Duplicate URLs between Train and Test: {len(dup_urls)}")

    # 5. Error Analysis: False Negatives & False Positives
    print("\n--- 5. ERROR ANALYSIS (FALSE NEGATIVES & FALSE POSITIVES) ---")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    
    test_indices = X_test.index
    df_test_res = df_te.copy()
    df_test_res['y_pred'] = y_pred
    if y_prob is not None:
        df_test_res['y_prob'] = y_prob
        
    fn_df = df_test_res[(df_test_res['y_clean'] == 1) & (df_test_res['y_pred'] == 0)] # Phishing missed
    fp_df = df_test_res[(df_test_res['y_clean'] == 0) & (df_test_res['y_pred'] == 1)] # Safe wrongly flagged
    
    print(f"Total False Negatives (Phishing missed): {len(fn_df)}")
    print(f"Total False Positives (Safe flagged)   : {len(fp_df)}")
    
    print("\n--- SAMPLE FALSE NEGATIVES (10 Representative Cases) ---")
    for idx, row in fn_df.head(10).reset_index().iterrows():
        prob_str = f" (Phish Prob: {row['y_prob']:.4f})" if 'y_prob' in row else ""
        print(f"FN #{idx+1}: [Domain: {row['domain']}]{prob_str}")
        print(f"     Length: {len(row[url_col])} | is_https: {1 if row[url_col].startswith('https') else 0} | path_len: {len(urlparse(row[url_col]).path)}")
        
    print("\n--- SAMPLE FALSE POSITIVES (10 Representative Cases) ---")
    for idx, row in fp_df.head(10).reset_index().iterrows():
        prob_str = f" (Phish Prob: {row['y_prob']:.4f})" if 'y_prob' in row else ""
        print(f"FP #{idx+1}: [Domain: {row['domain']}]{prob_str}")
        print(f"     Length: {len(row[url_col])} | is_https: {1 if row[url_col].startswith('https') else 0} | path_len: {len(urlparse(row[url_col]).path)}")

if __name__ == "__main__":
    run_audit()
