import os
import sys
import time
import json
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
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RAW_CSV_PATH = os.path.join(DATA_DIR, "PhiUSIIL_Phishing_URL_Dataset.csv")

MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "domain_split_random_forest.pkl")
RESULTS_OUTPUT_PATH = os.path.join(MODELS_DIR, "domain_split_results.json")

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

def run_domain_split_experiment():
    start_time = time.time()
    print("==================================================")
    print("  CYBER-GUARD DOMAIN-BASED EVALUATION EXPERIMENT ")
    print("==================================================")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if not os.path.exists(RAW_CSV_PATH):
        print(f"Error: Dataset CSV not found at {RAW_CSV_PATH}")
        sys.exit(1)
        
    print(f"\n[1/6] Loading raw dataset: {RAW_CSV_PATH}...")
    df_raw = pd.read_csv(RAW_CSV_PATH)
    raw_count = len(df_raw)
    print(f"      Loaded {raw_count:,} raw records.")
    
    # Locate columns
    url_col = 'URL' if 'URL' in df_raw.columns else 'url'
    label_col = 'label' if 'label' in df_raw.columns else 'label'
    
    # 2 & 3. Clean and Deduplicate
    print("\n[2/6] Deduplicating and deriving domain grouping keys...")
    df = df_raw.dropna(subset=[url_col, label_col]).drop_duplicates(subset=[url_col]).copy()
    clean_count = len(df)
    print(f"      Clean unique URLs: {clean_count:,}")
    
    # Remap labels: PhiUSIIL 1 -> 0 (Safe), 0 -> 1 (Phishing)
    df['y_target'] = np.where(df[label_col].astype(int) == 1, 0, 1)
    
    # Derive Domain Groups
    df['domain_group'] = df[url_col].apply(extract_domain_group)
    
    # 5. Extract 24 ML Features
    print("\n[3/6] Extracting 24 URL-only features using feature_extractor.py...")
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
    
    # 6. Group-Based 80/20 Train/Test Split on Domain
    print("\n[4/6] Performing Group-Based 80/20 Split (GroupShuffleSplit on domain)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X_df, y, groups=groups))
    
    X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    train_domains = set(groups[train_idx])
    test_domains = set(groups[test_idx])
    domain_overlap = train_domains.intersection(test_domains)
    overlap_pct = (len(domain_overlap) / len(test_domains)) * 100 if test_domains else 0.0
    
    # 7. Explicit Verification
    print(f"      Training samples : {len(X_train):,}")
    print(f"      Testing samples  : {len(X_test):,}")
    print(f"      Training domains : {len(train_domains):,}")
    print(f"      Testing domains  : {len(test_domains):,}")
    print(f"      Domain Overlap   : {len(domain_overlap)} domains ({overlap_pct:.2f}%)")
    assert len(domain_overlap) == 0, "CRITICAL ERROR: Domain overlap detected!"
    print("      Verification SUCCESS: 0 domain overlap confirmed.")
    
    # 8. Train RandomForestClassifier
    print("\n[5/6] Training RandomForestClassifier (100 trees, random_state=42, n_jobs=-1)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    train_start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"      Model training completed in {train_time:.2f} seconds.")
    
    # 9. Model Evaluation
    print("\n[6/6] Evaluating on UNSEEN domains (20% test group)...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n--- DOMAIN SPLIT PERFORMANCE METRICS ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    
    print("\n--- CONFUSION MATRIX ---")
    print(f"True Negatives  (Safe correctly identified):     {cm[0][0]:,}")
    print(f"False Positives (Safe wrongly flagged):         {cm[0][1]:,}")
    print(f"False Negatives (Phishing missed):               {cm[1][0]:,}")
    print(f"True Positives  (Phishing correctly caught):    {cm[1][1]:,}")
    
    print("\n--- ALL 24 FEATURE IMPORTANCES ---")
    importances = clf.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    for rank, row in enumerate(feat_imp_df.itertuples(), 1):
        print(f"  {rank:2d}. {row.Feature:<28} : {row.Importance * 100:.2f}%")
        
    https_imp = float(feat_imp_df[feat_imp_df["Feature"] == "is_https"]["Importance"].values[0]) * 100
    print(f"\n--- SPECIFIC FEATURE REPORT ---")
    print(f"is_https Feature Importance: {https_imp:.2f}%")
    
    # Save artifacts
    joblib.dump(clf, MODEL_OUTPUT_PATH, compress=3)
    print(f"\nSaved experiment model to: {MODEL_OUTPUT_PATH}")
    
    results_data = {
        "experiment": "Domain-Based Group Split (GroupShuffleSplit)",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_domains": int(len(train_domains)),
        "test_domains": int(len(test_domains)),
        "domain_overlap_count": int(len(domain_overlap)),
        "domain_overlap_percentage": float(overlap_pct),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "is_https_importance_pct": float(https_imp),
        "feature_importances": dict(zip(feature_names, [float(i) for i in importances]))
    }
    
    with open(RESULTS_OUTPUT_PATH, "w") as f:
        json.dump(results_data, f, indent=4)
    print(f"Saved experiment metrics to: {RESULTS_OUTPUT_PATH}")
    
    print("\n==================================================")
    print("           EXPERIMENT COMPARISON SUMMARY           ")
    print("==================================================")
    print("Random row split:")
    print("Accuracy  = 99.58%")
    print("Precision = 99.88%")
    print("Recall    = 99.13%")
    print("F1        = 99.51%")
    print("")
    print("Domain split (Unseen Domains):")
    print(f"Accuracy  = {acc * 100:.2f}%")
    print(f"Precision = {prec * 100:.2f}%")
    print(f"Recall    = {rec * 100:.2f}%")
    print(f"F1        = {f1 * 100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    run_domain_split_experiment()
