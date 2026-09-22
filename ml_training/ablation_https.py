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
    confusion_matrix
)

# Ensure ml_training is in path for feature_extractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_ml_features

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RAW_CSV_PATH = os.path.join(DATA_DIR, "PhiUSIIL_Phishing_URL_Dataset.csv")

MODEL_A_PATH = os.path.join(MODELS_DIR, "ablation_all_features.pkl")
MODEL_B_PATH = os.path.join(MODELS_DIR, "ablation_without_https.pkl")
RESULTS_OUTPUT_PATH = os.path.join(MODELS_DIR, "ablation_https_results.json")

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

def run_ablation_experiment():
    start_time = time.time()
    print("==================================================")
    print("  CYBER-GUARD HTTPS ABLATION EXPERIMENT           ")
    print("==================================================")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if not os.path.exists(RAW_CSV_PATH):
        print(f"Error: Dataset CSV not found at {RAW_CSV_PATH}")
        sys.exit(1)
        
    print(f"\n[1/6] Loading dataset and extracting features...")
    df_raw = pd.read_csv(RAW_CSV_PATH)
    url_col = 'URL' if 'URL' in df_raw.columns else 'url'
    label_col = 'label' if 'label' in df_raw.columns else 'label'
    
    # Deduplicate & Clean
    df = df_raw.dropna(subset=[url_col, label_col]).drop_duplicates(subset=[url_col]).copy()
    clean_count = len(df)
    
    # Target label: PhiUSIIL 1->0 (Safe), 0->1 (Phishing)
    df['y_target'] = np.where(df[label_col].astype(int) == 1, 0, 1)
    df['domain_group'] = df[url_col].apply(extract_domain_group)
    
    urls = df[url_col].tolist()
    features_list = []
    chunk_size = 50000
    for i, u in enumerate(urls):
        features_list.append(extract_ml_features(u))
        if (i + 1) % chunk_size == 0 or (i + 1) == clean_count:
            print(f"      Processed {i + 1:,} / {clean_count:,} URLs...")
            
    X_df_all = pd.DataFrame(features_list)
    y = df['y_target'].values
    groups = df['domain_group'].values
    all_feature_names = X_df_all.columns.tolist()
    
    # 23 Features (without is_https)
    features_no_https = [col for col in all_feature_names if col != "is_https"]
    X_df_no_https = X_df_all[features_no_https].copy()
    
    # Group-based split
    print("\n[2/6] Performing GroupShuffleSplit (80% train / 20% test on domain)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X_df_all, y, groups=groups))
    
    train_domains = set(groups[train_idx])
    test_domains = set(groups[test_idx])
    domain_overlap = train_domains.intersection(test_domains)
    assert len(domain_overlap) == 0, "CRITICAL ERROR: Domain overlap detected!"
    print(f"      Domain split verified: 0 domain overlap across {len(test_domains):,} test domains.")
    
    # Model A: All 24 Features
    print("\n[3/6] Training Model A (All 24 Features)...")
    X_tr_A, X_te_A = X_df_all.iloc[train_idx], X_df_all.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    clf_A = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    t0 = time.time()
    clf_A.fit(X_tr_A, y_tr)
    t_A = time.time() - t0
    
    y_pred_A = clf_A.predict(X_te_A)
    acc_A = accuracy_score(y_te, y_pred_A)
    prec_A = precision_score(y_te, y_pred_A)
    rec_A = recall_score(y_te, y_pred_A)
    f1_A = f1_score(y_te, y_pred_A)
    cm_A = confusion_matrix(y_te, y_pred_A)
    
    print(f"      Model A trained in {t_A:.2f}s | F1: {f1_A*100:.2f}% | Acc: {acc_A*100:.2f}%")
    
    # Model B: 23 Features (Without is_https)
    print("\n[4/6] Training Model B (23 Features - WITHOUT is_https)...")
    X_tr_B, X_te_B = X_df_no_https.iloc[train_idx], X_df_no_https.iloc[test_idx]
    
    clf_B = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    t0 = time.time()
    clf_B.fit(X_tr_B, y_tr)
    t_B = time.time() - t0
    
    y_pred_B = clf_B.predict(X_te_B)
    acc_B = accuracy_score(y_te, y_pred_B)
    prec_B = precision_score(y_te, y_pred_B)
    rec_B = recall_score(y_te, y_pred_B)
    f1_B = f1_score(y_te, y_pred_B)
    cm_B = confusion_matrix(y_te, y_pred_B)
    
    print(f"      Model B trained in {t_B:.2f}s | F1: {f1_B*100:.2f}% | Acc: {acc_B*100:.2f}%")
    
    # 5. Feature Importances
    imp_A = clf_A.feature_importances_
    imp_B = clf_B.feature_importances_
    
    df_imp_A = pd.DataFrame({"Feature": all_feature_names, "Importance": imp_A}).sort_values(by="Importance", ascending=False)
    df_imp_B = pd.DataFrame({"Feature": features_no_https, "Importance": imp_B}).sort_values(by="Importance", ascending=False)
    
    https_imp_A = float(df_imp_A[df_imp_A["Feature"] == "is_https"]["Importance"].values[0]) * 100
    
    print("\n--- TOP 10 FEATURES IN MODEL A (All 24 Features) ---")
    for r, row in enumerate(df_imp_A.head(10).itertuples(), 1):
        print(f"  {r:2d}. {row.Feature:<28} : {row.Importance * 100:.2f}%")
        
    print("\n--- TOP 10 FEATURES IN MODEL B (Without is_https) ---")
    for r, row in enumerate(df_imp_B.head(10).itertuples(), 1):
        print(f"  {r:2d}. {row.Feature:<28} : {row.Importance * 100:.2f}%")
        
    # Save artifacts
    joblib.dump(clf_A, MODEL_A_PATH, compress=3)
    joblib.dump(clf_B, MODEL_B_PATH, compress=3)
    
    results = {
        "experiment": "Controlled is_https Ablation Experiment",
        "train_samples": int(len(X_tr_A)),
        "test_samples": int(len(X_te_A)),
        "model_A_all_features": {
            "accuracy": float(acc_A),
            "precision": float(prec_A),
            "recall": float(rec_A),
            "f1_score": float(f1_A),
            "confusion_matrix": {
                "TN": int(cm_A[0][0]),
                "FP": int(cm_A[0][1]),
                "FN": int(cm_A[1][0]),
                "TP": int(cm_A[1][1])
            },
            "is_https_importance_pct": float(https_imp_A),
            "feature_importances": dict(zip(all_feature_names, [float(i) for i in imp_A]))
        },
        "model_B_without_https": {
            "accuracy": float(acc_B),
            "precision": float(prec_B),
            "recall": float(rec_B),
            "f1_score": float(f1_B),
            "confusion_matrix": {
                "TN": int(cm_B[0][0]),
                "FP": int(cm_B[0][1]),
                "FN": int(cm_B[1][0]),
                "TP": int(cm_B[1][1])
            },
            "feature_importances": dict(zip(features_no_https, [float(i) for i in imp_B]))
        }
    }
    
    with open(RESULTS_OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved results to: {RESULTS_OUTPUT_PATH}")
    
    # 6. Final Comparison Table
    print("\n==================================================")
    print("        HTTPS ABLATION FINAL COMPARISON           ")
    print("==================================================")
    print(f"{'Metric':<18} | {'Model A (All 24 Feats)':<22} | {'Model B (No is_https)':<22} | {'Delta (B - A)':<15}")
    print("-" * 80)
    print(f"{'Accuracy':<18} | {acc_A*100:<21.2f}% | {acc_B*100:<21.2f}% | {(acc_B - acc_A)*100:<+14.2f}%")
    print(f"{'Precision':<18} | {prec_A*100:<21.2f}% | {prec_B*100:<21.2f}% | {(prec_B - prec_A)*100:<+14.2f}%")
    print(f"{'Recall':<18} | {rec_A*100:<21.2f}% | {rec_B*100:<21.2f}% | {(rec_B - rec_A)*100:<+14.2f}%")
    print(f"{'F1-Score':<18} | {f1_A*100:<21.2f}% | {f1_B*100:<21.2f}% | {(f1_B - f1_A)*100:<+14.2f}%")
    print("-" * 80)
    print(f"{'True Negatives':<18} | {cm_A[0][0]:<22,} | {cm_B[0][0]:<22,} | {cm_B[0][0] - cm_A[0][0]:<+15,}")
    print(f"{'False Positives':<18} | {cm_A[0][1]:<22,} | {cm_B[0][1]:<22,} | {cm_B[0][1] - cm_A[0][1]:<+15,}")
    print(f"{'False Negatives':<18} | {cm_A[1][0]:<22,} | {cm_B[1][0]:<22,} | {cm_B[1][0] - cm_A[1][0]:<+15,}")
    print(f"{'True Positives':<18} | {cm_A[1][1]:<22,} | {cm_B[1][1]:<22,} | {cm_B[1][1] - cm_A[1][1]:<+15,}")
    print("==================================================")
    
    total_time = time.time() - start_time
    print(f"\nABLATION EXPERIMENT COMPLETE IN {total_time:.2f} SECONDS.")

if __name__ == "__main__":
    run_ablation_experiment()
