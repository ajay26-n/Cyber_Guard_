import os
import sys
import json
import time
import pandas as pd
import numpy as np

# Ensure ml_training path for feature_extractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_ml_features

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RAW_CSV_PATH = os.path.join(DATA_DIR, "PhiUSIIL_Phishing_URL_Dataset.csv")
JSON_OUTPUT_PATH = os.path.join(MODELS_DIR, "dataset_bias_analysis.json")

BINARY_FEATURES = [
    "has_ip", "is_https", "has_at_symbol", "has_query",
    "suspicious_tld", "has_punycode", "has_percent_encoding"
]

TARGET_FOCUS_FEATURES = [
    "is_https", "hostname_length", "path_length",
    "path_special_char_count", "url_length",
    "suspicious_keyword_count", "has_ip"
]

def analyze_bias():
    start_time = time.time()
    print("==================================================")
    print("   CYBER-GUARD PHIUSIIL DATASET BIAS ANALYSIS    ")
    print("==================================================")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if not os.path.exists(RAW_CSV_PATH):
        print(f"Error: Dataset CSV not found at {RAW_CSV_PATH}")
        sys.exit(1)
        
    print(f"\n[1/4] Loading raw dataset: {RAW_CSV_PATH}...")
    df_raw = pd.read_csv(RAW_CSV_PATH)
    url_col = 'URL' if 'URL' in df_raw.columns else 'url'
    label_col = 'label' if 'label' in df_raw.columns else 'label'
    
    df = df_raw.dropna(subset=[url_col, label_col]).drop_duplicates(subset=[url_col]).copy()
    clean_count = len(df)
    
    # Label convention:
    # Raw PhiUSIIL: 1 = Legitimate (Safe), 0 = Phishing
    # Cyber-Guard internal: 0 = Safe, 1 = Phishing
    df['is_phishing'] = np.where(df[label_col].astype(int) == 0, 1, 0)
    
    total_samples = len(df)
    safe_samples = int((df['is_phishing'] == 0).sum())
    phish_samples = int((df['is_phishing'] == 1).sum())
    
    print("\n--- 1. DATASET CLASS DISTRIBUTION ---")
    print(f"Total Unique Clean URLs : {total_samples:,}")
    print(f"Safe / Legitimate (0)   : {safe_samples:,} ({safe_samples/total_samples*100:.2f}%)")
    print(f"Phishing (1)            : {phish_samples:,} ({phish_samples/total_samples*100:.2f}%)")
    
    # Extract 24 features
    print("\n[2/4] Extracting 24 URL-only features using feature_extractor.py...")
    urls = df[url_col].tolist()
    features_list = []
    chunk_size = 50000
    for i, u in enumerate(urls):
        features_list.append(extract_ml_features(u))
        if (i + 1) % chunk_size == 0 or (i + 1) == total_samples:
            print(f"      Processed {i + 1:,} / {total_samples:,} URLs...")
            
    X_df = pd.DataFrame(features_list)
    X_df['is_phishing'] = df['is_phishing'].values
    
    safe_df = X_df[X_df['is_phishing'] == 0]
    phish_df = X_df[X_df['is_phishing'] == 1]
    
    feature_names = [col for col in X_df.columns if col != 'is_phishing']
    
    # 2. Binary Features Analysis
    print("\n--- 2. BINARY FEATURES DISTRIBUTION ---")
    binary_stats = {}
    print(f"{'Binary Feature':<25} | {'Safe % (feat=1)':<18} | {'Phish % (feat=1)':<18} | {'Abs Diff %':<12}")
    print("-" * 80)
    for b_feat in BINARY_FEATURES:
        safe_pct = (safe_df[b_feat] == 1).mean() * 100
        phish_pct = (phish_df[b_feat] == 1).mean() * 100
        diff = abs(phish_pct - safe_pct)
        binary_stats[b_feat] = {
            "safe_pct_active": float(safe_pct),
            "phish_pct_active": float(phish_pct),
            "abs_diff_pct": float(diff)
        }
        print(f"{b_feat:<25} | {safe_pct:<17.2f}% | {phish_pct:<17.2f}% | {diff:<11.2f}%")

    # 3. All 24 Features Statistics (Mean, Median, Min, Max)
    print("\n--- 3. NUMERIC FEATURES SUMMARY (SAFE vs PHISHING) ---")
    numeric_stats = {}
    diff_scores = []
    
    for feat in feature_names:
        s_mean, s_med = safe_df[feat].mean(), safe_df[feat].median()
        s_min, s_max = safe_df[feat].min(), safe_df[feat].max()
        
        p_mean, p_med = phish_df[feat].mean(), phish_df[feat].median()
        p_min, p_max = phish_df[feat].min(), phish_df[feat].max()
        
        # Standardized difference measure for ranking
        std_comb = np.std(X_df[feat]) if np.std(X_df[feat]) > 0 else 1.0
        abs_mean_diff = abs(p_mean - s_mean)
        norm_diff = abs_mean_diff / std_comb
        
        numeric_stats[feat] = {
            "safe": {"mean": float(s_mean), "median": float(s_med), "min": float(s_min), "max": float(s_max)},
            "phishing": {"mean": float(p_mean), "median": float(p_med), "min": float(p_min), "max": float(p_max)},
            "abs_mean_diff": float(abs_mean_diff),
            "normalized_diff": float(norm_diff)
        }
        diff_scores.append((feat, abs_mean_diff, norm_diff))

    # 4. Top 10 Features with Largest Differences
    diff_scores.sort(key=lambda x: x[2], reverse=True)
    top_10_features = [x[0] for x in diff_scores[:10]]
    print("\n--- 4. TOP 10 FEATURES WITH LARGEST DISTRIBUTION DIFFERENCES ---")
    for r, (f_name, m_diff, n_diff) in enumerate(diff_scores[:10], 1):
        s_m = numeric_stats[f_name]["safe"]["mean"]
        p_m = numeric_stats[f_name]["phishing"]["mean"]
        print(f"  {r:2d}. {f_name:<28} : Safe Mean = {s_m:8.2f} | Phish Mean = {p_m:8.2f} | (Norm Diff: {n_diff:.4f})")

    # 5. Targeted Feature Focus Analysis
    print("\n--- 5. TARGETED FEATURE INVESTIGATION ---")
    target_stats = {}
    for t_feat in TARGET_FOCUS_FEATURES:
        t_data = numeric_stats[t_feat]
        target_stats[t_feat] = t_data
        print(f"\nFeature: [{t_feat}]")
        print(f"  - Safe (0)   : Mean = {t_data['safe']['mean']:.4f} | Median = {t_data['safe']['median']:.4f} | Range = [{t_data['safe']['min']}, {t_data['safe']['max']}]")
        print(f"  - Phishing(1): Mean = {t_data['phishing']['mean']:.4f} | Median = {t_data['phishing']['median']:.4f} | Range = [{t_data['phishing']['min']}, {t_data['phishing']['max']}]")

    # 6. Suspicious Keyword Count Breakdown
    print("\n--- 6. SUSPICIOUS_KEYWORD_COUNT DETAILED BREAKDOWN ---")
    def get_kw_counts(df_sub):
        n = len(df_sub)
        c0 = (df_sub['suspicious_keyword_count'] == 0).sum() / n * 100
        c1 = (df_sub['suspicious_keyword_count'] == 1).sum() / n * 100
        c2 = (df_sub['suspicious_keyword_count'] == 2).sum() / n * 100
        c3p = (df_sub['suspicious_keyword_count'] >= 3).sum() / n * 100
        return {"0_kw_pct": float(c0), "1_kw_pct": float(c1), "2_kw_pct": float(c2), "3plus_kw_pct": float(c3p)}

    safe_kw_bdown = get_kw_counts(safe_df)
    phish_kw_bdown = get_kw_counts(phish_df)
    
    print("Safe URLs (Label 0):")
    print(f"  - 0 Keywords  : {safe_kw_bdown['0_kw_pct']:.2f}%")
    print(f"  - 1 Keyword   : {safe_kw_bdown['1_kw_pct']:.2f}%")
    print(f"  - 2 Keywords  : {safe_kw_bdown['2_kw_pct']:.2f}%")
    print(f"  - 3+ Keywords : {safe_kw_bdown['3plus_kw_pct']:.2f}%")
    
    print("Phishing URLs (Label 1):")
    print(f"  - 0 Keywords  : {phish_kw_bdown['0_kw_pct']:.2f}%")
    print(f"  - 1 Keyword   : {phish_kw_bdown['1_kw_pct']:.2f}%")
    print(f"  - 2 Keywords  : {phish_kw_bdown['2_kw_pct']:.2f}%")
    print(f"  - 3+ Keywords : {phish_kw_bdown['3plus_kw_pct']:.2f}%")

    # 7 & 8. Categorization of Features
    direct_separators = []
    if binary_stats["is_https"]["abs_diff_pct"] > 80:
        direct_separators.append("is_https (Near-binary separator in PhiUSIIL dataset)")

    feature_categories = {
        "potentially_useful": [
            "path_special_char_count", "path_length", "url_length",
            "url_depth", "hostname_length", "dot_count", "special_char_count"
        ],
        "weakly_informative": [
            "has_at_symbol", "has_punycode", "has_percent_encoding",
            "hostname_hyphen_count", "query_parameter_count"
        ],
        "strongly_dataset_dependent": [
            "is_https", "has_ip", "suspicious_tld", "suspicious_keyword_count"
        ]
    }
    
    print("\n--- 7. DIRECT LABEL SEPARATOR CHECK ---")
    if direct_separators:
        print("  ALERT: The following feature(s) act as near-direct label separators in PhiUSIIL:")
        for ds in direct_separators:
            print(f"   - {ds}")
    else:
        print("  No single feature acts as a 100% perfect label separator.")

    print("\n--- 8. FEATURE UTILITY CATEGORIZATION (PhiUSIIL Dataset) ---")
    print("  1. Potentially Useful (High structural signal):")
    print("     " + ", ".join(feature_categories["potentially_useful"]))
    print("  2. Weakly Informative (Low variance / rare occurrence):")
    print("     " + ", ".join(feature_categories["weakly_informative"]))
    print("  3. Strongly Dataset-Dependent (Dataset specific bias):")
    print("     " + ", ".join(feature_categories["strongly_dataset_dependent"]))

    # Export JSON
    json_data = {
        "dataset_analysis": "PhiUSIIL Dataset Bias & Feature Distribution Analysis",
        "sample_counts": {
            "total_clean_urls": total_samples,
            "safe_urls": safe_samples,
            "phishing_urls": phish_samples
        },
        "binary_feature_distributions": binary_stats,
        "all_numeric_feature_stats": numeric_stats,
        "top_10_distinguishing_features": top_10_features,
        "targeted_investigation": target_stats,
        "keyword_count_breakdown": {
            "safe": safe_kw_bdown,
            "phishing": phish_kw_bdown
        },
        "direct_label_separators": direct_separators,
        "feature_categorization": feature_categories
    }
    
    with open(JSON_OUTPUT_PATH, "w") as f:
        json.dump(json_data, f, indent=4)
        
    print(f"\nSaved full bias analysis JSON to: {JSON_OUTPUT_PATH}")
    print("==================================================")

if __name__ == "__main__":
    analyze_bias()
