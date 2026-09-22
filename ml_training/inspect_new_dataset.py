import os
import sys
import pandas as pd
import numpy as np
from urllib.parse import urlparse

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "phishing_site_urls.csv")

def inspect_dataset():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        sys.exit(1)
        
    print("==================================================")
    print("   INSPECTING DATASET: phishing_site_urls.csv     ")
    print("==================================================")
    
    df = pd.read_csv(CSV_PATH)
    
    print(f"1. CSV Filename : {os.path.basename(CSV_PATH)}")
    print(f"2. Number of Rows : {len(df):,}")
    print(f"3. Column Names   : {df.columns.tolist()}")
    
    url_col = 'URL' if 'URL' in df.columns else ('url' if 'url' in df.columns else df.columns[0])
    label_col = 'Label' if 'Label' in df.columns else ('label' if 'label' in df.columns else df.columns[1])
    
    print(f"\n4. Label Value Counts:")
    print(df[label_col].value_counts().to_dict())
    
    missing_urls = df[url_col].isna().sum()
    duplicate_urls = df[url_col].duplicated().sum()
    
    print(f"5. Missing URLs   : {missing_urls}")
    print(f"6. Duplicate URLs : {duplicate_urls:,}")
    
    # Clean temporary dataframe for protocol & path stats
    df_clean = df.dropna(subset=[url_col, label_col]).copy()
    urls = df_clean[url_col].astype(str).tolist()
    labels = df_clean[label_col].astype(str).tolist()
    
    # Compute HTTPS and Path presence
    has_https = []
    has_path = []
    
    for u in urls:
        u_lower = u.lower().strip()
        is_sec = u_lower.startswith("https://")
        has_https.append(is_sec)
        
        # Check path
        parse_target = u_lower if u_lower.startswith(("http://", "https://")) else "http://" + u_lower
        try:
            parsed = urlparse(parse_target)
            p = parsed.path
            path_exists = len(p) > 0 and p != "/"
        except Exception:
            path_exists = False
        has_path.append(path_exists)
        
    df_clean['is_https'] = has_https
    df_clean['has_path'] = has_path
    
    print("\n--- 7. HTTPS & PATH STATS BY CLASS ---")
    groups = df_clean.groupby(label_col)
    
    for name, group in groups:
        n = len(group)
        https_pct = (group['is_https'].sum() / n) * 100
        path_pct = (group['has_path'].sum() / n) * 100
        print(f"Class [{name}] (Total: {n:,}):")
        print(f"  - HTTPS Percentage : {https_pct:.2f}%")
        print(f"  - Path Percentage  : {path_pct:.2f}%")
        
    print("==================================================")

if __name__ == "__main__":
    inspect_dataset()
