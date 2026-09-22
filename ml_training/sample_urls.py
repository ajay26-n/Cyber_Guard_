import os
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "phishing_site_urls.csv")

def sample_urls():
    if not os.path.exists(CSV_PATH):
        print("CSV not found.")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    good_samples = df[df['Label'] == 'good']['URL'].sample(n=10, random_state=42).tolist()
    bad_samples = df[df['Label'] == 'bad']['URL'].sample(n=10, random_state=42).tolist()
    
    print("==================================================")
    print("  10 RANDOM 'GOOD' (LEGITIMATE) URL STRINGS       ")
    print("==================================================")
    for i, u in enumerate(good_samples, 1):
        print(f" {i:2d}. {u}")
        
    print("\n==================================================")
    print("  10 RANDOM 'BAD' (PHISHING) URL STRINGS          ")
    print("==================================================")
    for i, u in enumerate(bad_samples, 1):
        print(f" {i:2d}. {u}")
    print("==================================================")

if __name__ == "__main__":
    sample_urls()
