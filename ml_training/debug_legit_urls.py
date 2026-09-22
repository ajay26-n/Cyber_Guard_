import os
import sys
import joblib
import pandas as pd

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
from features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "new_phishing_model.pkl")

test_urls = [
    "https://google.com",
    "https://github.com",
    "https://example.com"
]

def debug_legit_urls():
    print("==================================================")
    print("  DEBUGGING LEGITIMATE URL PREDICTIONS (READ-ONLY) ")
    print("==================================================")
    
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found.")
        sys.exit(1)
        
    model = joblib.load(MODEL_PATH)
    feature_names = model.feature_names_in_.tolist()
    
    print("\n--- MODEL TOP FEATURE IMPORTANCES ---")
    feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
    feat_imp = feat_imp.sort_values(by="Importance", ascending=False)
    for r, row in enumerate(feat_imp.head(10).itertuples(), 1):
        print(f"  {r:2d}. {row.Feature:<28} : {row.Importance * 100:.2f}%")
        
    print("\n--- 24 EXTRACTED FEATURE VALUES FOR EACH TARGET URL ---")
    for url in test_urls:
        feat_dict = extract_features(url)
        df_feat = pd.DataFrame([feat_dict])
        
        pred = model.predict(df_feat)[0]
        prob = model.predict_proba(df_feat)[0][1]
        
        print(f"\nTarget URL: {url}")
        print(f"  - ML Prediction : {pred} ({'PHISHING' if pred == 1 else 'SAFE'})")
        print(f"  - Phishing Prob : {prob * 100:.2f}%")
        print("  - Extracted 24 Features:")
        for k, v in feat_dict.items():
            print(f"      {k:<28} = {v}")

if __name__ == "__main__":
    debug_legit_urls()
