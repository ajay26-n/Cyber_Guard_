import os
import sys
import joblib
import pandas as pd

# Add backend to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend"))
sys.path.insert(0, BASE_DIR)

from features import extract_features
from security_checks import is_reachable, check_redirects, check_domain_age, check_ssl, suspicious_tld, brand_impersonation, detect_login_form, suspicious_hosting, detect_download

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "new_phishing_model.pkl"))

test_urls = [
    "google.com",
    "github.com",
    "example.com/login/verify/account"
]

def run_integration_test():
    print("==================================================")
    print("   CYBER-GUARD INTEGRATION TEST (NEW 24-FEAT ML)  ")
    print("==================================================")
    
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found.")
        sys.exit(1)
        
    model = joblib.load(MODEL_PATH)
    print("1. New 24-Feature ML Model successfully loaded.")
    print(f"   Model Features Expected ({len(model.feature_names_in_)}): {model.feature_names_in_.tolist()}")
    
    for url_input in test_urls:
        url = url_input.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
            
        print(f"\n--------------------------------------------------")
        print(f"Target URL: {url_input} ({url})")
        
        feat_dict = extract_features(url)
        print(f"  - Extracted Features ({len(feat_dict)}): {list(feat_dict.keys())[:5]}... Total: {len(feat_dict)}")
        
        df_features = pd.DataFrame([feat_dict])
        
        # Verify columns match exactly
        assert list(df_features.columns) == list(model.feature_names_in_), "Mismatch in feature column names!"
        
        ml_pred = int(model.predict(df_features)[0])
        ml_prob = float(model.predict_proba(df_features)[0][1]) if hasattr(model, "predict_proba") else float(ml_pred)
        
        print(f"  - ML Model Prediction : {ml_pred} ({'PHISHING' if ml_pred == 1 else 'SAFE'})")
        print(f"  - ML Phishing Prob    : {ml_prob * 100:.2f}%")
        
        # Calculate Heuristics
        url_length = feat_dict["url_length"]
        keywords = feat_dict["suspicious_keyword_count"]
        subdomains = feat_dict["subdomain_count"]
        has_ip = feat_dict["has_ip"]
        special_chars = feat_dict["special_char_count"]

        # Fast heuristic calculations (mocking/running reachability safely)
        bad_tld = suspicious_tld(url)
        brand_fake = brand_impersonation(url)
        bad_host = suspicious_hosting(url)
        download = detect_download(url)
        
        risk_score = 0
        if has_ip: risk_score += 30
        if keywords > 0: risk_score += 15
        if special_chars > 2: risk_score += 10
        if bad_tld: risk_score += 20
        if brand_fake: risk_score += 40
        if bad_host: risk_score += 40
        
        # Proportional / High-Confidence ML Scoring matching app.py
        if ml_prob >= 0.80:
            risk_score += 35
        elif ml_prob >= 0.50 or ml_pred == 1:
            risk_score += 20
            
        if download: risk_score += 40
        
        verdict = "PHISHING" if risk_score >= 50 else "SAFE"
        
        print(f"  - Heuristic Risk Score: {risk_score} points")
        print(f"  - Overall Verdict     : {verdict}")
        
    print("==================================================")
    print("       INTEGRATION TEST SUCCESSFUL!               ")
    print("==================================================")

if __name__ == "__main__":
    run_integration_test()
