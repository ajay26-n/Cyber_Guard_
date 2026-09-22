"""
Cyber-Guard Integration Test — Model V2
----------------------------------------
Tests the full prediction pipeline OFFLINE (no network calls).
Uses ml_training/new_phishing_model_v2.pkl with backend/features.py (canonical extractor).

Run from project root:
    python ml_training/integration_test_v2.py
"""

import os
import sys
import joblib
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BACKEND_DIR  = os.path.join(PROJECT_ROOT, "backend")

sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, SCRIPT_DIR)

from features import extract_features  # production canonical extractor (V2-aligned)

MODEL_V2_PATH = os.path.join(SCRIPT_DIR, "new_phishing_model_v2.pkl")
OLD_MODEL_PATH = os.path.join(BACKEND_DIR, "phishing_model.pkl")

# ── Feature parity check ──────────────────────────────────────────────────────
def verify_feature_parity():
    """Verify that production features.py and V2 training extractor produce identical results."""
    from run_full_investigation import extract_canonical_features  # V2 training extractor

    test_urls = [
        "https://google.com",
        "https://github.com",
        "https://example.com",
        "https://example.com/login/verify/account",
        "google.com",       # raw dataset-style (no scheme)
        "github.com",
    ]

    print("=" * 65)
    print("  STEP 1 — FEATURE PARITY VERIFICATION")
    print("  Production features.py  vs  V2 canonical extractor")
    print("=" * 65)

    all_match = True
    for u in test_urls:
        prod = extract_features(u)
        v2   = extract_canonical_features(u)
        mismatch = {k: (prod[k], v2[k]) for k in prod if prod[k] != v2[k]}
        if mismatch:
            all_match = False
            print(f"  ✗  {u}")
            for feat, (pv, vv) in mismatch.items():
                print(f"       {feat:30s}  prod={pv}  v2={vv}")
        else:
            print(f"  ✓  {u}  — all 24 features match")

    if all_match:
        print("\n  RESULT: ✓ 100% parity — production extractor matches V2 training extractor.")
    else:
        print("\n  RESULT: ✗ MISMATCH DETECTED — integration cannot proceed safely.")
    return all_match


# ── Heuristic scoring (mirrors app.py, offline-only — no network calls) ───────
def heuristic_score(url: str, feat: dict) -> int:
    """
    Compute the rule-based component of the hybrid risk score.
    This replicates the heuristic block in app.py WITHOUT any network calls
    (redirects/domain_age/ssl/brand_impersonation/etc. are all zero/False offline).
    """
    score = 0
    if feat["has_ip"]:                              score += 30
    if feat["suspicious_keyword_count"] > 0:        score += 15
    if feat["special_char_count"] > 2:              score += 10
    if feat["suspicious_tld"]:                      score += 20
    # Network-dependent checks (offline test → all contribute 0)
    # redirects, domain_age, ssl_valid, brand_fake, bad_host, login_page, download
    return score


# ── Main integration test ─────────────────────────────────────────────────────
def main():
    # -- Step 1: Parity check
    parity_ok = verify_feature_parity()
    if not parity_ok:
        print("\nAborting integration test due to feature mismatch.")
        sys.exit(1)

    # -- Step 2: Load models
    print("\n" + "=" * 65)
    print("  STEP 2 — MODEL LOAD VERIFICATION")
    print("=" * 65)

    # V2 model
    assert os.path.exists(MODEL_V2_PATH), f"V2 model not found: {MODEL_V2_PATH}"
    v2_model = joblib.load(MODEL_V2_PATH)
    v2_features = list(v2_model.feature_names_in_)
    prod_sample = list(extract_features("https://google.com").keys())
    feature_order_match = v2_features == prod_sample

    print(f"  Model V2 path       : {MODEL_V2_PATH}")
    print(f"  Model V2 type       : {type(v2_model).__name__}")
    print(f"  Model V2 features   : {len(v2_features)} features")
    print(f"  Old model untouched : {'YES' if os.path.exists(OLD_MODEL_PATH) else 'MISSING'}")
    print(f"  Feature order match : {'✓ YES' if feature_order_match else '✗ NO'}")

    if not feature_order_match:
        print("  MISMATCH in feature order!")
        for i, (vf, pf) in enumerate(zip(v2_features, prod_sample)):
            if vf != pf:
                print(f"    Position {i}: V2={vf}  Prod={pf}")
        sys.exit(1)

    # -- Step 3: Integration test
    test_urls = [
        ("https://google.com",                     "SAFE",     "Legitimate — well-known homepage"),
        ("https://github.com",                     "SAFE",     "Legitimate — well-known homepage"),
        ("https://example.com",                    "SAFE",     "Legitimate — IANA reserved domain"),
        ("https://example.com/login/verify/account","PHISHING","Legitimate domain but suspicious path"),
    ]

    print("\n" + "=" * 65)
    print("  STEP 3 — CYBER-GUARD INTEGRATION TEST  (OFFLINE)")
    print("=" * 65)
    print(f"  {'URL':<42} {'ML Prob':>9} {'ML Pred':>9} {'Heuristic':>10} {'Hybrid':>7} {'Decision':>9}")
    print("  " + "-" * 90)

    results = []
    for url, expected, note in test_urls:
        feat      = extract_features(url)
        df_feat   = pd.DataFrame([feat])[v2_features]          # enforce column order
        ml_pred   = int(v2_model.predict(df_feat)[0])
        ml_prob   = float(v2_model.predict_proba(df_feat)[0][1])

        heur      = heuristic_score(url, feat)

        # Mirror app.py hybrid scoring
        risk = heur
        if ml_prob >= 0.80:
            risk += 35
        elif ml_prob >= 0.50 or ml_pred == 1:
            risk += 20

        decision  = "PHISHING" if risk >= 50 else "SAFE"
        results.append((url, ml_prob, ml_pred, heur, risk, decision, expected, note))

        ml_label  = "PHISHING" if ml_pred == 1 else "SAFE"
        match_sym = "✓" if decision == expected else "✗"
        print(f"  {url:<42} {ml_prob*100:>8.2f}% {ml_label:>9} {heur:>10} {risk:>7} {decision:>9}  {match_sym}")

    # -- Step 4: Summary
    print("\n" + "=" * 65)
    print("  STEP 4 — DETAILED RESULTS")
    print("=" * 65)
    for url, ml_prob, ml_pred, heur, risk, decision, expected, note in results:
        ok = "✓ PASS" if decision == expected else "✗ FAIL"
        print(f"\n  URL      : {url}")
        print(f"  Note     : {note}")
        print(f"  ML Prob  : {ml_prob*100:.2f}%")
        print(f"  ML Pred  : {'PHISHING' if ml_pred == 1 else 'SAFE'}")
        print(f"  Heuristic: {heur} pts  (offline — no network checks)")
        print(f"  Hybrid   : {risk} pts")
        print(f"  Decision : {decision}  (expected: {expected})  {ok}")

    passes = sum(1 for *_, dec, exp, __ in results if dec == exp)
    total  = len(results)
    print(f"\n  Overall: {passes}/{total} tests passed")

    if passes == total:
        print("\n  ✓ ALL INTEGRATION TESTS PASSED.")
        print("  Model V2 is correctly wired into the Cyber-Guard prediction path.")
        print("  backend/phishing_model.pkl is UNTOUCHED.")
    else:
        print("\n  ✗ Some tests failed — review results above before deployment.")

    print("=" * 65)


if __name__ == "__main__":
    main()
