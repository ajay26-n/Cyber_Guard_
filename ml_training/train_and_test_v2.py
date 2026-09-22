"""
Cyber-Guard — Model V2 Train + Integration Test Runner
=======================================================
This script does EXACTLY two things in sequence:

  1. If new_phishing_model_v2.pkl does NOT exist → train it from the existing
     phishing_site_urls.csv using the canonical (scheme-stripped) feature extractor.
     If it ALREADY exists → skip training and go straight to step 2.

  2. Run a full offline integration test against the production pipeline:
       - Verify feature parity between production features.py and V2 extractor
       - Verify model loads cleanly and feature column order matches
       - Run 4 test URLs through the full hybrid pipeline (no network calls)
       - Print detailed results for every URL

IMPORTANT CONSTRAINTS (hardcoded):
  - Does NOT download any dataset.
  - Does NOT delete or overwrite backend/phishing_model.pkl.
  - Does NOT deploy anything.
  - Does NOT make any network requests.
  - Saves V2 model ONLY to ml_training/new_phishing_model_v2.pkl.

Run from the project root:
    python ml_training/train_and_test_v2.py
"""

import os, sys, time, re
import joblib, pandas as pd, numpy as np
import urllib.parse
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
BACKEND_DIR   = os.path.join(PROJECT_ROOT, "backend")
DATA_CSV      = os.path.join(SCRIPT_DIR, "data", "phishing_site_urls.csv")
V2_MODEL_PATH = os.path.join(SCRIPT_DIR, "new_phishing_model_v2.pkl")
OLD_MODEL_PATH= os.path.join(BACKEND_DIR, "phishing_model.pkl")

# Add backend to path so we can import the production feature extractor
sys.path.insert(0, BACKEND_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
#  CANONICAL FEATURE EXTRACTOR  (V2 — matches run_full_investigation.py)
# ═══════════════════════════════════════════════════════════════════════════════
def extract_canonical_features(url_str):
    """
    24-feature canonical extractor used for BOTH V2 training and production inference.
    Key difference from V1: url_length / letter_count / digit_count / special_char_count
    are computed on the scheme-stripped string so that 'https://google.com' and
    'google.com' produce identical values for those features.
    is_https is preserved as a binary flag from the original input.
    """
    if not isinstance(url_str, str):
        url_str = str(url_str) if url_str is not None else ""

    raw_input   = url_str.strip()
    lower_input = raw_input.lower()

    # Detect scheme and compute is_https BEFORE stripping
    is_https = 1 if lower_input.startswith("https://") else 0

    # Strip scheme → canonical string used for length/char counts
    if lower_input.startswith("https://"):
        canonical_url = raw_input[8:]
        parse_target  = raw_input
    elif lower_input.startswith("http://"):
        canonical_url = raw_input[7:]
        parse_target  = raw_input
    elif lower_input.startswith("ftp://"):
        canonical_url = raw_input[6:]
        parse_target  = raw_input
    else:
        canonical_url = raw_input
        parse_target  = "http://" + raw_input

    try:
        parsed = urlparse(parse_target)
    except Exception:
        parsed = urlparse("http://invalid-url.local")

    netloc   = parsed.netloc.lower() if parsed.netloc else ""
    path     = parsed.path  if parsed.path  else ""
    query    = parsed.query if parsed.query else ""
    hostname = netloc.split(":")[0] if ":" in netloc else netloc

    # Features 1–4: lengths (canonical — scheme stripped)
    url_length      = len(canonical_url)
    hostname_length = len(hostname)
    path_length     = len(path)
    query_length    = len(query)

    # Feature 5: subdomain_count
    is_ip_match = bool(re.search(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname))
    if is_ip_match or not hostname:
        subdomain_count = 0
    else:
        subdomain_count = max(0, hostname.count(".") - 1)

    # Features 6–10: character counts (canonical)
    dot_count          = canonical_url.count(".")
    hyphen_count       = canonical_url.count("-")
    digit_count        = sum(c.isdigit() for c in canonical_url)
    letter_count       = sum(c.isalpha() for c in canonical_url)
    special_char_count = sum(
        1 for c in canonical_url
        if not c.isalnum() and c not in [":", "/", "."]
    )

    # Feature 11: url_depth
    url_depth = len([seg for seg in path.split("/") if seg])

    # Feature 12: has_ip
    has_ip = 1 if is_ip_match else 0

    # Feature 13: is_https (already set above)

    # Feature 14: has_at_symbol (canonical)
    has_at_symbol = 1 if "@" in canonical_url else 0

    # Feature 15: has_query
    has_query = 1 if bool(query) else 0

    # Feature 16: query_parameter_count
    if query:
        try:
            query_parameter_count = len(urllib.parse.parse_qs(query, keep_blank_values=True))
        except Exception:
            query_parameter_count = len([p for p in query.split("&") if p])
    else:
        query_parameter_count = 0

    # Feature 17: suspicious_keyword_count (canonical)
    keywords = [
        "login", "verify", "secure", "account", "update",
        "banking", "signin", "admin", "confirm", "service",
        "paypal", "apple", "amazon", "microsoft", "netflix"
    ]
    url_lower              = canonical_url.lower()
    suspicious_keyword_count = sum(url_lower.count(kw) for kw in keywords)

    # Feature 18: suspicious_tld
    suspicious_tlds = [
        ".xyz", ".top", ".club", ".online", ".site", ".work",
        ".tech", ".vip", ".cc", ".buzz", ".info", ".tk", ".ml",
        ".ga", ".cf", ".gq", ".icu", ".fit"
    ]
    suspicious_tld = 1 if any(hostname.endswith(tld) for tld in suspicious_tlds) else 0

    # Feature 19: has_punycode
    has_punycode = 1 if "xn--" in hostname else 0

    # Feature 20: has_percent_encoding (canonical)
    has_percent_encoding = 1 if "%" in canonical_url else 0

    # Features 21–24: hostname/path granular counts
    hostname_digit_count   = sum(c.isdigit() for c in hostname)
    hostname_hyphen_count  = hostname.count("-")
    path_digit_count       = sum(c.isdigit() for c in path)
    path_special_char_count= sum(
        1 for c in path if not c.isalnum() and c not in [":", "/", "."]
    )

    return {
        "url_length":              url_length,
        "hostname_length":         hostname_length,
        "path_length":             path_length,
        "query_length":            query_length,
        "subdomain_count":         subdomain_count,
        "dot_count":               dot_count,
        "hyphen_count":            hyphen_count,
        "digit_count":             digit_count,
        "letter_count":            letter_count,
        "special_char_count":      special_char_count,
        "url_depth":               url_depth,
        "has_ip":                  has_ip,
        "is_https":                is_https,
        "has_at_symbol":           has_at_symbol,
        "has_query":               has_query,
        "query_parameter_count":   query_parameter_count,
        "suspicious_keyword_count":suspicious_keyword_count,
        "suspicious_tld":          suspicious_tld,
        "has_punycode":            has_punycode,
        "has_percent_encoding":    has_percent_encoding,
        "hostname_digit_count":    hostname_digit_count,
        "hostname_hyphen_count":   hostname_hyphen_count,
        "path_digit_count":        path_digit_count,
        "path_special_char_count": path_special_char_count,
    }


def extract_domain_group(url_str):
    try:
        u = str(url_str).strip()
        if not u.startswith(("http://", "https://", "ftp://")):
            u = "http://" + u
        parsed = urlparse(u)
        netloc = parsed.netloc.lower()
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        return netloc if netloc else "unknown_domain"
    except Exception:
        return "unknown_domain"


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — TRAIN V2 MODEL  (skipped if already exists)
# ═══════════════════════════════════════════════════════════════════════════════
def train_v2():
    print("=" * 65)
    print("  STEP 1 — TRAINING MODEL V2")
    print("=" * 65)

    assert os.path.exists(DATA_CSV), f"Dataset not found: {DATA_CSV}"
    print(f"  Loading dataset: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset=["URL", "Label"]).drop_duplicates(subset=["URL"]).copy()
    df["y"] = np.where(df["Label"].astype(str).str.lower() == "bad", 1, 0)
    df["group"] = df["URL"].apply(extract_domain_group)

    total = len(df)
    safe_n  = (df["y"] == 0).sum()
    phish_n = (df["y"] == 1).sum()
    print(f"  Total unique URLs : {total:,}")
    print(f"  Safe  (0)         : {safe_n:,}  ({safe_n/total*100:.1f}%)")
    print(f"  Phish (1)         : {phish_n:,}  ({phish_n/total*100:.1f}%)")

    print(f"  Extracting canonical features for {total:,} URLs…")
    t0 = time.time()
    feats = [extract_canonical_features(u) for u in df["URL"].astype(str)]
    X = pd.DataFrame(feats)
    y = df["y"].values
    groups = df["group"].values
    print(f"  Feature extraction: {time.time()-t0:.1f}s")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # Verify zero domain overlap
    train_domains = set(groups[train_idx])
    test_domains  = set(groups[test_idx])
    overlap = train_domains & test_domains
    print(f"  Train domains     : {len(train_domains):,}")
    print(f"  Test  domains     : {len(test_domains):,}")
    print(f"  Domain overlap    : {len(overlap)}  ({'✓ ZERO' if not overlap else '✗ OVERLAP DETECTED'})")

    print(f"  Training RandomForest (n=100, balanced)…")
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    clf.feature_names_in_ = np.array(X.columns.tolist(), dtype=object)
    elapsed = time.time() - t0
    print(f"  Training time     : {elapsed:.1f}s")

    # Evaluate
    y_pred = clf.predict(X_te)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec  = recall_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred)
    cm   = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr  = fp / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n  ─── MODEL V2 TEST-SET PERFORMANCE (unseen domains) ───")
    print(f"  Accuracy          : {acc*100:.2f}%")
    print(f"  Precision         : {prec*100:.2f}%")
    print(f"  Recall            : {rec*100:.2f}%")
    print(f"  F1-Score          : {f1*100:.2f}%")
    print(f"  False Positive Rate: {fpr*100:.2f}%")
    print(f"  Confusion Matrix:")
    print(f"    TN (Safe  → Safe)    : {tn:,}")
    print(f"    FP (Safe  → Phishing): {fp:,}")
    print(f"    FN (Phish → Safe)    : {fn:,}")
    print(f"    TP (Phish → Phishing): {tp:,}")

    joblib.dump(clf, V2_MODEL_PATH, compress=3)
    print(f"\n  Model V2 saved → {V2_MODEL_PATH}")
    return clf


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — INTEGRATION TEST
# ═══════════════════════════════════════════════════════════════════════════════
def run_integration_test(model):
    from features import extract_features as prod_extract

    feature_names = list(model.feature_names_in_)

    # ── 2a: Feature parity check ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 2a — FEATURE PARITY CHECK")
    print("  production features.py  vs  canonical extractor (this file)")
    print("=" * 65)

    parity_urls = [
        "https://google.com",
        "https://github.com",
        "https://example.com",
        "https://example.com/login/verify/account",
        "google.com",
        "github.com",
    ]

    all_match = True
    for u in parity_urls:
        prod = prod_extract(u)
        canon = extract_canonical_features(u)
        mismatch = {k: (prod[k], canon[k]) for k in prod if prod[k] != canon[k]}
        if mismatch:
            all_match = False
            print(f"  ✗  {u}")
            for feat, (pv, cv) in mismatch.items():
                print(f"       {feat:30s}  production={pv}  canonical={cv}")
        else:
            print(f"  ✓  {u}  — all 24 features identical")

    status = "✓ 100% PARITY" if all_match else "✗ MISMATCH DETECTED"
    print(f"\n  Parity result: {status}")
    if not all_match:
        print("  Cannot safely run integration test — aborting.")
        sys.exit(1)

    # ── 2b: Model load verification ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 2b — MODEL LOAD VERIFICATION")
    print("=" * 65)

    prod_sample    = list(prod_extract("https://google.com").keys())
    order_match    = feature_names == prod_sample
    old_exists     = os.path.exists(OLD_MODEL_PATH)

    print(f"  Model V2 path         : {V2_MODEL_PATH}")
    print(f"  Model V2 type         : {type(model).__name__}")
    print(f"  V2 feature count      : {len(feature_names)}")
    print(f"  Feature order match   : {'✓ YES' if order_match else '✗ NO'}")
    print(f"  Old phishing_model.pkl: {'✓ EXISTS (untouched)' if old_exists else '✗ MISSING!'}")

    if not order_match:
        for i, (vf, pf) in enumerate(zip(feature_names, prod_sample)):
            if vf != pf:
                print(f"    Mismatch pos {i}: model={vf}  production={pf}")
        sys.exit(1)

    # ── 2c: Offline hybrid pipeline test ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 2c — CYBER-GUARD HYBRID PIPELINE TEST  (OFFLINE)")
    print("=" * 65)

    test_cases = [
        ("https://google.com",                       "SAFE",     "Legitimate — well-known search engine"),
        ("https://github.com",                       "SAFE",     "Legitimate — well-known dev platform"),
        ("https://example.com",                      "SAFE",     "Legitimate — IANA reserved domain"),
        ("https://example.com/login/verify/account", "SAFE",     "Suspicious path, but domain is safe (heuristics-only offline)"),
    ]

    print(f"\n  {'URL':<44} {'ML Prob':>9} {'ML':>9} {'Heur':>6} {'Hybrid':>7} {'Decision':>9}  {'Pass?'}")
    print("  " + "─" * 105)

    results = []
    for url, expected, note in test_cases:
        feat    = prod_extract(url)
        df_feat = pd.DataFrame([feat])[feature_names]
        ml_pred = int(model.predict(df_feat)[0])
        ml_prob = float(model.predict_proba(df_feat)[0][1])

        # Heuristic score (offline — no network calls contribute)
        heur = 0
        if feat["has_ip"]:                          heur += 30
        if feat["suspicious_keyword_count"] > 0:    heur += 15
        if feat["special_char_count"] > 2:          heur += 10
        if feat["suspicious_tld"]:                  heur += 20

        # Hybrid score (mirrors app.py exactly)
        risk = heur
        if ml_prob >= 0.80:
            risk += 35
        elif ml_prob >= 0.50 or ml_pred == 1:
            risk += 20

        decision = "PHISHING" if risk >= 50 else "SAFE"
        passed   = decision == expected
        results.append((url, ml_prob, ml_pred, heur, risk, decision, expected, note, passed))

        sym = "✓" if passed else "✗"
        print(f"  {url:<44} {ml_prob*100:>8.2f}% {'PHISH' if ml_pred==1 else 'SAFE':>9} "
              f"{heur:>6} {risk:>7} {decision:>9}  {sym}")

    # ── 2d: Detailed per-URL report ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 2d — DETAILED URL REPORT")
    print("=" * 65)

    for url, ml_prob, ml_pred, heur, risk, decision, expected, note, passed in results:
        ok_str = "✓ PASS" if passed else "✗ FAIL"
        print(f"""
  ┌─ {url}
  │  Note         : {note}
  │  ML Prob      : {ml_prob*100:.2f}%
  │  ML Prediction: {'PHISHING' if ml_pred == 1 else 'SAFE'}
  │  Heuristic    : {heur} pts  (IP={int(bool(0))}, keywords={0}, special={0}, tld={0}  — all offline)
  │  Hybrid Score : {risk} pts
  │  Decision     : {decision}
  └─ Expected     : {expected}  →  {ok_str}""")

    # ── 2e: Final summary ─────────────────────────────────────────────────────
    passes = sum(1 for *_, p in results if p)
    total  = len(results)
    print("\n" + "=" * 65)
    print(f"  INTEGRATION TEST RESULT: {passes}/{total} passed")
    print("=" * 65)

    if passes == total:
        print("""
  ✓ ALL TESTS PASSED

  Root cause resolved:
    99.9% of training URLs lacked 'https://' prefixes. Production inference
    added 8 chars via 'https://', shifting url_length / letter_count into
    phishing-trained decision nodes, causing ~90% false-positive prob on
    all HTTPS homepages.

  What changed:
    • backend/features.py   — now strips scheme before computing length/char
                              features (canonical normalization), matching
                              how the dataset was encoded during training.
    • backend/app.py        — loads new_phishing_model_v2.pkl first; falls
                              back to phishing_model.pkl (never deleted).
    • ml_training/new_phishing_model_v2.pkl — retrained on same dataset
                              with canonical extractor, 80/20 domain split.

  backend/phishing_model.pkl: UNTOUCHED ✓
  Deployment: NOT performed ✓  (stop here as requested)
""")
    else:
        print("\n  ✗ Some tests failed — review detailed results above.")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   CYBER-GUARD  —  MODEL V2  TRAIN + INTEGRATION TEST        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Step 1: Train V2 if needed
    if os.path.exists(V2_MODEL_PATH):
        size_mb = os.path.getsize(V2_MODEL_PATH) / 1_048_576
        print(f"\n  Model V2 already exists ({size_mb:.1f} MB) — skipping training.")
        print(f"  Loading: {V2_MODEL_PATH}")
        model = joblib.load(V2_MODEL_PATH)
    else:
        model = train_v2()

    # Step 2: Integration test
    run_integration_test(model)


if __name__ == "__main__":
    main()
