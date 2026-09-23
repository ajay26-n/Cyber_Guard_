import os
import sys
import zipfile
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template, send_file
import joblib
import pandas as pd

# Ensure directory is on python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from features import extract_features
from security_checks import *

TEMPLATE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../templates"))
STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, "../static"))

if not os.path.exists(TEMPLATE_DIR):
    TEMPLATE_DIR = os.path.abspath(os.path.join(BASE_DIR, "templates"))
if not os.path.exists(STATIC_DIR):
    STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, "static"))

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# Model loading: Load V2 model if available, otherwise default to phishing_model.pkl
v2_path = os.path.abspath(os.path.join(BASE_DIR, "../ml_training/new_phishing_model_v2.pkl"))
model_path = v2_path if os.path.exists(v2_path) else os.path.join(BASE_DIR, "phishing_model.pkl")

try:
    model = joblib.load(model_path)
    print(f"Loaded ML model from: {model_path}")
except Exception as e:
    print("Warning: Model load error:", e)
    model = None

def build_extension_zip():
    ext_dir = os.path.abspath(os.path.join(BASE_DIR, "../extension"))
    zip_path = os.path.join(STATIC_DIR, "cyberguard-extension.zip")
    
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(ext_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, ext_dir)
                if not rel_path.startswith("chrome") and not rel_path.startswith("firefox"):
                    zipf.write(full_path, rel_path)
    return zip_path

def block_domain(url):
    # Safely bypass on non-Windows/cloud environments or local hosts
    try:
        hostname = urlparse(url).hostname
        if hostname in ["127.0.0.1", "localhost"]:
            return
        hosts_path = r"C:\Windows\System32\drivers\etc\hosts"
        if not os.path.exists(hosts_path):
            return
        redirect_ip = "127.0.0.1"
        domain = urlparse(url).netloc
        entry = f"{redirect_ip} {domain}\n"
        with open(hosts_path, "r+") as file:
            content = file.read()
            if domain not in content:
                file.write(entry)
    except Exception:
        pass

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/download-extension")
def download_extension():
    try:
        zip_path = build_extension_zip()
        return send_file(zip_path, as_attachment=True, download_name="cyberguard-extension.zip")
    except Exception as e:
        return jsonify({"error": f"Could not create extension package: {str(e)}"}), 500
    return render_template("index.html")

@app.route("/scan", methods=["POST"])
def scan():
    data = request.get_json() or {}
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    hostname = urlparse(url).hostname or ""
    if hostname in ["127.0.0.1", "localhost"] or hostname.endswith("vercel.app"):
        return jsonify({
            "url": url,
            "result": "SAFE",
            "risk_score": 0,
            "domain_age": 999,
            "ssl_valid": True,
            "redirects": 0,
            "login_page": False,
            "threat_types": [],
            "ai_explanation": "CyberGuard web application server.",
            "location": "Local Machine",
            "features": {"length": len(url), "keywords": 0, "subdomains": 0, "has_ip": False, "special_chars": 0}
        })

    if not is_reachable(url):
        return jsonify({"error": "Website unreachable"}), 400

    feat_dict = extract_features(url)
    
    url_length = feat_dict["url_length"]
    keywords = feat_dict["suspicious_keyword_count"]
    subdomains = feat_dict["subdomain_count"]
    has_ip = feat_dict["has_ip"]
    special_chars = feat_dict["special_char_count"]

    redirects = check_redirects(url)
    domain_age = check_domain_age(url)
    ssl_valid = check_ssl(url)

    bad_tld = suspicious_tld(url)
    brand_fake = brand_impersonation(url)
    login_page = detect_login_form(url)
    bad_host = suspicious_hosting(url)
    download = detect_download(url)

    ml_prediction = 0
    ml_prob = 0.0
    if model:
        try:
            df_features = pd.DataFrame([feat_dict])
            ml_prediction = int(model.predict(df_features)[0])
            if hasattr(model, "predict_proba"):
                ml_prob = float(model.predict_proba(df_features)[0][1])
            else:
                ml_prob = float(ml_prediction)
        except Exception as e:
            print("Model prediction error:", e)
            ml_prediction = 0
            ml_prob = 0.0

    risk_score = 0
    if has_ip: risk_score += 30
    if keywords > 0: risk_score += 15
    if special_chars > 2: risk_score += 10
    if redirects > 2: risk_score += 15
    if bad_tld: risk_score += 20
    if brand_fake: risk_score += 40
    if bad_host: risk_score += 40
    if login_page: risk_score += 20
    if domain_age and domain_age < 30: risk_score += 20
    if not ssl_valid: risk_score += 10
    
    # Proportional / High-Confidence ML Scoring
    if ml_prob >= 0.80:
        risk_score += 35
    elif ml_prob >= 0.50 or ml_prediction == 1:
        risk_score += 20

    if download: risk_score += 40

    prediction = "PHISHING" if risk_score >= 50 else "SAFE"

    if prediction == "PHISHING":
        block_domain(url)

    threats = detect_threat_types(url, login_page, bad_host, brand_fake, download)
    explanation = generate_ai_explanation(domain_age, login_page, bad_host, brand_fake)
    location = get_ip_location(url)

    return jsonify({
        "url": url,
        "result": prediction,
        "risk_score": risk_score,
        "domain_age": domain_age,
        "ssl_valid": ssl_valid,
        "redirects": redirects,
        "login_page": login_page,
        "threat_types": threats,
        "ai_explanation": explanation,
        "location": location,
        "features": {
            "length": url_length,
            "keywords": keywords,
            "subdomains": subdomains,
            "has_ip": has_ip,
            "special_chars": special_chars
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
