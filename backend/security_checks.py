import requests
import ssl
import socket
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    import whois
except ImportError:
    whois = None

def is_reachable(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    # 1. Try HTTPS/HTTP GET request (verify=False to avoid local SSL bundle issues)
    try:
        r = requests.get(url, headers=headers, timeout=5, verify=False)
        return r.status_code < 500
    except Exception:
        pass

    # 2. Try HTTP fallback if HTTPS failed
    if url.startswith("https://"):
        try:
            http_url = "http://" + url[8:]
            r = requests.get(http_url, headers=headers, timeout=5, verify=False)
            return r.status_code < 500
        except Exception:
            pass

    # 3. Try DNS resolution fallback (for offline, take-down, or DNS-only reachable domains)
    try:
        domain = urlparse(url).netloc
        if domain:
            domain = domain.split(":")[0]
            socket.gethostbyname(domain)
            return True
    except Exception:
        pass

    # 4. Scanner resilience fallback: if domain structure is valid, allow feature scanning
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        if "." in domain and len(domain.split(".")[-1]) >= 2:
            return True
    except Exception:
        pass

    return False

def check_redirects(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, headers=headers, allow_redirects=True, timeout=5, verify=False)
        return len(r.history)
    except Exception:
        return 0

def check_domain_age(url):
    if not whois:
        return None
    try:
        domain = urlparse(url).netloc
        if not domain:
            return None
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date:
            age = (datetime.now() - creation_date).days
            return age
    except:
        pass
    return None

def check_ssl(url):
    try:
        hostname = urlparse(url).netloc
        if not hostname:
            return False
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                return True
    except:
        return False

def suspicious_tld(url):
    suspicious = [".xyz", ".top", ".club", ".online", ".site", ".work", ".tech", ".vip"]
    domain = urlparse(url).netloc
    return any(domain.endswith(tld) for tld in suspicious)

def brand_impersonation(url):
    brands = ["paypal", "google", "apple", "amazon", "bankofamerica", "netflix", "microsoft", "facebook"]
    domain = urlparse(url).netloc.lower()
    for b in brands:
        if b in domain and not domain.endswith(f"{b}.com"):
            return True
    return False

def detect_login_form(url):
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=5, verify=False)
        soup = BeautifulSoup(r.text, "html.parser")
        inputs = soup.find_all("input")
        for i in inputs:
            if i.get("type") == "password":
                return True
    except:
        pass
    return False

def suspicious_hosting(url):
    suspicious = ["ngrok", "serveo", "localtunnel", "pagekite", "000webhostapp"]
    domain = urlparse(url).netloc
    return any(sh in domain for sh in suspicious)

def detect_download(url):
    suspicious_ext = [".exe", ".apk", ".bat", ".cmd", ".msi", ".zip"]
    path = urlparse(url).path
    return any(path.endswith(ext) for ext in suspicious_ext)

def detect_threat_types(url, login_page, bad_host, brand_fake, download):
    threats = []
    if brand_fake:
        threats.append("Brand Impersonation")
    if login_page:
        threats.append("Credential Harvesting")
    if bad_host:
        threats.append("Suspicious Infrastructure")
    if download:
        threats.append("Malicious File Download")
    return threats

def generate_ai_explanation(domain_age, login_page, bad_host, brand_fake):
    reasons = []
    if brand_fake:
        reasons.append("The domain name closely mimics a well-known brand.")
    if login_page:
        reasons.append("The website contains password fields that could capture credentials.")
    if bad_host:
        reasons.append("The website is hosted on a temporary tunneling service.")
    if domain_age and domain_age < 30:
        reasons.append(f"The domain was registered recently ({domain_age} days ago).")
    if not reasons:
        return "No high-risk threat indicators detected."
    return " ".join(reasons)

def get_ip_location(url):
    try:
        hostname = urlparse(url).netloc
        ip = socket.gethostbyname(hostname)
        r = requests.get(f"http://ip-api.com/json/{ip}", timeout=3).json()
        if r.get("status") == "success":
            return f"{r.get('city')}, {r.get('country')}"
    except:
        pass
    return "Unknown"
