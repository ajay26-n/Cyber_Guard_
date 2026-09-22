import re
import urllib.parse

def extract_ml_features(url):
    """
    Extracts 24 URL-only lexical features from a URL string.
    Never performs network requests or DNS/WHOIS lookups.
    Handles malformed URLs safely.
    Returns a dictionary of exactly 24 features in consistent order.
    """
    if not isinstance(url, str):
        url = str(url) if url is not None else ""
        
    url = url.strip()
    
    # Handle missing scheme safely for parsing
    parse_url = url
    if not parse_url.startswith(("http://", "https://", "ftp://")):
        parse_url = "http://" + parse_url

    try:
        parsed = urllib.parse.urlparse(parse_url)
    except Exception:
        parsed = urllib.parse.urlparse("http://invalid-url.local")

    scheme = parsed.scheme.lower() if parsed.scheme else ""
    netloc = parsed.netloc.lower() if parsed.netloc else ""
    path = parsed.path if parsed.path else ""
    query = parsed.query if parsed.query else ""
    
    # Remove port from netloc if present
    hostname = netloc.split(":")[0] if ":" in netloc else netloc

    # 1. url_length
    url_length = len(url)

    # 2. hostname_length
    hostname_length = len(hostname)

    # 3. path_length
    path_length = len(path)

    # 4. query_length
    query_length = len(query)

    # 5. subdomain_count
    # Check if IP address first
    is_ip_match = bool(re.search(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname))
    if is_ip_match or not hostname:
        subdomain_count = 0
    else:
        dots_in_host = hostname.count(".")
        subdomain_count = max(0, dots_in_host - 1)

    # 6. dot_count
    dot_count = url.count(".")

    # 7. hyphen_count
    hyphen_count = url.count("-")

    # 8. digit_count
    digit_count = sum(c.isdigit() for c in url)

    # 9. letter_count
    letter_count = sum(c.isalpha() for c in url)

    # 10. special_char_count (Excludes standard syntax ':', '/', '.')
    special_char_count = sum(1 for c in url if not c.isalnum() and c not in [":", "/", "."])

    # 11. url_depth
    path_segments = [seg for seg in path.split("/") if seg]
    url_depth = len(path_segments)

    # 12. has_ip
    has_ip = 1 if is_ip_match else 0

    # 13. is_https
    # Check actual original URL or parsed scheme
    is_https = 1 if url.lower().startswith("https://") or scheme == "https" else 0

    # 14. has_at_symbol
    has_at_symbol = 1 if "@" in url else 0

    # 15. has_query
    has_query = 1 if bool(query) else 0

    # 16. query_parameter_count
    if query:
        try:
            params = urllib.parse.parse_qs(query, keep_blank_values=True)
            query_parameter_count = len(params)
        except Exception:
            query_parameter_count = len([p for p in query.split("&") if p])
    else:
        query_parameter_count = 0

    # 17. suspicious_keyword_count
    keywords = [
        "login", "verify", "secure", "account", "update", 
        "banking", "signin", "admin", "confirm", "service",
        "paypal", "apple", "amazon", "microsoft", "netflix"
    ]
    url_lower = url.lower()
    suspicious_keyword_count = sum(url_lower.count(kw) for kw in keywords)

    # 18. suspicious_tld
    suspicious_tlds = [
        ".xyz", ".top", ".club", ".online", ".site", ".work", 
        ".tech", ".vip", ".cc", ".buzz", ".info", ".tk", ".ml", 
        ".ga", ".cf", ".gq", ".icu", ".fit"
    ]
    suspicious_tld = 1 if any(hostname.endswith(tld) for tld in suspicious_tlds) else 0

    # 19. has_punycode
    has_punycode = 1 if "xn--" in hostname else 0

    # 20. has_percent_encoding
    has_percent_encoding = 1 if "%" in url else 0

    # 21. hostname_digit_count
    hostname_digit_count = sum(c.isdigit() for c in hostname)

    # 22. hostname_hyphen_count
    hostname_hyphen_count = hostname.count("-")

    # 23. path_digit_count
    path_digit_count = sum(c.isdigit() for c in path)

    # 24. path_special_char_count (Excludes standard syntax ':', '/', '.')
    path_special_char_count = sum(1 for c in path if not c.isalnum() and c not in [":", "/", "."])

    return {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "query_length": query_length,
        "subdomain_count": subdomain_count,
        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "digit_count": digit_count,
        "letter_count": letter_count,
        "special_char_count": special_char_count,
        "url_depth": url_depth,
        "has_ip": has_ip,
        "is_https": is_https,
        "has_at_symbol": has_at_symbol,
        "has_query": has_query,
        "query_parameter_count": query_parameter_count,
        "suspicious_keyword_count": suspicious_keyword_count,
        "suspicious_tld": suspicious_tld,
        "has_punycode": has_punycode,
        "has_percent_encoding": has_percent_encoding,
        "hostname_digit_count": hostname_digit_count,
        "hostname_hyphen_count": hostname_hyphen_count,
        "path_digit_count": path_digit_count,
        "path_special_char_count": path_special_char_count
    }
