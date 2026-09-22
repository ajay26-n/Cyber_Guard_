import json
from feature_extractor import extract_ml_features

test_urls = [
    "https://google.com",
    "https://github.com",
    "http://example.com/login/verify/account",
    "https://example.com/products/item",
    "https://192.168.1.10/login"
]

def main():
    print("==================================================")
    print("       CYBER-GUARD ML FEATURE EXTRACTOR TEST      ")
    print("==================================================")
    
    for i, url in enumerate(test_urls, 1):
        features = extract_ml_features(url)
        print(f"\n--- [TEST {i}] {url} ---")
        print(f"Feature Count: {len(features)}")
        print("Dictionary Result:")
        print(json.dumps(features, indent=4))
        
        # Verify 24 features presence
        assert len(features) == 24, f"Expected 24 features, got {len(features)}"

if __name__ == "__main__":
    main()
