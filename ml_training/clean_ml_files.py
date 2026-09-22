import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Large redundant model files to remove (saves >120MB and fixes GitHub 100MB file limit)
REDUNDANT_MODELS = [
    os.path.join(BASE_DIR, "new_phishing_model.pkl"),
    os.path.join(BASE_DIR, "models", "ablation_all_features.pkl"),
    os.path.join(BASE_DIR, "models", "ablation_without_https.pkl"),
    os.path.join(BASE_DIR, "models", "domain_split_random_forest.pkl"),
    os.path.join(BASE_DIR, "models", "ablation_https_results.json"),
    os.path.join(BASE_DIR, "models", "dataset_bias_analysis.json"),
    os.path.join(BASE_DIR, "models", "domain_split_results.json"),
]

# One-off experimental research scripts no longer needed for production
EXPERIMENTAL_SCRIPTS = [
    "ablation_https.py",
    "analyze_dataset_bias.py",
    "audit_model_leakage.py",
    "compare_models.py",
    "debug_legit_urls.py",
    "download_and_inspect.py",
    "inspect_new_dataset.py",
    "prepare_dataset.py",
    "run_full_investigation.py",
    "sample_urls.py",
    "test_app_integration.py",
    "test_feature_extractor.py",
    "train_domain_split.py",
    "train_model.py",
    "train_new_dataset_model.py"
]

def clean():
    print("==================================================")
    print("      CLEANING UNNECESSARY EXPERIMENTAL FILES     ")
    print("==================================================")
    
    freed_bytes = 0

    # 1. Remove heavy pickle files
    for filepath in REDUNDANT_MODELS:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            os.remove(filepath)
            freed_bytes += size
            print(f"✓ Removed redundant model file ({size / 1024 / 1024:.2f} MB): {os.path.basename(filepath)}")

    # Remove models dir if empty
    models_dir = os.path.join(BASE_DIR, "models")
    if os.path.exists(models_dir):
        try:
            if not os.listdir(models_dir):
                os.rmdir(models_dir)
                print("✓ Removed empty models/ folder.")
        except Exception:
            pass

    # 2. Remove intermediate experimental scripts
    for script_name in EXPERIMENTAL_SCRIPTS:
        script_path = os.path.join(BASE_DIR, script_name)
        if os.path.exists(script_path):
            os.remove(script_path)
            print(f"✓ Removed experimental script: {script_name}")

    print(f"\nClean up complete! Freed approx {freed_bytes / 1024 / 1024:.2f} MB.")
    print("Maintained essential scripts:\n  - augment_and_retrain.py\n  - feature_extractor.py\n  - integration_test_v2.py")

if __name__ == "__main__":
    clean()
