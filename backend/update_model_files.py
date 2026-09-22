import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROD_MODEL_PATH = os.path.join(BASE_DIR, "phishing_model.pkl")
BACKUP_MODEL_PATH = os.path.join(BASE_DIR, "phishing_model_old_backup.pkl")
V2_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../ml_training/new_phishing_model_v2.pkl"))

def consolidate_models():
    print("==================================================")
    print("      CONSOLIDATING TO SINGLE PRODUCTION MODEL    ")
    print("==================================================")
    
    # Copy Model V2 to backend/phishing_model.pkl if present
    if os.path.exists(V2_MODEL_PATH):
        shutil.copy2(V2_MODEL_PATH, PROD_MODEL_PATH)
        print(f"✓ Updated {PROD_MODEL_PATH} with Model V2.")
        os.remove(V2_MODEL_PATH)
        print(f"✓ Removed redundant temporary model: {V2_MODEL_PATH}")
    else:
        print(f"Notice: {V2_MODEL_PATH} already merged or not found.")

    # Remove old backup model if exists
    if os.path.exists(BACKUP_MODEL_PATH):
        os.remove(BACKUP_MODEL_PATH)
        print(f"✓ Removed old backup model: {BACKUP_MODEL_PATH}")

    print(f"\nResult: Single production model active at:\n  {PROD_MODEL_PATH}")

if __name__ == "__main__":
    consolidate_models()
