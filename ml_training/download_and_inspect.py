import os
import sys
import zipfile
import urllib.request
import pandas as pd

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ZIP_PATH = os.path.join(DATA_DIR, "phiusiil.zip")
CSV_PATH = os.path.join(DATA_DIR, "PhiUSIIL_Phishing_URL_Dataset.csv")

os.makedirs(DATA_DIR, exist_ok=True)

UCI_URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"

print(f"Downloading official dataset from UCI Repository: {UCI_URL} ...")

try:
    req = urllib.request.Request(UCI_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(ZIP_PATH, 'wb') as out_file:
        out_file.write(response.read())
    print("Download complete. Extracting CSV...")
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith('.csv'):
                file_info.filename = "PhiUSIIL_Phishing_URL_Dataset.csv"
                zip_ref.extract(file_info, DATA_DIR)
                break
    
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        
    print(f"Successfully saved to: {CSV_PATH}")
    
except Exception as e:
    print(f"Download/Extraction error: {e}")
    sys.exit(1)

# Inspect CSV with pandas
print("\n--- DATASET INSPECTION RESULTS ---")
df = pd.read_csv(CSV_PATH)

print(f"1. Number of Rows: {len(df):,}")
print(f"2. Number of Columns: {len(df.columns)}")
print(f"3. Exact Column Names:\n{df.columns.tolist()}")

if 'label' in df.columns:
    print(f"\n4. Label Value Counts:\n{df['label'].value_counts().to_dict()}")
else:
    print("\n4. Label Column: Not Found")

url_col = 'URL' if 'URL' in df.columns else ('url' if 'url' in df.columns else None)
if url_col:
    num_missing_urls = df[url_col].isna().sum()
    num_duplicate_urls = df[url_col].duplicated().sum()
    print(f"5. Number of Duplicate URLs: {num_duplicate_urls:,}")
    print(f"6. Number of Missing URLs: {num_missing_urls}")
else:
    print("5. URL Column: Not Found")

label_col = 'label' if 'label' in df.columns else None
if label_col:
    num_missing_labels = df[label_col].isna().sum()
    print(f"7. Number of Missing Labels: {num_missing_labels}")
