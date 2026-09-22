import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_dataset.joblib")
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "new_phishing_model.pkl")

def train_and_evaluate():
    start_time = time.time()
    print("==================================================")
    print("  CYBER-GUARD ML MODEL TRAINING & EVALUATION PIPELINE ")
    print("==================================================")
    
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Processed dataset not found at {PROCESSED_DATA_PATH}")
        print("Please run python ml_training/prepare_dataset.py first.")
        sys.exit(1)
        
    print(f"\n[1/4] Loading processed dataset artifact...")
    dataset = joblib.load(PROCESSED_DATA_PATH)
    
    X_train = dataset["X_train"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"]
    y_test = dataset["y_test"]
    feature_names = dataset["feature_names"]
    
    print(f"      Features Count: {len(feature_names)}")
    print(f"      Training Samples: {len(X_train):,}")
    print(f"      Testing Samples:  {len(X_test):,}")
    
    # 2. Train RandomForestClassifier
    print("\n[2/4] Training RandomForestClassifier (100 trees, parallel execution)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    
    train_start = time.time()
    clf.fit(X_train, y_train)
    train_elapsed = time.time() - train_start
    print(f"      Model training completed in {train_elapsed:.2f} seconds.")
    
    # 3. Model Evaluation
    print("\n[3/4] Evaluating model performance on 20% unseen test set...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n--- MODEL PERFORMANCE METRICS ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    
    print("\n--- CONFUSION MATRIX ---")
    print(f"True Negatives  (Safe correctly identified):     {cm[0][0]:,}")
    print(f"False Positives (Safe wrongly flagged):         {cm[0][1]:,}")
    print(f"False Negatives (Phishing missed):               {cm[1][0]:,}")
    print(f"True Positives  (Phishing correctly caught):    {cm[1][1]:,}")
    
    print("\n--- DETAILED CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=["Safe (0)", "Phishing (1)"]))
    
    print("--- TOP 10 FEATURE IMPORTANCES ---")
    importances = clf.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    for rank, row in enumerate(feat_imp_df.head(10).itertuples(), 1):
        print(f"  {rank:2d}. {row.Feature:<28} : {row.Importance * 100:.2f}%")
        
    # 4. Export Model Artifact
    print(f"\n[4/4] Exporting trained model artifact to: {MODEL_OUTPUT_PATH}...")
    joblib.dump(clf, MODEL_OUTPUT_PATH, compress=3)
    
    total_elapsed = time.time() - start_time
    print(f"\n==================================================")
    print(f" TRAINING PIPELINE COMPLETE IN {total_elapsed:.2f} SECONDS ")
    print(f"==================================================")

if __name__ == "__main__":
    train_and_evaluate()
