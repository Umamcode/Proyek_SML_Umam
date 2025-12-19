import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_data(filepath):
    """Load data bersih dari folder lokal."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {filepath}")
    return pd.read_csv(filepath)

def eval_metrics(actual, pred):
    """Menghitung metrik evaluasi."""
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, zero_division=0)
    recall = recall_score(actual, pred, zero_division=0)
    f1 = f1_score(actual, pred, zero_division=0)
    return accuracy, precision, recall, f1

def run_tuning_and_logging():  
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("Eksperimen_Jantung_Umam")
        
    # Path data 
    data_path = "heart_disease_clean.csv" 
    
    print("[INFO] Loading data...")
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        print("ERROR: File csv belum ada.")
        return

    # Split Data
    X = df.drop('Target', axis=1)
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Starting Hyperparameter Tuning...")

    # Tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluasi
    y_pred = best_model.predict(X_test)
    acc, prec, rec, f1 = eval_metrics(y_test, y_pred)
    
    print(f"[RESULT] Accuracy: {acc:.4f}")

    # Logging ke MLflow
    print("[INFO] Logging to MLflow...")
    
    with mlflow.start_run(run_name="Best_Model_Tuning"):
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(best_model, "model_best_tuning")
        
        # Artifact Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(7,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.tight_layout()
        plt.savefig("confusion_matrix_tuned.jpg")
        mlflow.log_artifact("confusion_matrix_tuned.jpg")
        
        if os.path.exists("confusion_matrix_tuned.jpg"):
            os.remove("confusion_matrix_tuned.jpg")

        print("LOGGING SUKSES! Cek Dashboard.")

if __name__ == "__main__":
    run_tuning_and_logging()