import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_model():
    clean_data_path = "../preprocessing/heart_disease_clean.csv"
    
    print("[INFO] Starting Model Training Pipeline...")
    
    # Validasi dataset
    if not os.path.exists(clean_data_path):
        print(f"[ERROR] Clean data not found at: {clean_data_path}")
        print("[HINT] Jalankan script 'automate_Umam.py' di folder preprocessing terlebih dahulu.")
        return

    # 2. Load Data
    print(f"[INFO] Loading processed data from: {clean_data_path}")
    df = pd.read_csv(clean_data_path)
    
    # Pisahkan Fitur (X) dan Target (y)
    if 'Target' not in df.columns:
        print("[ERROR] Column 'Target' not found in dataset.")
        return

    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"[INFO] Training Data: {X_train.shape[0]} rows")
    print(f"[INFO] Testing Data : {X_test.shape[0]} rows")

    #3. Setup MLflow
    experiment_name = "Eksperimen_Jantung_Umam"
    mlflow.set_experiment(experiment_name)
    
    print(f"[INFO] MLflow Experiment: {experiment_name}")
    
    with mlflow.start_run(run_name="RandomForest_CleanData"):
        
        #4. Define Hyperparameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
        
        #5. Training
        print(f"[INFO] Training model with params: {params}")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        #6. Evaluation
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0)
        }
        
        print(f"[RESULT] Accuracy : {metrics['accuracy']:.4f}")
        print(f"[RESULT] F1 Score : {metrics['f1_score']:.4f}")

        #7. Logging to MLflow
        print("[INFO] Logging to MLflow...")
        
        # Log Parameters
        mlflow.log_params(params)
        
        # Log Metrics
        mlflow.log_metrics(metrics)
        
        # Log Model
        mlflow.sklearn.log_model(model, "model_jantung")
        
        # Log Confusion Matrix (Artifact)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        artifact_name = "confusion_matrix.png"
        plt.savefig(artifact_name)
        mlflow.log_artifact(artifact_name)
        
        # cleaning file lokal
        if os.path.exists(artifact_name):
            os.remove(artifact_name)
            
        print("[SUCCESS] Pipeline finished. Check MLflow Dashboard.")

if __name__ == "__main__":
    train_model()