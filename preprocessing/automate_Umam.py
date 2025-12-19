import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at: {path}")
    return pd.read_csv(path)

def clean_data(df):
    df = df.copy()
    
    # 1. Rename Kolom (Standarisasi)
    rename_mapping = {
        'age': 'Umur', 'sex': 'Jenis_Kelamin', 'cp': 'Tipe_Nyeri_Dada',
        'trestbps': 'Tekanan_Darah', 'chol': 'Kolesterol', 'fbs': 'Gula_Darah_Puasa',
        'restecg': 'Hasil_EKG', 'thalch': 'Detak_Jantung_Maks', 
        'exang': 'Angina_Olahraga', 'oldpeak': 'Depresi_ST', 'slope': 'Kemiringan_ST',
        'ca': 'Jml_Pembuluh_Utama', 'thal': 'Thalassemia', 'num': 'Keparahan_Penyakit'
    }
    df.rename(columns=rename_mapping, inplace=True)
    
    # 2. Feature Engineering: Target Biner
    if 'Keparahan_Penyakit' in df.columns:
        # 0 = Sehat, >0 = Sakit
        df['Target'] = df['Keparahan_Penyakit'].apply(lambda x: 1 if x > 0 else 0)
        df.drop(columns=['Keparahan_Penyakit'], inplace=True)
    
    # 3. Drop kolom sampah
    unused_cols = ['id', 'dataset', 'origin']
    for col in unused_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # 4. Handle Missing Values
    # Numerik diisi Mean
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        if col != 'Target':
            df[col].fillna(df[col].mean(), inplace=True)
            
    # Kategorikal diisi Modus
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns
    for col in cat_cols:
        if len(df[col].mode()) > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # 5. Encoding Label
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            df[col] = le.fit_transform(df[col].astype(str))
            
    return df

def split_data(df):
    """Membagi data menjadi Train dan Test set."""
    X = df.drop('Target', axis=1)
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("[INFO] Starting Data Automation...")
    
    # 1. LOAD DATA
    # Path relative ke folder preprocessing
    dataset_path = '../raw_data/heart_disease_uci.csv'
    
    try:
        df = load_data(dataset_path)
        print(f"[INFO] Data loaded successfully from {dataset_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        exit()
    
    # 2. CLEAN DATA
    df_clean = clean_data(df)
    print("[INFO] Data cleaning completed.")
    
    # 3. SAVE DATA (PENTING: Menyimpan hasil preprocessing)
    output_filename = 'heart_disease_clean.csv'
    try:
        df_clean.to_csv(output_filename, index=False)
        print(f"[SUCCESS] Clean data saved to: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"[ERROR] Failed to save file: {e}")

    # 4. SPLIT DATA (Untuk validasi shape)
    X_train, X_test, y_train, y_test = split_data(df_clean)
    
    print("-" * 30)
    print(f"[INFO] Training Data Shape : {X_train.shape}")
    print(f"[INFO] Testing Data Shape  : {X_test.shape}")
    print("-" * 30)