# train_light_pi_ready.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import zipfile
import io
import joblib

# --------------------- 1. Load & concat (unchanged) ---------------------
train_files = ['train_1.zip', 'train_2.zip', 'train_3.zip', 'train_4.zip']
dfs = []
for zip_path in train_files:
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_name = [n for n in z.namelist() if n.lower().endswith('.csv')][0]
        with z.open(csv_name) as f:
            df = pd.read_csv(io.StringIO(f.read().decode('utf-8', errors='replace')),
                             sep=',', engine='python', on_bad_lines='skip')
            dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
print(f"Raw dataset loaded: {df.shape}")

# --------------------- 2. Preprocessing (100% identical to yours) ---------------------
feature_cols = ['Time', 'ID', 'Signal1', 'Signal2', 'Signal3', 'Signal4']
X = df[feature_cols].copy()

signal_cols = ['Signal1', 'Signal2', 'Signal3', 'Signal4']
imputer = SimpleImputer(strategy='constant', fill_value=0)
X[signal_cols] = imputer.fit_transform(X[signal_cols])

X_sorted = X.sort_values(['ID', 'Time']).reset_index(drop=True)
X_sorted['Time_delta'] = X_sorted.groupby('ID')['Time'].diff().fillna(1.0)

for sig in signal_cols:
    X_sorted[f'{sig}_delta'] = X_sorted.groupby('ID')[sig].diff().fillna(0)
    X_sorted[f'{sig}_abs_delta'] = np.abs(X_sorted[f'{sig}_delta'])

for sig in signal_cols:
    delta_col = f'{sig}_delta'
    X_sorted[f'{delta_col}_roll_var'] = X_sorted.groupby('ID')[delta_col]\
        .transform(lambda s: s.rolling(5, min_periods=1).var()).fillna(0)
    X_sorted[f'{delta_col}_roll_mean'] = X_sorted.groupby('ID')[delta_col]\
        .transform(lambda s: s.rolling(5, min_periods=1).mean()).fillna(0)

X = X_sorted.drop(['Time'] + signal_cols, axis=1).sort_index().reset_index(drop=True)

le = LabelEncoder()
X['ID_encoded'] = le.fit_transform(X['ID'])
X = X.drop('ID', axis=1)

numeric_cols = [c for c in X.columns if c != 'ID_encoded']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print(f"After full feature engineering: {X.shape}")

# --------------------- 3. Light subsampling (still representative) ---------------------
X = X.sample(frac=0.08, random_state=42)   # 8% instead of 5% → better coverage, still fast
print(f"Final training size: {X.shape}")

# --------------------- 4. Pi-friendly IsolationForest ---------------------
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

model = IsolationForest(
    n_estimators=150,        # good trade-off
    max_samples=256,         # THIS IS THE KEY FOR LOW MEMORY
    contamination=0.001,
    bootstrap=True,
    random_state=42,
    n_jobs=1                 # crucial for Pi, also safe on Windows
)

print("Training lightweight IsolationForest...")
model.fit(X_train)

# --------------------- 5. Save everything (tiny files) ---------------------
joblib.dump(model, 'syncan_pi_ready_model.pkl')      # ← usually 20–45 MB
joblib.dump(le, 'id_encoder.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("All artifacts saved – ready for Raspberry Pi 3B+!")

# --------------------- 6. Quick validation ---------------------
scores = model.decision_function(X_val)
threshold = np.percentile(scores, 0.1)   # bottom 0.1% = strongest anomalies
anomaly_ratio = (scores < threshold).mean()
print(f"Detected anomaly fraction on validation: {anomaly_ratio:.5f}")