import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import joblib
import argparse
import sys
import time
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

class CANMonitor:
    def __init__(self, model_path, threshold, input_source='stdin', buffer_size=10):
        self.threshold = threshold
        self.buffer_size = buffer_size
        self.buffers = defaultdict(lambda: deque(maxlen=buffer_size))  # Per ID buffer for deltas/rolling
        self.id_encoder = joblib.load(f'{model_path.replace("model.pkl", "")}id_encoder.pkl')
        self.imputer = joblib.load(f'{model_path.replace("model.pkl", "")}imputer.pkl')
        self.scaler = joblib.load(f'{model_path.replace("model.pkl", "")}scaler.pkl')
        self.ensemble = joblib.load(model_path)
        self.if_model = self.ensemble['if']
        self.ocsvm_model = self.ensemble['ocsvm']
        self.signal_cols = ['Signal1', 'Signal2', 'Signal3', 'Signal4']
        self.input_source = input_source
        self.msg_count = 0
        print(f"Loaded ensemble model from {model_path}. Threshold: {threshold}. Input: {input_source}")

    def preprocess_message(self, row):
        # Incremental deltas/rolling for new message
        id_val = row['ID']
        buffer = self.buffers[id_val]
        buffer.append(row)

        # Time_delta (diff from last)
        if len(buffer) > 1:
            time_delta = row['Time'] - list(buffer)[-2]['Time']
        else:
            time_delta = 1.0  # Default ms

        # Signal deltas (diff from last)
        deltas = {sig: row[sig] - list(buffer)[-2][sig] if len(buffer) > 1 else 0 for sig in self.signal_cols}
        abs_deltas = {sig: abs(d) for sig, d in deltas.items()}

        # Rolling var/mean (simple deque-based for low compute)
        if len(buffer) >= 5:
            delta_values = np.array([msg[sig] for msg in list(buffer)[-5:] for sig in self.signal_cols if msg[sig] is not None])
            roll_var = np.var(delta_values) if len(delta_values) > 0 else 0
            roll_mean = np.mean(delta_values) if len(delta_values) > 0 else 0
        else:
            roll_var = 0
            roll_mean = 0

        # Features vector (Time_delta, deltas, abs_deltas, roll_var/mean, ID_encoded)
        features = [time_delta]
        for sig in self.signal_cols:
            features += [deltas[sig], abs_deltas[sig]]
        features += [roll_var, roll_mean] * 4  # 4 signals, but shared roll for simplicity (adjust if per-sig)
        features.append(self.id_encoder.transform([id_val])[0])

        # Impute and scale
        features = self.imputer.transform([features])[0]  # Shape to match
        features = self.scaler.transform([features])[0]

        return np.array(features).reshape(1, -1)

    def predict_anomaly(self, features):
        if_scores = self.if_model.decision_function(features)
        ocsvm_scores = self.ocsvm_model.score_samples(features)
        ensemble_score = (if_scores + ocsvm_scores) / 2
        proba = -ensemble_score[0]  # Invert for anomaly proba
        is_anomaly = proba > self.threshold
        return proba, is_anomaly

    def run(self):
        if self.input_source == 'stdin':
            for line in sys.stdin:
                row = pd.DataFrame([dict(line.strip().split(','))], columns=['Label', 'Time', 'ID'] + self.signal_cols)  # Parse CSV line
                row['Time'] = pd.to_numeric(row['Time'])
                for sig in self.signal_cols:
                    row[sig] = pd.to_numeric(row[sig], errors='coerce')
                features = self.preprocess_message(row.iloc[0])
                proba, is_anomaly = self.predict_anomaly(features)
                self.msg_count += 1
                print(f"Msg {self.msg_count}: Proba {proba:.4f}, Anomaly: {is_anomaly} (threshold {self.threshold})")
                if is_anomaly:
                    print("ALERT: Intrusion detected!")
                time.sleep(0.001)  # Simulate real-time delay
        else:
            # For real CAN bus (e.g., python-can)
            from can import interface
            bus = interface.Bus(channel=self.input_source, bustype='socketcan')
            while True:
                msg = bus.recv(timeout=1.0)
                if msg:
                    # Parse msg.data to row (assume arbitration_id = ID, data = signals, timestamp = Time)
                    row = {'ID': str(msg.arbitration_id), 'Time': msg.timestamp * 1000, 'Label': 0}
                    for i, sig in enumerate(self.signal_cols):
                        row[sig] = msg.data[i] if i < len(msg.data) else np.nan
                    features = self.preprocess_message(pd.DataFrame([row]))
                    proba, is_anomaly = self.predict_anomaly(features)
                    self.msg_count += 1
                    print(f"Msg {self.msg_count}: Proba {proba:.4f}, Anomaly: {is_anomaly}")
                    if is_anomaly:
                        print("ALERT: Intrusion detected!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time CAN Monitor')
    parser.add_argument('--model', default='syncan_ensemble_model.pkl', help='Path to ensemble model')
    parser.add_argument('--threshold', type=float, default=-5.93, help='Anomaly threshold')
    parser.add_argument('--input', default='stdin', help='Input source (stdin or can0)')
    args = parser.parse_args()
    monitor = CANMonitor(args.model, args.threshold, args.input)
    monitor.run()
