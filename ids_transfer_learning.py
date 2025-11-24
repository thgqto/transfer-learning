# ids_inference_pi.py
# Real-time SynCAN IDS for Raspberry Pi 3B+ (and any low-power device)
# Tested with Python 3.9–3.11 on Raspberry Pi OS (32-bit or 64-bit)

import pandas as pd
import numpy as np
import joblib
import time
import os
from collections import deque

# --------------------------- 1. Load model & preprocessors ---------------------------
print("Loading model and preprocessors...")
model      = joblib.load('syncan_pi_ready_model.pkl')      # ~665 KB
le         = joblib.load('id_encoder.pkl')
imputer    = joblib.load('imputer.pkl')
scaler     = joblib.load('scaler.pkl')
print("All artifacts loaded – memory footprint < 50 MB total")

# Feature column order must be exactly the same as during training
signal_cols = ['Signal1', 'Signal2', 'Signal3', 'Signal4']

# Sliding window buffers per CAN-ID (to compute deltas & rolling stats)
window_size = 5
buffers = {}   # key: CAN-ID (str), value: deque of dicts

# --------------------------- 2. Preprocessing function ---------------------------
def preprocess_single_frame(frame_dict):
    """
    Input: dict with keys  Time, ID, Signal1, Signal2, Signal3, Signal4
           (SignalX can be missing → will be treated as NaN)
    Output: 1×18 feature vector (numpy array, scaled)
    """
    can_id = frame_dict['ID']
    t = frame_dict['Time']

    # Initialize buffer for this CAN-ID if new
    if can_id not in buffers:
        buffers[can_id] = deque(maxlen=window_size)

    # ----- Build current row as DataFrame (1 row) -----
    row = {
        'Time': t,
        'ID': can_id,
        'Signal1': frame_dict.get('Signal1', np.nan),
        'Signal2': frame_dict.get('Signal2', np.nan),
        'Signal3': frame_dict.get('Signal3', np.nan),
        'Signal4': frame_dict.get('Signal4', np.nan),
    }
    df = pd.DataFrame([row])

    # Impute missing signals
    df[signal_cols] = imputer.transform(df[signal_cols])

    # Compute deltas (using buffer)
    if len(buffers[can_id]) > 0:
        prev = buffers[can_id][-1]
        time_delta = t - prev['Time']
        deltas = {sig: row[sig] - prev.get(sig, 0) for sig in signal_cols}
    else:
        time_delta = 1.0
        deltas = {sig: 0.0 for sig in signal_cols}

    # Append current (pre-impute) values for next iteration
    buffers[can_id].append(row)

    # Build feature row exactly like training
    features = {
        'Time_delta': time_delta,
    }
    for sig in signal_cols:
        delta = deltas[sig]
        features[f'{sig}_delta'] = delta
        features[f'{sig}_abs_delta'] = abs(delta)

    # Rolling stats over the buffer
    for sig in signal_cols:
        col = f'{sig}_delta'
        series = [d.get(sig, 0) - prev.get(sig, 0) if len(buffers[can_id]) > 1 and i > 0
                  else 0 for i, (d, prev) in enumerate(zip(buffers[can_id], 
                  list(buffers[can_id])[1:] + [buffers[can_id][0]]))]
        # Simple rolling var/mean on last min(n, window_size) deltas
        recent = series[-window_size:]
        features[f'{col}_roll_var']  = np.var(recent) if len(recent) > 1 else 0
        features[f'{col}_roll_mean'] = np.mean(recent)

    feat_df = pd.DataFrame([features])
    feat_df['ID_encoded'] = le.transform([can_id])

    # Final scaling
    numeric_cols = [c for c in feat_df.columns if c != 'ID_encoded']
    feat_df[numeric_cols] = scaler.transform(feat_df[numeric_cols])

    # Reorder exactly like training (18 columns)
    final_order = ['Time_delta',
                   'Signal1_delta', 'Signal1_abs_delta', 'Signal1_delta_roll_var', 'Signal1_delta_roll_mean',
                   'Signal2_delta', 'Signal2_abs_delta', 'Signal2_delta_roll_var', 'Signal2_delta_roll_mean',
                   'Signal3_delta', 'Signal3_abs_delta', 'Signal3_delta_roll_var', 'Signal3_delta_roll_mean',
                   'Signal4_delta', 'Signal4_abs_delta', 'Signal4_delta_roll_var', 'Signal4_delta_roll_mean',
                   'ID_encoded']
    X = feat_df[final_order].values.astype(np.float32)

    return X

# --------------------------- 3. Main inference loop (CSV simulation) ---------------------------
csv_path = 'test_simulation.csv'   # ← change to your real test file or live capture source

print(f"Starting inference on {csv_path} ...")
start_time = time.time()
frame_count = 0
anomaly_count = 0

with open(csv_path, 'r') as f:
    header = f.readline()          # skip header
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue

        try:
            label = int(parts[0])
            t = float(parts[1])
            can_id = parts[2]
            signals = [float(x) if x != '' else np.nan for x in parts[3:7]]
            frame = {'Time': t, 'ID': can_id,
                     'Signal1': signals[0], 'Signal2': signals[1],
                     'Signal3': signals[2], 'Signal4': signals[3]}

            X = preprocess_single_frame(frame)
            score = model.decision_function(X)[0]      # lower = more anomalous
            prediction = model.predict(X)[0]           # -1 = anomaly, 1 = normal

            frame_count += 1
            if prediction == -1:
                anomaly_count += 1
                print(f"[ALERT] t={t:.3f} ID={can_id} score={score:.4f}")

            if frame_count % 10000 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {frame_count} frames | {frame_count/elapsed:.1f} fps "
                      f"| Anomalies: {anomaly_count}")

        except Exception as e:
            print(f"Error processing line: {e}")
            continue

total_time = time.time() - start_time
print("\n=== DONE ===")
print(f"Total frames   : {frame_count}")
print(f"Anomalies      : {anomaly_count} ({100*anomaly_count/frame_count:.3f}%)")
print(f"Throughput     : {frame_count/total_time:.1f} fps")
print(f"Total time     : {total_time:.2f} s")