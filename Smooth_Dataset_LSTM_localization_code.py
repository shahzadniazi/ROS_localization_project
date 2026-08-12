import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ===============================
# === 1. Load All Data
# ===============================
df = pd.read_csv("all_sensor_data.csv")
timestamps = df['timestamp'].values

# Parse and Convert LiDAR
def parse_lidar_column(row):
    return [float(val) if val != 'nan' else 0.0 for val in row.split(';')]

df['lidar_ranges'] = df['lidar_ranges'].apply(parse_lidar_column)

def convert_to_cartesian(ranges):
    angles = np.radians(np.linspace(0, 2 * np.pi, len(ranges)))
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return x.tolist() + y.tolist()

df['lidar_cartesian'] = df['lidar_ranges'].apply(convert_to_cartesian)

# LiDAR + IMU features
lidar_features = np.stack(df['lidar_cartesian'].values)
imu_yaw = df['imu_yaw'].values.reshape(-1, 1)
combined_features = np.hstack((lidar_features, imu_yaw))
labels = df[['odom_vx_global', 'odom_vy_global']].values

# Sliding window
def create_sliding_window(X, y, window_size, stride):
    Xs, ys = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        Xs.append(X[i:i + window_size])
        ys.append(y[i + window_size - 1])
    return np.array(Xs), np.array(ys)

window_size = 2
stride = 1
X_windowed, y_windowed = create_sliding_window(combined_features, labels, window_size, stride)
timestamps_windowed = timestamps[window_size - 1:]

# ===============================
# === 2. LiDAR+IMU Prediction
# ===============================
# CORRECT: Use joblib for scikit-learn scalers
scaler_X = joblib.load('scaler_X.save')
scaler_Y = joblib.load('scaler_Y.save')
# CORRECT: Use load_model for Keras models
model_lidar = load_model("LiDAR_weights.keras")

X_shape = X_windowed.shape
X_reshaped = X_windowed.reshape(-1, X_shape[2])
X_scaled = scaler_X.transform(X_reshaped).reshape(X_shape)
y_lidar_pred = model_lidar.predict(X_scaled, verbose=0)
y_lidar_pred = scaler_Y.inverse_transform(y_lidar_pred)

# Integrate predicted velocities
positions_pred = [(0.0, 0.0)]
for i in range(1, len(y_lidar_pred)):
    dt = timestamps_windowed[i] - timestamps_windowed[i - 1]
    last_x, last_y = positions_pred[-1]
    vx, vy = y_lidar_pred[i]
    new_x = last_x + vx * dt
    new_y = last_y + vy * dt
    positions_pred.append((new_x, new_y))
positions_pred = np.array(positions_pred)

# Ground truth integration
positions_true = [(0.0, 0.0)]
for i in range(1, len(y_windowed)):
    dt = timestamps_windowed[i] - timestamps_windowed[i - 1]
    last_x, last_y = positions_true[-1]
    vx, vy = y_windowed[i]
    new_x = last_x + vx * dt
    new_y = last_y + vy * dt
    positions_true.append((new_x, new_y))
positions_true = np.array(positions_true)

# ===============================
# === 3. UWB Prediction
# ===============================
X_uwb = df[['uwb_A0', 'uwb_A1', 'uwb_A2', 'uwb_A3']].values / 1000.0
X_uwb = X_uwb.reshape((X_uwb.shape[0], 1, X_uwb.shape[1]))

uwb_model = Sequential([
    LSTM(64, input_shape=(X_uwb.shape[1], X_uwb.shape[2])),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(2)
])
uwb_model.compile(optimizer='adam', loss='mse')
# CORRECT: Use load_weights for .h5 weight files
uwb_model.load_weights('UWB_weights.h5')

y_uwb_pred = uwb_model.predict(X_uwb, verbose=0)

# ===============================
# === 4. ORB+IMU Prediction
# ===============================
# CORRECT: Use load_model for Keras models
orb_model = load_model("Video_weight.keras")
orb = cv2.ORB_create(nfeatures=500)

frame_folder = 'frames_with_timestamps'
frames = sorted(os.listdir(frame_folder))

def extract_orb_features(prev_frame, curr_frame):
    kp1, des1 = orb.detectAndCompute(prev_frame, None)
    kp2, des2 = orb.detectAndCompute(curr_frame, None)
    if des1 is None or des2 is None:
        return [0, 0.0]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) == 0:
        return [0, 0.0]
    distances = [m.distance for m in matches]
    return [len(matches), np.mean(distances)]

X_orb_test, X_yaw_test, timestamps_orb = [], [], []

for i in range(1, len(frames)):
    curr_frame_path = os.path.join(frame_folder, frames[i])
    prev_frame_path = os.path.join(frame_folder, frames[i-1])
    
    if not os.path.exists(curr_frame_path) or not os.path.exists(prev_frame_path):
        continue

    curr_frame = cv2.imread(curr_frame_path, cv2.IMREAD_GRAYSCALE)
    prev_frame = cv2.imread(prev_frame_path, cv2.IMREAD_GRAYSCALE)
    if curr_frame is None or prev_frame is None:
        continue
    curr_frame = cv2.resize(curr_frame, (64, 64))
    prev_frame = cv2.resize(prev_frame, (64, 64))

    timestamp = float(frames[i].split('_')[1].split('.')[0])
    imu = df.iloc[(df['timestamp'] - timestamp).abs().argmin()]
    yaw = imu['imu_yaw']
    orb_feat = extract_orb_features(prev_frame, curr_frame)

    X_orb_test.append(orb_feat)
    X_yaw_test.append([yaw])
    timestamps_orb.append(timestamp)

X_orb_test = np.array(X_orb_test)
X_yaw_test = np.array(X_yaw_test)
predicted_velocities_orb = orb_model.predict([X_orb_test, X_yaw_test], verbose=0)

def integrate_velocity(velocities, timestamps):
    x, y = [0], [0]
    for i in range(1, len(velocities)):
        dt = timestamps[i] - timestamps[i - 1]
        dx = velocities[i][0] * dt
        dy = velocities[i][1] * dt
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
    return np.array(x), np.array(y)

orb_pred_x, orb_pred_y = integrate_velocity(predicted_velocities_orb, timestamps_orb)

# ===============================
# === 5. GPS Processing
# ===============================
# ===============================
# === 5. GPS Processing (1Hz & Smooth)
# ===============================
REFERENCE_LATITUDE = 37.7749
REFERENCE_LONGITUDE = -122.4194
EARTH_RADIUS = 6378137.0


def latlon_to_xy(lat, lon, r_lat, r_lon):
    # Standard conversion
    d_lat = np.radians(lat - r_lat)
    d_lon = np.radians(lon - r_lon)
    mean_lat = np.radians((lat + r_lat) / 2.0)
    
    # x = East (Longitude), y = North (Latitude)
    curr_x = -EARTH_RADIUS * d_lon * np.cos(mean_lat)
    curr_y = EARTH_RADIUS * d_lat
    return curr_x, curr_y

# 1. Initial conversion
gps_x, gps_y = latlon_to_xy(df['gps_lat'].values, df['gps_lon'].values, REFERENCE_LATITUDE, REFERENCE_LONGITUDE)


# 3. Add ultra-low noise (0.01m as requested)
np.random.seed(12)
gps_x += np.random.normal(0, 0.25, size=gps_x.shape)
gps_y += np.random.normal(0, 0.25, size=gps_y.shape)

# 4. 1Hz Sample-and-Hold
gps_positions = np.zeros((len(timestamps), 2))
last_upd = timestamps[0]
cur_val = [gps_x[0], gps_y[0]]

for i in range(len(timestamps)):
    if timestamps[i] - last_upd >= 1.0:
        cur_val = [gps_x[i], gps_y[i]]
        last_upd = timestamps[i]
    gps_positions[i] = cur_val
gps_positions_corrected = gps_positions[:, [1, 0]] # Your refined swap

gps_positions = np.array(gps_positions_corrected)
# ===============================
# === 6. EKF Implementation
# ===============================
class CorrectedEKF:
    def __init__(self):
        self.x = np.zeros((4, 1))  # [x, y, vx, vy]
        self.P = np.eye(4) * 1.0
        self.Q = np.eye(4) * 0.01
        self.R_uwb = np.eye(2) * 0.5
        self.R_gps = np.eye(2) * 0.5

    def predict(self, u, dt):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0 ],
                      [0, 0, 0, 1 ]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        self.x[2, 0] = u[0]
        self.x[3, 0] = u[1]

    def correct(self, z, sensor_type='gps'):
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        R = self.R_gps if sensor_type == 'gps' else self.R_uwb
        z = z.reshape(2, 1)
        y = z - (H @ self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

ekf = CorrectedEKF()
positions_ekf = []
last_time = timestamps_windowed[0]

for i in range(1, len(timestamps_windowed)):
    dt = timestamps_windowed[i] - last_time
    last_time = timestamps_windowed[i]

    # Average predicted velocities from LiDAR and ORB
    if i < len(y_lidar_pred) and i < len(predicted_velocities_orb):
        pred_vel = 0.5 * (y_lidar_pred[i] + predicted_velocities_orb[i])
    elif i < len(y_lidar_pred):
        pred_vel = y_lidar_pred[i]
    elif i < len(predicted_velocities_orb):
        pred_vel = predicted_velocities_orb[i]
    else:
        pred_vel = np.zeros(2)  # Default if no velocity available

    # EKF Prediction Step
    ekf.predict(pred_vel, dt)

    # EKF Correction Step using both UWB and GPS if available
    has_uwb = i < len(y_uwb_pred)
    has_gps = i < len(gps_positions)

    if has_uwb and has_gps:
        # First correct with UWB, then refine with GPS
        ekf.correct(y_uwb_pred[i], sensor_type='uwb')
        ekf.correct(gps_positions[i], sensor_type='gps')
    elif has_uwb:
        ekf.correct(y_uwb_pred[i], sensor_type='uwb')
    elif has_gps:
        ekf.correct(gps_positions[i], sensor_type='gps')

    # Save EKF estimated position (x, y)
    positions_ekf.append(ekf.x[:2].flatten())

positions_ekf = np.array(positions_ekf)




# ===============================
# === 7. Visualization
# ===============================
plt.figure(figsize=(16, 10))
plt.plot(positions_true[:, 0], positions_true[:, 1], label='Ground Truth')
plt.plot(positions_pred[:, 0], positions_pred[:, 1], label='LiDAR+IMU LSTM Predicted')
plt.plot(orb_pred_x, orb_pred_y, label='ORB+IMU LSTM Predicted')
plt.plot(y_uwb_pred[:, 0], y_uwb_pred[:, 1], 'r-', label='UWB LSTM Predicted')
plt.plot(gps_positions[:, 0], gps_positions[:, 1], label='GPS')
plt.plot(positions_ekf[:, 0], positions_ekf[:, 1], label='EKF Fused', linewidth=2.5)
plt.xlabel('X Position (m)', fontsize=18); plt.ylabel('Y Position (m)', fontsize=18)
plt.title('Trajectory Comparison with Hybrid Fusion', fontsize=22)
plt.legend(fontsize=16); plt.grid(); plt.tight_layout(); plt.show()

# ===============================
# === 8. Function to Compute ALL Metrics
# ===============================
def compute_all_metrics(predicted, true):
    min_len = min(len(predicted), len(true))
    pred = predicted[:min_len]
    gt = true[:min_len]
    
    # Euclidean Errors
    errors = np.linalg.norm(pred - gt, axis=1)
    
    mean_err = np.mean(errors)
    rmse_err = np.sqrt(np.mean(errors**2))
    max_err = np.max(errors)
    
    return errors, mean_err, rmse_err, max_err

# ===============================
# === 9. Compute Errors
# ===============================
orb_positions = np.column_stack((orb_pred_x, orb_pred_y))

e_lidar, m_lidar, r_lidar, mx_lidar = compute_all_metrics(positions_pred, positions_true)
e_orb, m_orb, r_orb, mx_orb = compute_all_metrics(orb_positions, positions_true)
e_uwb, m_uwb, r_uwb, mx_uwb = compute_all_metrics(y_uwb_pred, positions_true)
e_gps, m_gps, r_gps, mx_gps = compute_all_metrics(gps_positions_corrected, positions_true)
e_ekf, m_ekf, r_ekf, mx_ekf = compute_all_metrics(positions_ekf, positions_true)

# ===============================
# === 10. Print Detailed Error Table
# ===============================
print("\n" + "="*75)
print(f"{'Sensor Source':<20} | {'Mean (m)':<12} | {'RMSE (m)':<12} | {'Max (m)':<10}")
print("-" * 75)
print(f"{'LiDAR+IMU LSTM':<20} | {m_lidar:<12.4f} | {r_lidar:<12.4f} | {mx_lidar:<10.4f}")
print(f"{'ORB+IMU LSTM':<20} | {m_orb:<12.4f} | {r_orb:<12.4f} | {mx_orb:<10.4f}")
print(f"{'UWB LSTM':<20} | {m_uwb:<12.4f} | {r_uwb:<12.4f} | {mx_uwb:<10.4f}")
print(f"{'GPS (1Hz Noisy)':<20} | {m_gps:<12.4f} | {r_gps:<12.4f} | {mx_gps:<10.4f}")
print(f"{'EKF Fused':<20} | {m_ekf:<12.4f} | {r_ekf:<12.4f} | {mx_ekf:<10.4f}")
print("="*75)

# ===============================
# === 11. Plot Error Over Time
# ===============================
plt.figure(figsize=(16, 10))
plt.plot(e_lidar, label='LiDAR+IMU LSTM')
plt.plot(e_orb, label='ORB+IMU LSTM')
plt.plot(e_uwb, label='UWB LSTM')
plt.plot(e_gps, label='GPS Error')
plt.plot(e_ekf, label='EKF Fused Error', linewidth=2.5)
plt.xlabel("Timestep", fontsize=18); plt.ylabel("Position Error (meters)", fontsize=18)
plt.title("Sensor Position Errors Comparison", fontsize=22)
plt.legend(fontsize=16); plt.grid(True); plt.tight_layout(); plt.show()
