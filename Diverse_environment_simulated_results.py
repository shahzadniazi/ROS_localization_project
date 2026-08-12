import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Ensure TF runs on CPU if needed

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from collections import deque # For history buffers

# === Pre-computation for LiDAR feature dimensions ===
scaler_X_path = 'scaler_X.save'
scaler_Y_path = 'scaler_Y.save'
model_lidar_path = "LiDAR_weights.keras"

if not os.path.exists(scaler_X_path): raise FileNotFoundError(f"Scaler X not found: {scaler_X_path}")

scaler_X_instance = joblib.load(scaler_X_path)
NUM_EXPECTED_FEATURES_FOR_SCALER = scaler_X_instance.n_features_in_
NUM_IMU_FEATURES = 1
EXPECTED_LIDAR_CARTESIAN_FEATURES = NUM_EXPECTED_FEATURES_FOR_SCALER - NUM_IMU_FEATURES

if EXPECTED_LIDAR_CARTESIAN_FEATURES <= 0 or EXPECTED_LIDAR_CARTESIAN_FEATURES % 2 != 0:
    raise ValueError(
        f"Scaler expects {NUM_EXPECTED_FEATURES_FOR_SCALER} total features. "
        f"With {NUM_IMU_FEATURES} IMU features, this leaves {EXPECTED_LIDAR_CARTESIAN_FEATURES} for LiDAR Cartesian coordinates. "
        "This must be a positive even number."
    )
DYNAMIC_EXPECTED_LIDAR_POINTS = int(EXPECTED_LIDAR_CARTESIAN_FEATURES / 2)

# ===============================
# === 1. Load All Data
# ===============================
df = pd.read_csv("all_sensor_uwb_data.csv")
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce') # Ensure timestamp is numeric
df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp couldn't be converted
timestamps = df['timestamp'].values


def parse_lidar_column(row_str_or_list, num_expected_range_points_for_default):
    if isinstance(row_str_or_list, str):
        return [float(val) if val != 'nan' else 0.0 for val in row_str_or_list.split(';')]
    elif isinstance(row_str_or_list, (list, np.ndarray)):
        return [float(val) if not (isinstance(val, float) and np.isnan(val)) else 0.0 for val in row_str_or_list]
    return [0.0] * num_expected_range_points_for_default

def convert_to_cartesian(ranges_list, num_target_lidar_points_for_output_dim):
    num_actual_ranges = len(ranges_list)
    if num_actual_ranges == 0:
        return [0.0] * (2 * num_target_lidar_points_for_output_dim)
    angles = np.radians(np.linspace(0, 2 * np.pi, num_actual_ranges, endpoint=False))
    np_ranges = np.array(ranges_list)
    x_coords = np_ranges * np.cos(angles)
    y_coords = np_ranges * np.sin(angles)
    cartesian_coords_list = x_coords.tolist() + y_coords.tolist()
    target_cartesian_len = 2 * num_target_lidar_points_for_output_dim
    current_cartesian_len = len(cartesian_coords_list)
    if current_cartesian_len < target_cartesian_len:
        cartesian_coords_list.extend([0.0] * (target_cartesian_len - current_cartesian_len))
    elif current_cartesian_len > target_cartesian_len:
        cartesian_coords_list = cartesian_coords_list[:target_cartesian_len]
    return cartesian_coords_list

df['lidar_ranges'] = df['lidar_ranges'].apply(
    lambda r: parse_lidar_column(r, DYNAMIC_EXPECTED_LIDAR_POINTS)
)
df['lidar_cartesian'] = df['lidar_ranges'].apply(
    lambda r: convert_to_cartesian(r, DYNAMIC_EXPECTED_LIDAR_POINTS)
)

if not df['lidar_cartesian'].empty:
    lidar_features = np.stack(df['lidar_cartesian'].values)
else:
    lidar_features = np.empty((0, EXPECTED_LIDAR_CARTESIAN_FEATURES))

imu_yaw = df['imu_yaw'].values.reshape(-1, 1)
min_rows = min(len(lidar_features), len(imu_yaw))

if min_rows == 0 and len(df) > 0 :
    raise ValueError("LiDAR features or IMU yaw data is empty after processing, but source DataFrame was not empty.")

lidar_features = lidar_features[:min_rows]
imu_yaw = imu_yaw[:min_rows]
df_sliced = df.iloc[:min_rows].copy() # Use .copy() to avoid SettingWithCopyWarning
timestamps_sliced = timestamps[:min_rows]

combined_features = np.hstack((lidar_features, imu_yaw))
labels = df_sliced[['odom_vx_global', 'odom_vy_global']].values

def create_sliding_window(X, y, window_size, stride):
    Xs, ys = [], []
    if len(X) < window_size:
        return np.array(Xs), np.array(ys)
    for i in range(0, len(X) - window_size + 1, stride):
        Xs.append(X[i:i + window_size])
        ys.append(y[i + window_size - 1])
    return np.array(Xs), np.array(ys)

window_size = 2
stride = 1
X_windowed, y_windowed = create_sliding_window(combined_features, labels, window_size, stride)

if X_windowed.shape[0] == 0 and len(combined_features) >=window_size :
    raise ValueError("Sliding window creation resulted in no data.")

timestamps_windowed = timestamps_sliced[window_size - 1 : window_size - 1 + len(X_windowed)]

# ===============================
# === 2. LiDAR+IMU Prediction
# ===============================
if not os.path.exists(scaler_Y_path): raise FileNotFoundError(f"Scaler Y not found: {scaler_Y_path}")
if not os.path.exists(model_lidar_path): raise FileNotFoundError(f"LiDAR model not found: {model_lidar_path}")

scaler_X = scaler_X_instance
scaler_Y = joblib.load(scaler_Y_path)
model_lidar = load_model(model_lidar_path)

y_lidar_pred = np.array([])
if X_windowed.size > 0:
    X_shape = X_windowed.shape
    X_reshaped = X_windowed.reshape(-1, X_shape[2])
    if X_shape[2] != scaler_X.n_features_in_:
        raise ValueError(f"CRITICAL: Mismatch. X_windowed has {X_shape[2]} features, scaler_X expects {scaler_X.n_features_in_}.")
    X_scaled = scaler_X.transform(X_reshaped).reshape(X_shape)
    y_lidar_pred = model_lidar.predict(X_scaled, verbose=0)
    y_lidar_pred = scaler_Y.inverse_transform(y_lidar_pred)

positions_pred = [(0.0, 0.0)]
if len(y_lidar_pred) > 0 and len(timestamps_windowed) > 0:
    for i in range(1, min(len(y_lidar_pred), len(timestamps_windowed))): # Ensure min length
        dt = timestamps_windowed[i] - timestamps_windowed[i - 1]
        if dt <= 0: dt = 0.01
        last_x, last_y = positions_pred[-1]
        vx, vy = y_lidar_pred[i]
        new_x = last_x + vx * dt
        new_y = last_y + vy * dt
        positions_pred.append((new_x, new_y))
positions_pred = np.array(positions_pred)

positions_true = [(0.0, 0.0)]
if len(y_windowed) > 0 and len(timestamps_windowed) > 0:
    for i in range(1, min(len(y_windowed), len(timestamps_windowed))): # Ensure min length
        dt = timestamps_windowed[i] - timestamps_windowed[i - 1]
        if dt <= 0: dt = 0.01
        last_x, last_y = positions_true[-1]
        vx, vy = y_windowed[i]
        new_x = last_x + vx * dt
        new_y = last_y + vy * dt
        positions_true.append((new_x, new_y))
positions_true = np.array(positions_true)

# ===============================
# === 3. UWB Prediction
# ===============================
uwb_model_path = 'UWB_weights.h5'
if not os.path.exists(uwb_model_path): raise FileNotFoundError(f"UWB model weights not found: {uwb_model_path}")

# --- MODIFIED UWB DATA HANDLING ---
uwb_columns = ['uwb_A0', 'uwb_A1', 'uwb_A2', 'uwb_A3']

# Ensure UWB columns are numeric, coercing errors to NaN
for col in uwb_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"Warning: UWB column {col} not found in DataFrame. It will be treated as missing.")
        df[col] = np.nan # Create it as NaN if missing entirely

# Apply forward fill to propagate last known good values
df[uwb_columns] = df[uwb_columns].ffill()

# After ffill, if there are still NaNs (e.g., at the very beginning of the dataset),
# fill them with a default value (e.g., 0 or a large distance if that's more appropriate for "no signal")
df[uwb_columns] = df[uwb_columns].fillna(0) 

X_uwb_raw = df[uwb_columns].values / 1000.0
X_uwb = X_uwb_raw # NaNs should be handled by ffill and fillna
# --- END OF MODIFIED UWB DATA HANDLING ---

X_uwb = X_uwb.reshape((X_uwb.shape[0], 1, X_uwb.shape[1]))

uwb_model = Sequential([
    LSTM(64, input_shape=(X_uwb.shape[1], X_uwb.shape[2])),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(2)
])
uwb_model.compile(optimizer='adam', loss='mse')
uwb_model.load_weights(uwb_model_path)
y_uwb_pred = uwb_model.predict(X_uwb, verbose=0)

# ===============================
# === 4. ORB+IMU Prediction
# ===============================
orb_model_path = "Video_weight.keras"
if not os.path.exists(orb_model_path): raise FileNotFoundError(f"ORB model not found: {orb_model_path}")
orb_model = load_model(orb_model_path)
orb = cv2.ORB_create(nfeatures=500)

frame_folder = 'frames_with_timestamps'
frames_list = []
if os.path.exists(frame_folder) and os.path.isdir(frame_folder):
    frames_list = sorted(os.listdir(frame_folder))
else:
    print(f"Warning: Frame folder '{frame_folder}' not found. ORB+IMU path will be empty.")

def extract_orb_features(prev_frame_img, curr_frame_img):
    kp1, des1 = orb.detectAndCompute(prev_frame_img, None)
    kp2, des2 = orb.detectAndCompute(curr_frame_img, None)
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0: return [0, 0.0]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches: return [0, 0.0]
    distances = [m.distance for m in matches]
    return [len(matches), np.mean(distances) if distances else 0.0]

X_orb_test, X_yaw_test, timestamps_orb = [], [], []
black_frame_flags = [] 

if len(frames_list) > 1:
    if 'timestamp_numeric' not in df.columns:
         df['timestamp_numeric'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df_for_orb = df.dropna(subset=['timestamp_numeric', 'imu_yaw'])

    for i in range(1, len(frames_list)):
        curr_frame_path = os.path.join(frame_folder, frames_list[i])
        prev_frame_path = os.path.join(frame_folder, frames_list[i-1])
        
        if not (os.path.exists(curr_frame_path) and os.path.exists(prev_frame_path)): 
            continue
            
        curr_img = cv2.imread(curr_frame_path, cv2.IMREAD_GRAYSCALE)
        prev_img = cv2.imread(prev_frame_path, cv2.IMREAD_GRAYSCALE)
        
        if curr_img is None or prev_img is None: 
            continue
            
        curr_resized = cv2.resize(curr_img, (64, 64))
        prev_resized = cv2.resize(prev_img, (64, 64))

        curr_is_black = (cv2.countNonZero(curr_resized) == 0)
        prev_is_black = (cv2.countNonZero(prev_resized) == 0)
        is_current_step_black = curr_is_black or prev_is_black
        
        try:
            timestamp_str = frames_list[i].split('_')[1].split('.')[0]
            timestamp_val = float(timestamp_str.replace('-', '.'))
        except (IndexError, ValueError): 
            continue
            
        if df_for_orb.empty: 
            yaw_val = 0.0
        else:
            closest_idx = (df_for_orb['timestamp_numeric'] - timestamp_val).abs().idxmin()
            yaw_val = df_for_orb.loc[closest_idx, 'imu_yaw']
            
        orb_feat = extract_orb_features(prev_resized, curr_resized)
        
        X_orb_test.append(orb_feat)
        X_yaw_test.append([yaw_val])
        timestamps_orb.append(timestamp_val)
        black_frame_flags.append(is_current_step_black) 

predicted_velocities_orb = np.array([])
if X_orb_test and X_yaw_test:
    X_orb_test_np = np.array(X_orb_test)
    X_yaw_test_np = np.array(X_yaw_test)
    
    if X_orb_test_np.ndim == 1 and X_orb_test_np.shape[0] == 2 : 
        X_orb_test_np = X_orb_test_np.reshape(1, -1)
    if X_yaw_test_np.ndim == 1 : 
        X_yaw_test_np = X_yaw_test_np.reshape(-1, 1)
        
    try:
        if X_orb_test_np.shape[0] > 0 and X_yaw_test_np.shape[0] > 0:
             raw_predicted_velocities = orb_model.predict([X_orb_test_np, X_yaw_test_np], verbose=0)
             predicted_velocities_orb = np.array(raw_predicted_velocities) 

             if len(predicted_velocities_orb) == len(black_frame_flags):
                 for k in range(len(predicted_velocities_orb)):
                     if black_frame_flags[k]:
                         predicted_velocities_orb[k] = [0.0, 0.0] 
             else:
                 print("Warning: Mismatch between ORB predictions and black_frame_flags. Black frame override skipped.")
                 predicted_velocities_orb = raw_predicted_velocities
    except Exception as e:
        print(f"Error in ORB model prediction: {e}. Shapes: ORB {X_orb_test_np.shape}, Yaw {X_yaw_test_np.shape}")

# Standard integration function (unchanged)
def integrate_velocity(velocities_arr, ts_vel_arr):
    x_coords, y_coords = [0.0], [0.0]
    num_steps = len(velocities_arr)
    if num_steps == 0: # Handle empty velocity array
        return np.array(x_coords), np.array(y_coords)
    if num_steps == 1: # If only one velocity, path is just (0,0) and the first integrated point
        # This case is not well-defined by the loop range(1, num_steps).
        # For consistency, if num_steps is 1, perhaps it should just return [0.0], [0.0]
        # or handle dt appropriately if a t_initial is assumed.
        # The original loop range(1,num_steps) means for num_steps=1, loop is empty, returns [0.0],[0.0]
        # Let's keep it that way.
        if num_steps <=1: # Covers 0 and 1
             return np.array(x_coords), np.array(y_coords)


    # The loop computes num_steps-1 additional points. Total points = 1 (initial) + num_steps -1 = num_steps.
    for i in range(1, num_steps):
        if i >= len(ts_vel_arr) or i-1 >= len(ts_vel_arr) : # Boundary check for ts_vel_arr
            # This can happen if ts_vel_arr is shorter than velocities_arr, which shouldn't be the case.
            print(f"Warning: Timestamp array too short in integrate_velocity at index {i}.")
            x_coords.append(x_coords[-1]) # No movement if ts data is missing
            y_coords.append(y_coords[-1])
            continue

        dt = ts_vel_arr[i] - ts_vel_arr[i-1]
        if dt <= 0: dt = 0.01

        if hasattr(velocities_arr[i], '__getitem__') and len(velocities_arr[i]) >= 2:
            dx, dy = velocities_arr[i][0] * dt, velocities_arr[i][1] * dt
            x_coords.append(x_coords[-1] + dx)
            y_coords.append(y_coords[-1] + dy)
        else:
            x_coords.append(x_coords[-1])
            y_coords.append(y_coords[-1])
    return np.array(x_coords), np.array(y_coords)

# Initial ORB positions (will be used by EKF if ORB is an input, but later re-calculated for plotting)
orb_pred_x_raw, orb_pred_y_raw = integrate_velocity(predicted_velocities_orb, timestamps_orb)
orb_positions_raw = np.column_stack((orb_pred_x_raw, orb_pred_y_raw)) if len(orb_pred_x_raw) > 0 and len(orb_pred_y_raw) > 0 and len(orb_pred_x_raw) == len(orb_pred_y_raw) else np.array([])


# ===============================
# === 5. GPS Processing
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
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 1.0
        self.Q = np.diag([0.005, 0.005, 0.05, 0.05]) # Process noise: uncertainty in pos_dot, vel_dot
        self.R_uwb = np.eye(2) * 5.0 # Measurement noise for UWB (position)
        self.R_gps = np.eye(2) * 2.50 # Measurement noise for GPS (position)

    def predict(self, u_vel, dt_val): # u_vel is [vx, vy] from sensor fusion
        F = np.array([[1, 0, dt_val, 0], [0, 1, 0, dt_val], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.x = F @ self.x 
        # Add process noise related to velocity control input if u_vel is a control
        # If u_vel is a *measurement* of velocity, it's handled differently.
        # Here, u_vel is used to directly set the velocity state.
        self.P = F @ self.P @ F.T + self.Q 
        self.x[2, 0] = u_vel[0] # vx
        self.x[3, 0] = u_vel[1] # vy


    def correct(self, z_meas, sensor_type='gps'): # z_meas is [x, y] position
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) # Observation model: measures x, y
        R_mat = self.R_gps if sensor_type == 'gps' else self.R_uwb
        
        z_col = z_meas.reshape(2, 1)
        y_res = z_col - (H @ self.x) # Innovation or residual
        
        S_mat = H @ self.P @ H.T + R_mat # Innovation covariance
        if np.linalg.det(S_mat) < 1e-9: 
            return
        try:
            S_inv = np.linalg.inv(S_mat)
        except np.linalg.LinAlgError:
            return
            
        K_gain = self.P @ H.T @ S_inv # Kalman gain
        self.x = self.x + K_gain @ y_res
        self.P = (np.eye(4) - K_gain @ H) @ self.P

ekf = CorrectedEKF()
positions_ekf = []
N_HISTORY = 3
FAILURE_THRESHOLD_DIST_SQ = (0.01)**2 
uwb_history = deque(maxlen=N_HISTORY)
gps_history = deque(maxlen=N_HISTORY)

np_timestamps_orb = np.array(timestamps_orb) 

if len(timestamps_windowed) > 1:
    last_time = timestamps_windowed[0]
    if positions_true.shape[0] > 0:
        ekf.x[0,0] = positions_true[0,0]
        ekf.x[1,0] = positions_true[0,1]
    positions_ekf.append(ekf.x[:2].flatten().copy())


    for i in range(1, len(timestamps_windowed)): 
        dt_ekf = timestamps_windowed[i] - last_time
        if dt_ekf <= 0: 
            dt_ekf = 0.01 
        last_time = timestamps_windowed[i]

        current_timestamp_for_ekf_step = timestamps_windowed[i]
        
        lidar_vel_component = np.zeros(2)
        lidar_vel_available = False
        if i < len(y_lidar_pred): 
            lidar_vel_component = y_lidar_pred[i]
            lidar_vel_available = True

        orb_vel_component = np.zeros(2)
        orb_vel_available_and_synced = False
        # Use predicted_velocities_orb (raw, with 0 for black frames) for EKF input
        if predicted_velocities_orb.size > 0 and np_timestamps_orb.size > 0:
            orb_time_idx = (np.abs(np_timestamps_orb - current_timestamp_for_ekf_step)).argmin()
            if np.abs(np_timestamps_orb[orb_time_idx] - current_timestamp_for_ekf_step) < (2 * dt_ekf): 
                if orb_time_idx < len(predicted_velocities_orb): 
                    orb_vel_component = predicted_velocities_orb[orb_time_idx]
                    orb_vel_available_and_synced = True
        
        num_vel_sources = 0
        current_pred_vel_ekf = np.zeros(2)
        if lidar_vel_available:
            current_pred_vel_ekf += lidar_vel_component
            num_vel_sources += 1
        if orb_vel_available_and_synced:
            current_pred_vel_ekf += orb_vel_component
            num_vel_sources += 1
        
        if num_vel_sources > 0:
            current_pred_vel_ekf /= num_vel_sources
        
        ekf.predict(current_pred_vel_ekf, dt_ekf)

        uwb_valid = False
        current_uwb_pos = None
        if y_uwb_pred.size > 0: 
            uwb_df_idx = (np.abs(df['timestamp'].values - current_timestamp_for_ekf_step)).argmin()
            if uwb_df_idx < len(y_uwb_pred): 
                 _uwb = y_uwb_pred[uwb_df_idx]
                 if not np.isnan(_uwb).any(): 
                    current_uwb_pos = _uwb
                    uwb_history.append(current_uwb_pos.copy()) 
                    if len(uwb_history) == N_HISTORY: 
                        is_stuck = all(np.sum((pt - uwb_history[0])**2) <= FAILURE_THRESHOLD_DIST_SQ for pt in uwb_history)
                        if not is_stuck: uwb_valid = True
                    elif len(uwb_history) > 0 : 
                        uwb_valid = True

        gps_valid = False
        current_gps_pos = None
        if gps_positions.size > 0: 
            gps_df_idx = (np.abs(df['timestamp'].values - current_timestamp_for_ekf_step)).argmin()
            if gps_df_idx < len(gps_positions): 
                _gps = gps_positions[gps_df_idx] 
                if not np.isnan(_gps).any(): 
                    current_gps_pos = _gps
                    gps_history.append(current_gps_pos.copy()) 
                    if len(gps_history) == N_HISTORY:
                        is_stuck = all(np.sum((pt - gps_history[0])**2) <= FAILURE_THRESHOLD_DIST_SQ for pt in gps_history)
                        if not is_stuck: gps_valid = True
                    elif len(gps_history) > 0:
                        gps_valid = True
        
        if gps_valid and uwb_valid:
            ekf.correct(current_gps_pos, sensor_type='gps')
            ekf.correct(current_uwb_pos, sensor_type='uwb')
        elif gps_valid:
            ekf.correct(current_gps_pos, sensor_type='gps')
        elif uwb_valid:
            ekf.correct(current_uwb_pos, sensor_type='uwb')
        
        positions_ekf.append(ekf.x[:2].flatten().copy()) 
else:
    print("Warning: Not enough timestamps_windowed for EKF loop. EKF path might be empty or just initial point.")
    if not positions_ekf: 
        positions_ekf.append(ekf.x[:2].flatten().copy())

positions_ekf = np.array(positions_ekf)
if positions_ekf.size == 0: 
    positions_ekf = np.array([[0.0,0.0]]) 

# === MODIFIED: ORB Path Integration with EKF Reset ===
def integrate_orb_velocity_with_reset(
    velocities_arr,           # ORB velocities (with 0,0 for black frames)
    ts_vel_arr,               # timestamps_orb
    black_frame_flags_arr,    # black_frame_flags from ORB processing
    ekf_positions_path,       # positions_ekf (full path)
    ekf_timestamps_path       # timestamps_windowed (timestamps for EKF path)
):
    _orb_positions_x = [0.0]
    _orb_positions_y = [0.0]
    _orb_system_failed_state = False 
    MAX_EKF_SYNC_TIME_DIFF = 0.5 # seconds, max time diff to use EKF state for reset

    if len(velocities_arr) <= 1: # If 0 or 1 velocity, path is just (0,0)
        return np.array(_orb_positions_x), np.array(_orb_positions_y)

    for i in range(1, len(velocities_arr)):
        # dt for integrating velocities_arr[i]
        dt = ts_vel_arr[i] - ts_vel_arr[i-1]
        if dt <= 0: dt = 0.01 

        current_velocity_timestamp = ts_vel_arr[i] # Timestamp of velocities_arr[i]
        is_black_this_step = black_frame_flags_arr[i] # Failure flag for current velocity

        vx, vy = velocities_arr[i] # This velocity already has 0,0 if it was a black frame step

        if is_black_this_step:
            _orb_system_failed_state = True
            # vx, vy are already [0,0] due to earlier processing
        else: # Not a black frame
            if _orb_system_failed_state:
                # Just recovered from failure. Attempt to reset position to EKF.
                if ekf_positions_path.size > 0 and ekf_timestamps_path.size > 0:
                    # Find EKF state closest to current_velocity_timestamp
                    idx_in_ekf_ts = np.searchsorted(ekf_timestamps_path, current_velocity_timestamp)
                    
                    best_ekf_idx = -1
                    min_time_diff_to_ekf = float('inf')

                    # Check candidate EKF timestamp at idx_in_ekf_ts
                    if idx_in_ekf_ts < len(ekf_timestamps_path):
                        diff = abs(ekf_timestamps_path[idx_in_ekf_ts] - current_velocity_timestamp)
                        if diff < min_time_diff_to_ekf:
                            min_time_diff_to_ekf = diff
                            best_ekf_idx = idx_in_ekf_ts
                    
                    # Check candidate EKF timestamp at idx_in_ekf_ts - 1
                    if idx_in_ekf_ts > 0:
                        prev_idx = idx_in_ekf_ts - 1
                        diff = abs(ekf_timestamps_path[prev_idx] - current_velocity_timestamp)
                        if diff < min_time_diff_to_ekf: # Prefer earlier if equidistant
                            min_time_diff_to_ekf = diff
                            best_ekf_idx = prev_idx
                    
                    if best_ekf_idx != -1 and min_time_diff_to_ekf < MAX_EKF_SYNC_TIME_DIFF:
                        # Reset the *starting point* of this integration segment (_orb_positions_x/y[-1])
                        _orb_positions_x[-1] = ekf_positions_path[best_ekf_idx, 0]
                        _orb_positions_y[-1] = ekf_positions_path[best_ekf_idx, 1]
                        # print(f"ORB Reset at ORB time {current_velocity_timestamp:.2f} to EKF pos from EKF time {ekf_timestamps_path[best_ekf_idx]:.2f}")
                    # else:
                        # print(f"ORB Recovery: No suitable EKF state for reset at ORB time {current_velocity_timestamp:.2f}. Min diff: {min_time_diff_to_ekf:.3f}s")
                        
                _orb_system_failed_state = False # ORB system has now recovered
        
        # Integrate: pos_i = pos_{i-1} + vel_i * dt
        # _orb_positions_x/y[-1] is pos_{i-1} (potentially reset)
        new_x = _orb_positions_x[-1] + vx * dt
        new_y = _orb_positions_y[-1] + vy * dt
        _orb_positions_x.append(new_x)
        _orb_positions_y.append(new_y)

    return np.array(_orb_positions_x), np.array(_orb_positions_y)

# After EKF has run, re-calculate ORB path with potential EKF resets
orb_pred_x_corrected, orb_pred_y_corrected = integrate_orb_velocity_with_reset(
    predicted_velocities_orb, # Velocities (0,0 for black frames)
    np.array(timestamps_orb),        # Timestamps for ORB velocities
    np.array(black_frame_flags),     # Black frame flags for ORB
    positions_ekf,            # EKF path
    timestamps_windowed       # Timestamps for EKF path (assuming positions_ekf aligns with this)
)

if len(orb_pred_x_corrected) > 0 :
    orb_positions = np.column_stack((orb_pred_x_corrected, orb_pred_y_corrected))
else: # Fallback if integration produced nothing (e.g. very few ORB points)
    orb_positions = np.array([])


# ===============================
# === 7. Visualization & Error Calculation
# ===============================
plt.figure(figsize=(16, 10))
ax = plt.gca()

def robust_scatter_plot(data, label, color, s=10):
    if data is not None and isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] > 0 and data.shape[1] == 2:
        valid_data = data[~np.isnan(data).any(axis=1)]
        if valid_data.shape[0] > 0:
            plt.scatter(valid_data[:, 0], valid_data[:, 1], label=label, color=color, s=s)
        else:
            print(f"Skipping plot for {label}: Data contains only NaNs or is empty after NaN removal.")
    else: 
        shape_info = data.shape if hasattr(data, 'shape') else 'N/A'
        print(f"Skipping plot for {label}: Data empty/malformed. Shape: {shape_info}, Type: {type(data)}")

robust_scatter_plot(positions_true, 'Ground Truth', 'black', s=20)
robust_scatter_plot(positions_pred, 'LiDAR+IMU LSTM', 'blue')
robust_scatter_plot(orb_positions, 'ORB+IMU LSTM (EKF Reset)', 'green') # Now uses corrected path
robust_scatter_plot(y_uwb_pred, 'UWB LSTM (Raw)', 'red') 
robust_scatter_plot(gps_positions, 'GPS (Raw)', 'orange') 

if positions_ekf.shape[0] > 1 or (positions_ekf.shape[0] == 1 and not np.all(np.isclose(positions_ekf, 0.0))):
    robust_scatter_plot(positions_ekf, 'EKF Fused', 'purple', s=15)
else: 
    print(f"Skipping plot for EKF Fused: Data is default or empty. Shape: {positions_ekf.shape}")


plt.xlabel('X Position (m)', fontsize=24)
plt.ylabel('Y Position (m)', fontsize=24)
plt.title('Trajectory Comparison with Hybrid Fusion', fontsize=26)
plt.legend(fontsize=22)
plt.grid(True)
plt.axis('equal')
ax.tick_params(axis='both', which='major', labelsize=22)
plt.tight_layout()
plt.savefig("full_trajectory_comparison_with_gps_EKF_scatter_colored_MODIFIED_ORB_RESET.png", dpi=300)
plt.show()

def compute_position_error_aligned(predicted_path, true_path_full, predicted_timestamps=None, true_timestamps=None):
    if not isinstance(predicted_path, np.ndarray) or predicted_path.size == 0 or \
       not isinstance(true_path_full, np.ndarray) or true_path_full.size == 0:
        return np.array([np.nan]), np.nan

    if predicted_timestamps is None or true_timestamps is None or \
       len(predicted_timestamps) == 0 or len(true_timestamps) == 0:
        # Fallback to simple length-based alignment if timestamps are not suitable
        min_len = min(len(predicted_path), len(true_path_full))
        if min_len == 0: return np.array([np.nan]), np.nan
        pred_final_segment = predicted_path[:min_len]
        true_final_segment = true_path_full[:min_len]
    else:
        aligned_true_indices = []
        valid_pred_indices = []
        for i, pred_ts in enumerate(predicted_timestamps):
            if np.isnan(pred_ts): continue
            
            insert_idx = np.searchsorted(true_timestamps, pred_ts)
            best_true_idx = -1
            min_dt_align = float('inf')

            if insert_idx < len(true_timestamps):
                dt_align = abs(true_timestamps[insert_idx] - pred_ts)
                if dt_align < min_dt_align:
                    min_dt_align = dt_align
                    best_true_idx = insert_idx
            
            if insert_idx > 0: 
                dt_align = abs(true_timestamps[insert_idx-1] - pred_ts)
                if dt_align < min_dt_align:
                    min_dt_align = dt_align
                    best_true_idx = insert_idx -1
            
            # Max time diff for alignment (e.g. 1 second, depends on data rates)
            # This threshold should be more generous than EKF reset sync diff
            MAX_ALIGN_TIME_DIFF = 1.0 
            if best_true_idx != -1 and min_dt_align < MAX_ALIGN_TIME_DIFF:
                if not np.isnan(predicted_path[i]).any() and not np.isnan(true_path_full[best_true_idx]).any():
                    aligned_true_indices.append(best_true_idx)
                    valid_pred_indices.append(i)
        
        if not valid_pred_indices:
            return np.array([np.nan]), np.nan
            
        pred_final_segment = predicted_path[valid_pred_indices]
        true_final_segment = true_path_full[aligned_true_indices]

    if pred_final_segment.shape[0] == 0: return np.array([np.nan]), np.nan
    
    # Final NaN check on segments before error calculation
    valid_idx_pred = ~np.isnan(pred_final_segment).any(axis=1)
    valid_idx_true = ~np.isnan(true_final_segment).any(axis=1)
    valid_idx = valid_idx_pred & valid_idx_true
    
    pred_final_segment = pred_final_segment[valid_idx]
    true_final_segment = true_final_segment[valid_idx]

    if pred_final_segment.shape[0] == 0: return np.array([np.nan]), np.nan

    error_vals = np.linalg.norm(pred_final_segment - true_final_segment, axis=1)
    return error_vals, np.mean(error_vals)


print("\n--- Mean Position Errors (meters) ---")
# LiDAR+IMU: positions_pred aligns with timestamps_windowed
error_lidar, mean_error_lidar = compute_position_error_aligned(
    positions_pred, positions_true, 
    predicted_timestamps=timestamps_windowed[:len(positions_pred)], 
    true_timestamps=timestamps_windowed[:len(positions_true)]       
)
print(f"LiDAR+IMU LSTM:   {mean_error_lidar:.4f}")

# ORB+IMU (EKF Reset): orb_positions aligns with timestamps_orb
# Note: timestamps_orb might be shorter than orb_positions if integrate_orb_velocity_with_reset returns [0,0] for 1 velocity.
# The path length from integrate_orb_velocity_with_reset is len(velocities_arr) if len > 1, or 1 if len <=1.
# So, len(orb_positions) should be len(timestamps_orb) if len(timestamps_orb)>1.
# If len(timestamps_orb) <=1, len(orb_positions) is 1.
# We need to ensure predicted_timestamps matches length of predicted_path.
orb_ts_for_error = np.array(timestamps_orb)[:len(orb_positions)] if len(orb_positions) <= len(timestamps_orb) else np.array(timestamps_orb)

error_orb, mean_error_orb = compute_position_error_aligned(
    orb_positions, positions_true, 
    predicted_timestamps=orb_ts_for_error, 
    true_timestamps=timestamps_windowed[:len(positions_true)]     
)
print(f"ORB+IMU LSTM (EKF Reset): {mean_error_orb:.4f}")


# UWB LSTM (Raw): y_uwb_pred aligns with df['timestamp']
error_uwb, mean_error_uwb = compute_position_error_aligned(
    y_uwb_pred, positions_true,
    predicted_timestamps=df['timestamp'].values[:len(y_uwb_pred)], 
    true_timestamps=timestamps_windowed[:len(positions_true)]     
)
print(f"UWB LSTM (Raw):   {mean_error_uwb:.4f}")

# GPS (Raw): gps_positions aligns with df['timestamp']
error_gps, mean_error_gps = compute_position_error_aligned(
    gps_positions, positions_true,
    predicted_timestamps=df['timestamp'].values[:len(gps_positions)], 
    true_timestamps=timestamps_windowed[:len(positions_true)]        
)
print(f"GPS (Raw):        {mean_error_gps:.4f}")

# EKF Fused: positions_ekf aligns with timestamps_windowed
ekf_ts_for_error = timestamps_windowed[:len(positions_ekf)] 
error_ekf, mean_error_ekf = compute_position_error_aligned(
    positions_ekf, positions_true,
    predicted_timestamps=ekf_ts_for_error,
    true_timestamps=timestamps_windowed[:len(positions_true)]
)
print(f"EKF Fused:        {mean_error_ekf:.4f}")


plt.figure(figsize=(16, 10))
ax_err = plt.gca()

def plot_error_if_valid(error_array, label, **kwargs):
    if error_array.size > 0 and not np.all(np.isnan(error_array)):
        # Ensure we are plotting against a valid number of timesteps for this error array
        plt.plot(np.arange(len(error_array)), error_array, label=label, **kwargs)


plot_error_if_valid(error_lidar, 'LiDAR+IMU LSTM Error')
plot_error_if_valid(error_orb, 'ORB+IMU LSTM (EKF Reset) Error')
plot_error_if_valid(error_uwb, 'UWB LSTM (Raw) Error')
plot_error_if_valid(error_gps, 'GPS (Raw) Error')
plot_error_if_valid(error_ekf, 'EKF Fused Error', linewidth=2)

plt.xlabel("Aligned Timestep Index", fontsize=24)
plt.ylabel("Position Error (meters)", fontsize=24)
plt.title("Sensor Position Errors Compared to Ground Truth", fontsize=26)
plt.legend(fontsize=22)
plt.grid(True)
ax_err.tick_params(axis='both', which='major', labelsize=22)
plt.tight_layout()
plt.savefig("position_errors_over_time_MODIFIED_ORB_RESET.png", dpi=300)
plt.show()

print("\n--- Maximum Position Errors (meters) ---")
print(f"LiDAR+IMU LSTM:   {np.nanmax(error_lidar) if error_lidar.size > 0 and not np.all(np.isnan(error_lidar)) else np.nan:.4f}")
print(f"ORB+IMU LSTM (EKF Reset): {np.nanmax(error_orb) if error_orb.size > 0 and not np.all(np.isnan(error_orb)) else np.nan:.4f}")
print(f"UWB LSTM (Raw):   {np.nanmax(error_uwb) if error_uwb.size > 0 and not np.all(np.isnan(error_uwb)) else np.nan:.4f}")
print(f"GPS (Raw):        {np.nanmax(error_gps) if error_gps.size > 0 and not np.all(np.isnan(error_gps)) else np.nan:.4f}")
print(f"EKF Fused:        {np.nanmax(error_ekf) if error_ekf.size > 0 and not np.all(np.isnan(error_ekf)) else np.nan:.4f}")

def compute_trajectory_rmse(predicted, true):
    # 1. Calculate Euclidean distance for each point: sqrt((x2-x1)^2 + (y2-y1)^2)
    distances = np.linalg.norm(predicted - true, axis=1)
    
    # 2. Square the distances, take the mean, then the square root
    return np.sqrt(np.mean(distances**2))

# ===============================
# === 8. Final Detailed Error Summary
# ===============================

def calculate_final_metrics(error_array):
    """Helper to calculate Mean, RMSE, and Max from existing error distance arrays."""
    if error_array.size == 0 or np.all(np.isnan(error_array)):
        return np.nan, np.nan, np.nan
    
    # Remove any potential NaNs to ensure clean calculation
    clean_errors = error_array[~np.isnan(error_array)]
    
    mean_err = np.mean(clean_errors)
    rmse_err = np.sqrt(np.mean(clean_errors**2))
    max_err = np.max(clean_errors)
    
    return mean_err, rmse_err, max_err

# Store results in a dictionary for easy display
sensor_results = {
    "LiDAR+IMU LSTM":   error_lidar,
    "ORB+IMU (Reset)":  error_orb,
    "UWB LSTM (Raw)":   error_uwb,
    "GPS (Raw)":        error_gps,
    "EKF Fused":        error_ekf
}

print("\n" + "="*85)
print(f"{'Sensor Source':<25} | {'Mean (m)':<12} | {'RMSE (m)':<12} | {'Max (m)':<10}")
print("-" * 85)

for name, err_arr in sensor_results.items():
    m_e, r_e, mx_e = calculate_final_metrics(err_arr)
    # Check if we have valid numbers to print
    if not np.isnan(m_e):
        print(f"{name:<25} | {m_e:<12.4f} | {r_e:<12.4f} | {mx_e:<10.4f}")
    else:
        print(f"{name:<25} | {'No Data':<12} | {'No Data':<12} | {'No Data':<10}")

print("="*85)

print("Processing complete.")

print("Processing complete.")
