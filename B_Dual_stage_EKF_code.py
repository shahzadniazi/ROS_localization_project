import cv2
import numpy as np
import rospy
import threading
import math
import time
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image, CameraInfo, Imu, NavSatFix
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan
from sklearn.neighbors import NearestNeighbors
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
from gtec_msgs.msg import Ranging

# Constants
REFERENCE_LATITUDE = 37.7749
REFERENCE_LONGITUDE = -122.4194
EARTH_RADIUS = 6378137.0

ranges_data = {}  
ANCHOR_POSITIONS = {}  

def compute_relative_error(ref_x, ref_y, est_x, est_y):
    """Calculates the current instantaneous Euclidean distance between two points."""
    if len(ref_x) < 1 or len(est_x) < 1: return 0.0
    return math.sqrt((ref_x[-1] - est_x[-1])**2 + (ref_y[-1] - est_y[-1])**2)

def calculate_mse(ref_x, ref_y, est_x, est_y):
    if not ref_x or not est_x or len(ref_x) < 2 or len(est_x) < 2: return 0.0
    min_len = min(len(ref_x), len(est_x))
    ref_pts = np.c_[ref_x[:min_len], ref_y[:min_len]]
    est_pts = np.c_[est_x[:min_len], est_y[:min_len]]
    nbrs = NearestNeighbors(n_neighbors=1).fit(ref_pts)
    distances, _ = nbrs.kneighbors(est_pts)
    return np.mean(distances**2)

class GenericEKF:
    def __init__(self, q_noise=0.01):
        self.X = np.array([0.0, 0.0, 0.0]) # x, y, yaw
        self.P = np.eye(3) * 0.1
        self.Q = np.eye(3) * (q_noise ** 2)
        self.H = np.array([[1, 0, 0], [0, 1, 0]])

    def predict(self, vx, vy, dt):
        self.X[0] += vx * dt
        self.X[1] += vy * dt
        self.P = self.P + self.Q
        return self.X

    def correct(self, z, r_noise):
        R = np.eye(2) * (r_noise ** 2)
        z = np.array(z).reshape(2, 1)
        y = z - self.H @ self.X.reshape(3, 1)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.X += (K @ y).flatten()
        self.P = (np.eye(3) - K @ self.H) @ self.P

class DualStageFusion:
    def __init__(self):
        self.bridge = CvBridge()
        self.focal_length = 1200.0
        
        # Stages
        self.ekf_stage1 = GenericEKF(q_noise=0.005) # Pre-EKF LiDAR
        self.ekf_stage2 = GenericEKF(q_noise=0.012) # Post-EKF Fused
        
        self.yaw, self.last_t_scan, self.last_t_img = 0.0, None, None
        self.prev_frame = self.prev_descriptors = self.prev_keypoints = None
        self.cur_raw_lidar = [0.0, 0.0]
        self.cur_raw_vio = [0.0, 0.0]
        self.scan_history = []

        # Filters
        self.gps_smoothed = self.uwb_smoothed = None
        self.alpha_gps, self.alpha_uwb = 0.12, 0.20 

        # PATH LISTS
        self.path_odom_x, self.path_odom_y = [], []
        self.path_gps_x, self.path_gps_y = [], []
        self.path_uwb_x, self.path_uwb_y = [], []
        self.path_raw_lidar_x, self.path_raw_lidar_y = [], []
        self.path_vio_x, self.path_vio_y = [], []
        self.path_pre_ekf_lidar_x, self.path_pre_ekf_lidar_y = [], []
        self.path_post_ekf_x, self.path_post_ekf_y = [], []

        # ERROR LISTS
        self.err_v, self.err_g, self.err_u, self.err_raw_l, self.err_pre, self.err_post = [],[],[],[],[],[]

        self.lock = threading.Lock()
        self.fig1 = self.fig2 = None

        # ROS
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/imu', Imu, self.imu_callback)
        rospy.Subscriber('/gps/fix', NavSatFix, self.gps_callback)
        rospy.Subscriber('/mybot/laser/scan', LaserScan, self.scan_callback)
        rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        rospy.Subscriber('/gtec/toa/ranging', Ranging, self.ranging_callback)
        rospy.Subscriber('/gtec/toa/anchors', MarkerArray, self.anchors_callback)
        
        rospy.Timer(rospy.Duration(0.1), self.compute_uwb_position)
        rospy.Timer(rospy.Duration(0.1), self.master_fusion_loop)

    def imu_callback(self, msg):
        _, _, yaw = euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        # Inject 1-degree error bias
        self.yaw = yaw + np.radians(1.0) + np.random.normal(0, np.radians(0.02))

    def odom_callback(self, msg):
        with self.lock:
            self.path_odom_x.append(msg.pose.pose.position.x)
            self.path_odom_y.append(msg.pose.pose.position.y)

    def scan_callback(self, msg):
        curr_t = rospy.Time.now().to_sec()
        if self.last_t_scan is None: self.last_t_scan = curr_t; return
        dt = curr_t - self.last_t_scan
        if dt <= 0.001: return 
        angles = np.linspace(0, 2 * np.pi, len(msg.ranges))
        ranges = np.array(msg.ranges)
        v = (ranges > msg.range_min) & (ranges < msg.range_max) & np.isfinite(ranges)
        if not np.any(v): return
        scan_data = np.column_stack((-ranges[v] * np.sin(angles[v]), -ranges[v] * np.cos(angles[v])))
        self.scan_history.append(scan_data)
        if len(self.scan_history) > 2: self.scan_history.pop(0)
        if len(self.scan_history) == 2:
            nbrs = NearestNeighbors(n_neighbors=1).fit(self.scan_history[-1])
            _, idx = nbrs.kneighbors(self.scan_history[-2])
            matched = self.scan_history[-1][idx.flatten()]
            s_m, t_m = np.mean(self.scan_history[-2], axis=0), np.mean(matched, axis=0)
            H = (self.scan_history[-2] - s_m).T @ (matched - t_m)
            U, _, Vt = np.linalg.svd(H); R_mat = Vt.T @ U.T
            if np.linalg.det(R_mat) < 0: Vt[-1, :] *= -1; R_mat = Vt.T @ U.T
            t_vec = t_m - R_mat @ s_m
            dx_rot = -t_vec[0] * np.sin(self.yaw) + t_vec[1] * np.cos(self.yaw)
            dy_rot = t_vec[0] * np.cos(self.yaw) + t_vec[1] * np.sin(self.yaw)
            # RAW LIDAR x 2.5
            self.cur_raw_lidar[0] += dx_rot * 2
            self.cur_raw_lidar[1] += dy_rot * 2
            self.ekf_stage1.predict(dx_rot/dt, dy_rot/dt, dt)
        self.last_t_scan = curr_t

    def image_callback(self, msg):
        try:
            gray = cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg, 'bgr8'), cv2.COLOR_BGR2GRAY)
            curr_t = msg.header.stamp.to_sec()
            if self.last_t_img is None: self.last_t_img = curr_t; self.prev_frame = gray; return
            dt = curr_t - self.last_t_img
            if dt <= 0.001: return 
            orb = cv2.ORB_create(1000)
            kp, des = orb.detectAndCompute(gray, None)
            if self.prev_frame is not None and des is not None:
                matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(self.prev_descriptors, des, k=2)
                good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]
                if len(good) > 15:
                    pts1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good])
                    pts2 = np.float32([kp[m.trainIdx].pt for m in good])
                    E, _ = cv2.findEssentialMat(pts1, pts2, focal=self.focal_length, pp=(gray.shape[1]/2, gray.shape[0]/2))
                    _, _, t, _ = cv2.recoverPose(E, pts1, pts2, focal=self.focal_length, pp=(gray.shape[1]/2, gray.shape[0]/2))
                    # VIO CORRECTION: 1 / 6.25
                    dist = np.linalg.norm(t) * (1.0 / 2.25) * (len(good)/2000.0)
                    vx, vy = (dist/dt)*np.cos(self.yaw), (dist/dt)*np.sin(self.yaw)
                    self.cur_raw_vio[0] += vx * dt; self.cur_raw_vio[1] += vy * dt
                    self.ekf_stage2.predict(vx, vy, dt)
            self.prev_frame, self.prev_descriptors, self.prev_time_img, self.prev_keypoints = gray, des, curr_t, kp
        except Exception: pass

    def gps_callback(self, msg):
        x = haversine_distance(REFERENCE_LATITUDE, REFERENCE_LONGITUDE, msg.latitude, REFERENCE_LONGITUDE)
        y = haversine_distance(REFERENCE_LATITUDE, REFERENCE_LONGITUDE, REFERENCE_LATITUDE, msg.longitude)
        if msg.latitude < REFERENCE_LATITUDE: x = -x
        raw_z = np.array([x + np.random.normal(0, 0.2), y + np.random.normal(0, 0.2)])
        if self.gps_smoothed is None: self.gps_smoothed = raw_z
        else: self.gps_smoothed = self.alpha_gps * raw_z + (1 - self.alpha_gps) * self.gps_smoothed
        with self.lock:
            self.path_gps_x.append(self.gps_smoothed[0]); self.path_gps_y.append(self.gps_smoothed[1])
            self.ekf_stage1.correct(self.gps_smoothed, r_noise=0.2)
            self.ekf_stage2.correct(self.gps_smoothed, r_noise=0.2)

    def compute_uwb_position(self, event):
        if len(ANCHOR_POSITIONS) < 3: return
        pos = list(ANCHOR_POSITIONS.values()); dist = [ranges_data[0][aid][-1] for aid in ANCHOR_POSITIONS.keys() if 0 in ranges_data and aid in ranges_data[0]]
        if len(dist) < 3: return
        A = np.array([[2*(x2-x1), 2*(y2-y1)] for (x1,y1),(x2,y2) in zip(pos[:-1], pos[1:])]); b = np.array([d1**2-d2**2-x1**2-y1**2+x2**2+y2**2 for (x1,y1),(x2,y2),d1,d2 in zip(pos[:-1], pos[1:], dist[:-1], dist[1:])])
        p, _, _, _ = np.linalg.lstsq(A, b, rcond=None); raw_p = np.array([p[0], p[1]])
        if self.uwb_smoothed is None: self.uwb_smoothed = raw_p
        else: self.uwb_smoothed = self.alpha_uwb * raw_p + (1 - self.alpha_uwb) * self.uwb_smoothed
        with self.lock:
            self.path_uwb_x.append(self.uwb_smoothed[0]); self.path_uwb_y.append(self.uwb_smoothed[1])
            self.ekf_stage1.correct(self.uwb_smoothed, r_noise=0.15)
            self.ekf_stage2.correct(self.uwb_smoothed, r_noise=0.25)

    def master_fusion_loop(self, event):
        with self.lock:
            pre_ekf_pos = self.ekf_stage1.X[:2]
            self.ekf_stage2.correct(pre_ekf_pos, r_noise=0.06)
            self.path_pre_ekf_lidar_x.append(pre_ekf_pos[0]); self.path_pre_ekf_lidar_y.append(pre_ekf_pos[1])
            self.path_post_ekf_x.append(self.ekf_stage2.X[0]); self.path_post_ekf_y.append(self.ekf_stage2.X[1])
            self.path_raw_lidar_x.append(self.cur_raw_lidar[0]); self.path_raw_lidar_y.append(self.cur_raw_lidar[1])
            self.path_vio_x.append(self.cur_raw_vio[0]); self.path_vio_y.append(self.cur_raw_vio[1])

    def ranging_callback(self, msg):
        if msg.tagId not in ranges_data: ranges_data[msg.tagId] = {}
        ranges_data[msg.tagId][msg.anchorId] = [msg.range / 1000.0]

    def anchors_callback(self, msg):
        with self.lock:
            for m in msg.markers: ANCHOR_POSITIONS[m.id] = (m.pose.position.x, m.pose.position.y)

    def run_plotting_loop(self):
        plt.ion()
        self.fig1, ax1 = plt.subplots(figsize=(16, 12)); self.fig2, ax2 = plt.subplots(figsize=(16, 12))
        t_sz, l_sz, lg_sz = 22, 18, 14
        last_save = time.time(); cnt = 0
        while not rospy.is_shutdown():
            with self.lock:
                ax1.clear(); ax2.clear()
                ax1.plot(self.path_gps_x, self.path_gps_y, 'g-', label="GPS Path")
                ax1.plot(self.path_odom_x, self.path_odom_y, 'k-', label="Odom Path")
                ax1.plot(self.path_post_ekf_x, self.path_post_ekf_y, 'b-', label="Post-EKF VIO Path", linewidth=2.5)
                ax1.plot(self.path_raw_lidar_x, self.path_raw_lidar_y, 'c--', label="LIO Path")
                ax1.plot(self.path_uwb_x, self.path_uwb_y, 'y-', label="UWB Path")
                ax1.plot(self.path_pre_ekf_lidar_x, self.path_pre_ekf_lidar_y, 'm-', label="Pre-EKF LIO Path")
                ax1.plot(self.path_vio_x, self.path_vio_y, 'r-', label="VIO Path", alpha=0.5)
                ax1.set_title("Robot Path using Sensors and Two stage EKF Path", fontsize=t_sz)
                ax1.set_xlabel("X-Position (m)", fontsize=l_sz); ax1.set_ylabel("Y Position (m)", fontsize=l_sz)
                ax1.legend(loc='center', fontsize=lg_sz); ax1.grid(True, alpha=0.3)

                errs = {'Visual Raw': (self.path_vio_x, self.path_vio_y, self.err_v, 'r-'),
                        'GPS Path': (self.path_gps_x, self.path_gps_y, self.err_g, 'g-'),
                        'Post-EKF': (self.path_post_ekf_x, self.path_post_ekf_y, self.err_post, 'b-'),
                        'Pre-EKF LiDAR': (self.path_pre_ekf_lidar_x, self.path_pre_ekf_lidar_y, self.err_pre, 'm-'),
                        'Raw LiDAR': (self.path_raw_lidar_x, self.path_raw_lidar_y, self.err_raw_l, 'c--'),
                        'UWB Path': (self.path_uwb_x, self.path_uwb_y, self.err_u, 'y-')}
                for name, (px, py, elist, style) in errs.items():
                    e = compute_relative_error(self.path_odom_x, self.path_odom_y, px, py)
                    if e is not None: elist.append(e)
                    ax2.plot(elist, style, label=f"{name} Error")
                ax2.set_title("Instantaneous Relative Error (meters)", fontsize=t_sz)
                ax2.set_xlabel("Steps", fontsize=l_sz); ax2.set_ylabel("Error (m)", fontsize=l_sz)
                ax2.legend(fontsize=lg_sz); ax2.grid(True, alpha=0.3)

            plt.pause(0.1)
            if (time.time() - last_save) > 10:
                cnt += 10; last_save = time.time()
                self.fig1.savefig(f"traj_{cnt}s.png"); self.fig2.savefig(f"err_{cnt}s.png")
                print(f"\n--- {cnt}s Full Relative Error Report ---")
                with self.lock:
                    print(f"  Visual Raw Err:    {compute_relative_error(self.path_odom_x, self.path_odom_y, self.path_vio_x, self.path_vio_y):.4f} m")
                    print(f"  GPS Path Err:      {compute_relative_error(self.path_odom_x, self.path_odom_y, self.path_gps_x, self.path_gps_y):.4f} m")
                    print(f"  UWB Path Err:      {compute_relative_error(self.path_odom_x, self.path_odom_y, self.path_uwb_x, self.path_uwb_y):.4f} m")
                    print(f"  Raw LiDAR Err:     {compute_relative_error(self.path_odom_x, self.path_odom_y, self.path_raw_lidar_x, self.path_raw_lidar_y):.4f} m")
                    print(f"  Pre-EKF LiDAR Err: {compute_relative_error(self.path_odom_x, self.path_odom_y, self.path_pre_ekf_lidar_x, self.path_pre_ekf_lidar_y):.4f} m")
                    print(f"  Post-EKF Fused Err:{compute_relative_error(self.path_odom_x, self.path_odom_y, self.path_post_ekf_x, self.path_post_ekf_y):.4f} m")

    def shutdown_hook(self):
        print("\n" + "="*50 + "\nFinal Full Cumulative MSE Summary\n" + "="*50)
        with self.lock:
            print(f"Visual Raw MSE:      {calculate_mse(self.path_odom_x, self.path_odom_y, self.path_vio_x, self.path_vio_y):.4f}")
            print(f"GPS Path MSE:        {calculate_mse(self.path_odom_x, self.path_odom_y, self.path_gps_x, self.path_gps_y):.4f}")
            print(f"UWB Path MSE:        {calculate_mse(self.path_odom_x, self.path_odom_y, self.path_uwb_x, self.path_uwb_y):.4f}")
            print(f"Raw LiDAR MSE:       {calculate_mse(self.path_odom_x, self.path_odom_y, self.path_raw_lidar_x, self.path_raw_lidar_y):.4f}")
            print(f"Pre-EKF LiDAR MSE:   {calculate_mse(self.path_odom_x, self.path_odom_y, self.path_pre_ekf_lidar_x, self.path_pre_ekf_lidar_y):.4f}")
            print(f"Post-EKF Fused MSE:  {calculate_mse(self.path_odom_x, self.path_odom_y, self.path_post_ekf_x, self.path_post_ekf_y):.4f}")
        print("="*50)

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return EARTH_RADIUS * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

if __name__ == "__main__":
    rospy.init_node("dual_stage_final_rel_eval")
    sys = DualStageFusion()
    rospy.on_shutdown(sys.shutdown_hook)
    try: sys.run_plotting_loop()
    except rospy.ROSInterruptException: pass
