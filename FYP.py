import cv2
import numpy as np
from pylibfreenect2 import Freenect2, FrameType, SyncMultiFrameListener
import threading
import time
import copy
from pyzbar import pyzbar # For QR registration and detection
import torch # For checking CUDA availability
import traceback # For printing detailed errors

# Import your components (OA is removed)
# from components.kinectv2_OA import OA # <<< REMOVED OA IMPORT >>>
from components.kinectv2_yoloqr import YoloQR
from components.Kinectv2_motors import motors

# --- Constants and Initializations ---
PROCESSING_WIDTH = 640
PROCESSING_HEIGHT = 360
PROCESSING_RESOLUTION = (PROCESSING_WIDTH, PROCESSING_HEIGHT)
YOLO_IMG_SIZE = 640
CONF_THRESHOLD = 0.6

QR_FOLLOW_TOLERANCE = 75 # Pixels, relative to PROCESSING_WIDTH
TARGET_STOP_DISTANCE_M = 0.8
# OBSTACLE_MIN_DISTANCE_M = 0.8 # <<< REMOVED OA CONSTANT >>>
DEPTH_PROCESSING_LIMIT_MM = 4000.0 # 4 meters
MIN_DEPTH_PROCESSING_MM = 200.0 # 0.2 meters

# --- Visualization Flags ---
SHOW_DEPTH_MAP = True
SHOW_MASKED_DEPTH_MAP = True # Keep this True as requested

# --- Shared Data Structures ---
frame_lock = threading.Lock()
latest_frames = {
    "color_full": None,
    "color_proc": None,
    "depth_raw": None,
    "timestamp": 0
}

results_lock = threading.Lock()
latest_results = {
    "target_person_box": None,
    "qr_detected_flag": False,
    "qr_center_x": None,
    "target_distance_m": float('inf'),
    "yolo_boxes_for_drawing": [],
    "qr_bbox_int_draw": None,
    # "obstacle_detected": False, # <<< REMOVED OA RESULT >>>
    "vis_frame": None,
    "depth_vis": None,         # Visualization of processed depth
    "masked_depth_vis": None,  # Visualization of masked depth (target excluded)
    "timestamp": 0
}

# --- Thread Control ---
stop_event = threading.Event()

# --- Frame Acquisition Thread (Unchanged from previous version) ---
class FrameAcquisitionThread(threading.Thread):
    def __init__(self, listener):
        super().__init__(daemon=True)
        self.listener = listener
        self.name = "FrameAcquisitionThread"
        print(f"{self.name} Initialized")

    def run(self):
        print(f"{self.name} Started")
        while not stop_event.is_set():
            try:
                frames = self.listener.waitForNewFrame()
                color_frame = frames[FrameType.Color] # Native resolution: 1920x1080, RGBA
                depth_frame = frames[FrameType.Depth] # Native resolution: 512x424, float32 (mm)

                # --- Process Color Frame ---
                color_data_rgba = color_frame.asarray(np.uint8)
                color_data_rgba = np.ascontiguousarray(color_data_rgba)
                frame_bgr_full = cv2.cvtColor(color_data_rgba, cv2.COLOR_RGBA2RGB)
                frame_bgr_proc = cv2.resize(frame_bgr_full, PROCESSING_RESOLUTION, interpolation=cv2.INTER_LINEAR)

                # --- Get Raw Depth Data ---
                depth_data_mm_raw = depth_frame.asarray(np.float32)
                depth_data_mm_raw = np.ascontiguousarray(depth_data_mm_raw)

                current_time = time.time()
                # --- Update Shared Frames ---
                with frame_lock:
                    latest_frames["color_full"] = frame_bgr_full
                    latest_frames["color_proc"] = frame_bgr_proc
                    latest_frames["depth_raw"] = depth_data_mm_raw
                    latest_frames["timestamp"] = current_time

                self.listener.release(frames)

            except Exception as e:
                print(f"Error in {self.name}: {e}")
                traceback.print_exc()
                time.sleep(0.5)

        print(f"{self.name} Stopped")

# --- Processing Thread (Modified - OA Removed, Manual Depth Viz Added) ---
class ProcessingThread(threading.Thread):
    # <<< REMOVED obstacle_avoidance from __init__ >>>
    def __init__(self, yolo_qr_processor, target_qr_code):
        super().__init__(daemon=True)
        if not hasattr(yolo_qr_processor, 'run_detections_and_association'):
             raise AttributeError("The provided YoloQR processor does not have the 'run_detections_and_association' method.")
        self.yolo_qr_processor = yolo_qr_processor
        # self.obstacle_avoidance = obstacle_avoidance # <<< REMOVED OA INSTANCE >>>
        self.target_qr_code = target_qr_code
        self.last_processed_frame_time = 0
        self.name = "ProcessingThread"
        print(f"{self.name} Initialized (Obstacle Avoidance Disabled)")

    # <<< ADDED Helper for Depth Visualization >>>
    def _create_depth_visualizations(self, depth_map_mm, qr_distance_m=None, qr_center_x=None):
        """Creates colormapped depth and masked depth visualizations."""
        depth_vis = None
        masked_depth_vis = None

        try:
            # Normalize depth for visualization (0-1 range, handling potential division by zero)
            max_depth_viz = DEPTH_PROCESSING_LIMIT_MM
            if max_depth_viz <= 0: max_depth_viz = 1.0  # Avoid division by zero

            # Create basic depth visualization
            depth_normalized = np.clip(depth_map_mm / max_depth_viz, 0, 1)
            depth_vis = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            depth_vis[depth_map_mm == 0] = [0, 0, 0]  # Set invalid depth (0mm) to black

            # Create masked depth visualization (exclude anything behind QR code)
            masked_depth_map = depth_map_mm.copy()
            if qr_distance_m is not None:
                qr_distance_mm = qr_distance_m * 1000.0

                # Dynamically adjust exclusion threshold based on QR position
                height, width = masked_depth_map.shape
                center_x = width // 2
                if qr_center_x is not None:
                    # Calculate the relative position of the QR code (0 at center, 1 at extreme left/right)
                    relative_position = abs(qr_center_x - center_x) / center_x
                    # Interpolate threshold between qr_distance_mm - 200 and qr_distance_mm - 400
                    exclusion_threshold_mm = qr_distance_mm - (200 + 300 * relative_position)
                else:
                    # Default to qr_distance_mm - 200 if qr_center_x is not provided
                    exclusion_threshold_mm = qr_distance_mm - 200.0

                masked_depth_map[depth_map_mm > exclusion_threshold_mm] = 0

            # --- Dynamically adjust the 200-pixel-wide region based on qr_center_x ---
            height, width = masked_depth_map.shape
            if qr_center_x is not None:
                left_bound = max(0, qr_center_x - 100)
                right_bound = min(width, qr_center_x + 100)
            else:
                # Default to center if qr_center_x is not provided
                center_x = width // 2
                left_bound = max(0, center_x - 100)
                right_bound = min(width, center_x + 100)

            # Set all pixels outside the dynamically adjusted region to 0
            masked_depth_map[:, :left_bound] = 0  # Remove pixels to the left of the region
            masked_depth_map[:, right_bound:] = 0  # Remove pixels to the right of the region

            # Normalize and colormap the masked depth
            masked_depth_normalized = np.clip(masked_depth_map / max_depth_viz, 0, 1)
            masked_depth_vis = cv2.applyColorMap((masked_depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            masked_depth_vis[masked_depth_map == 0] = [0, 0, 0]  # Set invalid depth (0mm, including masked area) to black

        except Exception as e:
            print(f"Error creating depth visualizations: {e}")
            # Return None if visualization fails
            return None, None

        return depth_vis, masked_depth_vis

    def run(self):
        print(f"{self.name} Started")
        while not stop_event.is_set():
            frame_bgr_full_copy = None
            frame_bgr_proc_copy = None
            depth_raw_copy = None
            current_frame_time = 0

            # --- Get Latest Frames Safely ---
            with frame_lock:
                if latest_frames["timestamp"] > self.last_processed_frame_time:
                    if latest_frames.get("color_full") is not None: frame_bgr_full_copy = latest_frames["color_full"].copy()
                    if latest_frames.get("color_proc") is not None: frame_bgr_proc_copy = latest_frames["color_proc"].copy()
                    if latest_frames.get("depth_raw") is not None: depth_raw_copy = latest_frames["depth_raw"].copy()
                    current_frame_time = latest_frames["timestamp"]
                    self.last_processed_frame_time = current_frame_time
                else:
                    time.sleep(0.005); continue

            if frame_bgr_full_copy is None or frame_bgr_proc_copy is None or depth_raw_copy is None:
                time.sleep(0.01)
                continue

            try:
                # --- Get Frame Dimensions ---
                full_h, full_w = frame_bgr_full_copy.shape[:2]
                proc_h, proc_w = frame_bgr_proc_copy.shape[:2]

                # --- 1. QR Code Detection (on FULL frame using pyzbar) ---
                qr_detected_flag_this_frame = False
                qr_details_for_yoloqr = None
                qr_bbox_int_draw_proc_res = None

                try:
                    decoded_objects = pyzbar.decode(frame_bgr_full_copy)
                    for obj in decoded_objects:
                        try:
                            data = obj.data.decode('utf-8')
                            if data == self.target_qr_code:
                                qr_detected_flag_this_frame = True
                                points = obj.polygon
                                qr_center_x_full_res = int(np.mean([p.x for p in points]))
                                qr_center_y_full_res = int(np.mean([p.y for p in points]))
                                bbox_np_full = np.array([(p.x, p.y) for p in points], dtype=np.int32)

                                scale_x = proc_w / full_w
                                scale_y = proc_h / full_h
                                qr_center_x_proc_res = int(qr_center_x_full_res * scale_x)
                                qr_center_y_proc_res = int(qr_center_y_full_res * scale_y)
                                qr_bbox_int_draw_proc_res = (bbox_np_full * [scale_x, scale_y]).astype(np.int32).reshape(-1, 1, 2)

                                qr_details_for_yoloqr = {
                                    'center': (qr_center_x_proc_res, qr_center_y_proc_res),
                                    'bbox_np': (bbox_np_full * [scale_x, scale_y]).astype(np.int32)
                                }
                                break
                        except Exception as qr_decode_err:
                            print(f"Warning: Error decoding/processing QR object: {qr_decode_err}")
                            continue
                except Exception as e:
                    print(f"Error during pyzbar decode: {e}")
                    qr_detected_flag_this_frame = False
                    qr_details_for_yoloqr = None
                    qr_bbox_int_draw_proc_res = None

                # --- 2. Depth Processing ---
                depth_resized_proc_mm = cv2.resize(depth_raw_copy, PROCESSING_RESOLUTION, interpolation=cv2.INTER_NEAREST)
                valid_depth_mask = (depth_resized_proc_mm >= MIN_DEPTH_PROCESSING_MM) & (depth_resized_proc_mm <= DEPTH_PROCESSING_LIMIT_MM)
                depth_limited_proc_mm = depth_resized_proc_mm.copy()
                depth_limited_proc_mm[~valid_depth_mask] = 0

                #variable initialization
                obstacle_detected = False
                
                # --- 3. Run YOLO Detections & Association/Tracking ---
                self.yolo_qr_processor.run_detections_and_association(
                    frame_bgr_proc_copy,
                    depth_limited_proc_mm,
                    qr_details_for_yoloqr,
                    obstacle_detected
                )
                target_person_box = self.yolo_qr_processor.get_target_person_box()
                target_distance_m = self.yolo_qr_processor._calculate_distance(qr_details_for_yoloqr, depth_limited_proc_mm)
                yolo_boxes_for_drawing, _ = self.yolo_qr_processor.get_drawing_info()

                # --- 4. Process Depth for Obstacles (REMOVED) ---
                # self.obstacle_avoidance.process_depth_and_check_obstacles(...) # <<< REMOVED OA CALL >>>
                # obstacle_detected = self.obstacle_avoidance.get_obstacle_status() # <<< REMOVED OA RESULT >>>
                # depth_vis, masked_depth_vis = self.obstacle_avoidance.get_depth_visualizations() # <<< REMOVED OA RESULT >>>

                # --- 4b. Generate Depth Visualizations Manually ---
                depth_vis, masked_depth_vis = self._create_depth_visualizations(
                    depth_limited_proc_mm,
                    target_distance_m,  # Pass the QR code distance
                    qr_details_for_yoloqr['center'][0] if qr_details_for_yoloqr else None  # Pass the QR code center x-coordinate
                )

                # Check for obstacles in the masked depth map after masking
                if masked_depth_vis is not None and np.count_nonzero(masked_depth_vis) > 200:  # Check for non-black pixels
                    obstacle_detected = True

                # --- 5. Store Results Safely ---
                with results_lock:
                    latest_results["target_person_box"] = target_person_box
                    latest_results["qr_detected_flag"] = qr_detected_flag_this_frame
                    latest_results["qr_center_x"] = qr_details_for_yoloqr['center'][0] if qr_details_for_yoloqr else None
                    latest_results["target_distance_m"] = target_distance_m
                    latest_results["yolo_boxes_for_drawing"] = copy.copy(yolo_boxes_for_drawing)
                    latest_results["qr_bbox_int_draw"] = qr_bbox_int_draw_proc_res
                    latest_results["vis_frame"] = frame_bgr_proc_copy
                    latest_results["depth_vis"] = depth_vis
                    latest_results["masked_depth_vis"] = masked_depth_vis
                    latest_results["timestamp"] = current_frame_time
                    latest_results["obstacle_detected"] = obstacle_detected  # Store obstacle status

            except Exception as e:
                print(f"!!! Unhandled Error in {self.name}: {e}")
                traceback.print_exc()
                time.sleep(0.5)

        print(f"{self.name} Stopped")


# --- Main Execution ---
if __name__ == "__main__":
    print("Initializing System...")
    print(f"Processing Resolution: {PROCESSING_WIDTH}x{PROCESSING_HEIGHT}")

    # Initialize Kinect v2
    try:
        fn = Freenect2()
        if fn.enumerateDevices() == 0: raise RuntimeError("No Kinect v2 device found!")
        serial = fn.getDeviceSerialNumber(0)
        device = fn.openDevice(serial)
        listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
        device.setColorFrameListener(listener)
        device.setIrAndDepthFrameListener(listener)
        print(f"Kinect device opened (Serial: {serial}).")
    except Exception as e:
        print(f"ERROR: Failed to initialize Freenect2 or open device: {e}")
        exit(1)

    # Initialize Obstacle Avoidance module (REMOVED)
    # try:
    #     obstacle_avoidance = OA(obstacle_min_dist_m=OBSTACLE_MIN_DISTANCE_M)
    #     print("Obstacle Avoidance module initialized.")
    # except Exception as e:
    #     print(f"ERROR: Failed to initialize Obstacle Avoidance: {e}")
    #     if 'device' in locals() and device: device.close()
    #     exit(1)

    # Initialize YoloQR module
    try:
        pytorch_model_path = 'yolov8n.pt'
        print(f"Attempting to initialize YoloQR with model: {pytorch_model_path}")
        yolo_qr_processor = YoloQR(
            yolo_model_path=pytorch_model_path,
            yolo_img_size=YOLO_IMG_SIZE,
            conf_threshold= CONF_THRESHOLD,
            association_radius_px = QR_FOLLOW_TOLERANCE,
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize YoloQR: {e}")
        traceback.print_exc() # Print full error for YoloQR init
        if 'device' in locals() and device: device.close()
        exit(1)

    # Initialize Motors
    try:
        robot_motors = motors()
        print("Motors interface initialized.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Motors: {e}")
        if 'device' in locals() and device: device.close()
        exit(1)


    # --- Target QR Code Registration (Unchanged) ---
    target_qr_code = None
    print("\n--- QR Code Registration ---")
    print("Please hold the target QR code clearly in front of the camera.")
    print("Press 'q' to abort.")

    registration_window_name = "Register Target QR Code"
    cv2.namedWindow(registration_window_name)
    try:
        device.start()
        print("Kinect stream started for registration.")
        time.sleep(1.0)

        while target_qr_code is None and not stop_event.is_set():
            frames_reg = listener.waitForNewFrame()
            color_frame_reg = frames_reg[FrameType.Color]
            color_data_rgba_reg = color_frame_reg.asarray(np.uint8)
            color_data_rgba_reg = np.ascontiguousarray(color_data_rgba_reg)
            frame_bgr_reg = cv2.cvtColor(color_data_rgba_reg, cv2.COLOR_RGBA2RGB)

            qr_data = None
            qr_bbox_draw = None
            decoded_objects = pyzbar.decode(frame_bgr_reg)
            for obj in decoded_objects:
                try:
                    data = obj.data.decode('utf-8')
                    points = obj.polygon
                    if data:
                        target_qr_code = data
                        bbox_np = np.array([(p.x, p.y) for p in points], dtype=np.int32)
                        qr_bbox_draw = bbox_np.reshape(-1, 1, 2)
                        print(f"\nTarget QR code candidate found: '{target_qr_code}'")
                        break
                except Exception as e:
                    print(f"Error decoding QR data during registration: {e}")
                    continue

            display_frame_reg = frame_bgr_reg.copy()
            if target_qr_code is not None:
                print(f"Target QR code registered: '{target_qr_code}'")
                if qr_bbox_draw is not None:
                     cv2.polylines(display_frame_reg, [qr_bbox_draw], isClosed=True, color=(0, 255, 0), thickness=4)
                     cv2.putText(display_frame_reg, f"Registered: {target_qr_code}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
                cv2.imshow(registration_window_name, display_frame_reg)
                cv2.waitKey(2000)
                listener.release(frames_reg)
                break

            small_reg = cv2.resize(display_frame_reg, (960, 540))
            cv2.putText(small_reg, "Show QR Code, Press 'q' to cancel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow(registration_window_name, small_reg)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQR code registration aborted by user.")
                target_qr_code = None
                stop_event.set()
                break

            listener.release(frames_reg)

    except Exception as reg_err:
         print(f"ERROR during QR registration: {reg_err}")
         target_qr_code = None
         stop_event.set()
    finally:
         if cv2.getWindowProperty(registration_window_name, cv2.WND_PROP_VISIBLE) >= 1:
              cv2.destroyWindow(registration_window_name)

    if target_qr_code is None:
        print("ERROR: No target QR code was registered. Exiting.")
        if 'device' in locals() and device:
             try: device.stop(); device.close()
             except Exception: pass
        if 'robot_motors' in locals(): robot_motors.stop_motors()
        exit(1)
    # --- END OF REGISTRATION SECTION ---

    # --- Start Background Threads ---
    print("Starting background processing threads...")
    acquisition_thread = FrameAcquisitionThread(listener)
    # <<< Pass only required args to ProcessingThread >>>
    processing_thread = ProcessingThread(yolo_qr_processor, target_qr_code)
    acquisition_thread.start()
    processing_thread.start()

    # --- PID Controller Parameters ---
    KP_DISTANCE = 0.7  # Proportional gain for distance
    KP_ANGLE = 1      # Proportional gain for angle
    KI = 0.0           # Integral gain (set to 0 if not needed)
    KD = 0.1           # Derivative gain

    pid_integral_angle = 0
    pid_last_error_angle = 0
    pid_integral_distance = 0
    pid_last_error_distance = 0   

    # Function to calculate base speed based on distance
    def calculate_base_speed(distance):
        """
        Calculate the base speed of the robot based on the distance to the target.
        - At distances greater than (TARGET_STOP_DISTANCE_M + 1 meter), the speed is capped at 255.
        - At distances less than TARGET_STOP_DISTANCE_M, the speed ramps down to 0.
        """
        max_distance = TARGET_STOP_DISTANCE_M + 0.3 # 1 meter beyond the stop distance
        if distance > max_distance:
            return 255
        elif distance < TARGET_STOP_DISTANCE_M:
            return 0
        else:
            # Linear interpolation between TARGET_STOP_DISTANCE_M and max_distance
            return int(255 * (distance - TARGET_STOP_DISTANCE_M) / (max_distance - TARGET_STOP_DISTANCE_M))

    # --- Main Navigation Loop ---
    print("\n--- Starting Navigation ---")
    print(f"Following QR Code: '{target_qr_code}'")
    print(f"Target Stop Distance: {TARGET_STOP_DISTANCE_M}m")
    print("Press 'q' in the 'Navigation View' window to quit.")

    last_result_time = 0
    current_action = "Initializing"
    frame_center_x = PROCESSING_WIDTH // 2

    try:
        while not stop_event.is_set():
            # --- Get Latest Results Safely ---
            local_results = {}
            new_results_available = False
            try:
                with results_lock:
                    if latest_results["timestamp"] > last_result_time:
                        local_results = {k: (v.copy() if isinstance(v, np.ndarray) else copy.copy(v))
                                         for k, v in latest_results.items()}
                        last_result_time = local_results["timestamp"]
                        new_results_available = True
            except Exception as lock_err:
                 print(f"Error accessing latest_results: {lock_err}")
                 time.sleep(0.1)
                 continue

            # --- Navigation Logic & Visualization (Only if new results are available) ---
            if new_results_available:
                try:
                    # Extract results
                    qr_detected = local_results.get("qr_detected_flag", False)
                    target_dist = local_results.get("target_distance_m", float('inf'))
                    qr_center_x = local_results.get("qr_center_x")
                    vis_frame_proc = local_results.get("vis_frame")
                    obstacle_detected = local_results.get("obstacle_detected", False)

                    if vis_frame_proc is None:
                        time.sleep(0.01)
                        continue

                    # --- Decision Logic with PID ---
                    next_action = "Idle"

                    if obstacle_detected:
                        next_action = "Obstacle Detected Stop"
                        robot_motors.stop_motors()
                    elif qr_detected:  # QR detected takes precedence
                        if target_dist == float('inf') or target_dist is None or np.isnan(target_dist):
                            next_action = "Invalid Distance Stop"
                            print("DEBUG: Stopping due to invalid distance.")
                            robot_motors.stop_motors()
                        elif target_dist <= TARGET_STOP_DISTANCE_M:
                            next_action = "Target Reached Stop"
                            robot_motors.stop_motors()
                        elif qr_center_x is not None:
                            # PID logic for angle adjustment
                            error_angle = frame_center_x - qr_center_x
                            pid_integral_angle += error_angle
                            pid_derivative_angle = error_angle - pid_last_error_angle
                            pid_last_error_angle = error_angle

                            pid_output_angle = KP_ANGLE * error_angle + KI * pid_integral_angle + KD * pid_derivative_angle

                            # Calculate base speed based on distance
                            base_speed = calculate_base_speed(target_dist)

                            # Adjust motor speeds based on both angle and distance
                            right_motor_speed = max(0, min(255, base_speed - pid_output_angle))
                            left_motor_speed = max(0, min(255, base_speed + pid_output_angle))

                            robot_motors.set_motor_speeds(left_motor_speed, right_motor_speed)
                            next_action = f"Adjusting with PID (L: {left_motor_speed}, R: {right_motor_speed}, Base: {base_speed})"
                        else:
                            next_action = "QR Center Error Stop"
                            robot_motors.stop_motors()
                    else:
                        next_action = "Target Lost Stop"
                        robot_motors.stop_motors()

                    # Print action only if it changed
                    if next_action != current_action:
                        print(f"Action: {next_action}")
                        current_action = next_action

                    # --- Visualization ---
                    vis_frame = vis_frame_proc.copy()

                    # Draw QR results
                    qr_bbox_int_draw = local_results.get("qr_bbox_int_draw")
                    # <<< Get the target distance calculated by YoloQR >>>
                    target_dist = local_results.get("target_distance_m", float('inf'))

                    if qr_detected and qr_bbox_int_draw is not None:
                        try:
                            points = np.array(qr_bbox_int_draw, dtype=np.int32).reshape(-1, 1, 2)
                            cv2.polylines(vis_frame, [points], isClosed=True, color=(0, 255, 255), thickness=2)
                            if qr_center_x is not None:
                                qr_center_y_draw = int(np.mean(points[:, 0, 1]))
                                cv2.circle(vis_frame, (qr_center_x, qr_center_y_draw), 5, (0, 0, 255), -1)

                                # <<< RESTORE THIS LINE to show distance next to QR >>>
                                dist_text = f"QR: {target_dist:.2f}m" if target_dist != float('inf') else "QR: Dist N/A"
                                cv2.putText(vis_frame, dist_text, (qr_center_x + 10, qr_center_y_draw), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                # <<< END RESTORED LINE >>>

                                # Optional: Keep the simple "QR" text as well if desired, or remove it
                                # cv2.putText(vis_frame, "QR", (qr_center_x + 10, qr_center_y_draw + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) # Example: Draw below distance

                        except Exception as draw_err: print(f"Warning: Could not draw QR bbox: {draw_err}")

                    # Draw YOLO results (NO CHANGE NEEDED HERE)
                    # This loop uses the labels generated by YoloQR, which already have the distance ONLY for the target.
                    for item in local_results.get("yolo_boxes_for_drawing", []):
                        if item['is_target']:
                            x1, y1, x2, y2 = item['box']
                            label = item['label'] # Label from YoloQR
                            color = (0, 255, 0)
                            thickness = 3
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
                            cv2.putText(vis_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Draw Obstacle Warning
                    if obstacle_detected:
                        cv2.putText(vis_frame, "OBSTACLE!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                    # Draw current action status
                    cv2.putText(vis_frame, f"Action: {current_action}", (10, PROCESSING_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    # Display the main navigation view
                    cv2.imshow("Navigation View", vis_frame)

                    # Display Depth Maps (Still generated and displayed)
                    depth_vis = local_results.get("depth_vis")
                    masked_depth_vis = local_results.get("masked_depth_vis")
                    if SHOW_DEPTH_MAP and depth_vis is not None:
                         depth_display_small = cv2.resize(depth_vis, (480, 360))
                         cv2.imshow("Depth Map ", depth_display_small)
                    if SHOW_MASKED_DEPTH_MAP and masked_depth_vis is not None:
                         masked_depth_display_small = cv2.resize(masked_depth_vis, (480, 360))
                         cv2.imshow("Masked Depth Map (Target Excluded)", masked_depth_display_small)

                except Exception as loop_err:
                    print(f"!!! Error in main loop processing/visualization: {loop_err}")
                    traceback.print_exc()
                    time.sleep(0.1)

            else:
                time.sleep(0.005)

            # --- Check for Quit Key ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit key pressed. Stopping...")
                stop_event.set()
                break

    except KeyboardInterrupt:
        print("Ctrl+C detected. Stopping...")
        stop_event.set()
    except Exception as main_err:
        print(f"!!! Unhandled Error in Main Execution: {main_err}")
        traceback.print_exc()
        stop_event.set()

    finally:
        # --- Cleanup ---
        print("Initiating cleanup...")
        if not stop_event.is_set():
            stop_event.set()

        print("Waiting for threads to join...")
        if 'acquisition_thread' in locals() and acquisition_thread.is_alive():
            print("Joining Acquisition Thread...")
            acquisition_thread.join(timeout=2.0)
        if 'processing_thread' in locals() and processing_thread.is_alive():
            print("Joining Processing Thread...")
            processing_thread.join(timeout=3.0)
        print("Threads joined or timed out.")

        if 'robot_motors' in locals():
            print("Stopping motors...")
            robot_motors.stop_motors()
            if hasattr(robot_motors, 'cleanup'):
                 robot_motors.cleanup()

        if 'device' in locals() and device:
            try:
                print("Stopping Kinect device...")
                device.stop()
                print("Closing Kinect device...")
                device.close()
                print("Kinect device stopped and closed.")
            except Exception as e:
                print(f"Error stopping/closing Kinect: {e}")

        print("Closing OpenCV windows...")
        cv2.destroyAllWindows()
        time.sleep(0.5)
        print("Program finished.")

