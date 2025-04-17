# /home/aleeya/FYP/fyp2/FYP/FYP.py
import cv2
import numpy as np
from pylibfreenect2 import Freenect2, FrameType, SyncMultiFrameListener
import threading
# from ultralytics import YOLO # YOLO is used within YoloQR
import time
import copy
from pyzbar import pyzbar # For initial QR registration

# Import your components
# Make sure these components are adapted to potentially receive
# frames/depth maps at the new PROCESSING_RESOLUTION
from components.kinectv2_OA import OA
from components.kinectv2_yoloqr import YoloQR
from components.Kinectv2_motors import motors # Assuming this is your motor control

# --- Constants and Initializations ---
# <<< OPTIMIZATION: Define a standard processing resolution >>>
# Lower resolution significantly speeds up YOLO and other processing.
# Adjust if needed based on accuracy requirements.
PROCESSING_WIDTH = 640
PROCESSING_HEIGHT = 480
PROCESSING_RESOLUTION = (PROCESSING_WIDTH, PROCESSING_HEIGHT)

# <<< OPTIMIZATION: YOLO Input Size - Should ideally match or be compatible with PROCESSING_WIDTH >>>
# Ensure YoloQR uses an appropriate imgsz, e.g., 640
YOLO_IMG_SIZE = 640

QR_FOLLOW_TOLERANCE = 50 # Pixels, relative to PROCESSING_WIDTH
TARGET_STOP_DISTANCE_M = 0.6
OBSTACLE_MIN_DISTANCE_M = 0.8
# Increased depth limit for processing (adjust as needed)
DEPTH_PROCESSING_LIMIT_MM = 4000.0 # 4 meters
MIN_DEPTH_PROCESSING_MM = 200.0 # 0.2 meters (ignore too close readings)

# --- Shared Data Structures ---
# Use locks for thread-safe access
frame_lock = threading.Lock()
latest_frames = {
    "color_proc": None, # Stores BGR frame resized to PROCESSING_RESOLUTION
    "depth_raw": None,  # Stores raw float32 depth frame (512x424)
    "timestamp": 0
}

results_lock = threading.Lock()
latest_results = {
    "target_person_box": None, # Coords relative to PROCESSING_RESOLUTION
    "qr_detected_flag": False,
    "qr_center_x": None,       # X-coord relative to PROCESSING_RESOLUTION
    "target_distance_m": float('inf'),
    "yolo_boxes_for_drawing": [], # Coords relative to PROCESSING_RESOLUTION
    "qr_bbox_int_draw": None,     # Coords relative to PROCESSING_RESOLUTION
    "obstacle_detected": False,
    "vis_frame": None,         # Stores the PROC_RES frame used for results
    "depth_vis": None,         # Visualization of processed depth
    "masked_depth_vis": None,  # Visualization of masked depth
    "timestamp": 0
}

# --- Thread Control ---
stop_event = threading.Event()

# --- Frame Acquisition Thread ---
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
                # Ensure contiguous array for safety after asarray/cvtColor
                color_data_rgba = np.ascontiguousarray(color_data_rgba)
                # Convert RGBA to BGR
                frame_bgr_full = cv2.cvtColor(color_data_rgba, cv2.COLOR_RGBA2BGR)
                # <<< OPTIMIZATION: Resize color frame ONCE to processing resolution >>>
                frame_bgr_proc = cv2.resize(frame_bgr_full, PROCESSING_RESOLUTION, interpolation=cv2.INTER_LINEAR) # Use INTER_LINEAR for color

                # --- Get Raw Depth Data ---
                depth_data_mm_raw = depth_frame.asarray(np.float32)
                # Ensure contiguous array
                depth_data_mm_raw = np.ascontiguousarray(depth_data_mm_raw)

                current_time = time.time()
                # --- Update Shared Frames ---
                with frame_lock:
                    latest_frames["color_proc"] = frame_bgr_proc # Store processed BGR
                    latest_frames["depth_raw"] = depth_data_mm_raw # Store raw depth
                    latest_frames["timestamp"] = current_time

                self.listener.release(frames)
                # time.sleep(0.01) # Optional small sleep if CPU is still too high

            except Exception as e:
                print(f"Error in {self.name}: {e}")
                time.sleep(0.5) # Avoid busy-looping on error

        print(f"{self.name} Stopped")

# --- Processing Thread ---
class ProcessingThread(threading.Thread):
    def __init__(self, yolo_qr_processor, obstacle_avoidance, target_qr_code):
        super().__init__(daemon=True)
        # Ensure the YoloQR instance is capable of segmentation
        if not hasattr(yolo_qr_processor, 'get_target_person_mask'):
             raise AttributeError("The provided YoloQR processor does not have the 'get_target_person_mask' method. Ensure it's the updated version.")
        self.yolo_qr_processor = yolo_qr_processor
        self.obstacle_avoidance = obstacle_avoidance
        self.target_qr_code = target_qr_code
        self.last_processed_frame_time = 0
        self.name = "ProcessingThread"
        print(f"{self.name} Initialized")

    def run(self):
        print(f"{self.name} Started")
        while not stop_event.is_set():
            frame_bgr_proc_copy = None
            depth_raw_copy = None
            current_frame_time = 0

            # --- Get Latest Frames Safely ---
            with frame_lock:
                if latest_frames["timestamp"] > self.last_processed_frame_time:
                    # Use copy() for numpy arrays (shallow copy, efficient)
                    if latest_frames["color_proc"] is not None:
                        frame_bgr_proc_copy = latest_frames["color_proc"].copy()
                    if latest_frames["depth_raw"] is not None:
                        depth_raw_copy = latest_frames["depth_raw"].copy()
                    current_frame_time = latest_frames["timestamp"]
                    self.last_processed_frame_time = current_frame_time
                else:
                    time.sleep(0.01)
                    continue

            if frame_bgr_proc_copy is None or depth_raw_copy is None:
                continue

            try:
                # --- Perform Processing on PROC_RESOLUTION data ---

                # <<< OPTIMIZATION: Resize depth ONCE to processing resolution >>>
                # Resize the raw depth map (512x424) to match the processed color frame (e.g., 640x480)
                depth_resized_proc_mm = cv2.resize(
                    depth_raw_copy,
                    PROCESSING_RESOLUTION,
                    interpolation=cv2.INTER_NEAREST
                )

                # --- Apply Depth Limits ---
                # Create a mask for valid depth ranges on the *resized* depth map
                valid_depth_mask = (depth_resized_proc_mm >= MIN_DEPTH_PROCESSING_MM) & \
                                   (depth_resized_proc_mm <= DEPTH_PROCESSING_LIMIT_MM)
                # Create a limited depth map (set invalid depths to 0 or NaN)
                # Using 0 is common for compatibility with functions expecting non-negative depth
                depth_limited_proc_mm = depth_resized_proc_mm.copy()
                depth_limited_proc_mm[~valid_depth_mask] = 0 # Set invalid depths to 0

                # --- Run Detections (YOLO + QR) ---
                # Pass the processed color frame and the *limited, resized* depth map
                # Ensure YoloQR expects/uses imgsz compatible with PROCESSING_RESOLUTION
                self.yolo_qr_processor.run_detections(
                    frame_bgr_proc_copy, # Already PROC_RESOLUTION
                    self.target_qr_code,
                    depth_limited_proc_mm
                )
                # <<< CHANGE: Get the mask INSTEAD of just the box for OA >>>
                target_person_mask = self.yolo_qr_processor.get_target_person_mask() # Get the boolean mask
                # --- Keep getting the box for potential drawing/other logic ---
                target_person_box = self.yolo_qr_processor.get_target_person_box()
                # --- Get other details ---
                qr_detected_flag, qr_center_x, target_distance_m = self.yolo_qr_processor.get_qr_details()
                yolo_boxes_for_drawing, qr_bbox_int_draw = self.yolo_qr_processor.get_drawing_info()

                # --- Process Depth for Obstacles ---
                # <<< CHANGE: Pass the target_person_mask to OA >>>
                self.obstacle_avoidance.process_depth_and_check_obstacles(
                    depth_limited_proc_mm, # Resized and limited depth
                    target_person_mask,    # <<< Pass the mask here
                    PROCESSING_RESOLUTION  # Pass the resolution for context
                )
                obstacle_detected = self.obstacle_avoidance.get_obstacle_status()
                depth_vis, masked_depth_vis = self.obstacle_avoidance.get_depth_visualizations()

                # --- Store Results Safely ---
                with results_lock:
                    latest_results["target_person_box"] = target_person_box # Still store box if needed elsewhere
                    latest_results["qr_detected_flag"] = qr_detected_flag
                    latest_results["qr_center_x"] = qr_center_x
                    latest_results["target_distance_m"] = target_distance_m
                    latest_results["yolo_boxes_for_drawing"] = copy.copy(yolo_boxes_for_drawing)
                    latest_results["qr_bbox_int_draw"] = qr_bbox_int_draw
                    latest_results["obstacle_detected"] = obstacle_detected
                    latest_results["vis_frame"] = frame_bgr_proc_copy
                    latest_results["depth_vis"] = depth_vis
                    latest_results["masked_depth_vis"] = masked_depth_vis
                    latest_results["timestamp"] = current_frame_time

            except Exception as e:
                print(f"Error in {self.name}: {e}")
                import traceback
                traceback.print_exc() # Print detailed traceback for debugging
                time.sleep(0.5) # Avoid busy-looping on error

        print(f"{self.name} Stopped")

# ... (rest of the fypnew.py script remains the same) ...

# --- Main Execution ---
if __name__ == "__main__":
    print("Initializing System...")
    print(f"Processing Resolution: {PROCESSING_WIDTH}x{PROCESSING_HEIGHT}")

    # Initialize Kinect v2
    try:
        fn = Freenect2()
        if fn.enumerateDevices() == 0:
            print("ERROR: No Kinect v2 device found!")
            exit(1)
        serial = fn.getDeviceSerialNumber(0)
        device = fn.openDevice(serial)
        listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
        device.setColorFrameListener(listener)
        device.setIrAndDepthFrameListener(listener)
        print(f"Kinect device opened (Serial: {serial}).")
    except Exception as e:
        print(f"ERROR: Failed to initialize Freenect2 or open device: {e}")
        exit(1)

    # Initialize Obstacle Avoidance module
    # Ensure OA uses OBSTACLE_MIN_DISTANCE_M correctly
    obstacle_avoidance = OA(obstacle_min_dist_m=OBSTACLE_MIN_DISTANCE_M)
    print("Obstacle Avoidance module initialized.")

    # Initialize YoloQR module
    try:
        # Ensure YoloQR is initialized with settings compatible with PROC_RESOLUTION
        # e.g., pass YOLO_IMG_SIZE if needed
        yolo_qr_processor = YoloQR(yolo_img_size=YOLO_IMG_SIZE) # Example: Pass size if needed
        print("YoloQR module initialized.")
    except Exception as e:
        print(f"ERROR: Failed to initialize YoloQR: {e}")
        if 'device' in locals() and device:
            try: device.stop(); device.close()
            except Exception: pass
        exit(1)

    # Initialize Motors
    try:
        robot_motors = motors() # Ensure this class/module is correctly imported and initialized
        print("Motors interface initialized.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Motors: {e}")
        if 'device' in locals() and device:
            try: device.stop(); device.close()
            except Exception: pass
        exit(1)


    # --- Target QR Code Registration (Using pyzbar on full frame initially for ease) ---
    target_qr_code = None
    print("\n--- QR Code Registration ---")
    print("Please hold the target QR code clearly in front of the camera.")
    print("Press 'q' to abort.")

    registration_window_name = "Register Target QR Code"
    cv2.namedWindow(registration_window_name)
    device.start() # Start stream for registration
    print("Kinect stream started for registration.")
    time.sleep(1.0) # Allow stream to stabilize

    while target_qr_code is None and not stop_event.is_set():
        frames_reg = listener.waitForNewFrame()
        color_frame_reg = frames_reg[FrameType.Color]
        color_data_rgba_reg = color_frame_reg.asarray(np.uint8)
        color_data_rgba_reg = np.ascontiguousarray(color_data_rgba_reg)
        frame_bgr_reg = cv2.cvtColor(color_data_rgba_reg, cv2.COLOR_RGBA2BGR) # Full res BGR

        # --- Use pyzbar for detection on the full frame for better chance ---
        qr_data = None
        qr_bbox_draw = None
        decoded_objects = pyzbar.decode(frame_bgr_reg)
        for obj in decoded_objects:
            try:
                data = obj.data.decode('utf-8')
                points = obj.polygon
                if data: # Found a QR code
                    target_qr_code = data # Register the first one found
                    # Prepare bbox for drawing (relative to full frame)
                    bbox_np = np.array([(p.x, p.y) for p in points], dtype=np.int32)
                    qr_bbox_draw = bbox_np.reshape(-1, 1, 2)
                    print(f"\nTarget QR code candidate found: '{target_qr_code}'")
                    break # Stop after finding the first QR code
            except Exception as e:
                print(f"Error decoding QR data: {e}")
                continue

        # --- Display and confirmation ---
        display_frame_reg = frame_bgr_reg.copy()
        if target_qr_code is not None:
            print(f"Target QR code registered: '{target_qr_code}'")
            if qr_bbox_draw is not None:
                 cv2.polylines(display_frame_reg, [qr_bbox_draw], isClosed=True, color=(0, 255, 0), thickness=4)
                 cv2.putText(display_frame_reg, f"Registered: {target_qr_code}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

            # Display full res confirmation briefly
            cv2.imshow(registration_window_name, display_frame_reg)
            cv2.waitKey(2000) # Show registered code for 2 seconds
            cv2.destroyWindow(registration_window_name)
            listener.release(frames_reg)
            break # Exit registration loop

        # --- Display if no QR code found yet (resized for preview) ---
        small_reg = cv2.resize(display_frame_reg, (960, 540)) # Preview size
        cv2.putText(small_reg, "Show QR Code, Press 'q' to cancel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(registration_window_name, small_reg)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQR code registration aborted by user.")
            target_qr_code = None # Ensure it's None
            stop_event.set() # Signal exit
            break # Exit loop

        listener.release(frames_reg)


    if target_qr_code is None:
        print("ERROR: No target QR code was registered. Exiting.")
        if 'device' in locals() and device:
            try: device.stop(); device.close()
            except Exception: pass
        if 'robot_motors' in locals():
             robot_motors.stop_motors()
        exit(1)
    # --- END OF REGISTRATION SECTION ---

    # --- Start Background Threads ---
    print("Starting background processing threads...")
    acquisition_thread = FrameAcquisitionThread(listener)
    # Pass the registered target_qr_code to the ProcessingThread
    processing_thread = ProcessingThread(yolo_qr_processor, obstacle_avoidance, target_qr_code)
    acquisition_thread.start()
    processing_thread.start()

    # --- Main Navigation Loop (Uses latest results) ---
    print("\n--- Starting Navigation ---")
    print(f"Following QR Code: '{target_qr_code}'")
    print(f"Target Stop Distance: {TARGET_STOP_DISTANCE_M}m")
    print(f"Obstacle Min Distance: {obstacle_avoidance.obstacle_min_distance_mm}m")
    print("Press 'q' in the 'Navigation View' window to quit.")

    last_result_time = 0
    current_action = "Initializing" # Start with an initial state
    frame_center_x = PROCESSING_WIDTH // 2 # Center relative to processing resolution

    try:
        while not stop_event.is_set():
            # --- Get Latest Results Safely ---
            local_results = {}
            new_results_available = False
            with results_lock:
                if latest_results["timestamp"] > last_result_time:
                    # Use copy() for numpy arrays, dicts/lists containing numpy arrays might need care
                    # If components modify results in place, deepcopy might be safer but slower
                    local_results = {k: (v.copy() if isinstance(v, np.ndarray) else copy.copy(v))
                                     for k, v in latest_results.items()}
                    last_result_time = local_results["timestamp"]
                    new_results_available = True

            # --- Navigation Logic (Only if new results are available) ---
            if new_results_available:
                # Extract results for easier access
                qr_detected = local_results["qr_detected_flag"]
                obstacle_detected = local_results["obstacle_detected"]
                target_dist = local_results["target_distance_m"]
                qr_center_x = local_results["qr_center_x"] # Relative to PROC_WIDTH, can be None
                vis_frame_proc = local_results["vis_frame"] # The PROC_RES frame used for processing

                if vis_frame_proc is None: # Skip if processing hasn't produced a frame yet
                    time.sleep(0.01) # Wait briefly
                    continue

                # --- Decision Logic ---
                next_action = "Idle" # Default state

                if obstacle_detected:
                    next_action = "Obstacle Stop"
                    if current_action != next_action: print(f"Decision: Obstacle detected! Stopping.")
                    robot_motors.stop_motors()
                elif qr_detected:
                    if target_dist == float('inf') or target_dist is None or np.isnan(target_dist):
                        next_action = "Invalid Distance Stop"
                        if current_action != next_action: print("Decision: Target QR detected, but distance invalid. Stopping.")
                        robot_motors.stop_motors()
                    elif target_dist <= TARGET_STOP_DISTANCE_M:
                        next_action = "Target Reached Stop"
                        if current_action != next_action: print(f"Decision: Target ({target_dist:.2f}m) is close. Stopping.")
                        robot_motors.stop_motors()
                    elif qr_center_x is None:
                        next_action = "QR Center Error Stop"
                        if current_action != next_action: print("Decision: Target QR detected, but center invalid. Stopping.")
                        robot_motors.stop_motors()
                    elif qr_center_x < frame_center_x - QR_FOLLOW_TOLERANCE:
                        next_action = "Turning Left"
                        if current_action != next_action: print("Decision: Target left. Turning left.")
                        robot_motors.turn_left()
                    elif qr_center_x > frame_center_x + QR_FOLLOW_TOLERANCE:
                        next_action = "Turning Right"
                        if current_action != next_action: print("Decision: Target right. Turning right.")
                        robot_motors.turn_right()
                    else: # Centered, no obstacle, not too close, valid distance
                        next_action = "Moving Forward"
                        if current_action != next_action: print("Decision: Target centered. Moving forward.")
                        robot_motors.move_forward()
                else: # Target QR not detected
                    next_action = "Target Lost Stop"
                    if current_action != next_action: print("Decision: Target QR code not detected. Stopping.")
                    robot_motors.stop_motors()

                # Update the current action state
                current_action = next_action

                # --- Visualization (Draw on the PROC_RES frame) ---
                vis_frame = vis_frame_proc.copy() # Draw on a copy

                # Draw QR results (coords are already relative to PROC_RES)
                qr_bbox_int_draw = local_results["qr_bbox_int_draw"]
                if qr_detected and qr_bbox_int_draw is not None:
                    # Assuming qr_bbox_int_draw is [(x1,y1), (x2,y2), ...] or similar
                    try:
                        points = np.array(qr_bbox_int_draw, dtype=np.int32).reshape(-1, 1, 2)
                        cv2.polylines(vis_frame, [points], isClosed=True, color=(0, 255, 255), thickness=2)
                        if qr_center_x is not None:
                            qr_center_y_draw = int(np.mean(points[:, 0, 1])) # Estimate Y center
                            cv2.circle(vis_frame, (qr_center_x, qr_center_y_draw), 5, (0, 0, 255), -1)
                            dist_text = f"Target QR: {target_dist:.2f}m" if target_dist != float('inf') else "Target QR: Dist Invalid"
                            cv2.putText(vis_frame, dist_text, (qr_center_x + 10, qr_center_y_draw),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    except Exception as draw_err:
                         print(f"Warning: Could not draw QR bbox: {draw_err}, bbox data: {qr_bbox_int_draw}")


                # Draw YOLO results (coords are already relative to PROC_RES)
                for item in local_results["yolo_boxes_for_drawing"]:
                    x1, y1, x2, y2 = item['box']
                    label = item['label']
                    is_target = item['is_target']
                    color = (255, 0, 0) if is_target else (0, 255, 0)
                    thickness = 3 if is_target else 2
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(vis_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw Obstacle Warning
                if obstacle_detected:
                    cv2.putText(vis_frame, "OBSTACLE!", (PROCESSING_WIDTH // 2 - 90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

                # Draw current action status
                cv2.putText(vis_frame, f"Action: {current_action}", (10, PROCESSING_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Display the main navigation view (resized from PROC_RES if needed)
                # display_nav = cv2.resize(vis_frame, (960, 540)) # Optional resize for larger display
                display_nav = vis_frame # Display at processing resolution
                cv2.imshow("Navigation View", display_nav)

                # Display Depth Maps (resized for consistent display size)
                depth_vis = local_results["depth_vis"]
                masked_depth_vis = local_results["masked_depth_vis"]
                if depth_vis is not None:
                     depth_display_small = cv2.resize(depth_vis, (480, 360)) # Smaller display size
                     cv2.imshow("Depth Map (Processed)", depth_display_small)
                if masked_depth_vis is not None:
                     masked_depth_display_small = cv2.resize(masked_depth_vis, (480, 360))
                     cv2.imshow("Masked Depth Map (Obstacles)", masked_depth_display_small)

            else:
                # No new results, yield CPU briefly
                time.sleep(0.005) # Shorter sleep when idle

            # --- Check for Quit Key ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit key pressed. Stopping...")
                stop_event.set() # Signal threads to stop
                break

    finally:
        # --- Cleanup ---
        print("Initiating cleanup...")
        if not stop_event.is_set():
            stop_event.set() # Ensure threads are signaled

        print("Waiting for threads to join...")
        if 'acquisition_thread' in locals() and acquisition_thread.is_alive():
            print("Joining Acquisition Thread...")
            acquisition_thread.join(timeout=2.0)
        if 'processing_thread' in locals() and processing_thread.is_alive():
            print("Joining Processing Thread...")
            processing_thread.join(timeout=3.0) # Allow more time for processing to finish
        print("Threads joined or timed out.")

        if 'robot_motors' in locals():
            print("Stopping motors...")
            robot_motors.stop_motors()

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
        print("Program finished.")
