# /home/aleeya/FYP/fyp2/FYP/FYP.py
import cv2
import numpy as np
from pylibfreenect2 import Freenect2, FrameType, SyncMultiFrameListener
import threading
# from ultralytics import YOLO # YOLO is used within YoloQR, not directly here
import time
import copy
from pyzbar import pyzbar # <-- Import pyzbar here

# Import your components
from components.kinectv2_OA import OA
from components.kinectv2_yoloqr import YoloQR
from components.Kinectv2_motors import motors # Assuming this is your placeholder/actual motor control

# --- Constants and Initializations ---
QR_FOLLOW_TOLERANCE = 50
TARGET_STOP_DISTANCE_M = 0.6
OBSTACLE_MIN_DISTANCE_M = 0.8

# --- Shared Data Structures ---
# Use locks to ensure thread-safe access
frame_lock = threading.Lock()
latest_frames = {
    "color": None, # Will store BGR frame
    "depth_raw": None, # Will store raw float32 depth frame
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
    "obstacle_detected": False,
    "vis_frame": None, # Store the frame used for processing/drawing
    "depth_vis": None,
    "masked_depth_vis": None,
    "timestamp": 0
}

# --- Thread Control ---
stop_event = threading.Event()

# --- Frame Acquisition Thread ---
class FrameAcquisitionThread(threading.Thread):
    def __init__(self, listener):
        super().__init__(daemon=True) # Daemon threads exit when main program exits
        self.listener = listener
        self.name = "FrameAcquisitionThread"
        print(f"{self.name} Initialized")

    def run(self):
        print(f"{self.name} Started")
        while not stop_event.is_set():
            try:
                frames = self.listener.waitForNewFrame()
                color_frame = frames[FrameType.Color]
                depth_frame = frames[FrameType.Depth]

                # Process color frame (RGBA -> BGR) right away
                color_data_rgba = color_frame.asarray(np.uint8)
                # Ensure the array is contiguous (sometimes needed after asarray/cvtColor)
                color_data_rgba = np.ascontiguousarray(color_data_rgba)
                frame_bgr = cv2.cvtColor(color_data_rgba, cv2.COLOR_RGBA2BGR)

                # Get raw depth data (float32, mm)
                depth_data_mm_raw = depth_frame.asarray(np.float32)

                current_time = time.time()
                with frame_lock:
                    latest_frames["color"] = frame_bgr # Store processed BGR
                    latest_frames["depth_raw"] = depth_data_mm_raw
                    latest_frames["timestamp"] = current_time

                self.listener.release(frames)
                # time.sleep(0.01) # Optional small sleep if CPU usage is too high

            except Exception as e:
                print(f"Error in {self.name}: {e}")
                # Decide if the error is critical, maybe set stop_event?
                time.sleep(0.5) # Avoid busy-looping on error

        print(f"{self.name} Stopped")

# --- Processing Thread ---
class ProcessingThread(threading.Thread):
    def __init__(self, yolo_qr_processor, obstacle_avoidance, target_qr_code):
        super().__init__(daemon=True)
        self.yolo_qr_processor = yolo_qr_processor
        self.obstacle_avoidance = obstacle_avoidance
        self.target_qr_code = target_qr_code
        self.last_processed_frame_time = 0
        self.name = "ProcessingThread"
        print(f"{self.name} Initialized")

    def run(self):
        print(f"{self.name} Started")
        # Define the depth limit in millimeters
        # --- INCREASED DEPTH LIMIT ---
        DEPTH_LIMIT_MM = 4000.0 # 4 meters (Increased from 1000.0)

        while not stop_event.is_set():
            frame_bgr_copy = None
            depth_raw_copy = None
            current_frame_time = 0

            # Get the latest frames safely
            with frame_lock:
                if latest_frames["timestamp"] > self.last_processed_frame_time:
                    # Use copy to avoid holding the lock during long processing
                    if latest_frames["color"] is not None:
                        frame_bgr_copy = latest_frames["color"].copy()
                    if latest_frames["depth_raw"] is not None:
                        depth_raw_copy = latest_frames["depth_raw"].copy()
                    current_frame_time = latest_frames["timestamp"]
                    self.last_processed_frame_time = current_frame_time
                else:
                    # No new frame, wait a bit
                    time.sleep(0.01)
                    continue # Go back to check for stop_event/new frame

            if frame_bgr_copy is None or depth_raw_copy is None:
                continue # Skip if frames weren't ready

            try:
                # --- Perform Processing ---
                frame_height, frame_width = frame_bgr_copy.shape[:2]

                # --- Apply Depth Limit ---
                # Create a copy to modify
                depth_raw_limited = depth_raw_copy.copy()
                # Set values beyond the limit to 0 (invalid/ignored)
                # Also set values below a minimum reasonable distance (e.g., 0.2m) to 0
                depth_raw_limited[(depth_raw_limited > DEPTH_LIMIT_MM) | (depth_raw_limited < 200.0)] = 0

                # Resize the *limited* depth map (needed for YoloQR distance calc)
                depth_data_resized_mm_limited = cv2.resize(
                    depth_raw_limited, # Use the limited version
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_NEAREST
                )

                # Run Detections and Target Identification using the limited resized depth
                self.yolo_qr_processor.run_detections(
                    frame_bgr_copy,
                    self.target_qr_code,
                    depth_data_resized_mm_limited # Pass the limited depth
                )
                target_person_box = self.yolo_qr_processor.get_target_person_box()
                qr_detected_flag, qr_center_x, target_distance_m = self.yolo_qr_processor.get_qr_details()
                yolo_boxes_for_drawing, qr_bbox_int_draw = self.yolo_qr_processor.get_drawing_info()

                # Process Depth, Mask Target, and Check Obstacles using the limited raw depth
                self.obstacle_avoidance.process_depth_and_check_obstacles(
                    depth_raw_limited, # Pass the limited raw depth
                    target_person_box,
                    (frame_width, frame_height)
                )
                obstacle_detected = self.obstacle_avoidance.get_obstacle_status()
                depth_vis, masked_depth_vis = self.obstacle_avoidance.get_depth_visualizations()

                # --- Store Results Safely ---
                with results_lock:
                    latest_results["target_person_box"] = target_person_box # Store potentially None value
                    latest_results["qr_detected_flag"] = qr_detected_flag
                    latest_results["qr_center_x"] = qr_center_x
                    latest_results["target_distance_m"] = target_distance_m
                    # Use deepcopy for lists/dicts if they might be modified later by main thread (safer)
                    latest_results["yolo_boxes_for_drawing"] = copy.deepcopy(yolo_boxes_for_drawing)
                    latest_results["qr_bbox_int_draw"] = qr_bbox_int_draw # Can be None
                    latest_results["obstacle_detected"] = obstacle_detected
                    latest_results["vis_frame"] = frame_bgr_copy # Store the frame used for this result set
                    latest_results["depth_vis"] = depth_vis # Can be None
                    latest_results["masked_depth_vis"] = masked_depth_vis # Can be None
                    latest_results["timestamp"] = current_frame_time

            except Exception as e:
                print(f"Error in {self.name}: {e}")
                # Avoid busy-looping on error
                time.sleep(0.5)

        print(f"{self.name} Stopped")


# --- Main Execution ---
if __name__ == "__main__":
    print("Initializing System...")

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
        print("Kinect device opened.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Freenect2 or open device: {e}")
        exit(1)

    # Initialize Obstacle Avoidance module
    obstacle_avoidance = OA(obstacle_min_dist_m=OBSTACLE_MIN_DISTANCE_M)

    # Initialize YoloQR module
    try:
        # Pass the target QR code string here if needed by YoloQR init, otherwise None
        yolo_qr_processor = YoloQR()
    except Exception as e:
        print(f"Failed to initialize YoloQR: {e}")
        if 'device' in locals() and device:
            try:
                device.stop()
                device.close()
            except Exception: pass # Ignore errors during cleanup on failure
        exit(1)

    # Initialize Motors
    robot_motors = motors() # Make sure this class has the methods: move_forward, turn_left, turn_right, stop_motors

    print("System Initialized.")

    # --- Target QR Code Registration (Using pyzbar directly) --- # MODIFIED SECTION
    target_qr_code = None
    print("\n--- QR Code Registration ---")
    print("Please hold the target QR code clearly in front of the camera.")
    print("Press 'q' to abort.")

    registration_window_name = "Register Target QR Code"
    cv2.namedWindow(registration_window_name)
    device.start() # Start stream for registration
    print("Kinect stream started for registration.")

    while target_qr_code is None:
        frames_reg = listener.waitForNewFrame()
        color_frame_reg = frames_reg[FrameType.Color]
        color_data_rgba_reg = color_frame_reg.asarray(np.uint8)
        # Ensure contiguous array
        color_data_rgba_reg = np.ascontiguousarray(color_data_rgba_reg)
        frame_bgr_reg = cv2.cvtColor(color_data_rgba_reg, cv2.COLOR_RGBA2BGR)

        # --- Use pyzbar for detection ---
        qr_data = None
        qr_bbox_draw = None
        decoded_objects = pyzbar.decode(frame_bgr_reg)
        for obj in decoded_objects:
            data = obj.data.decode('utf-8')
            points = obj.polygon
            if data: # Found a QR code
                target_qr_code = data # Register the first one found
                # Prepare bbox for drawing
                bbox_np = np.array([(p.x, p.y) for p in points], dtype=np.int32)
                qr_bbox_draw = bbox_np.reshape(-1, 1, 2) # Reshape for polylines
                break # Stop after finding the first QR code

        # --- Display and confirmation ---
        display_frame_reg = frame_bgr_reg.copy()
        if target_qr_code is not None: # Use the variable set inside the loop
            print(f"\nTarget QR code registered: '{target_qr_code}'")
            if qr_bbox_draw is not None:
                 cv2.polylines(display_frame_reg, [qr_bbox_draw], isClosed=True, color=(0, 255, 0), thickness=4)
                 cv2.putText(display_frame_reg, f"Registered: {target_qr_code}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(registration_window_name, cv2.resize(display_frame_reg, (640, 480)))
            cv2.waitKey(2000) # Show registered code for 2 seconds
            cv2.destroyWindow(registration_window_name)
            listener.release(frames_reg)
            break # Exit registration loop

        # --- Display if no QR code found yet ---
        small_reg = cv2.resize(display_frame_reg, (640, 480))
        cv2.putText(small_reg, "Show QR Code, Press 'q' to cancel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(registration_window_name, small_reg)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQR code registration aborted by user.")
            target_qr_code = None # Ensure it's None
            cv2.destroyWindow(registration_window_name)
            listener.release(frames_reg)
            print("Exiting program as no target QR code was registered.")
            # Cleanup before exiting
            if 'device' in locals() and device:
                try:
                    device.stop()
                    device.close()
                except Exception: pass
            robot_motors.stop_motors()
            exit(0)

        listener.release(frames_reg)
    # --- END OF MODIFIED REGISTRATION SECTION ---

    if target_qr_code is None:
        print("ERROR: Failed to register a target QR code after loop.")
        # Cleanup before exiting
        if 'device' in locals() and device:
            try:
                device.stop()
                device.close()
            except Exception: pass
        robot_motors.stop_motors()
        exit(1)

    # --- Start Background Threads ---
    print("Starting background threads...")
    acquisition_thread = FrameAcquisitionThread(listener)
    # Pass the registered target_qr_code to the ProcessingThread
    processing_thread = ProcessingThread(yolo_qr_processor, obstacle_avoidance, target_qr_code)
    acquisition_thread.start()
    processing_thread.start()

    # --- Main Navigation Loop (Uses latest results) ---
    print("\n--- Starting Navigation ---")
    print(f"Following QR Code: '{target_qr_code}'")
    print(f"Stopping if target closer than {TARGET_STOP_DISTANCE_M}m.")
    print(f"Stopping if obstacle closer than {obstacle_avoidance.obstacle_min_distance_m}m in central region.")
    print("Press 'q' to quit.")

    last_result_time = 0
    # Initialize last_action to ensure it exists before the loop
    current_action = "Initializing" # Start with an initial state

    try:
        while True:
            # --- Get Latest Results Safely ---
            local_results = {}
            new_results_available = False
            with results_lock:
                if latest_results["timestamp"] > last_result_time:
                    # Use deepcopy to ensure main thread works with a consistent snapshot
                    local_results = copy.deepcopy(latest_results)
                    last_result_time = local_results["timestamp"]
                    new_results_available = True

            # --- Navigation Logic (Only if new results are available) ---
            if new_results_available:
                # Extract results for easier access
                qr_detected = local_results["qr_detected_flag"]
                obstacle_detected = local_results["obstacle_detected"]
                target_dist = local_results["target_distance_m"]
                qr_center_x = local_results["qr_center_x"] # Can be None
                vis_frame_base = local_results["vis_frame"] # The frame used for processing

                if vis_frame_base is None: # Skip if processing hasn't produced a frame yet
                    time.sleep(0.02) # Wait briefly
                    continue

                frame_height, frame_width = vis_frame_base.shape[:2]
                frame_center_x = frame_width // 2

                # --- Decision Logic ---
                # Determine the *next* action based on current sensor data
                # The actual motor command and print will happen within the condition block
                next_action = "Idle" # Default state if no other condition met

                if obstacle_detected:
                    next_action = "Obstacle Stop"
                    print(f"Obstacle detected within {obstacle_avoidance.obstacle_min_distance_m}m! Stopping.") # Print always
                    robot_motors.stop_motors() # Command always
                elif qr_detected:
                    if target_dist == float('inf'):
                        next_action = "Invalid Distance Stop"
                        print("Target QR detected, but distance is invalid/out of range. Stopping.") # Print always
                        robot_motors.stop_motors() # Command always
                    elif target_dist <= TARGET_STOP_DISTANCE_M:
                        next_action = "Target Reached Stop"
                        print(f"Target ({target_dist:.2f}m) is close. Stopping.") # Print always
                        robot_motors.stop_motors() # Command always
                    elif qr_center_x is None:
                        next_action = "QR Center Error Stop"
                        print("Target QR detected, but center calculation failed. Stopping.") # Print always
                        robot_motors.stop_motors() # Command always
                    elif qr_center_x < frame_center_x - QR_FOLLOW_TOLERANCE:
                        next_action = "Turning Left"
                        print("Target left. Turning left.") # Print always
                        robot_motors.turn_left() # Command always
                    elif qr_center_x > frame_center_x + QR_FOLLOW_TOLERANCE:
                        next_action = "Turning Right"
                        print("Target right. Turning right.") # Print always
                        robot_motors.turn_right() # Command always
                    else: # Centered, no obstacle, not too close, valid distance
                        next_action = "Moving Forward"
                        print("Target centered. Moving forward.") # Print always
                        robot_motors.move_forward() # Command always
                else: # Target QR not detected
                    next_action = "Target Lost Stop"
                    print("Target QR code not detected. Stopping.") # Print always
                    robot_motors.stop_motors() # Command always

                # Update the current action state *after* determining the action for this frame
                current_action = next_action

                # --- Visualization (using results from the processing thread) ---
                vis_frame = vis_frame_base.copy() # Draw on a copy

                # Draw QR results
                qr_bbox_int_draw = local_results["qr_bbox_int_draw"] # Shape might be (1, 4, 2)
                if qr_detected and qr_bbox_int_draw is not None and qr_bbox_int_draw.shape == (1, 4, 2):
                    points = qr_bbox_int_draw[0].astype(np.int32)
                    cv2.polylines(vis_frame, [points.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 255), thickness=2)
                    if qr_center_x is not None:
                        # Estimate Y center for drawing text (might be slightly off if QR tilted)
                        qr_center_y_draw = int(np.mean(points[:, 1]))
                        cv2.circle(vis_frame, (qr_center_x, qr_center_y_draw), 5, (0, 0, 255), -1)
                        dist_text = f"Target QR: {target_dist:.2f}m" if target_dist != float('inf') else "Target QR: Dist Invalid/OOR"
                        cv2.putText(vis_frame, dist_text, (qr_center_x + 10, qr_center_y_draw),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                elif qr_detected and qr_bbox_int_draw is not None:
                     print(f"Warning: QR bbox has unexpected shape for drawing: {qr_bbox_int_draw.shape}")


                # Draw YOLO results
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
                    cv2.putText(vis_frame, "OBSTACLE!", (frame_width // 2 - 90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

                # Draw current action status
                cv2.putText(vis_frame, f"Action: {current_action}", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Display the main navigation view
                display_small = cv2.resize(vis_frame, (960, 540))
                cv2.imshow("Navigation View", display_small)

                # Display Depth Maps
                depth_vis = local_results["depth_vis"]
                masked_depth_vis = local_results["masked_depth_vis"]
                if depth_vis is not None:
                     depth_display_small = cv2.resize(depth_vis, (640, 360))
                     cv2.imshow("Depth Map (Resized, with Obstacle ROI)", depth_display_small)
                # --- UNCOMMENTED MASKED DEPTH DISPLAY ---
                if masked_depth_vis is not None: # Uncomment to show masked depth
                     masked_depth_display_small = cv2.resize(masked_depth_vis, (640, 360))
                     cv2.imshow("Masked Depth Map (Target Excluded)", masked_depth_display_small)
                # --- END UNCOMMENTED ---

            else:
                # No new results, maybe sleep briefly to yield CPU
                time.sleep(0.01)


            # --- Check for Quit Key ---
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit key pressed. Stopping...")
                stop_event.set() # Signal threads to stop
                break

    finally:
        # --- Cleanup ---
        print("Cleaning up resources...")
        if not stop_event.is_set():
            stop_event.set() # Ensure threads are signaled to stop

        # Wait for threads to finish
        print("Waiting for threads to join...")
        if 'acquisition_thread' in locals() and acquisition_thread.is_alive():
            acquisition_thread.join(timeout=2.0) # Add timeout
        if 'processing_thread' in locals() and processing_thread.is_alive():
            processing_thread.join(timeout=2.0) # Add timeout
        print("Threads joined.")

        robot_motors.stop_motors() # Ensure motors are stopped on exit

        if 'device' in locals() and device:
            try:
                device.stop()
                device.close()
                print("Kinect device stopped and closed.")
            except Exception as e:
                print(f"Error stopping/closing Kinect: {e}")

        cv2.destroyAllWindows()
        print("OpenCV windows closed.")
        print("Program finished.")

