# components/kinectv2_yoloqr.py
import cv2
import numpy as np
from ultralytics import YOLO
from pyzbar import pyzbar # <-- Added for pyzbar QR detection

class YoloQR:
    # COCO class indices (only person needed for target association)
    CLASS_NAMES = {0: 'person'} # Simplified, only need person for target logic

    def __init__(self, yolo_model_path='yolov8n.pt', confidence_threshold=0.5):
        print("Initializing YoloQR Module...")
        # Load YOLOv8 model
        try:
            self.model = YOLO(yolo_model_path)
            # Only detect 'person' class (index 0)
            self.model.overrides['classes'] = [0]
            self.confidence_threshold = confidence_threshold
            print(f"YOLO model '{yolo_model_path}' loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load YOLO model from {yolo_model_path}: {e}")
            raise # Re-raise the exception to stop initialization if model fails

        # QR Code Detector (Using pyzbar now)
        # self.qr_detector = cv2.QRCodeDetector() # <-- Removed cv2.QRCodeDetector
        print("Using pyzbar for QR detection.")

        # --- Internal state to store results ---
        self._reset_results()

    def _reset_results(self):
        """Resets the internal state variables."""
        self.target_person_box = None # (x1, y1, x2, y2) integer coords
        self.qr_detected_flag = False
        self.qr_center_x = None       # Integer x-coordinate
        self.target_distance_m = float('inf')
        self.yolo_boxes_for_drawing = [] # List of dicts: {'box': (x1,y1,x2,y2), 'label': str, 'is_target': bool}
        self.qr_bbox_int_draw = None   # Numpy array shape (1, 4, 2) for drawing compatibility

    def _get_distance_from_depth(self, depth_map_mm, center_x, center_y, box_radius=10): # <-- Increased box_radius
        """
        Safely calculates the median distance in a small area around the center point.
        Args:
            depth_map_mm: The depth map (aligned with color frame, in mm).
            center_x: The x-coordinate of the center.
            center_y: The y-coordinate of the center.
            box_radius: Half the side length of the square region to sample depth.
        Returns:
            Distance in meters, or float('inf') if invalid.
        """
        h, w = depth_map_mm.shape
        # Ensure coordinates are within bounds
        # Use int() conversions here to ensure slicing works correctly
        center_x_int = int(round(center_x))
        center_y_int = int(round(center_y))
        y_min = max(0, center_y_int - box_radius)
        y_max = min(h, center_y_int + box_radius)
        x_min = max(0, center_x_int - box_radius)
        x_max = min(w, center_x_int + box_radius)

        if y_min >= y_max or x_min >= x_max:
            # print("Warning: Invalid depth sampling region.") # Debugging
            return float('inf') # Invalid region

        depth_region = depth_map_mm[y_min:y_max, x_min:x_max]
        valid_depths = depth_region[depth_region > 0] # Filter out 0 values (invalid/too far)

        if valid_depths.size == 0:
            # print("Warning: No valid depth points in sampling region.") # Debugging
            return float('inf') # No valid depth points in the region

        # Use median to be robust against noise/outliers
        median_depth_mm = np.median(valid_depths)
        return median_depth_mm / 1000.0 # Convert mm to meters

    # --- New QR Detection Method using pyzbar ---
    def _detect_qr_pyzbar(self, frame_bgr, target_qr_data_str):
        """Detects QR codes using pyzbar and checks for the target."""
        qr_detected_flag = False
        qr_center_x = None
        qr_center_y = None # Also get Y for distance calculation
        best_match_bbox_raw = None # Store the raw (4, 2) bbox

        try:
            # Decode QR codes using pyzbar
            decoded_objects = pyzbar.decode(frame_bgr)

            for obj in decoded_objects:
                # Extract data and bounding box polygon
                data = obj.data.decode('utf-8')
                points = obj.polygon # List of Point(x,y) objects

                # Convert points to NumPy array format (4, 2)
                bbox_np = np.array([(p.x, p.y) for p in points], dtype=np.float32)

                # Check if this is the target QR code
                if data == target_qr_data_str:
                    qr_detected_flag = True
                    # Calculate center x and y
                    center = np.mean(bbox_np, axis=0)
                    qr_center_x = int(center[0])
                    qr_center_y = int(center[1])
                    # Store the raw bounding box for distance calculation
                    best_match_bbox_raw = bbox_np.astype(int) # Use int for consistency
                    # Optional: Break if you only care about the first match
                    break # Assuming only one target QR code exists/is needed

            # If the target was found, prepare bbox for drawing in the expected format
            if qr_detected_flag and best_match_bbox_raw is not None:
                 # Reshape to (1, 4, 2) for compatibility with main loop drawing
                 self.qr_bbox_int_draw = best_match_bbox_raw.reshape(1, 4, 2)
            else:
                 self.qr_bbox_int_draw = None # Ensure it's None if not detected

        except Exception as e:
            print(f"Error during pyzbar QR detection: {e}")
            # Ensure flags/values are in a safe state
            qr_detected_flag = False
            qr_center_x = None
            qr_center_y = None
            best_match_bbox_raw = None
            self.qr_bbox_int_draw = None

        # Return values needed by run_detections
        return qr_detected_flag, qr_center_x, qr_center_y

    # --- Updated run_detections ---
    def run_detections(self, frame_bgr, target_qr_code, depth_map_mm):
        """
        Processes a single frame for YOLO person detection and target QR code detection (using pyzbar).
        Updates internal state variables.
        Args:
            frame_bgr: The input color frame (BGR format).
            target_qr_code: The string data of the target QR code.
            depth_map_mm: The depth map aligned with the color frame (in mm).
        """
        self._reset_results() # Clear previous results
        height, width = frame_bgr.shape[:2]
        # frame_center_x = width // 2 # Not directly used in this method anymore

        # --- 1. QR Code Detection (using pyzbar) ---
        self.qr_detected_flag, self.qr_center_x, qr_center_y = self._detect_qr_pyzbar(frame_bgr, target_qr_code)
        # self.qr_bbox_int_draw is set within _detect_qr_pyzbar

        # --- Calculate Distance (if QR detected) ---
        if self.qr_detected_flag and self.qr_center_x is not None and qr_center_y is not None:
            # --- TEMPORARY DEBUG --- START ---
            try:
                # Ensure coordinates are valid before accessing depth map
                h_depth, w_depth = depth_map_mm.shape
                y_coord = int(round(qr_center_y))
                x_coord = int(round(self.qr_center_x))
                if 0 <= y_coord < h_depth and 0 <= x_coord < w_depth:
                    raw_depth_at_center = depth_map_mm[y_coord, x_coord]
                    print(f"DEBUG: Raw depth at QR center ({x_coord},{y_coord}): {raw_depth_at_center} mm")
                else:
                    print(f"DEBUG: QR center ({x_coord},{y_coord}) out of bounds for depth map shape {depth_map_mm.shape}")
            except Exception as e_debug: # Catch potential errors during debugging itself
                print(f"DEBUG: Error getting raw depth: {e_debug}")
            # --- TEMPORARY DEBUG --- END ---

            self.target_distance_m = self._get_distance_from_depth(depth_map_mm, self.qr_center_x, qr_center_y)
        else:
            self.target_distance_m = float('inf') # Ensure distance is inf if QR not found or center invalid


        # --- 2. YOLO Person Detection ---
        # Decide whether to run YOLO always or only if QR is detected
        # Running always allows detecting people even if QR is lost temporarily
        try:
            # Use a slightly smaller imgsz if performance is an issue, but frame width is often good
            results = self.model.predict(frame_bgr, imgsz=width, conf=self.confidence_threshold, verbose=False)

            person_boxes = [] # Store detected person boxes (x1, y1, x2, y2)
            person_confidences = [] # Store confidences separately

            if results and results[0].boxes:
                for box_data in results[0].boxes.data.tolist():
                    x1, y1, x2, y2, confidence, cls = box_data
                    if int(cls) == 0: # Class index for 'person'
                        box_coords = tuple(map(int, [x1, y1, x2, y2]))
                        person_boxes.append(box_coords)
                        person_confidences.append(confidence) # Store confidence

                # Prepare drawing info for all detected persons (initially non-target)
                for i, box_coords in enumerate(person_boxes):
                    self.yolo_boxes_for_drawing.append({
                        'box': box_coords,
                        'label': f"Person {person_confidences[i]:.2f}",
                        'is_target': False
                    })

                # --- 3. Target Person Identification (Associate QR with Person) ---
                # This logic runs only if the target QR was successfully detected in this frame
                if self.qr_detected_flag and self.qr_center_x is not None and qr_center_y is not None:
                    best_match_idx = -1
                    min_dist_sq = float('inf')

                    # Find the person box whose center is closest to the QR center
                    for i, person_box in enumerate(person_boxes):
                        px1, py1, px2, py2 = person_box
                        person_center_x = (px1 + px2) // 2
                        person_center_y = (py1 + py2) // 2

                        # Calculate squared distance between QR center and person center
                        dist_sq = (person_center_x - self.qr_center_x)**2 + (person_center_y - qr_center_y)**2

                        # Simple check: Is the QR center within the person's bounding box?
                        # This can be more robust than just closest center if QR is near edge
                        is_inside = (px1 <= self.qr_center_x <= px2) and (py1 <= qr_center_y <= py2)

                        # Prioritize the person box containing the QR code,
                        # otherwise fall back to the closest center.
                        # Add a threshold to avoid associating with very distant "closest" centers.
                        DISTANCE_THRESHOLD_SQ = (width * 0.2)**2 # Example: threshold is 20% of frame width

                        if is_inside and dist_sq < min_dist_sq:
                             min_dist_sq = dist_sq
                             best_match_idx = i
                        elif not is_inside and best_match_idx == -1 and dist_sq < min_dist_sq and dist_sq < DISTANCE_THRESHOLD_SQ:
                             # Only consider closest center if no containing box found yet AND it's reasonably close
                             min_dist_sq = dist_sq
                             best_match_idx = i


                    # If a best match was found, update the target info
                    if best_match_idx != -1:
                        self.target_person_box = person_boxes[best_match_idx] # Store the box coords
                        # Update the drawing info for the matched box
                        self.yolo_boxes_for_drawing[best_match_idx]['is_target'] = True
                        self.yolo_boxes_for_drawing[best_match_idx]['label'] = f"TARGET Person {person_confidences[best_match_idx]:.2f}"

        except Exception as e:
            print(f"Error during YOLO prediction or Target Association: {e}")
            # Clear potentially inconsistent results
            self.yolo_boxes_for_drawing = []
            self.target_person_box = None


    # --- Getter Methods (Unchanged) ---
    def get_target_person_box(self):
        """Returns the bounding box (x1, y1, x2, y2) of the identified target person, or None."""
        return self.target_person_box

    def get_qr_details(self):
        """Returns QR detection status, center x-coordinate, and distance."""
        return self.qr_detected_flag, self.qr_center_x, self.target_distance_m

    def get_drawing_info(self):
        """Returns information needed for drawing overlays."""
        # Returns the list of YOLO boxes and the QR bounding box polygon
        return self.yolo_boxes_for_drawing, self.qr_bbox_int_draw
