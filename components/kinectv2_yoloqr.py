# /home/aleeya/FYP/fyp2/FYP/components/kinectv2_yoloqr.py
import cv2
import numpy as np
from ultralytics import YOLO
# from pyzbar import pyzbar # No longer used directly
import traceback
import torch
# from ultralytics.nn.autobackend import AutoBackend # No longer needed

class YoloQR:
    # <<< REVERT: Default device back to 'cpu' or let caller specify >>>
    def __init__(self, yolo_model_path='yolov8n.pt', yolo_img_size=640, conf_threshold=0.5,
                 association_radius_px=75):
                

        # --- Device Check and YOLO Loading (Reverted) ---
        try:
            # Initialize YOLO object
            self.model = YOLO(yolo_model_path)
            print(f"YOLO object initialized with path: {yolo_model_path}")

            # --- Common Overrides ---
            self.model.overrides['classes'] = [0] # Detect only 'person'
            self.model.overrides['conf'] = conf_threshold
            self.yolo_img_size = yolo_img_size
            print(f"YOLO Overrides Applied: classes=[0], conf={conf_threshold}")

        except Exception as e:
            print(f"ERROR: Failed to load or configure YOLO model: {e}")
            traceback.print_exc()
            raise # Re-raise the exception

        # --- State Variables (Remain the same) ---
        self.current_target_box = None
        self.current_target_distance_m = float('inf')
        self.current_yolo_boxes_for_drawing = []
        self.association_radius_sq = association_radius_px * association_radius_px
        self.last_confirmed_target_box = None
        self.frames_since_last_confirmation = 0
        self.is_target_confirmed_in_current_frame = False


    def run_detections_and_association(self,
                                       frame_bgr_proc_res,
                                       depth_map_proc_res_mm,
                                       qr_code_details_from_processor,
                                       obstacle_detected): # Keep obstacle_detected argument
        
        # --- Reset Frame-Specific State ---
        self.current_yolo_boxes_for_drawing = []
        self.is_target_confirmed_in_current_frame = False
        # Don't clear current_target_box/mask/distance here, do it conditionally later

        proc_h, proc_w = frame_bgr_proc_res.shape[:2]
        target_size_wh_for_resize = (proc_w, proc_h) # Used for resizing masks

        # --- Stop if Obstacle is Detected ---
        if obstacle_detected:
            # ... (obstacle handling remains the same) ...
            print("Obstacle detected! Stopping tracking and clearing target.")
            self._clear_current_target_state() # Clear state when stopping for obstacle
            return

        # --- 1. YOLO Detection ---
        # ... (YOLO prediction remains the same) ...
        frame_bgr_proc_res_cont = np.ascontiguousarray(frame_bgr_proc_res)
        results = None
        try:
            results = self.model.predict(
                frame_bgr_proc_res_cont,
                imgsz=self.yolo_img_size,
                verbose=False
            )
        except Exception as e:
            # ... (error handling remains the same) ...
            print(f"Error during YOLO prediction: {e}"); traceback.print_exc()
            results = None
            self.frames_since_last_confirmation += 1
            self._clear_current_target_state()
            return

        # --- Process YOLO Results (Extract Boxes and Masks) ---
        persons_found = []
        masks_cpu = None
        try:
            if results and results[0].boxes:
                 has_masks = results[0].masks is not None
                 boxes_cpu = results[0].boxes.cpu()
                 masks_cpu = results[0].masks.cpu() if has_masks else None

                 for i, box in enumerate(boxes_cpu):
                     cls_id = int(box.cls.item())
                     if cls_id == 0: # Person
                         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                         confidence = float(box.conf.item())
                         persons_found.append({'box': [x1, y1, x2, y2], 'conf': confidence, 'index': i})
                         # <<< CHANGE: Initial label has NO distance >>>
                         self.current_yolo_boxes_for_drawing.append({
                             'box': [x1, y1, x2, y2],
                             'label': f"Person {confidence:.2f}", # Only show confidence
                             'is_target': False
                         })
            # else: print("DEBUG: No boxes found in YOLO results.")

        except Exception as e:
            # ... (error handling remains the same) ...
            print(f"Error processing YOLO results: {e}"); traceback.print_exc()
            persons_found = []
            masks_cpu = None
            self.frames_since_last_confirmation += 1

        # --- 2. Attempt Confirmation (Link QR to Person) ---
        # ... (association logic remains the same) ...
        confirmed_person_info = None
        if qr_code_details_from_processor and persons_found:
             qr_center = qr_code_details_from_processor['center']
             min_dist_sq = float('inf')
             best_match_person = None

             for person in persons_found:
                  p_box = person['box']
                  person_center_x = (p_box[0] + p_box[2]) // 2
                  person_center_y = (p_box[1] + p_box[3]) // 2
                  dist_sq = (qr_center[0] - person_center_x)**2 + (qr_center[1] - person_center_y)**2
                  is_within_box = p_box[0] <= qr_center[0] < p_box[2] and p_box[1] <= qr_center[1] < p_box[3]
                  is_close_enough = dist_sq < self.association_radius_sq
                  if is_within_box and is_close_enough and dist_sq < min_dist_sq:
                       min_dist_sq = dist_sq
                       best_match_person = person

             if best_match_person is not None:
                  confirmed_person_info = best_match_person
                  self.is_target_confirmed_in_current_frame = True
                  # print("DEBUG: Target confirmed by QR association.")
             # else: print("DEBUG: QR detected, but no suitable person association found.")
        # elif not qr_code_details_from_processor: print("DEBUG: No QR details received.")
        # elif not persons_found: print("DEBUG: No persons found by YOLO.")


        # --- 3. Update State based on Confirmation & Tracking ---
        if self.is_target_confirmed_in_current_frame:
             # Target confirmed by QR this frame
             assoc_idx = confirmed_person_info['index']
             confirmed_box = confirmed_person_info['box']
             self.last_confirmed_target_box = confirmed_box

             # Calculate distance using box only
             confirmed_distance_m = self._calculate_distance(confirmed_box, depth_map_proc_res_mm)

             # Update current state
             self.current_target_box = confirmed_box
             self.current_target_distance_m = confirmed_distance_m # <<< THIS IS THE DISTANCE TO DISPLAY

             # <<< CHANGE: Update drawing info for the TARGET >>>
             if assoc_idx < len(self.current_yolo_boxes_for_drawing):
                  self.current_yolo_boxes_for_drawing[assoc_idx]['is_target'] = True
                  # Format distance string, handle infinity
                  dist_str = f"{self.current_target_distance_m:.2f}m" if self.current_target_distance_m != float('inf') else "Dist N/A"
                  # Update label to include distance
                  self.current_yolo_boxes_for_drawing[assoc_idx]['label'] = f"TARGET {dist_str}"

        else:
                if self.current_target_box is not None:
                    print(f"Target lost")
                self._clear_current_target_state()

        # --- Final check: If no target box is set after all logic, clear state ---
        if self.current_target_box is None:
             self._clear_current_target_state() # Ensures distance is inf if no target

    # --- _calculate_distance (No changes needed here) ---
    def _calculate_distance(self, qr_details, depth_map_mm):
        """
        Calculate the distance to the QR code using the depth map.
        :param qr_details: Dictionary containing QR code details, including 'bbox_np' (bounding box as numpy array).
        :param depth_map_mm: Depth map in millimeters.
        :return: Distance to the QR code in meters.
        """
        if depth_map_mm is None or qr_details is None:
            return float('inf')

        distance_m = float('inf')
        h, w = depth_map_mm.shape

        try:
            # Extract the QR code bounding box
            bbox_np = qr_details.get('bbox_np')
            if bbox_np is not None:
                # Create a mask for the QR code region
                qr_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(qr_mask, [bbox_np], 255)

                # Extract depth values within the QR code region
                qr_depth_values = depth_map_mm[qr_mask == 255]
                valid_depth_values = qr_depth_values[qr_depth_values > 0]  # Exclude invalid depth values (0 mm)

                if valid_depth_values.size > 0:
                    # Use the median depth value for robustness
                    median_depth_mm = np.median(valid_depth_values)
                    if 100 < median_depth_mm < 10000:  # Ensure the depth is within a valid range
                        distance_m = median_depth_mm / 1000.0  # Convert mm to meters
        except Exception as e:
            print(f"Error calculating QR code distance: {e}")
            traceback.print_exc()
            distance_m = float('inf')

        return distance_m

    # --- _clear_current_target_state (No changes needed) ---
    def _clear_current_target_state(self):
        self.current_target_box = None
        self.current_target_distance_m = float('inf') # Ensure distance is reset

    # --- Getters for External Use (No changes needed) ---
    def get_target_person_box(self): return self.current_target_box
    def get_drawing_info(self):
        # This returns the list of boxes with labels already formatted
        return self.current_yolo_boxes_for_drawing, None # No separate QR bbox needed from here
