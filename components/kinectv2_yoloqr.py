# /home/aleeya/FYP/fyp2/FYP/components/kinectv2_yoloqr.py
import cv2
import numpy as np
from ultralytics import YOLO
# from pyzbar import pyzbar # No longer used directly
import traceback
import torch
import time
# from ultralytics.nn.autobackend import AutoBackend # No longer needed

class YoloQR:
    # <<< REVERT: Default device back to 'cpu' or let caller specify >>>
    def __init__(self, yolo_model_path='yolov8n-seg.pt', yolo_img_size=640, conf_threshold=0.5, device='cpu',
                 persistence_frames=10,
                 association_radius_px=75):
        """
        Initializes YOLO (.pt), associates with external QR detection, and uses KCF tracking for persistence.
        # ... (rest of docstring) ...
        """
        print(f"Initializing YoloQR (KCF Tracking) with model: {yolo_model_path}, img_size: {yolo_img_size}, requested device: {device}")
        self.device = device # Store requested device

        # --- Device Check and YOLO Loading (Reverted) ---
        try:
            # <<< REVERT: Simple PyTorch Loading >>>
            print("Loading PyTorch model (.pt).")
            # Check CUDA availability if requested
            if 'cuda' in self.device and not torch.cuda.is_available():
                print(f"WARNING: CUDA device '{self.device}' requested but not available. Falling back to CPU.")
                self.device = 'cpu' # Fallback internal device variable

            # Initialize YOLO object
            self.model = YOLO(yolo_model_path)
            print(f"YOLO object initialized with path: {yolo_model_path}")

            # Move the PyTorch model to the specified device
            print(f"Moving PyTorch model to device: {self.device}...")
            self.model.to(self.device)
            self.use_half = self.device != 'cpu' # Use half precision only on GPU
            print(f"PyTorch model loaded on {self.device}. Half precision: {self.use_half}")

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
        self.current_target_mask = None
        self.current_target_distance_m = float('inf')
        self.current_yolo_boxes_for_drawing = []
        self.persistence_frames = persistence_frames
        self.association_radius_sq = association_radius_px * association_radius_px
        self.last_confirmed_target_box = None
        self.frames_since_last_confirmation = 0
        self.is_target_confirmed_in_current_frame = False
        self.tracker = None
        self.is_tracker_initialized = False

        print("YoloQR Initialized (KCF Tracking).")

    def _initialize_tracker(self, frame, bbox):
        """ Creates and initializes a KCF tracker. """
        try:
            # Convert bbox [x1, y1, x2, y2] to [x, y, w, h] for tracker
            x, y = int(bbox[0]), int(bbox[1])
            w = int(bbox[2] - bbox[0])
            h = int(bbox[3] - bbox[1])
            if w <= 0 or h <= 0:
                 print("Warning: Invalid bounding box dimensions for tracker init.")
                 return False

            self.tracker = cv2.TrackerKCF_create()
            # self.tracker = cv2.TrackerCSRT_create() # Alternative: Slower but potentially more robust
            ok = self.tracker.init(frame, (x, y, w, h))
            if ok:
                print("KCF Tracker Initialized.")
                self.is_tracker_initialized = True
                return True
            else:
                print("ERROR: KCF Tracker initialization failed.")
                self.tracker = None
                self.is_tracker_initialized = False
                return False
        except Exception as e:
            print(f"Error initializing tracker: {e}")
            self.tracker = None
            self.is_tracker_initialized = False
            return False

    def run_detections_and_association(self,
                                       frame_bgr_proc_res,
                                       depth_map_proc_res_mm,
                                       qr_code_details_from_processor):
        """
        Runs YOLO, associates with external QR, uses KCF tracker for persistence.
        # ... (rest of docstring) ...
        """
        # --- Reset Frame-Specific State ---
        self.current_yolo_boxes_for_drawing = []
        self.is_target_confirmed_in_current_frame = False
        self._clear_current_target_state()

        proc_h, proc_w = frame_bgr_proc_res.shape[:2]
        target_size_wh_for_resize = (proc_w, proc_h)

        # --- 1. YOLO Detection ---
        frame_bgr_proc_res_cont = np.ascontiguousarray(frame_bgr_proc_res)
        results = None
        try:
            # <<< REVERT: Predict call with device and half arguments >>>
            results = self.model.predict(
                frame_bgr_proc_res_cont,
                imgsz=self.yolo_img_size,
                verbose=False,
                half=self.use_half,   # Use half precision based on device
                device=self.device    # Pass the device model is on
            )
        except Exception as e:
            print(f"Error during YOLO prediction: {e}")
            print("--- Traceback ---")
            traceback.print_exc()
            print("-----------------")
            results = None

        # --- Process YOLO Results ---
        # (The rest of the method remains the same)
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
                         self.current_yolo_boxes_for_drawing.append({
                             'box': [x1, y1, x2, y2], 'label': f"Person {confidence:.2f}", 'is_target': False
                         })

        except Exception as e:
            print(f"Error processing YOLO results: {e}")
            traceback.print_exc()
            persons_found = []
            masks_cpu = None
            self.frames_since_last_confirmation = self.persistence_frames + 1
            self.is_tracker_initialized = False
            self.tracker = None

        # --- 2. Attempt Confirmation (Link QR to Person) ---
        # ... (association logic remains the same) ...
        confirmed_person_info = None
        if qr_code_details_from_processor and persons_found:
             qr_center = qr_code_details_from_processor['center']
             min_dist_sq = float('inf')
             for person in persons_found:
                  p_box = person['box']
                  person_center_x = (p_box[0] + p_box[2]) // 2
                  person_center_y = (p_box[1] + p_box[3]) // 2
                  dist_sq = (qr_center[0] - person_center_x)**2 + (qr_center[1] - person_center_y)**2
                  is_within_box = p_box[0] <= qr_center[0] < p_box[2] and p_box[1] <= qr_center[1] < p_box[3]
                  is_close_enough = dist_sq < self.association_radius_sq
                  if is_within_box and is_close_enough and dist_sq < min_dist_sq:
                       min_dist_sq = dist_sq
                       confirmed_person_info = person
             if confirmed_person_info is not None:
                  self.is_target_confirmed_in_current_frame = True

        # --- 3. Update State based on Confirmation & Tracking ---
        # ... (state update logic remains the same) ...
        if self.is_target_confirmed_in_current_frame:
             # ... (handle confirmed target) ...
             assoc_idx = confirmed_person_info['index']
             confirmed_box = confirmed_person_info['box']
             self.last_confirmed_target_box = confirmed_box
             self._initialize_tracker(frame_bgr_proc_res, confirmed_box)
             confirmed_mask = None
             if masks_cpu is not None and assoc_idx < len(masks_cpu.data):
                  try:
                      mask_tensor = masks_cpu.data[assoc_idx]; mask_np_native = mask_tensor.numpy()
                      if mask_np_native.ndim != 2: mask_np_native = np.squeeze(mask_np_native)
                      if mask_np_native.ndim == 2:
                           resized_mask = cv2.resize(mask_np_native, target_size_wh_for_resize, interpolation=cv2.INTER_NEAREST)
                           confirmed_mask = resized_mask > 0.5
                  except Exception as mask_err: print(f"Error processing mask for confirmed target: {mask_err}")
             confirmed_distance_m = self._calculate_distance(confirmed_box, confirmed_mask, depth_map_proc_res_mm)
             self.current_target_box = confirmed_box
             self.current_target_mask = confirmed_mask
             self.current_target_distance_m = confirmed_distance_m
             self.frames_since_last_confirmation = 0
             if assoc_idx < len(self.current_yolo_boxes_for_drawing):
                  self.current_yolo_boxes_for_drawing[assoc_idx]['is_target'] = True
                  self.current_yolo_boxes_for_drawing[assoc_idx]['label'] = f"TARGET Person {confirmed_person_info['conf']:.2f}"

        else:
             # ... (handle tracking) ...
             self.frames_since_last_confirmation += 1
             if self.frames_since_last_confirmation <= self.persistence_frames and self.is_tracker_initialized:
                  # ... (update tracker) ...
                  tracker_ok = False; tracked_bbox_xywh = None
                  try:
                       tracker_ok, tracked_bbox_xywh = self.tracker.update(frame_bgr_proc_res)
                  except Exception as tracker_err: print(f"Error updating tracker: {tracker_err}"); tracker_ok = False

                  if tracker_ok:
                       # ... (use tracked box) ...
                       x1 = int(tracked_bbox_xywh[0]); y1 = int(tracked_bbox_xywh[1])
                       x2 = int(x1 + tracked_bbox_xywh[2]); y2 = int(y1 + tracked_bbox_xywh[3])
                       tracked_box_xyxy = [x1, y1, x2, y2]
                       self.current_target_box = tracked_box_xyxy
                       self.current_target_mask = None
                       self.current_target_distance_m = self._calculate_distance(self.current_target_box, self.current_target_mask, depth_map_proc_res_mm)
                       # Update drawing info
                       min_draw_dist_sq = float('inf'); closest_draw_idx = -1
                       tracked_center_x = (x1+x2)//2; tracked_center_y = (y1+y2)//2
                       for idx, draw_item in enumerate(self.current_yolo_boxes_for_drawing):
                            if not draw_item['is_target']:
                                 draw_box = draw_item['box']
                                 draw_center_x = (draw_box[0] + draw_box[2]) // 2; draw_center_y = (draw_box[1] + draw_box[3]) // 2
                                 draw_dist_sq = (tracked_center_x - draw_center_x)**2 + (tracked_center_y - draw_center_y)**2
                                 if draw_dist_sq < (75*75) and draw_dist_sq < min_draw_dist_sq:
                                      min_draw_dist_sq = draw_dist_sq; closest_draw_idx = idx
                       if closest_draw_idx != -1:
                            self.current_yolo_boxes_for_drawing[closest_draw_idx]['is_target'] = True
                            self.current_yolo_boxes_for_drawing[closest_draw_idx]['label'] = "TARGET (Tracked)"
                  else:
                       # ... (handle tracker failure) ...
                       print(f"Tracker failed on frame {self.frames_since_last_confirmation}. Target lost.")
                       self.is_tracker_initialized = False; self.tracker = None
                       self.frames_since_last_confirmation = self.persistence_frames + 1
                       self._clear_current_target_state()
             else:
                  # ... (handle persistence expired) ...
                  if self.frames_since_last_confirmation == self.persistence_frames + 1 and self.last_confirmed_target_box is not None:
                       print(f"Target lost (Exceeded {self.persistence_frames} persistence frames or tracker failed)")
                  if self.is_tracker_initialized:
                       print("Tracker stopped due to expired persistence.")
                       self.is_tracker_initialized = False; self.tracker = None
                  self._clear_current_target_state()

    # --- _calculate_distance (remains the same) ---
    def _calculate_distance(self, box, mask, depth_map_mm):
        # ... (implementation is the same) ...
        if depth_map_mm is None: return float('inf')
        distance_m = float('inf')
        try:
            # Prefer mask if available and valid
            if mask is not None and mask.shape == depth_map_mm.shape:
                target_depths_mm = depth_map_mm[mask]
                valid_target_depths_mm = target_depths_mm[target_depths_mm > 0]
                if valid_target_depths_mm.size > 5: # Require a few points for median
                    median_depth_mm = np.median(valid_target_depths_mm)
                    distance_m = median_depth_mm / 1000.0
            # Fallback to box center if mask failed or wasn't available
            if distance_m == float('inf') and box is not None:
                x1, y1, x2, y2 = map(int, box); center_x = (x1 + x2) // 2; center_y = (y1 + y2) // 2
                h, w = depth_map_mm.shape
                if 0 <= center_y < h and 0 <= center_x < w:
                    center_depth_mm = depth_map_mm[center_y, center_x]
                    if center_depth_mm > 0:
                        distance_m = center_depth_mm / 1000.0
            # Sanity check final distance
            if not (0.1 < distance_m < 10.0): distance_m = float('inf')
        except Exception as e: print(f"Error calculating distance: {e}"); distance_m = float('inf')
        return distance_m

    # --- _clear_current_target_state (remains the same) ---
    def _clear_current_target_state(self):
        self.current_target_box = None
        self.current_target_mask = None
        self.current_target_distance_m = float('inf')

    # --- Getters for External Use (remain the same) ---
    def get_target_person_box(self): return self.current_target_box
    def get_target_person_mask(self): return self.current_target_mask
    def get_qr_details(self):
        return None, None, self.current_target_distance_m
    def get_drawing_info(self):
        return self.current_yolo_boxes_for_drawing, None

