# /home/aleeya/FYP/fyp2/FYP/components/kinectv2_yoloqr.py
import cv2
import numpy as np
from ultralytics import YOLO
# from ultralytics.utils.ops import scale_image # We'll stick with cv2.resize for now
from pyzbar import pyzbar
import traceback # For detailed error printing

class YoloQR:
    def __init__(self, yolo_model_path='yolov8n-seg.pt', yolo_img_size=640, conf_threshold=0.5):
        """
        Initializes the YOLOv8 segmentation model and QR detector.
        Args:
            yolo_model_path (str): Path to the YOLO model file (e.g., 'yolov8n-seg.pt').
            yolo_img_size (int): Input image size for YOLO model (e.g., 640).
            conf_threshold (float): Confidence threshold for YOLO detections.
        """
        print(f"Initializing YoloQR with model: {yolo_model_path}, img_size: {yolo_img_size}")
        try:
            # Ensure you are loading a segmentation model ('-seg')
            if '-seg' not in yolo_model_path:
                print(f"WARNING: Model path '{yolo_model_path}' might not be a segmentation model. Masking will likely fail.")
            self.model = YOLO(yolo_model_path)
            # Define classes of interest (adjust as needed)
            # COCO: 0: person, ... (add others if needed)
            self.model.overrides['classes'] = [0] # Detect only 'person'
            self.model.overrides['conf'] = conf_threshold
            self.yolo_img_size = yolo_img_size # Store the image size for prediction
            print(f"YOLO model loaded successfully. Detecting classes: {self.model.overrides['classes']}")
        except Exception as e:
            print(f"ERROR: Failed to load YOLO model from {yolo_model_path}: {e}")
            raise # Re-raise the exception to be caught by the main script

        # Internal state variables
        self.target_person_box = None # Bbox [x1, y1, x2, y2] relative to processing res
        self.target_person_mask = None # Boolean mask (shape of processing res) for the target person
        self.qr_detected_flag = False
        self.qr_center_x = None # X-coordinate relative to processing width
        self.target_distance_m = float('inf')
        self.yolo_boxes_for_drawing = [] # List of dicts {'box': [x1,y1,x2,y2], 'label': str, 'is_target': bool}
        self.qr_bbox_int_draw = None # Bbox [[x1,y1], [x2,y2], ...] relative to processing res

    def run_detections(self, frame_bgr_proc_res, target_qr_code_data, depth_map_proc_res_mm):
        """
        Runs YOLO object/segmentation detection and QR code detection.
        Updates internal state including the target person's mask.

        Args:
            frame_bgr_proc_res (np.ndarray): Input BGR frame (resized to PROCESSING_RESOLUTION).
            target_qr_code_data (str): The data content of the target QR code.
            depth_map_proc_res_mm (np.ndarray): Depth map (float32, mm) resized to PROCESSING_RESOLUTION.
        """
        # Reset results from previous frame
        self.target_person_box = None
        self.target_person_mask = None # Reset mask
        self.qr_detected_flag = False
        self.qr_center_x = None
        self.target_distance_m = float('inf')
        self.yolo_boxes_for_drawing = []
        self.qr_bbox_int_draw = None

        proc_h, proc_w = frame_bgr_proc_res.shape[:2]
        # Ensure the target shape for resizing is correct (width, height)
        target_size_wh_for_resize = (proc_w, proc_h)

        # --- 1. YOLO Detection ---
        frame_bgr_proc_res_cont = np.ascontiguousarray(frame_bgr_proc_res)
        try:
            # Predict using the segmentation model
            results = self.model.predict(frame_bgr_proc_res_cont, imgsz=self.yolo_img_size, verbose=False)
        except Exception as e:
            print(f"Error during YOLO prediction: {e}")
            results = []

        persons_found = []
        # person_masks = [] # Not needed if we process directly when associated

        if results and results[0].boxes:
            # Check if masks are present (indicating segmentation model output)
            has_masks = results[0].masks is not None
            if not has_masks and '-seg' in self.model.ckpt_path: # Check if it should have masks
                 print("WARNING: Segmentation model loaded, but no masks found in results!")

            for i, box in enumerate(results[0].boxes):
                cls_id = int(box.cls.item())
                if cls_id == 0: # Person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf.item())
                    # Store index 'i' needed to retrieve the correct mask later
                    persons_found.append({'box': [x1, y1, x2, y2], 'conf': confidence, 'index': i})
                    self.yolo_boxes_for_drawing.append({
                        'box': [x1, y1, x2, y2],
                        'label': f"Person {confidence:.2f}",
                        'is_target': False
                    })
                    # Don't store raw masks here, process only the target one later

        # --- 2. QR Code Detection ---
        # (QR detection code remains the same as before)
        qr_found_target = None
        decoded_objects = pyzbar.decode(frame_bgr_proc_res)
        for obj in decoded_objects:
            try:
                data = obj.data.decode('utf-8')
                points = obj.polygon
                bbox_np = np.array([(p.x, p.y) for p in points], dtype=np.int32)
                bbox_int = bbox_np.reshape(-1, 1, 2)

                if data == target_qr_code_data:
                    self.qr_detected_flag = True
                    center_x = int(np.mean(bbox_np[:, 0]))
                    center_y = int(np.mean(bbox_np[:, 1]))
                    self.qr_center_x = center_x
                    self.qr_bbox_int_draw = bbox_int

                    if 0 <= center_y < depth_map_proc_res_mm.shape[0] and 0 <= center_x < depth_map_proc_res_mm.shape[1]:
                        depth_mm = depth_map_proc_res_mm[center_y, center_x]
                        self.target_distance_m = depth_mm / 1000.0 if depth_mm > 0 else float('inf')
                    else:
                        self.target_distance_m = float('inf')

                    qr_found_target = {'box_points': bbox_np, 'center': (center_x, center_y)}
                    # break # Optional: stop after finding the first target QR

            except Exception as e:
                print(f"Error decoding/processing QR data: {e}")
                continue

        # --- 3. Associate Target QR with Person & Extract Mask ---
        associated_person_info = None # Reset association for this frame
        if self.qr_detected_flag and qr_found_target and persons_found:
            qr_center = qr_found_target['center']
            min_dist_sq = float('inf')

            # --- Relaxed Association Parameter ---
            # Max distance squared between QR center and Person center
            # Increase this if the QR code might be held further away from the body center
            MAX_CENTER_DIST_SQ = (75 * 75) # Example: 75 pixels radius

            for person in persons_found:
                p_box = person['box']
                person_center_x = (p_box[0] + p_box[2]) // 2
                person_center_y = (p_box[1] + p_box[3]) // 2
                dist_sq = (qr_center[0] - person_center_x)**2 + (qr_center[1] - person_center_y)**2

                # --- Use Relaxed Association Criteria ---
                is_associated = dist_sq < MAX_CENTER_DIST_SQ

                if is_associated and dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    associated_person_info = person

            # --- If association found, update state and process mask ---
            if associated_person_info is not None:
                assoc_idx = associated_person_info['index']
                self.target_person_box = associated_person_info['box']

                # Update drawing info (find the correct box to mark as target)
                # Assuming order is preserved or index is reliable
                if assoc_idx < len(self.yolo_boxes_for_drawing):
                     self.yolo_boxes_for_drawing[assoc_idx]['is_target'] = True
                     self.yolo_boxes_for_drawing[assoc_idx]['label'] = f"TARGET Person {associated_person_info['conf']:.2f}"
                else:
                     print(f"Warning: assoc_idx {assoc_idx} out of bounds for yolo_boxes_for_drawing")


                # --- Extract and Resize Mask ---
                if results and results[0].masks and assoc_idx < len(results[0].masks.data):
                    try:
                        mask_tensor = results[0].masks.data[assoc_idx]
                        # Mask tensor is usually relative to the padded input size (e.g., 640x640)
                        mask_np_native_res = mask_tensor.cpu().numpy() # Shape e.g., (160, 160)

                        # Ensure mask is 2D
                        if mask_np_native_res.ndim != 2:
                            mask_np_native_res = np.squeeze(mask_np_native_res)
                        if mask_np_native_res.ndim != 2:
                             raise ValueError(f"Mask could not be reduced to 2D. Shape: {mask_np_native_res.shape}")

                        # --- Resize the mask to the *processing resolution* ---
                        # target_size_wh_for_resize was defined earlier as (proc_w, proc_h)
                        resized_mask = cv2.resize(
                            mask_np_native_res,
                            target_size_wh_for_resize, # Use (width, height) tuple
                            interpolation=cv2.INTER_NEAREST # Keep sharp edges for mask
                        )

                        # Threshold to create a final boolean mask
                        self.target_person_mask = resized_mask > 0.5 # Adjust threshold if needed (0.0-1.0)

                        # --- DEBUG (Uncomment ONE line temporarily if offset persists) ---
                        # print(f"DEBUG: Native mask shape: {mask_np_native_res.shape}, Target resize shape: {target_size_wh_for_resize}, Final mask shape: {self.target_person_mask.shape}")
                        # cv2.imshow("DEBUG Resized Mask", (self.target_person_mask * 255).astype(np.uint8)); cv2.waitKey(1) # Visualize the mask

                    except Exception as e:
                        print(f"Error processing/resizing mask for target person: {e}")
                        traceback.print_exc()
                        self.target_person_mask = None
                else:
                     # This branch means no mask data was available in results for the associated person
                     print(f"Warning: Mask data not available in results for associated person index {assoc_idx}.")
                     self.target_person_mask = None

            # else: # QR detected, but not associated
            #     pass # print("Target QR detected, but not associated with a person box.")

    def get_target_person_box(self):
        """Returns the bounding box [x1, y1, x2, y2] of the associated target person, or None."""
        return self.target_person_box

    # --- New Method ---
    def get_target_person_mask(self):
        """
        Returns the boolean segmentation mask (numpy array) for the target person,
        resized to the processing resolution, or None if no target/mask available.
        Mask is True where the target person is detected.
        """
        return self.target_person_mask

    def get_qr_details(self):
        """Returns QR detection status, center X, and distance."""
        return self.qr_detected_flag, self.qr_center_x, self.target_distance_m

    def get_drawing_info(self):
        """Returns lists/bboxes needed for drawing overlays."""
        return self.yolo_boxes_for_drawing, self.qr_bbox_int_draw
