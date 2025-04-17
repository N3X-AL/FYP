# /home/aleeya/FYP/fyp2/FYP/components/kinectv2_OA.py
import cv2
import numpy as np

class OA:
    def __init__(self, obstacle_min_dist_m=0.8, roi_width_ratio=0.4, roi_top_ratio=0.3, roi_bottom_ratio=0.9):
        """
        Initializes the Obstacle Avoidance module.
        Args:
            obstacle_min_dist_m (float): Minimum distance for an object to be considered an obstacle (in meters).
            roi_width_ratio (float): Width of the central ROI as a ratio of the frame width.
            roi_top_ratio (float): Top boundary of the ROI as a ratio of frame height.
            roi_bottom_ratio (float): Bottom boundary of the ROI as a ratio of frame height.
        """
        self.obstacle_min_distance_mm = obstacle_min_dist_m * 1000.0
        self.roi_width_ratio = roi_width_ratio
        self.roi_top_ratio = roi_top_ratio
        self.roi_bottom_ratio = roi_bottom_ratio
        print(f"Obstacle Avoidance initialized. Min distance: {obstacle_min_dist_m}m ({self.obstacle_min_distance_mm}mm)")

        self.obstacle_detected = False
        self.depth_vis = None
        self.masked_depth_vis = None

    # --- Updated Method Signature ---
    def process_depth_and_check_obstacles(self, depth_map_proc_res_mm, target_person_mask, processing_resolution):
        """
        Processes the depth map to check for obstacles in a central ROI, excluding the target person using their segmentation mask.

        Args:
            depth_map_proc_res_mm (np.ndarray): Depth map (float32, mm) resized to PROCESSING_RESOLUTION.
            target_person_mask (np.ndarray | None): Boolean mask (same shape as depth map) where True indicates the target person, or None.
            processing_resolution (tuple): (width, height) of the processed frame/depth map.
        """
        self.obstacle_detected = False
        self.depth_vis = None
        self.masked_depth_vis = None

        if depth_map_proc_res_mm is None:
            print("Warning [OA]: Received None depth map.")
            return

        proc_h, proc_w = depth_map_proc_res_mm.shape[:2]
        if proc_w != processing_resolution[0] or proc_h != processing_resolution[1]:
             print(f"Warning [OA]: Depth map shape {depth_map_proc_res_mm.shape} doesn't match processing_resolution {processing_resolution}")
             # Attempt to proceed, but ROI/mask calculations might be off

        # --- 1. Create Mask to Exclude Target Person ---
        # Start with a mask that includes everything (True means check this pixel for obstacles)
        obstacle_check_mask = np.ones_like(depth_map_proc_res_mm, dtype=bool)

        # If a target person mask is provided, set those pixels to False in our check mask
        if target_person_mask is not None:
            # Ensure the provided mask has the same shape as the depth map
            if target_person_mask.shape == obstacle_check_mask.shape:
                # Where target_person_mask is True (person is present),
                # set obstacle_check_mask to False (do not check these pixels).
                obstacle_check_mask[target_person_mask] = False
            else:
                print(f"Warning [OA]: Target person mask shape {target_person_mask.shape} "
                      f"does not match depth map shape {obstacle_check_mask.shape}. Ignoring mask.")
                # Keep obstacle_check_mask as all True if shapes mismatch

        # --- 2. Define Central ROI ---
        # (ROI definition remains the same)
        roi_w = int(proc_w * self.roi_width_ratio)
        roi_x_start = (proc_w - roi_w) // 2
        roi_x_end = roi_x_start + roi_w
        roi_y_start = int(proc_h * self.roi_top_ratio)
        roi_y_end = int(proc_h * self.roi_bottom_ratio)

        # Ensure ROI coordinates are valid
        roi_x_start, roi_y_start = max(0, roi_x_start), max(0, roi_y_start)
        roi_x_end, roi_y_end = min(proc_w, roi_x_end), min(proc_h, roi_y_end)

        if roi_x_start >= roi_x_end or roi_y_start >= roi_y_end:
            print("Warning [OA]: Invalid ROI calculated.")
            return

        # --- 3. Check for Obstacles in ROI (using the exclusion mask) ---
        # Extract the depth values within the ROI
        roi_depth = depth_map_proc_res_mm[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        # Extract the corresponding exclusion mask values for the ROI
        roi_obstacle_check_mask = obstacle_check_mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        # Find pixels within the ROI that are:
        # 1. Valid depth (> 0)
        # 2. Closer than the minimum obstacle distance
        # 3. NOT part of the target person (where roi_obstacle_check_mask is True)
        valid_obstacle_pixels = roi_depth[
            (roi_depth > 0) &
            (roi_depth < self.obstacle_min_distance_mm) &
            roi_obstacle_check_mask # Only check pixels where the mask is True
        ]

        if valid_obstacle_pixels.size > 0:
            self.obstacle_detected = True
            # min_obstacle_dist_mm = np.min(valid_obstacle_pixels) # Optional debug
            # print(f"Obstacle detected! Closest distance: {min_obstacle_dist_mm / 1000.0:.2f}m")


        # --- 4. Create Visualizations (Optional) ---
        try:
            # Depth colormap (same as before)
            depth_colormap = cv2.normalize(depth_map_proc_res_mm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)
            depth_colormap[depth_map_proc_res_mm == 0] = [0, 0, 0]
            cv2.rectangle(depth_colormap, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 255, 255), 2)
            self.depth_vis = depth_colormap

            # Masked depth visualization (using the actual mask now)
            masked_depth_map = depth_map_proc_res_mm.copy()
            # Black out the target area using the obstacle_check_mask (where it's False)
            masked_depth_map[~obstacle_check_mask] = 0
            masked_colormap = cv2.normalize(masked_depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            masked_colormap = cv2.applyColorMap(masked_colormap, cv2.COLORMAP_JET)
            masked_colormap[masked_depth_map == 0] = [0, 0, 0] # Black out invalid depth and target
            cv2.rectangle(masked_colormap, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 0, 255), 1)
            self.masked_depth_vis = masked_colormap

        except Exception as e:
            print(f"Warning [OA]: Failed to create visualizations: {e}")
            self.depth_vis = None
            self.masked_depth_vis = None


    def get_obstacle_status(self):
        """Returns True if an obstacle was detected in the ROI, False otherwise."""
        return self.obstacle_detected

    def get_depth_visualizations(self):
        """Returns the depth visualization frames (can be None)."""
        return self.depth_vis, self.masked_depth_vis

