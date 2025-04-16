# components/kinectv2_OA.py
import cv2
import numpy as np
import time

class OA:
    def __init__(self, obstacle_min_dist_m=0.8, check_width_px=300, roi_y_start_frac=1/3, roi_y_end_frac=2/3, mask_dilation_pixels=10): # Added mask_dilation_pixels
        print("Initializing Obstacle Avoidance Module...")
        self.obstacle_min_distance_m = obstacle_min_dist_m
        self.obstacle_check_width_px = check_width_px
        self.obstacle_check_roi_y_start = roi_y_start_frac
        self.obstacle_check_roi_y_end = roi_y_end_frac
        self.mask_dilation_pixels = mask_dilation_pixels # Pixels to expand the mask box

        # --- Attributes ---
        self.depth_map_resized_mm = None # Store the resized depth map
        self.masked_depth_mm = None      # Store the depth map with target masked
        self.obstacle_detected = False   # Flag for obstacle detection result
        self.obstacle_roi_coords = None  # Store ROI coords for drawing (x1, y1, x2, y2)
        self.dilated_mask_coords = None  # Store the dilated box coords for potential debugging/vis

    def _check_for_obstacles(self, masked_depth_data):
        """
        Internal helper to check for obstacles in the central ROI of the masked depth map.
        """
        height, frame_width = masked_depth_data.shape
        center_x = frame_width // 2

        roi_x1 = max(0, center_x - self.obstacle_check_width_px // 2)
        roi_x2 = min(frame_width, center_x + self.obstacle_check_width_px // 2)
        roi_y1 = int(height * self.obstacle_check_roi_y_start)
        roi_y2 = int(height * self.obstacle_check_roi_y_end)

        self.obstacle_roi_coords = (roi_x1, roi_y1, roi_x2, roi_y2) # Store for drawing

        if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
            # print("Warning: Invalid obstacle ROI") # Debugging
            return False # Invalid ROI

        obstacle_roi = masked_depth_data[roi_y1:roi_y2, roi_x1:roi_x2]

        # Filter out invalid depth values (0) and the masked values (e.g., 65535)
        # Convert valid readings from mm to meters for comparison
        valid_depths_mm = obstacle_roi[(obstacle_roi > 0) & (obstacle_roi < 65535)]
        if valid_depths_mm.size == 0:
            # print("No valid depth points in obstacle ROI") # Debugging
            return False # No valid depth points in ROI to check

        min_depth_in_roi_m = np.min(valid_depths_mm) / 1000.0
        # print(f"Min depth in obstacle ROI: {min_depth_in_roi_m:.2f}m") # Debugging

        return min_depth_in_roi_m < self.obstacle_min_distance_m

    def process_depth_and_check_obstacles(self, depth_frame_raw, target_person_box, target_shape_wh):
        """
        Resizes depth map, masks the target person (using a dilated box), and checks for obstacles.
        Updates internal attributes.
        Args:
            depth_frame_raw: Raw depth frame (numpy array, float32 mm).
            target_person_box: Bounding box (x1, y1, x2, y2) of the target person, or None.
            target_shape_wh: Target (W, H) for resizing (usually color frame shape).
        """
        # 1. Resize depth map
        # Using INTER_NEAREST is crucial for depth data
        self.depth_map_resized_mm = cv2.resize(
            depth_frame_raw,
            target_shape_wh, # (width, height)
            interpolation=cv2.INTER_NEAREST
        )

        # 2. Create Masked Depth Map
        self.masked_depth_mm = self.depth_map_resized_mm.copy()
        self.dilated_mask_coords = None # Reset dilated coords

        if target_person_box is not None:
            px1, py1, px2, py2 = target_person_box
            h, w = self.masked_depth_mm.shape # Get dimensions of the map we are masking

            # --- DILATE the bounding box ---
            # Subtract from top-left (x1, y1), add to bottom-right (x2, y2)
            # Ensure coordinates stay within the frame bounds (0 to w-1, 0 to h-1)
            dilated_x1 = max(0, px1 - self.mask_dilation_pixels)
            dilated_y1 = max(0, py1 - self.mask_dilation_pixels)
            dilated_x2 = min(w, px2 + self.mask_dilation_pixels) # Use w, h for upper bounds in slicing
            dilated_y2 = min(h, py2 + self.mask_dilation_pixels)
            # --- Store for debugging/visualization if needed ---
            self.dilated_mask_coords = (dilated_x1, dilated_y1, dilated_x2, dilated_y2)

            # Set depth within the *dilated* target person's box to a large value (e.g., 65535)
            # Use the calculated dilated coordinates
            if dilated_y1 < dilated_y2 and dilated_x1 < dilated_x2: # Check if valid area after dilation/clipping
                 self.masked_depth_mm[dilated_y1:dilated_y2, dilated_x1:dilated_x2] = 65535.0 # Use float if base is float

        # 3. Check for Obstacles using the masked map
        self.obstacle_detected = self._check_for_obstacles(self.masked_depth_mm)

    def get_obstacle_status(self):
        """Returns True if an obstacle was detected."""
        return self.obstacle_detected

    def get_depth_visualizations(self):
        """Returns visualized depth maps (raw resized and masked)."""
        if self.depth_map_resized_mm is None:
            return None, None

        # Raw Depth Map Visualization
        depth_image_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(self.depth_map_resized_mm, alpha=0.05), # Adjust alpha for visibility
            cv2.COLORMAP_JET
        )
        # Draw ROI on raw depth vis
        if self.obstacle_roi_coords:
            r_x1, r_y1, r_x2, r_y2 = self.obstacle_roi_coords
            cv2.rectangle(depth_image_vis, (r_x1, r_y1), (r_x2, r_y2), (255, 255, 255), 2) # White ROI box

        # Masked Depth Map Visualization
        masked_depth_vis = None
        if self.masked_depth_mm is not None:
            # Create a version for visualization where masked area is distinctly visible (e.g., black)
            vis_masked = self.masked_depth_mm.copy()
            vis_masked[vis_masked == 65535.0] = 0 # Set masked area to 0 for visualization
            masked_depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(vis_masked, alpha=0.05),
                cv2.COLORMAP_JET
            )
            # Make the actual masked area black on the color map for clarity
            masked_depth_vis[self.masked_depth_mm == 65535.0] = [0, 0, 0] # Black color

            # Draw ROI on masked depth vis
            if self.obstacle_roi_coords:
                 r_x1, r_y1, r_x2, r_y2 = self.obstacle_roi_coords
                 cv2.rectangle(masked_depth_vis, (r_x1, r_y1), (r_x2, r_y2), (255, 255, 255), 2) # White ROI box

            # Optional: Draw the dilated box used for masking (e.g., in red) for debugging
            # if self.dilated_mask_coords:
            #     dx1, dy1, dx2, dy2 = self.dilated_mask_coords
            #     cv2.rectangle(masked_depth_vis, (dx1, dy1), (dx2, dy2), (0, 0, 255), 1) # Red thin box


        return depth_image_vis, masked_depth_vis

