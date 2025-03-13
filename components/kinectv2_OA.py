import cv2
import numpy as np
import time

class OA:
    def __init__(self, min_distance=0.2, max_distance=2.0):
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.depth_map = None
        self.color_image = None
        self.depth_output = None

    # Function to process depth map and detect obstacles using edge detection and depth information
    def process_depth_map(self):
        if self.depth_map is not None:
            # Preprocess depth map
            valid_depth = (self.depth_map >= self.min_distance) & (self.depth_map <= self.max_distance)
            depth_map_filtered = np.where(valid_depth, self.depth_map, 0)

            # Extract region of interest (ROI)
            roi = depth_map_filtered  # Use the entire depth map

            # Edge Detection
            edges = cv2.Canny((roi * 255 / self.max_distance).astype('uint8'), 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Visualize Obstacles with Depth Information
            self.depth_output = cv2.cvtColor((roi / self.max_distance * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
            for contour in contours:
                cv2.drawContours(self.depth_output, [contour], -1, (0, 255, 0), 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
            time.sleep(1/24)  # 24 FPS

    # Function to process color image
    def process_color_image(self):
        if self.color_image is not None:
            # Process color image if needed
            pass
