import numpy as np
import cv2
import matplotlib.pyplot as plt
from openni import openni2
import threading
import time
from pyzbar.pyzbar import decode  # For QR code detection

# Initialize OpenNI2
openni2.initialize()

# Open the default device
device = openni2.Device.open_any()

# Create streams for depth and color
depth_stream = device.create_depth_stream()
color_stream = device.create_color_stream()

# Set video modes for depth and color (reduced resolution for better performance)
depth_stream.set_video_mode(openni2.VideoMode(pixelFormat=openni2.PIXEL_FORMAT_DEPTH_1_MM, resolutionX=320, resolutionY=240, fps=30))
color_stream.set_video_mode(openni2.VideoMode(pixelFormat=openni2.PIXEL_FORMAT_RGB888, resolutionX=320, resolutionY=240, fps=30))

# Start streams
depth_stream.start()
color_stream.start()

# Parameters for obstacle detection
min_distance = 0.2  # Minimum depth in meters
max_distance = 2.0  # Maximum depth in meters
region_of_interest = (10, 230, 10, 310)  # ROI: y_min, y_max, x_min, x_max (adjusted for lower resolution)

# Shared variables
depth_map = None
color_image = None
output = None

# Function to retrieve the depth map
def get_depth_map():
    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()
    depth_array = np.frombuffer(frame_data, dtype=np.uint16).reshape((240, 320))
    depth_array = np.flip(depth_array, axis=1)  # Flip across y-axis
    return depth_array / 1000.0  # Convert from mm to meters

# Function to retrieve the color image
def get_color_image():
    frame = color_stream.read_frame()
    frame_data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8)
    color_image = frame_data.reshape((240, 320, 3))  # RGB format
    color_image = np.flip(color_image, axis=1)  # Flip across y-axis
    return cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)  # Convert to BGR format for OpenCV

# Function to process depth map and detect obstacles using edge detection and depth information
def process_depth_map():
    global depth_map, output
    while True:
        depth_map = get_depth_map()

        # Preprocess depth map
        valid_depth = (depth_map >= min_distance) & (depth_map <= max_distance)
        depth_map_filtered = np.where(valid_depth, depth_map, 0)

        # Extract region of interest (ROI)
        y_min, y_max, x_min, x_max = region_of_interest
        roi = depth_map_filtered[y_min:y_max, x_min:x_max]

        # Edge Detection
        edges = cv2.Canny((roi * 255 / max_distance).astype('uint8'), 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Visualize Obstacles with Depth Information
        output = cv2.cvtColor((roi / max_distance * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
        for contour in contours:
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                mean_depth = np.mean(roi[cy, cx])
                cv2.putText(output, f"{mean_depth:.2f}m", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Draw bounding box with depth information
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(output, f"Depth: {mean_depth:.2f}m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Function to process color image
def process_color_image():
    global color_image
    while True:
        color_image = get_color_image()

        # Upscale the color image for display
        color_image_upscaled = cv2.resize(color_image, (640, 480), interpolation=cv2.INTER_LINEAR)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start depth map and color image processing in separate threads
depth_thread = threading.Thread(target=process_depth_map)
depth_thread.daemon = True
depth_thread.start()

color_thread = threading.Thread(target=process_color_image)
color_thread.daemon = True
color_thread.start()

# Main loop for displaying results
try:
    while True:  # Run indefinitely until 'q' is pressed
        if depth_map is not None and color_image is not None:
            # Display Results
            output_resized = cv2.resize(output, (640, 480), interpolation=cv2.INTER_LINEAR)

            # Show results
            cv2.imshow("Obstacle Detection", output_resized)
            cv2.imshow("RGB Stream", color_image)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    depth_stream.stop()
    color_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()
