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
min_cluster_size = 50  # Minimum pixel count to consider a valid obstacle (reduced for lower resolution)
region_of_interest = (10, 230, 10, 310)  # ROI: y_min, y_max, x_min, x_max (adjusted for lower resolution)

# Shared variables
depth_map = None
color_image = None
obstacle_clusters = []
cost_map = None
output = None

# QR Code Variables
original_qr_code = None
tracking_qr_code = False

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

# Function to colorize the depth map
def colorize_depth(depth_map, max_depth=3.0):
    normalized = np.clip(depth_map / max_depth, 0, 1)
    return cv2.applyColorMap((normalized * 255).astype('uint8'), cv2.COLORMAP_JET)

# Function to detect QR codes
def detect_qr_code(frame):
    global original_qr_code, tracking_qr_code
    decoded_objects = decode(frame)

    if not decoded_objects:
        if tracking_qr_code:
            print("Warning: Looking for the QR Code...")
        tracking_qr_code = False
    else:
        for obj in decoded_objects:
            qr_data = obj.data.decode('utf-8')
            if original_qr_code is None:  # Initial QR code scan
                original_qr_code = qr_data
                tracking_qr_code = True
                print(f"Following QR Code: {original_qr_code}")
            elif qr_data != original_qr_code:
                print(f"New QR Code Detected: {qr_data}. Not following. Finding original QR Code...")
            else:
                print("Following the original QR Code.")
                tracking_qr_code = True

    return frame

# Function to generate a color based on weight
def get_color_for_weight(weight):
    if weight < 0.2:
        return (0, 255, 0)  # Green for low weight
    elif weight < 0.4:
        return (255, 255, 0)  # Yellow for medium-low weight
    elif weight < 0.6:
        return (255, 165, 0)  # Orange for medium weight
    elif weight < 0.8:
        return (255, 69, 0)  # Red-Orange for medium-high weight
    else:
        return (255, 0, 0)  # Red for high weight

# Function to process depth map and detect obstacles
def process_depth_map():
    global depth_map, obstacle_clusters, cost_map, output
    while True:
        depth_map = get_depth_map()

        # Preprocess depth map
        valid_depth = (depth_map >= min_distance) & (depth_map <= max_distance)
        depth_map_filtered = np.where(valid_depth, depth_map, 0)

        # Extract region of interest (ROI)
        y_min, y_max, x_min, x_max = region_of_interest
        roi = depth_map_filtered[y_min:y_max, x_min:x_max]

        # Obstacle Detection
        binary_roi = (roi > 0).astype('uint8')

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_roi, connectivity=8)

        # Filter clusters
        obstacle_clusters = []
        for i in range(1, num_labels):  # Skip the background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_cluster_size:
                obstacle_clusters.append({
                    'label': i,
                    'area': area,
                    'bbox': stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT + 4],
                    'centroid': centroids[i],
                    'mean_depth': np.mean(roi[labels == i])
                })

        # Visualize Obstacles
        output = cv2.cvtColor((roi / max_distance * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                mean_depth = np.mean(roi[labels == labels[cy, cx]])
                cv2.putText(output, f"{mean_depth:.2f}m", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Generate 3D Weighted Cost Map
        cost_map = np.zeros((roi.shape[0], roi.shape[1], 3), dtype='uint8')
        for cluster in obstacle_clusters:
            distance = cluster['mean_depth']
            weight = 1 / (distance + 1e-6)  # Avoid division by zero
            color = get_color_for_weight(weight)
            cost_map[labels == cluster['label']] = color

# Function to process color image
def process_color_image():
    global color_image, tracking_qr_code
    while True:
        color_image = get_color_image()

        # If not tracking, prompt user to scan a QR code
        if not tracking_qr_code:
            cv2.putText(color_image, "Scan a QR Code to Start", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Detect QR code
        color_image = detect_qr_code(color_image)

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
            # Preprocess depth map
            valid_depth = (depth_map >= min_distance) & (depth_map <= max_distance)
            depth_map_filtered = np.where(valid_depth, depth_map, 0)

            # Display Results
            colorized_depth = colorize_depth(depth_map)

            # Draw contours on the color image
            color_image_with_contours = color_image.copy()
            contours, _ = cv2.findContours((depth_map_filtered > 0).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.drawContours(color_image_with_contours, [contour], -1, (0, 255, 0), 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    mean_depth = np.mean(depth_map_filtered[cy, cx])
                    cv2.putText(color_image_with_contours, f"{mean_depth:.2f}m", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Upscale the streams to 640x480
            colorized_depth = cv2.resize(colorized_depth, (640, 480), interpolation=cv2.INTER_LINEAR)
            output = cv2.resize(output, (640, 480), interpolation=cv2.INTER_LINEAR)
            color_image_with_contours = cv2.resize(color_image_with_contours, (640, 480), interpolation=cv2.INTER_LINEAR)
            cost_map_resized = cv2.resize(cost_map, (640, 480), interpolation=cv2.INTER_NEAREST)

            # Show results
            cv2.imshow("Obstacle Detection", output)
            cv2.imshow("RGB Stream with Obstacles", color_image_with_contours)
            cv2.imshow("Local Cost Map", cost_map_resized)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    depth_stream.stop()
    color_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()