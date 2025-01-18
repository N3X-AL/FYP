import numpy as np
import cv2
from openni import openni2
import threading
from pyzbar.pyzbar import decode  # For QR code detection

# Initialize OpenNI2
openni2.initialize()

# Open the default device
device = openni2.Device.open_any()

# Create streams for depth and color
depth_stream = device.create_depth_stream()
color_stream = device.create_color_stream()

# Set video modes for depth and color (reduced resolution for better performance)
depth_stream.set_video_mode(
    openni2.VideoMode(pixelFormat=openni2.PIXEL_FORMAT_DEPTH_1_MM, resolutionX=320, resolutionY=240, fps=30))
color_stream.set_video_mode(
    openni2.VideoMode(pixelFormat=openni2.PIXEL_FORMAT_RGB888, resolutionX=320, resolutionY=240, fps=30))

# Start streams
depth_stream.start()
color_stream.start()

# Parameters for obstacle detection
min_distance = 0.2  # Minimum depth in meters
max_distance = 3.0  # Maximum depth in meters
min_cluster_size = 50  # Minimum pixel count to consider a valid obstacle (reduced for lower resolution)
region_of_interest = (25, 215, 60, 260)  # ROI: y_min, y_max, x_min, x_max (adjusted for lower resolution)

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
    return depth_array / 1000.0  # Convert from mm to meters


# Function to retrieve the color image
def get_color_image():
    try:
        frame = color_stream.read_frame()
        frame_data = frame.get_buffer_as_uint8()
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)  # Convert ctypes array to NumPy array
        color_image = frame_array.reshape((240, 320, 3))  # RGB format
        return color_image
    except Exception as e:
        print(f"Error retrieving color image: {e}")
        return np.zeros((240, 320, 3), dtype=np.uint8)  # Return a black image if an error occurs


# Function to colorize the depth map
def colorize_depth(depth_map, max_depth=3.0):
    normalized = np.clip(depth_map / max_depth, 0, 1)
    return cv2.applyColorMap((normalized * 255).astype('uint8'), cv2.COLORMAP_JET)


# Function to detect QR codes
def detect_qr_code(frame):
    global original_qr_code, tracking_qr_code
    decoded_objects = decode(frame)

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

    return frame


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
        for cluster in obstacle_clusters:
            x, y, w, h = cluster['bbox']
            cx, cy = cluster['centroid']
            mean_depth = cluster['mean_depth']
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, f"{mean_depth:.2f}m", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Generate Cost Map
        cost_map = np.zeros_like(roi, dtype='uint8')
        for cluster in obstacle_clusters:
            cost_map[labels == cluster['label']] = 1


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

        # Show the color image
        cv2.imshow("QR Code Detection", color_image)

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
    while True:
        if depth_map is not None and color_image is not None:
            # Display Results
            colorized_depth = colorize_depth(depth_map)

            # Draw bounding boxes on the color image
            color_image_with_boxes = color_image.copy()
            for cluster in obstacle_clusters:
                x, y, w, h = cluster['bbox']
                cx, cy = cluster['centroid']
                mean_depth = cluster['mean_depth']

                # Draw rectangle and depth information on the color image
                cv2.rectangle(color_image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(color_image_with_boxes, f"{mean_depth:.2f}m",
                            (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Show results
            cv2.imshow("Colorized Depth Map", colorized_depth)
            cv2.imshow("Obstacle Detection", output)
            cv2.imshow("RGB Stream with Obstacles", color_image_with_boxes)
            cv2.imshow("Local Cost Map", cost_map * 255)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    depth_stream.stop()
    color_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()
