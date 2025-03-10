from pylibfreenect2 import Freenect2, Freenect2Device, FrameType, SyncMultiFrameListener
import cv2
import numpy as np
import threading
import time

# Initialize Kinect v2
fn = Freenect2()
if fn.enumerateDevices() == 0:
    print("No Kinect v2 device found!")
    exit(1)

device = fn.openDefaultDevice()
listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)
device.start()

# Parameters for obstacle detection
min_distance = 0.2  # Minimum depth in meters
max_distance = 2.0  # Maximum depth in meters
# region_of_interest = (10, 230, 10, 310)  # ROI: y_min, y_max, x_min, x_max (adjusted for lower resolution)

# Shared variables
depth_map = None
color_image = None
depth_output = None

# Function to process depth map and detect obstacles using edge detection and depth information
def process_depth_map():
    global depth_map, depth_output
    if depth_map is not None:
        # Preprocess depth map
        valid_depth = (depth_map >= min_distance) & (depth_map <= max_distance)
        depth_map_filtered = np.where(valid_depth, depth_map, 0)

        # Extract region of interest (ROI)
        # y_min, y_max, x_min, x_max = region_of_interest
        # roi = depth_map_filtered[y_min:y_max, x_min:x_max]
        roi = depth_map_filtered  # Use the entire depth map

        # Edge Detection
        edges = cv2.Canny((roi * 255 / max_distance).astype('uint8'), 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Visualize Obstacles with Depth Information
        depth_output = cv2.cvtColor((roi / max_distance * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
        for contour in contours:
            cv2.drawContours(depth_output, [contour], -1, (0, 255, 0), 2)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
        time.sleep(1/20)  # 20 FPS

# Function to process color image
def process_color_image():
    global color_image
    if color_image is not None:
        # Process color image if needed
        pass

# Main loop for retrieving frames and processing them
try:
    while True:  # Run indefinitely until 'q' is pressed
        frames = listener.waitForNewFrame()
        depth_map = frames["depth"].asarray() / 1000.0  # Convert from mm to meters
        color_image = cv2.cvtColor(frames["color"].asarray(), cv2.COLOR_BGRA2BGR)
        listener.release(frames)

        # Start depth map and color image processing in separate threads
        depth_thread = threading.Thread(target=process_depth_map)
        depth_thread.start()

        color_thread = threading.Thread(target=process_color_image)
        color_thread.start()

        depth_thread.join()
        color_thread.join()

        if depth_map is not None and color_image is not None:
            # Display Results upscaled
            color_image_upscaled = cv2.resize(color_image, (640, 360), interpolation=cv2.INTER_LINEAR)
            depth_output_resized = cv2.resize(depth_output, (640, 480), interpolation=cv2.INTER_LINEAR)

            # Show results
            cv2.imshow("Obstacle Detection", depth_output)
            cv2.imshow("RGB Stream", color_image_upscaled)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    device.stop()
    device.close()
    cv2.destroyAllWindows()

