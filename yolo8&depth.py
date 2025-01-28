import cv2
import numpy as np
from openni import openni2
from ultralytics import YOLO

# Initialize OpenNI
openni2.initialize()
device = openni2.Device.open_any()

# Create depth and color streams
depth_stream = device.create_depth_stream()
color_stream = device.create_color_stream()
depth_stream.start()
color_stream.start()

# Load YOLOv8 model
model = YOLO('yolov8n.pt')
model.overrides['classes'] = [0]  # Detect only 'person'

try:
    while True:
        # Depth Stream
        depth_frame = depth_stream.read_frame()
        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16)
        depth_data = depth_data.reshape((depth_stream.get_video_mode().resolutionY, depth_stream.get_video_mode().resolutionX))

        # Color Stream
        color_frame = color_stream.read_frame()
        color_data = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8)
        color_data = color_data.reshape((color_stream.get_video_mode().resolutionY, color_stream.get_video_mode().resolutionX, 3))
        frame_bgr = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)

        # Run YOLOv8 inference
        results = model.predict(frame_bgr, imgsz=640, conf=0.5)
        if results and results[0].boxes:  # Check if there are detections
            for box in results[0].boxes.data.tolist():
                x1, y1, x2, y2, confidence, cls = map(int, box[:6])  # Extract bounding box info

                # Calculate depth at the center of the bounding box
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                if 0 <= center_x < depth_stream.get_video_mode().resolutionX and 0 <= center_y < depth_stream.get_video_mode().resolutionY:
                    person_distance = depth_data[center_y, center_x] / 1000.0  # Convert to meters

                    # Create a combined label for the person
                    label = f"Person: {confidence:.2f}, {person_distance:.2f}m"

                    # Draw bounding box and overlay label
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                    cv2.putText(frame_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    print(f"Person detected at ({center_x}, {center_y}), Distance: {person_distance:.2f} meters")

        # Display the frames
        cv2.imshow("YOLOv8 - Person Detection with Distance", frame_bgr)
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=0.05), cv2.COLORMAP_JET)
        cv2.imshow("Depth Image", depth_image)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streams and unload OpenNI
    depth_stream.stop()
    color_stream.stop()
    openni2.unload()