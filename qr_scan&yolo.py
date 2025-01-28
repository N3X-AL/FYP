import cv2
import numpy as np
from openni import openni2
from ultralytics import YOLO

# COCO class indices
CLASS_NAMES = {0: 'person', 56: 'chair', 57: 'bench', 60: 'dining table', 64: 'potted plant'}

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

# Specify classes to detect (person, chair, bench, dining table, potted plant)
model.overrides['classes'] = [0, 56, 57, 60, 64]  # COCO indices for the selected classes

# Initialize QR Code detector
qr_detector = cv2.QRCodeDetector()

# Ask the user to scan the QR code to set the target
print("Please hold the QR code in front of the camera to register the target QR code.")
target_qr_code = None

while target_qr_code is None:
    # Read a frame from the color stream
    color_frame = color_stream.read_frame()
    color_data = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8)
    color_data = color_data.reshape((color_stream.get_video_mode().resolutionY, color_stream.get_video_mode().resolutionX, 3))
    frame_bgr = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)

    # Detect and decode QR code
    qr_data, _, _ = qr_detector.detectAndDecode(frame_bgr)
    if qr_data:
        target_qr_code = qr_data
        print(f"Target QR code registered: {target_qr_code}")
        break

    # Display the camera feed
    cv2.imshow("Register QR Code", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("QR code registration aborted.")
        break

try:
    while target_qr_code:
        # Depth Stream
        depth_frame = depth_stream.read_frame()
        depth_data = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16)
        depth_data = depth_data.reshape((depth_stream.get_video_mode().resolutionY, depth_stream.get_video_mode().resolutionX))

        # Color Stream
        color_frame = color_stream.read_frame()
        color_data = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8)
        color_data = color_data.reshape((color_stream.get_video_mode().resolutionY, color_stream.get_video_mode().resolutionX, 3))
        frame_bgr = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)

        # Detect and decode QR code
        qr_data, qr_bbox, _ = qr_detector.detectAndDecode(frame_bgr)

        if qr_data:
            if qr_data == target_qr_code:
                print("Following this QR code...")
                if qr_bbox is not None:
                    # Draw the bounding box around the detected QR code
                    qr_bbox = qr_bbox.astype(int)
                    for i in range(len(qr_bbox)):
                        cv2.line(frame_bgr, tuple(qr_bbox[i][0]), tuple(qr_bbox[(i + 1) % len(qr_bbox)][0]), (0, 255, 0), 2)
            else:
                print("New QR code detected, not going to follow that.")
                print("Finding original QR code...")
        else:
            print("No QR code detected, searching for the original QR code...")

        # Run YOLOv8 inference
        results = model.predict(frame_bgr, imgsz=640, conf=0.5)
        if results and results[0].boxes:  # Check if there are detections
            for box in results[0].boxes.data.tolist():
                x1, y1, x2, y2, confidence, cls = map(int, box[:6])  # Extract bounding box info

                # Get the class name
                class_name = CLASS_NAMES.get(cls, "Unknown")

                # Calculate depth at the center of the bounding box
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                if 0 <= center_x < depth_stream.get_video_mode().resolutionX and 0 <= center_y < depth_stream.get_video_mode().resolutionY:
                    object_distance = depth_data[center_y, center_x] / 1000.0  # Convert to meters

                    # Create a label with the class name and distance
                    label = f"{class_name}: {confidence:.2f}, {object_distance:.2f}m"

                    # Draw bounding box and overlay label
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                    cv2.putText(frame_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frames
        cv2.imshow("YOLOv8 - Object Detection with Distance", frame_bgr)
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
    cv2.destroyAllWindows()