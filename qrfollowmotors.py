import cv2
import numpy as np
from openni import openni2
from ultralytics import YOLO
#import time

# COCO class indices
CLASS_NAMES = {0: 'person', 56: 'chair', 57: 'bench', 60: 'dining table', 64: 'potted plant'}

# Initialize OpenNI
openni2.initialize()
device = openni2.Device.open_any()

# Create depth and color streams
depth_stream = device.create_depth_stream()
color_stream = device.create_color_stream()

# Set the resolution to 320x240
depth_stream.set_video_mode(openni2.VideoMode(openni2.PIXEL_FORMAT_DEPTH_1_MM, 320, 240, 30))
color_stream.set_video_mode(openni2.VideoMode(openni2.PIXEL_FORMAT_RGB888, 320, 240, 30))

depth_stream.start()
color_stream.start()

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Specify classes to detect (person, chair, bench, dining table, potted plant)
model.overrides['classes'] = [0, 56, 57, 60, 64]  # COCO indices for the selected classes

# Initialize QR Code detector
qr_detector = cv2.QRCodeDetector()

# Motor control functions (replace with actual commands for your hardware)
def move_forward():
    print("Motors: Moving forward...")

def turn_left():
    print("Motors: Turning left...")

def turn_right():
    print("Motors: Turning right...")

def stop_motors():
    print("Motors: Stopping...")

# Prompt the user to scan the QR code to set the target
print("Please hold the QR code in front of the camera to register the target QR code.")
target_qr_code = None

while target_qr_code is None:
    # Read a frame from the color stream
    color_frame = color_stream.read_frame()
    color_data = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8)
    color_data = color_data.reshape((color_stream.get_video_mode().resolutionY, color_stream.get_video_mode().resolutionX, 3))
    frame_bgr = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)

    # Upscale the frame to the original resolution
    frame_bgr = cv2.resize(frame_bgr, (640, 480))

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
tolerance=50
try:
    qr_lost = False  # Flag to handle QR code loss
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

        # Upscale the frame to the original resolution
        frame_bgr = cv2.resize(frame_bgr, (640, 480))

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

                    print(f"{class_name} detected at ({center_x}, {center_y}), Distance: {object_distance:.2f} meters")

        # Function to calculate QR code center based on bounding box
        def calculate_qr_center_from_bbox(qr_bbox):
            """
            Calculate the center of the QR code based on bounding box.
            :param qr_bbox: Bounding box of the QR code
            :return: (center_x, center_y) coordinates of the QR code center
            """
            if qr_bbox is not None and len(qr_bbox) == 4:
                center_x = int((qr_bbox[0][0][0] + qr_bbox[1][0][0] + qr_bbox[2][0][0] + qr_bbox[3][0][0]) / 4)
                center_y = int((qr_bbox[0][0][1] + qr_bbox[1][0][1] + qr_bbox[2][0][1] + qr_bbox[3][0][1]) / 4)
                return center_x, center_y
            return None, None

        # Function to calculate QR code center using contours
        def calculate_qr_center_using_contours(frame_bgr):
            """
            Calculate the center of the QR code using contours.
            :param frame_bgr: The frame containing the QR code
            :return: (center_x, center_y) coordinates of the QR code center
            """
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 150)

            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:
                    M = cv2.moments(approx)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        return center_x, center_y
            return None, None

        # Detect and decode QR code
        qr_data, qr_bbox, _ = qr_detector.detectAndDecode(frame_bgr)
        if qr_bbox is not None and qr_data:
            # Draw bounding box for QR code
            qr_bbox = qr_bbox.astype(int)
            for i in range(len(qr_bbox)):
                start_point = tuple(qr_bbox[i][0])
                end_point = tuple(qr_bbox[(i + 1) % len(qr_bbox)][0])
                cv2.line(frame_bgr, start_point, end_point, (0, 255, 255), 2)

            # Display the QR code data on the frame
            cv2.putText(frame_bgr, f"QR: {qr_data}", (qr_bbox[0][0][0], qr_bbox[0][0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            print(f"QR Code Detected: {qr_data}")

            # Calculate QR code position using contours
            center_x, center_y = calculate_qr_center_using_contours(frame_bgr)
            frame_center_x = frame_bgr.shape[1] // 2

            if center_x is not None and center_y is not None:
                # Adjust motor commands based on QR code position
                if center_x < frame_center_x - tolerance:  # QR code is to the left
                    print("QR code detected: Adjusting left...")
                    turn_left()
                elif center_x > frame_center_x + tolerance:  # QR code is to the right
                    print("QR code detected: Adjusting right...")
                    turn_right()
                else:  # QR code is centered
                    print("QR code is centered. Moving forward.")
                    move_forward()
            else:
                print("QR code center calculation failed.")
                stop_motors()
        else:
            print("QR code detection failed: No QR code detected or bounding box is incomplete.")
            stop_motors()

        cv2.imshow("YOLOv8 - Object Detection with Distance", frame_bgr)
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=0.05), cv2.COLORMAP_JET)
        depth_image = cv2.resize(depth_image, (640, 480))  # Upscale depth image
        cv2.imshow("Depth Image", depth_image)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streams and unload OpenNI
    stop_motors()
    depth_stream.stop()
    color_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()