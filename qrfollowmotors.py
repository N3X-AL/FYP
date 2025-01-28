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

        # Detect and decode QR code
        retval, qr_bbox, _ = qr_detector.detectAndDecode(frame_bgr)

        if not retval:
            print("QR code detection failed: No QR code detected.")
            stop_motors()
        elif qr_bbox is None or len(qr_bbox) < 4:
            print("QR code detection failed: Bounding box is None or incomplete.")
            stop_motors()
        else:
            print(f"QR code detected with bounding box: {qr_bbox}")

            # Calculate QR code position
            qr_bbox = qr_bbox.astype(int)
            center_x = int((qr_bbox[0][0][0] + qr_bbox[2][0][0]) / 2)
            center_y = int((qr_bbox[0][0][1] + qr_bbox[2][0][1]) / 2)
            frame_center_x = frame_bgr.shape[1] // 2

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

        '''retval,qr_data, qr_bbox, _ = qr_detector.detectAndDecodeMulti(frame_bgr)

        if qr_data:
            if qr_data == target_qr_code:
                qr_lost = False
                print("Following the target QR code...")

                if qr_bbox is not None:
                    # Calculate QR code position
                    qr_bbox = qr_bbox.astype(int)
                    center_x = int((qr_bbox[0][0][0] + qr_bbox[2][0][0]) / 2)
                    center_y = int((qr_bbox[0][0][1] + qr_bbox[2][0][1]) / 2)
                    frame_center_x = frame_bgr.shape[1] // 2
                    print('centre x:', center_x,'centre y:',center_y)
                    # Adjust motor commands based on QR code position
                    tolerance = 10  # Define tolerance for QR code centering
                    if center_x < frame_center_x - tolerance:  # QR code is to the left
                        print("QR code detected: Adjusting left...")
                        turn_left()
                    elif center_x > frame_center_x + tolerance:  # QR code is to the right
                        print("QR code detected: Adjusting right...")
                        turn_right()
                    else:  # QR code is centered
                        print("QR code is centered. Moving forward.")
                        move_forward()

                    # Draw the bounding box and mark the QR code center
                    for i in range(len(qr_bbox)):
                        cv2.line(frame_bgr, tuple(qr_bbox[i][0]), tuple(qr_bbox[(i + 1) % len(qr_bbox)][0]), (0, 255, 0), 2)
                    cv2.circle(frame_bgr, (center_x, center_y), 5, (0, 0, 255), -1)

                    # Calculate distance to the QR code
                    qr_distance = depth_data[center_y, center_x] / 1000.0  # Convert to meters
                    print(f"Distance to QR code: {qr_distance:.2f} meters")

                    # Stop if the QR code is too close
                    if qr_distance < 0.5:
                        print("QR code is too close! Stopping the trolley.")
                        stop_motors()

                last_time_seen = time.time()

            else:
                print("New QR code detected, not following it.")
        else:
            if not qr_lost:
                qr_lost = True
                print("Lost sight of the QR code! Searching...")
                turn_left()
                cv2.waitKey(500)  # Adjust timing as needed

        # Display the frames
        cv2.imshow("QR Code Tracking", frame_bgr)'''
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