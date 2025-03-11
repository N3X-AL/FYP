import cv2
import numpy as np
from pylibfreenect2 import Freenect2, Freenect2Device, FrameType, SyncMultiFrameListener, Frame
from ultralytics import YOLO

# COCO class indices
CLASS_NAMES = {0: 'person', 56: 'chair', 57: 'bench', 60: 'dining table', 64: 'potted plant'}

# Initialize Kinect v2
fn = Freenect2()
if fn.enumerateDevices() == 0:
    print("No Kinect v2 device found!")
    exit()

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial)

# Create a listener for RGB and Depth frames
listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)
device.start()

# Load YOLOv8 model
model = YOLO('yolov8n.pt')
model.overrides['classes'] = [0, 56, 57, 60, 64]  # Select specific classes

# QR Code Detector
qr_detector = cv2.QRCodeDetector()

def get_kinect_frames():
    frames = listener.waitForNewFrame()
    color_frame = frames[FrameType.Color]
    depth_frame = frames[FrameType.Depth]

    # Convert color frame
    color_img = color_frame.asarray()
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

    # Convert depth frame
    depth_img = depth_frame.asarray()
    
    listener.release(frames)
    return color_img, depth_img

def interpolate_depth_to_color(depth_data, color_res):
    return cv2.resize(depth_data, (color_res[1], color_res[0]), interpolation=cv2.INTER_NEAREST)

def move_forward():
    print("Motors: Moving forward...")

def turn_left():
    print("Motors: Turning left...")

def turn_right():
    print("Motors: Turning right...")

def stop_motors():
    print("Motors: Stopping...")

print("Please hold the QR code in front of the camera to register the target QR code.")
target_qr_code = None

while target_qr_code is None:
    frame_bgr, _ = get_kinect_frames()
    qr_data, _, _ = qr_detector.detectAndDecode(frame_bgr)
    if qr_data:
        target_qr_code = qr_data
        print(f"Target QR code registered: {target_qr_code}")
        break
    cv2.imshow("Register QR Code", cv2.resize(frame_bgr, (640, 480)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("QR code registration aborted.")
        break

tolerance = 50
frame_center_x = 1920 // 2

try:
    while True:
        frame_bgr, depth_data = get_kinect_frames()
        if frame_bgr is None or depth_data is None:
            continue
        interpolated_depth_data = interpolate_depth_to_color(depth_data, (1080, 1920))
        results = model.predict(frame_bgr, imgsz=640, conf=0.5)
        #bounding box for yolo
        
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, cls = map(int, box[:6])
            class_name = CLASS_NAMES.get(cls, "Unknown")
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            if 0 <= center_x < 1920 and 0 <= center_y < 1080:
                object_distance = interpolated_depth_data[center_y, center_x] / 1000.0
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"{class_name}: {confidence:.2f}, {object_distance:.2f}m", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"{class_name} at ({center_x}, {center_y}), Distance: {object_distance:.2f} meters")
        
        #bounding for qr
        qr_data, qr_bbox, _ = qr_detector.detectAndDecode(frame_bgr)
        if qr_bbox is not None and qr_data:
            qr_bbox = qr_bbox.astype(int)
            # Calculate the centroid of the QR code bounding box
            center_x, center_y = np.mean(qr_bbox, axis=(0, 1)).astype(int)

            # Draw the bounding box
            min_x = np.min(qr_bbox[:, :, 0])
            max_x = np.max(qr_bbox[:, :, 0])
            min_y = np.min(qr_bbox[:, :, 1])
            max_y = np.max(qr_bbox[:, :, 1])
            cv2.rectangle(frame_bgr, (min_x, min_y), (max_x, max_y), (0, 255, 255), 2)
            
            # Add QR code data to the display
            cv2.putText(frame_bgr, f"QR: {qr_data}", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if center_x < frame_center_x - tolerance:
                turn_left()
            elif center_x > frame_center_x + tolerance:
                turn_right()
            else:
                move_forward()
            
            if 0 <= center_x < 1920 and 0 <= center_y < 1080:
                object_distance = interpolated_depth_data[center_y, center_x] / 1000.0
                if object_distance <= 0.5:
                    stop_motors()
        else:
            stop_motors()
        
        cv2.imshow("YOLOv8 - Object Detection", cv2.resize(frame_bgr, (640, 320)))
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(interpolated_depth_data, alpha=0.05), cv2.COLORMAP_JET)
        cv2.imshow("Depth Image", cv2.resize(depth_image, (640, 480)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    stop_motors()
    device.stop()
    device.close()
    cv2.destroyAllWindows()
