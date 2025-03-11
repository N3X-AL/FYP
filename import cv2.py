import cv2
import numpy as np
from pylibfreenect2 import Freenect2, Freenect2Device, SyncMultiFrameListener, FrameType
from pylibfreenect2 import Registration, Frame
from ultralytics import YOLO

# COCO class indices
CLASS_NAMES = {0: 'person', 56: 'chair', 57: 'bench', 60: 'dining table', 64: 'potted plant'}

# Initialize Freenect2
fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No Kinect devices found!")
    exit(1)

device = fn.openDefaultDevice()

# Set up listeners
listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)

device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)
device.start()

# Registration object to align depth with color
registration = Registration(device.getIrCameraParams(), device.getColorCameraParams())

# Load YOLOv8 model
model = YOLO('yolov8n.pt')
model.overrides['classes'] = [0, 56, 57, 60, 64]  # Specify classes to detect

# Initialize QR Code detector
qr_detector = cv2.QRCodeDetector()

def move_forward():
    print("Motors: Moving forward...")

def turn_left():
    print("Motors: Turning left...")

def turn_right():
    print("Motors: Turning right...")

def stop_motors():
    print("Motors: Stopping...")

target_qr_code = None

print("Please hold the QR code in front of the camera to register the target QR code.")
while target_qr_code is None:
    frames = listener.waitForNewFrame()
    color_frame = frames[FrameType.Color]
    color_data = color_frame.asarray()
    frame_bgr = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)
    
    qr_data, _, _ = qr_detector.detectAndDecode(frame_bgr)
    if qr_data:
        target_qr_code = qr_data
        print(f"Target QR code registered: {target_qr_code}")
        cv2.destroyAllWindows()
        break
    
    small = cv2.resize(frame_bgr, (640, 480))
    cv2.imshow("Register QR Code", small)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("QR code registration aborted.")
        break
    
    listener.release(frames)

tolerance = 50

try:
    while True:
        frames = listener.waitForNewFrame()
        color_frame = frames[FrameType.Color]
        depth_frame = frames[FrameType.Depth]
        
        color_data = color_frame.asarray()
        depth_data = depth_frame.asarray()
        
        frame_bgr = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)
        depth_data = cv2.resize(depth_data, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        frame_center_x = frame_bgr.shape[1] // 2
        
        results = model.predict(frame_bgr, imgsz=640, conf=0.5)
        
        if results and results[0].boxes:
            for box in results[0].boxes.data.tolist():
                x1, y1, x2, y2, confidence, cls = map(int, box[:6])
                class_name = CLASS_NAMES.get(cls, "Unknown")
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                if 0 <= center_x < frame_bgr.shape[1] and 0 <= center_y < frame_bgr.shape[0]:
                    object_distance = depth_data[center_y, center_x] / 1000.0
                    label = f"{class_name}: {confidence:.2f}, {object_distance:.2f}m"
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    print(f"{class_name} detected at ({center_x}, {center_y}), Distance: {object_distance:.2f} meters")
        
        qr_data, qr_bbox, _ = qr_detector.detectAndDecode(frame_bgr)
        if qr_bbox is not None and qr_data:
            qr_bbox = qr_bbox.astype(int)
            for i in range(len(qr_bbox)):
                start_point = tuple(qr_bbox[i][0])
                end_point = tuple(qr_bbox[(i + 1) % len(qr_bbox)][0])
                cv2.line(frame_bgr, start_point, end_point, (0, 255, 255), 2)
            
            center_x, center_y = np.mean(qr_bbox[:, 0, :], axis=0).astype(int)
            if center_x < frame_center_x - tolerance:
                print("QR code detected: Adjusting left...")
                turn_left()
            elif center_x > frame_center_x + tolerance:
                print("QR code detected: Adjusting right...")
                turn_right()
            else:
                print("QR code is centered. Moving forward.")
                move_forward()
            
            object_distance = depth_data[center_y, center_x] / 1000.0
            if object_distance <= 0.5:
                print("QR code is within .5 meters. Stopping motors.")
                stop_motors()
        else:
            print("QR code detection failed.")
            stop_motors()
        
        display = cv2.resize(frame_bgr, (640, 480))
        cv2.imshow("YOLOv8 - Object Detection with Distance", display)
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=0.05), cv2.COLORMAP_JET)
        depth_image = cv2.resize(depth_image, (640, 480))
        cv2.imshow("Depth Image", depth_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        listener.release(frames)

finally:
    stop_motors()
    device.stop()
    device.close()
    cv2.destroyAllWindows()
