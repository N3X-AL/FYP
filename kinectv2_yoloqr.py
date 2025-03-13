import cv2
import numpy as np
from pylibfreenect2 import Freenect2, FrameType, SyncMultiFrameListener
from ultralytics import YOLO

class KinectUtils:
    # COCO class indices
    CLASS_NAMES = {0: 'person', 56: 'chair', 57: 'bench', 60: 'dining table', 64: 'potted plant'}

    def __init__(self):
        # Initialize Kinect v2
        self.fn = Freenect2()
        if self.fn.enumerateDevices() == 0:
            print("No Kinect v2 device found!")
            exit()

        serial = self.fn.getDeviceSerialNumber(0)
        self.device = self.fn.openDevice(serial)

        # Create a listener for RGB and Depth frames
        self.listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)

        # Register listeners
        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)
        self.device.start()

        # Load YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        self.model.overrides['classes'] = [0, 56, 57, 60, 64]  # Select specific classes

        # QR Code Detector
        self.qr_detector = cv2.QRCodeDetector()

    def get_kinect_frames(self):
        frames = self.listener.waitForNewFrame()
        color_frame = frames[FrameType.Color]
        depth_frame = frames[FrameType.Depth]

        # Convert color frame
        color_img = color_frame.asarray()
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

        # Convert depth frame
        depth_img = depth_frame.asarray()
        
        self.listener.release(frames)
        return color_img, depth_img

    def interpolate_depth_to_color(self, depth_data, color_res):
        return cv2.resize(depth_data, (color_res[1], color_res[0]), interpolation=cv2.INTER_NEAREST)

    def move_forward(self):
        print("Motors: Moving forward...")

    def turn_left(self):
        print("Motors: Turning left...")

    def turn_right(self):
        print("Motors: Turning right...")

    def stop_motors(self):
        print("Motors: Stopping...")

    def qr_code_detection(self, target_qr_code, tolerance, frame_center_x):
        while True:
            frame_bgr, depth_data = self.get_kinect_frames()
            if frame_bgr is None or depth_data is None:
                continue
            interpolated_depth_data = self.interpolate_depth_to_color(depth_data, (1080, 1920))
            
            qr_data, qr_bbox, _ = self.qr_detector.detectAndDecode(frame_bgr)
            if qr_bbox is not None and qr_data:
                qr_bbox = qr_bbox.astype(int)
                min_x = np.min(qr_bbox[:,0])
                max_x = np.max(qr_bbox[:,0])
                min_y = np.min(qr_bbox[:,1])
                max_y = np.max(qr_bbox[:,1])
                cv2.rectangle(frame_bgr, (min_x, min_y), (max_x, max_y), (0, 255, 255), 2)
                center_x, center_y = np.mean(qr_bbox, axis=0)[0].astype(int)
                if center_x < frame_center_x - tolerance:
                    self.turn_left()
                elif center_x > frame_center_x + tolerance:
                    self.turn_right()
                else:
                    self.move_forward()
                if 0 <= center_x < 1920 and 0 <= center_y < 1080:
                    object_distance = interpolated_depth_data[center_y, center_x] / 1000.0
                    if object_distance <= 0.5:
                        self.stop_motors()
            else:
                self.stop_motors()
            
            cv2.imshow("QR Code Detection", cv2.resize(frame_bgr, (640, 320)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def yolo_detection(self):
        while True:
            frame_bgr, depth_data = self.get_kinect_frames()
            if frame_bgr is None or depth_data is None:
                continue
            interpolated_depth_data = self.interpolate_depth_to_color(depth_data, (1080, 1920))
            results = self.model.predict(frame_bgr, imgsz=640, conf=0.5)
            
            for box in results[0].boxes.data.tolist():
                x1, y1, x2, y2, confidence, cls = map(int, box[:6])
                class_name = self.CLASS_NAMES.get(cls, "Unknown")
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                if 0 <= center_x < 1920 and 0 <= center_y < 1080:
                    object_distance = interpolated_depth_data[center_y, center_x] / 1000.0
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_bgr, f"{class_name}: {confidence:.2f}, {object_distance:.2f}m", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    print(f"{class_name} at ({center_x}, {center_y}), Distance: {object_distance:.2f} meters")
            
            cv2.imshow("YOLOv8 - Object Detection", cv2.resize(frame_bgr, (640, 320)))
            depth_image = cv2.applyColorMap(cv2.convertScaleAbs(interpolated_depth_data, alpha=0.05), cv2.COLORMAP_JET)
            cv2.imshow("Depth Image", cv2.resize(depth_image, (640, 480)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
