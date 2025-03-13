import cv2
import threading
from kinectv2_yoloqr import KinectUtils

# Initialize KinectUtils
kinect_utils = KinectUtils()

print("Please hold the QR code in front of the camera to register the target QR code.")
target_qr_code = None

while target_qr_code is None:
    frame_bgr, _ = kinect_utils.get_kinect_frames()
    qr_data, _, _ = kinect_utils.qr_detector.detectAndDecode(frame_bgr)
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
    qr_thread = threading.Thread(target=kinect_utils.qr_code_detection, args=(target_qr_code, tolerance, frame_center_x))
    yolo_thread = threading.Thread(target=kinect_utils.yolo_detection)
    
    qr_thread.start()
    yolo_thread.start()
    
    qr_thread.join()
    yolo_thread.join()

finally:
    kinect_utils.stop_motors()
    kinect_utils.device.stop()
    kinect_utils.device.close()
    cv2.destroyAllWindows()
