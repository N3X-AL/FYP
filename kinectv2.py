from pylibfreenect2 import Freenect2, Freenect2Device, FrameType, SyncMultiFrameListener
import cv2
import numpy as np
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

fn = Freenect2()
if fn.enumerateDevices() == 0:
    print("No Kinect v2 device found!")
    exit(1)

device = fn.openDefaultDevice()

listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)
device.start()

while True:
    frames = listener.waitForNewFrame()

    # Get color frame
    color = frames["color"].asarray()
    color = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)

    # Get depth frame
    depth = frames["depth"].asarray()

    cv2.imshow("Color", color)
    cv2.imshow("Depth", depth / np.max(depth))  # Normalize for display

    listener.release(frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

device.stop()
device.close()
cv2.destroyAllWindows()

