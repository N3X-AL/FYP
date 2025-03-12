from pylibfreenect2 import Freenect2, FrameType, SyncMultiFrameListener
import cv2
import threading
from components.kinectv2_functions import Kinect

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

# Initialize Kinect
kinect = Kinect()

# Main loop for retrieving frames and processing them
try:
    while True:  # Run indefinitely until 'q' is pressed
        frames = listener.waitForNewFrame()
        kinect.depth_map = frames["depth"].asarray() / 1000.0  # Convert from mm to meters
        kinect.color_image = cv2.cvtColor(frames["color"].asarray(), cv2.COLOR_BGRA2BGR)
        listener.release(frames)

        # Start depth map and color image processing in separate threads
        depth_thread = threading.Thread(target=kinect.process_depth_map)
        depth_thread.start()

        color_thread = threading.Thread(target=kinect.process_color_image)
        color_thread.start()

        depth_thread.join()
        color_thread.join()

        if kinect.depth_output is not None and kinect.color_image is not None:
            # Display Results upscaled
            color_image_upscaled = cv2.resize(kinect.color_image, (640, 360), interpolation=cv2.INTER_LINEAR)

            # Show results
            cv2.imshow("Obstacle Detection", kinect.depth_output)
            cv2.imshow("RGB Stream", color_image_upscaled)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    device.stop()
    device.close()
    cv2.destroyAllWindows()

