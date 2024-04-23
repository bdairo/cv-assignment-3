import cv2
import depthai as dai
import numpy as np
import time
import os

def create_pipeline():
    """Creates and configures the pipeline for stereo vision and depth estimation."""
    pipeline = dai.Pipeline()

    # Define sources for left and right cameras and rgb camera
    camLeft = pipeline.create(dai.node.MonoCamera)
    camRight = pipeline.create(dai.node.MonoCamera)
    camRgb = pipeline.create(dai.node.ColorCamera)

    camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)

    camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
  

    # Create outputs for the cameras
    xoutLeft = pipeline.create(dai.node.XLinkOut)
    xoutRight = pipeline.create(dai.node.XLinkOut)
    xoutRgb = pipeline.create(dai.node.XLinkOut)

    xoutLeft.setStreamName("left")
    xoutRight.setStreamName("right")
    xoutRgb.setStreamName("rgb")

    camLeft.out.link(xoutLeft.input)
    camRight.out.link(xoutRight.input)
    camRgb.preview.link(xoutRgb.input)

    # Create stereo depth node
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setConfidenceThreshold(200)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)
    camLeft.out.link(stereo.left)
    camRight.out.link(stereo.right)

    # Output depth
    xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutDepth.setStreamName("depth")
    stereo.depth.link(xoutDepth.input)

    return pipeline

def main():
    pipeline = create_pipeline()
    with dai.Device(pipeline) as device:
        qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)
        qRight = device.getOutputQueue(name="right", maxSize=4, blocking=False)
        device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        print('waiting for camera to stabilize...')
        time.sleep(5)
        i = 0
        dirName = "images"
        leftDir = os.path.join(dirName, "images_left")
        rightDir = os.path.join(dirName, "images_right")

        # Ensure directories exist
        os.makedirs(leftDir, exist_ok=True)
        os.makedirs(rightDir, exist_ok=True)

        while i < 2:
            inLeft = qLeft.get()
            inRight = qRight.get()
            i += 1

            if inLeft is not None and inRight is not None:
                frameLeft = inLeft.getCvFrame()
                frameRight = inRight.getCvFrame()

                cv2.imshow("left", frameLeft)
                cv2.imshow("right", frameRight)

                # Save the frames with a unique timestamp
                timestamp = int(time.time() * 1000)
                cv2.imwrite(f"{leftDir}/{timestamp}.png", frameLeft) 
                cv2.imwrite(f"{rightDir}/{timestamp}.png", frameRight) 

                if cv2.waitKey(1) == ord('q'):
                    break

        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
