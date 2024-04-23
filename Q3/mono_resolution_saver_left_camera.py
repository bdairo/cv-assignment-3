#!/usr/bin/env python3

from pathlib import Path
import depthai as dai
import time
import cv2  # Make sure cv2 (OpenCV) is imported as it's used for displaying and saving frames

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
monoLeft = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)

xoutLeft.setStreamName("left")

# Properties
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)  # Changed to use the left camera
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

# Linking
monoLeft.out.link(xoutLeft.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the grayscale frames from the output defined above
    qLeft = device.getOutputQueue(name="left", maxSize=4, blocking=False)

    dirName = "disparity_images_left"
    Path(dirName).mkdir(parents=True, exist_ok=True)

    while True:
        inLeft = qLeft.get()  # Blocking call, will wait until a new data has arrived
        # Data is originally represented as a flat 1D array, it needs to be converted into HxW form
        # Frame is transformed and ready to be shown
        cv2.imshow("left", inLeft.getCvFrame())

        # After showing the frame, it's being stored inside a target directory as a PNG image
        cv2.imwrite(f"{dirName}/{int(time.time() * 1000)}.png", inLeft.getCvFrame())  # Corrected method call

        if cv2.waitKey(1) == ord('q'):
            break
