import cv2
import depthai as dai
import numpy as np
from sklearn.cluster import DBSCAN

def create_pipeline():
    # Create a pipeline
    pipeline = dai.Pipeline()

    # Create a ColorCamera node
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Create an XLinkOut node
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    return pipeline

def cluster_points(points):
    # Using DBSCAN to cluster points
    clustering = DBSCAN(eps=20, min_samples=5).fit(points)
    labels = clustering.labels_

    # Number of clusters in labels, ignoring noise if present
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    clusters = [points[labels == i] for i in range(n_clusters_)]
    return clusters

def track_objects():
    # Initialize the pipeline
    pipeline = create_pipeline()

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        # Output queue for RGB frames
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)

        # Initialize SIFT detector with different parameters
        sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.03, edgeThreshold=8, nOctaveLayers=6)

        prev_frame = None
        prev_descriptors = None

        while True:
            in_rgb = q_rgb.get()  # Get RGB frame from camera
            frame = in_rgb.getCvFrame()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect features and compute descriptors
            keypoints, descriptors = sift.detectAndCompute(gray, None)

            if prev_frame is not None and prev_descriptors is not None:
                # Match descriptors using FLANN matcher
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(prev_descriptors, descriptors, k=2)

                # Filter matches using the Lowe's ratio test
                good_points = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_points.append(keypoints[m.trainIdx].pt)

                # Cluster good points
                if len(good_points) > 0:
                    clusters = cluster_points(np.array(good_points))
                    for cluster in clusters:
                        if len(cluster) > 0:
                            x, y, w, h = cv2.boundingRect(cluster.astype(np.float32))
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            prev_frame = gray
            prev_descriptors = descriptors
            cv2.imshow('Real-time Object Tracker (with SIFT and Clustering)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


# Call function to start tracking objects
track_objects()
