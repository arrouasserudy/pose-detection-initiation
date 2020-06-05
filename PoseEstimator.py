import time
import cv2
from matplotlib import pyplot as plt

from Pose import Pose
from PosePairs import POSE_PAIRS_COCO

protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
network = None


def get_network():
    global network
    if not network:
        network = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    return network


class PoseEstimator:
    def __init__(self, width=368, height=368, pose_pairs=POSE_PAIRS_COCO):
        self.width = width
        self.height = height
        self.pose_pairs = pose_pairs
        self.net = get_network()

    def predict(self, frame):
        t = time.time()
        # input image dimensions for the network
        input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (self.width, self.height),
                                           (0, 0, 0), swapRB=False, crop=False)

        self.net.setInput(input_blob)

        output = self.net.forward()
        print("time taken by network : {:.3f}".format(time.time() - t))
        return output

    def get_rescaled_points(self, prediction, frame, threshold=0.1):
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        height = prediction.shape[2]
        width = prediction.shape[3]

        # Empty list to store the detected keypoints
        points = []

        for i in range(len(self.pose_pairs)):
            # confidence map of corresponding body's part.
            prob_map = prediction[0, i, :, :]
            if i == 1:
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                plt.imshow(prob_map, alpha=0.6)

            # Find global maxima of the probMap.
            _, prob, _, point = cv2.minMaxLoc(prob_map)
            # Scale the point to fit on the original image
            x = (frame_width * point[0]) / width
            y = (frame_height * point[1]) / height

            if prob > threshold:
                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

        return Pose(points)
