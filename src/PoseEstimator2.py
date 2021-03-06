import cv2
from matplotlib import pyplot as plt

from src.Pose import Pose
from src.PosePairs import POSE_PAIRS_COCO
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

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
        self.net = TfPoseEstimator(get_graph_path('openpose_quantize'), target_size=(width, height))

    def predict(self, frame):
        # input image dimensions for the network
        # input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (self.width, self.height),
        #                                    (0, 0, 0), swapRB=False, crop=False)

        # self.net.setInput(input_blob)

        # output = self.net.forward()
        humans = self.net.inference(frame)

        return humans

    def get_rescaled_points(self, prediction, frame, threshold=0.05):
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
