import cv2
import numpy as np

from Drawer import Drawer
from PoseEstimator import PoseEstimator

IMG = "35-0.png"
VIDEO = "sample_video.mp4"


def process_video():
    cap = cv2.VideoCapture(VIDEO)
    hasFrame, frame = cap.read()

    vid_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                 (frame.shape[1], frame.shape[0]))

    estimator = PoseEstimator()

    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        frameCopy = np.copy(frame)
        if not hasFrame:
            cv2.waitKey()
            break

        drawer = Drawer(frameCopy)
        prediction = estimator.predict(frame)
        points = estimator.get_rescaled_points(prediction, frame, 0.2)
        drawer.draw_skeleton(points)

        cv2.imshow('Output-Skeleton', frameCopy)

        vid_writer.write(frameCopy)

    vid_writer.release()


def process_image():
    estimator = PoseEstimator()
    frame = cv2.imread(IMG)
    frameCopy = np.copy(frame)  # TODO Why copy?

    drawer = Drawer(frameCopy)

    prediction = estimator.predict(frame)
    points = estimator.get_rescaled_points(prediction, frame, 0.2)
    drawer.draw_skeleton(points)

    cv2.imshow('Output-Keypoints', frameCopy)
    cv2.imwrite('Output-Keypoints.jpg', frameCopy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    process_image()


if __name__ == '__main__':
    main()
