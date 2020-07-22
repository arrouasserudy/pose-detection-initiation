import cv2
from src.PosePairs import POSE_PAIRS_COCO


class Drawer:
    def __init__(self, frame):
        self.frame = frame

    def draw_skeleton(self, pose, pairs=POSE_PAIRS_COCO):
        points = pose.points
        for i, point in enumerate(points):
            if point:
                cv2.circle(self.frame, (point[0], point[1]), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(self.frame, "{}".format(i), (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                            lineType=cv2.LINE_AA)

        # Draw Skeleton
        for pair in pairs:
            part_a = pair['points'][0]
            part_b = pair['points'][1]

            if points[part_a] and points[part_b]:
                cv2.line(self.frame, points[part_a], points[part_b], pair['color'], 2)