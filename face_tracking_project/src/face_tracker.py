
import dlib
import cv2
import logging

class FaceTracker:
    def __init__(self):
        self.tracker = dlib.correlation_tracker()
        self.tracking_face = False

    def start_tracking(self, frame, rect):
        logging.debug("Starting face tracking")
        dlib_rect = dlib.rectangle(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
        self.tracker.start_track(frame, dlib_rect)
        self.tracking_face = True

    def update_tracking(self, frame):
        if self.tracking_face:
            logging.debug("Updating tracking")
            self.tracker.update(frame)
            pos = self.tracker.get_position()
            return int(pos.left()), int(pos.top()), int(pos.width()), int(pos.height())
        return None
