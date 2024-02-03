
import cv2
import os
import logging
from face_detector import detect_faces
from face_tracker import FaceTracker

class FaceTrackingVisualizer:
    def __init__(self, rtsp_url, face_cascade_path, save_faces=False, save_path='faces'):
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')
        logging.info("Initializing FaceTrackingVisualizer")
        self.rtsp_url = rtsp_url
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.save_faces = save_faces
        self.save_path = save_path
        self.face_tracker = FaceTracker()
        if save_faces and not os.path.exists(save_path):
            os.makedirs(save_path)

    def capture_stream(self):
        logging.info(f"Opening RTSP stream: {self.rtsp_url}")
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logging.error(f"Failed to open stream: {self.rtsp_url}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to get frame from stream")
                break

            self.process_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        faces = detect_faces(frame, self.face_cascade)

        for (x, y, w, h) in faces:
            if not self.face_tracker.tracking_face:
                self.face_tracker.start_tracking(frame, (x, y, w, h))

            tracking_info = self.face_tracker.update_tracking(frame)
            if tracking_info:
                cv2.rectangle(frame, (tracking_info[0], tracking_info[1]), 
                              (tracking_info[0] + tracking_info[2], tracking_info[1] + tracking_info[3]), 
                              (0, 255, 0), 2)

                if self.save_faces:
                    face_frame = frame[y:y+h, x:x+w]
                    cv2.imwrite(os.path.join(self.save_path, f'face_{x}_{y}.jpg'), face_frame)

        cv2.imshow("Face Tracking", frame)


rtsp_url = "rtsp://admin:Aa123456@79.142.95.210:49152/live"
face_cascade_path = "haarcascade_frontalface_default.xml"
visualizer = FaceTrackingVisualizer(rtsp_url, face_cascade_path, save_faces=True)
visualizer.capture_stream()
