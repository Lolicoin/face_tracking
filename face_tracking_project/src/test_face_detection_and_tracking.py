
import unittest
import cv2
from face_detector import detect_faces
from face_tracker import FaceTracker

class TestFaceDetectionAndTracking(unittest.TestCase):
    def test_face_detection(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        test_frame = cv2.imread('test_face.jpg')  # Assuming a test image with a face
        faces = detect_faces(test_frame, face_cascade)
        self.assertTrue(len(faces) > 0, "No faces detected")

    def test_face_tracking(self):
        face_tracker = FaceTracker()
        test_frame = cv2.imread('test_face.jpg')  # Assuming a test image with a face
        face_tracker.start_tracking(test_frame, (100, 100, 50, 50))  # Example values
        tracking_info = face_tracker.update_tracking(test_frame)
        self.assertIsNotNone(tracking_info, "Tracking info is None")

if __name__ == '__main__':
    unittest.main()
