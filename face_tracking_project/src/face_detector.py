
import cv2
import logging

def detect_faces(frame, face_cascade):
    logging.debug("Converting frame to grayscale")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    logging.debug("Detecting faces")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    logging.debug(f"{len(faces)} faces detected")
    return faces
