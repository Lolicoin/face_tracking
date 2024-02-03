
import cv2
import logging
import psutil
import threading
import time
from face_detector import detect_faces
from face_tracker import FaceTracker

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

def check_rtsp_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            logging.info("RTSP stream is active.")
            return True
    logging.error("Failed to connect to RTSP stream.")
    return False

def print_system_usage():
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        logging.info(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")
    except Exception as e:
        logging.error("Error in system usage monitoring: " + str(e))

def process_frame(frame, face_cascade, face_tracker):
    faces = detect_faces(frame, face_cascade)
    logging.debug(f"Detected {len(faces)} faces")

    for (x, y, w, h) in faces:
        if not face_tracker.tracking_face:
            face_tracker.start_tracking(frame, (x, y, w, h))
            logging.debug("Started face tracking")

    tracking_info = face_tracker.update_tracking(frame)
    if tracking_info:
        logging.debug("Updating tracking info")

def capture_stream(rtsp_url, face_cascade):
    try:
        cap = cv2.VideoCapture(rtsp_url)
        face_tracker = FaceTracker()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Failed to get frame from stream: {rtsp_url}")
                break

            threading.Thread(target=process_frame, args=(frame, face_cascade, face_tracker)).start()
            print_system_usage()

        cap.release()
    except Exception as e:
        logging.error("Error in capture stream: " + str(e))

def main():
    try:
        if not cv2.__version__:
            logging.error("OpenCV is not installed or configured properly.")
            return

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        rtsp_url = "rtsp://admin:Aa123456@79.142.95.210:49152/live"
        
        if check_rtsp_stream(rtsp_url):
            logging.info("Starting video capture")
            capture_stream(rtsp_url, face_cascade)
    except Exception as e:
        logging.error("Error in main: " + str(e))

if __name__ == "__main__":
    main()
