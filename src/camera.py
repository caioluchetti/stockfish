import time
import cv2
import logging

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, config):
        self.config = config
        self.cap = None
        self.back_sub = None

    def start(self):
        self.cap = cv2.VideoCapture(self.config.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

        if not self.cap.isOpened():
            logger.error("Could not open webcam (device %s).", self.config.device_id)
            raise RuntimeError(f"Could not open webcam device {self.config.device_id}")

        self._init_background_subtractor()

        logger.info("Webcam opened. Learning background...")
        logger.info("Please keep the camera perfectly still.")
        logger.info("Press 'q' to quit, 'c' to recalibrate background.")

    def _init_background_subtractor(self):
        self.back_sub = cv2.createBackgroundSubtractorKNN(detectShadows=False)

    def recalibrate(self):
        self._init_background_subtractor()
        logger.info("Background recalibrated. Keep camera still for a moment.")

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to grab frame.")
            return None

        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)

        return frame

    def apply_background_subtraction(self, frame):
        fg_mask = self.back_sub.apply(frame)
        return fg_mask

    def release(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released.")
