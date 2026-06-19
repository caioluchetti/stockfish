import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Detector:
    def __init__(self, config):
        self.config = config

    def process_mask(self, fg_mask):
        ret, thresh = cv2.threshold(
            fg_mask, self.config.motion_sensitivity, 255, cv2.THRESH_BINARY
        )

        kernel = np.ones(self.config.kernel_size, np.uint8)
        clean = cv2.dilate(thresh, kernel, iterations=self.config.dilate_iterations)
        clean = cv2.erode(clean, kernel, iterations=self.config.erode_iterations)

        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return clean, contours

    def detect_fish(self, contours, frame_width):
        if not contours:
            return False, 0, 0, "HOLD"

        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) <= self.config.min_fish_area:
            return False, 0, 0, "HOLD"

        M = cv2.moments(c)
        if M["m00"] == 0:
            return False, 0, 0, "HOLD"

        fish_x = int(M["m10"] / M["m00"])
        fish_y = int(M["m01"] / M["m00"])

        mid_x = frame_width // 2
        hold_margin = int(frame_width * self.config.hold_zone_ratio)

        if fish_x < (mid_x - hold_margin):
            decision = "BUY"
        elif fish_x > (mid_x + hold_margin):
            decision = "SELL"
        else:
            decision = "HOLD"

        return True, fish_x, fish_y, decision
