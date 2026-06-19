import datetime
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pytz
import logging

logger = logging.getLogger(__name__)

EST = pytz.timezone("US/Eastern")


class UI:
    def __init__(self, config):
        self.config = config
        self.font = None
        self._load_font()

    def _load_font(self):
        try:
            self.font = ImageFont.truetype(self.config.font_path, self.config.font_size)
        except Exception:
            logger.warning("Font '%s' not found. Using PIL default.", self.config.font_path)
            self.font = ImageFont.load_default()

    def render(self, frame, decision, pending_decision, current_stock, found_fish, fish_x, fish_y, market_is_open):
        frame_height, frame_width = frame.shape[:2]
        mid_x = frame_width // 2
        shadow = self.config.shadow_offset

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        draw.line([(mid_x, 0), (mid_x, frame_height)], fill=(255, 255, 255, 80), width=2)

        if found_fish:
            r = 20
            draw.ellipse(
                [fish_x - r, fish_y - r, fish_x + r, fish_y + r],
                outline=(0, 255, 0),
                width=3,
            )

        if market_is_open:
            if pending_decision:
                self._draw_label(draw, "BUY", mid_x // 4, 50, (34, 197, 94), shadow, anchor="left")
                self._draw_label(draw, "SELL", mid_x + mid_x // 4, 50, (34, 197, 94), shadow, anchor="left")

                wait_text = f"WAITING: {current_stock}" if current_stock else "WAITING..."
                bbox = draw.textbbox((0, 0), wait_text, font=self.font)
                tw = bbox[2] - bbox[0]
                self._draw_label(draw, wait_text, (frame_width - tw) // 2, 50, (239, 68, 68), shadow)

            decision_text = f">> {decision}"
            self._draw_label(draw, decision_text, 20, frame_height - 60, (255, 255, 255), shadow, anchor="left-bottom")
        else:
            closed_text = "MARKET CLOSED – SEE YOU NEXT SESSION"
            bbox = draw.textbbox((0, 0), closed_text, font=self.font)
            tw = bbox[2] - bbox[0]
            self._draw_label(draw, closed_text, (frame_width - tw) // 2, 50, (239, 68, 68), shadow)

            weekday_text = self._aquarium_mode(frame_pil, draw, frame_width, frame_height, shadow)
            if weekday_text:
                self._draw_label(draw, weekday_text, (frame_width - draw.textbbox((0, 0), weekday_text, font=self.font)[2]) // 2, frame_height - 60, (148, 163, 184), shadow)

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        return frame

    def _draw_label(self, draw, text, x, y, color, shadow, anchor="left-top"):
        draw.text((x + shadow, y + shadow), text, font=self.font, fill=(0, 0, 0))
        draw.text((x, y), text, font=self.font, fill=color)

    def _aquarium_mode(self, frame_pil, draw, frame_width, frame_height, shadow):
        est_time = datetime.datetime.now(EST)
        if est_time.weekday() >= 5:
            return "IT'S THE WEEKEND – RELAX, NO TRADING"
        current_hour = est_time.hour
        if current_hour < 9:
            return "MARKET OPENS AT 9:30 AM EST"
        if current_hour >= 16:
            return "MARKET CLOSED AT 4:00 PM – SEE YOU TOMORROW"
        return None
