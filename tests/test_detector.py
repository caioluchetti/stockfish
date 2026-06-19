import pytest
from unittest.mock import patch, MagicMock


class TestDetector:
    def test_empty_contours_returns_hold(self):
        from src.config import DetectorConfig
        from src.detector import Detector

        det = Detector(DetectorConfig())
        found, x, y, decision = det.detect_fish([], 1280)
        assert found is False
        assert decision == "HOLD"

    def test_small_contour_returns_hold(self):
        from src.config import DetectorConfig
        from src.detector import Detector
        import numpy as np

        det = Detector(DetectorConfig(min_fish_area=500))

        mock_contour = MagicMock()
        mock_contour.contourArea = 200

        with patch("cv2.contourArea", return_value=200):
            found, x, y, decision = det.detect_fish([mock_contour], 1280)
            assert found is False
            assert decision == "HOLD"

    def test_fish_left_returns_buy(self):
        from src.config import DetectorConfig
        from src.detector import Detector
        import numpy as np

        det = Detector(DetectorConfig(min_fish_area=500))
        frame_width = 1280

        mock_contour = MagicMock()
        mock_moment = {"m00": 1.0, "m10": 200.0, "m01": 360.0}

        with patch("cv2.contourArea", return_value=600):
            with patch("cv2.moments", return_value=mock_moment):
                found, x, y, decision = det.detect_fish([mock_contour], frame_width)
                assert found is True
                assert x == 200
                assert y == 360
                assert decision == "BUY"

    def test_fish_right_returns_sell(self):
        from src.config import DetectorConfig
        from src.detector import Detector

        det = Detector(DetectorConfig(min_fish_area=500))
        frame_width = 1280

        mock_contour = MagicMock()
        mock_moment = {"m00": 1.0, "m10": 1000.0, "m01": 360.0}

        with patch("cv2.contourArea", return_value=600):
            with patch("cv2.moments", return_value=mock_moment):
                found, x, y, decision = det.detect_fish([mock_contour], frame_width)
                assert found is True
                assert decision == "SELL"

    def test_fish_center_returns_hold(self):
        from src.config import DetectorConfig
        from src.detector import Detector

        det = Detector(DetectorConfig(min_fish_area=500, hold_zone_ratio=0.20))
        frame_width = 1280

        mock_contour = MagicMock()
        mock_moment = {"m00": 1.0, "m10": 640.0, "m01": 360.0}

        with patch("cv2.contourArea", return_value=600):
            with patch("cv2.moments", return_value=mock_moment):
                found, x, y, decision = det.detect_fish([mock_contour], frame_width)
                assert found is True
                assert decision == "HOLD"
