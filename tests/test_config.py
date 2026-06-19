import pytest
from src.config import Config, CameraConfig, DetectorConfig, TradingConfig, FirebaseConfig, UIConfig


class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.camera.device_id == 0
        assert cfg.camera.width == 1280
        assert cfg.camera.height == 720
        assert cfg.detector.motion_sensitivity == 220
        assert cfg.detector.min_fish_area == 500
        assert cfg.trading.log_interval == 5.0
        assert cfg.simulate is False
        assert cfg.dry_run is False
        assert cfg.no_firebase is False

    def test_dry_run_disables_firebase(self):
        cfg = Config(dry_run=True, no_firebase=False)
        assert cfg.no_firebase is True

    def test_simulate_mode(self):
        cfg = Config(simulate=True)
        assert cfg.simulate is True

    def test_trading_csv_path(self):
        cfg = Config()
        assert "sp500_table.csv" in cfg.trading.sp500_csv
