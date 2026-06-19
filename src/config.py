import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class CameraConfig:
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    flip_horizontal: bool = True


@dataclass
class DetectorConfig:
    motion_sensitivity: int = 220
    kernel_size: tuple = (2, 2)
    dilate_iterations: int = 3
    erode_iterations: int = 3
    min_fish_area: int = 500
    hold_zone_ratio: float = 0.20


@dataclass
class TradingConfig:
    log_interval: float = 5.0
    sp500_csv: str = field(default_factory=lambda: str(ROOT_DIR / "market_data" / "sp500_table.csv"))


@dataclass
class FirebaseConfig:
    credential_path: str = field(default_factory=lambda: os.getenv("CREDENTIAL_PATH", ""))
    app_id: str = field(default_factory=lambda: os.getenv("APP_ID", ""))
    trades_collection: str = "fish_trades"
    portfolio_collection: str = "portfolio_history"


@dataclass
class AlpacaConfig:
    api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("ALPACA_API_SECRET", ""))
    enabled: bool = False


@dataclass
class UIConfig:
    font_path: str = "arial.ttf"
    font_size: int = 40
    shadow_offset: int = 3


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    firebase: FirebaseConfig = field(default_factory=FirebaseConfig)
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    data_dir: Path = field(default_factory=lambda: ROOT_DIR / "data")
    dry_run: bool = False
    simulate: bool = False
    no_firebase: bool = False
