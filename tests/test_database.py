import pytest
from unittest.mock import patch, MagicMock
import json
import tempfile
import os


class TestDatabase:
    def test_local_save_fallback(self):
        from src.config import Config
        from src.database import Database

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(no_firebase=True)
            cfg.data_dir = __import__("pathlib").Path(tmpdir)

            db = Database(cfg)

            trade = {
                "decision": "BUY",
                "stock": "AAPL",
                "price": 150.0,
                "position_x": 200,
                "position_y": 300,
                "canvas_width": 1280,
            }

            db.log_trade(trade)

            trades_path = os.path.join(tmpdir, "trades.json")
            assert os.path.exists(trades_path)

            with open(trades_path) as f:
                data = json.load(f)
                assert len(data) == 1
                assert data[0]["decision"] == "BUY"
                assert data[0]["stock"] == "AAPL"
                assert data[0]["price"] == 150.0

    def test_multiple_trades(self):
        from src.config import Config
        from src.database import Database

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(no_firebase=True)
            cfg.data_dir = __import__("pathlib").Path(tmpdir)

            db = Database(cfg)

            for i in range(3):
                db.log_trade({
                    "decision": "BUY" if i % 2 == 0 else "SELL",
                    "stock": f"STOCK{i}",
                    "price": 100.0 + i,
                    "position_x": 200,
                    "position_y": 300,
                    "canvas_width": 1280,
                })

            trades_path = os.path.join(tmpdir, "trades.json")
            with open(trades_path) as f:
                data = json.load(f)
                assert len(data) == 3
