import datetime
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, config):
        self.config = config
        self.trades_ref = None
        self.portfolio_ref = None
        self.connected = False
        self._local_trades = []

        if not config.no_firebase:
            self._init_firebase()

    def _init_firebase(self):
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            cred_path = self.config.firebase.credential_path
            if not cred_path:
                logger.warning("CREDENTIAL_PATH not set. Firebase disabled.")
                return

            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)

            db = firestore.client()
            app_id = self.config.firebase.app_id
            self.trades_ref = (
                db.collection("artifacts")
                .document(app_id)
                .collection(self.config.firebase.trades_collection)
            )
            self.portfolio_ref = (
                db.collection("artifacts")
                .document(app_id)
                .collection(self.config.firebase.portfolio_collection)
            )
            self.connected = True
            logger.info("Firebase connected. Logging to artifacts/%s/fish_trades.", app_id)

        except Exception as e:
            logger.error("Firebase init failed: %s. Falling back to local storage.", e)
            self.connected = False

    def log_trade(self, decision_data):
        trade = {
            **decision_data,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "status": "pending",
        }

        self._save_local(trade)

        if self.connected:
            try:
                self.trades_ref.add(trade)
                logger.info(
                    "Trade logged to Firebase: %s | %s | $%s",
                    trade["decision"],
                    trade["stock"],
                    trade.get("price"),
                )
            except Exception as e:
                logger.error("Firebase write failed: %s", e)
        else:
            logger.info(
                "Trade logged locally: %s | %s | $%s",
                trade["decision"],
                trade["stock"],
                trade.get("price"),
            )

    def _save_local(self, trade):
        trade["timestamp"] = (
            trade["timestamp"].isoformat()
            if isinstance(trade["timestamp"], datetime.datetime)
            else trade["timestamp"]
        )
        self._local_trades.append(trade)

        data_dir = self.config.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        trades_file = data_dir / "trades.json"

        try:
            with open(trades_file, "w") as f:
                json.dump(self._local_trades, f, indent=2, default=str)
        except Exception as e:
            logger.error("Failed to save local trades: %s", e)
