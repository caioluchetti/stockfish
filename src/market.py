import datetime
import random
import logging
import time
from pathlib import Path

import pandas as pd
import pytz
import yfinance as yf

logger = logging.getLogger(__name__)

EST = pytz.timezone("US/Eastern")
MARKET_OPEN_TIME = datetime.time(9, 30)
MARKET_CLOSE_TIME = datetime.time(16, 0)


class Market:
    def __init__(self, config):
        self.config = config
        self.tickers = []
        self._price_cache = {}
        self._cache_ttl = 60
        self._load_tickers()

    def _load_tickers(self):
        csv_path = Path(self.config.trading.sp500_csv)
        if not csv_path.exists():
            logger.warning("SP500 CSV not found at %s. Using fallback ticker list.", csv_path)
            self.tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            return

        stocks = pd.read_csv(csv_path)
        self.tickers = stocks["Symbol"].tolist()
        logger.info("Loaded %d tickers from %s.", len(self.tickers), csv_path)

    def is_market_open(self):
        now_est = datetime.datetime.now(EST)
        weekday = now_est.weekday()
        current_time = now_est.time()

        return weekday < 5 and MARKET_OPEN_TIME <= current_time <= MARKET_CLOSE_TIME

    def pick_random_ticker(self):
        return random.choice(self.tickers)

    def get_price(self, ticker):
        now = time.time()
        if ticker in self._price_cache:
            cached_price, cached_time = self._price_cache[ticker]
            if now - cached_time < self._cache_ttl:
                return cached_price

        try:
            stock_data = yf.download(
                tickers=ticker, period="1d", interval="1m", progress=False
            )
            if not stock_data.empty:
                price = float(stock_data["Close"].iloc[-1])
                self._price_cache[ticker] = (price, now)
                return price
        except Exception as e:
            logger.error("Error fetching price for %s: %s", ticker, e)

        return None
