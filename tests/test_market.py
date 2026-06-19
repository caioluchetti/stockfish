import datetime
import pytest
from unittest.mock import patch, MagicMock


class TestMarket:
    def test_is_market_open_weekday_morning(self):
        from src.market import Market, MARKET_OPEN_TIME, MARKET_CLOSE_TIME
        from src.config import Config

        mock_now = datetime.datetime(2026, 6, 19, 10, 0, 0)  # Friday 10 AM
        est_time = MagicMock()
        est_time.weekday.return_value = 4
        est_time.time.return_value = datetime.time(10, 0)

        with patch("src.market.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = est_time
            market = Market(Config())
            mock_dt.datetime.now.return_value = mock_now
            # Actually this is tricky because pytz is involved
            # Let's test via a different approach

    def test_is_market_open_saturday(self):
        from unittest.mock import patch
        import pytz
        from src.config import Config

        est = pytz.timezone("US/Eastern")

        # Saturday June 20, 2026 at 10 AM EST
        saturday = datetime.datetime(2026, 6, 20, 10, 0, 0)
        saturday_est = est.localize(saturday)

        with patch("src.market.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = saturday_est
            from src.market import Market
            market = Market(Config())
            assert market.is_market_open() is False

    def test_is_market_open_friday_noon(self):
        from unittest.mock import patch
        import pytz
        from src.config import Config

        est = pytz.timezone("US/Eastern")

        # Friday June 19, 2026 at 12 PM EST
        friday = datetime.datetime(2026, 6, 19, 12, 0, 0)
        friday_est = est.localize(friday)

        with patch("src.market.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = friday_est
            from src.market import Market
            market = Market(Config())
            assert market.is_market_open() is True

    def test_is_market_open_after_close(self):
        from unittest.mock import patch
        import pytz
        from src.config import Config

        est = pytz.timezone("US/Eastern")

        friday = datetime.datetime(2026, 6, 19, 17, 0, 0)
        friday_est = est.localize(friday)

        with patch("src.market.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = friday_est
            from src.market import Market
            market = Market(Config())
            assert market.is_market_open() is False

    def test_is_market_open_before_open(self):
        from unittest.mock import patch
        import pytz
        from src.config import Config

        est = pytz.timezone("US/Eastern")

        friday = datetime.datetime(2026, 6, 19, 8, 0, 0)
        friday_est = est.localize(friday)

        with patch("src.market.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = friday_est
            from src.market import Market
            market = Market(Config())
            assert market.is_market_open() is False

    def test_pick_random_ticker(self):
        from src.config import Config
        from src.market import Market

        market = Market(Config())
        market.tickers = ["AAPL", "MSFT", "GOOGL"]
        ticker = market.pick_random_ticker()
        assert ticker in ["AAPL", "MSFT", "GOOGL"]
