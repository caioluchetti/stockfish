import logging
from .base import Broker, Order

logger = logging.getLogger(__name__)


class AlpacaBroker(Broker):
    def __init__(self, config):
        self.config = config
        self.api = None
        self._init_api()

    def _init_api(self):
        try:
            from alpaca.trading.client import TradingClient

            self.api = TradingClient(
                api_key=self.config.api_key,
                secret_key=self.config.api_secret,
                paper=True,
            )
            account = self.api.get_account()
            logger.info(
                "Alpaca Paper connected. Buying power: $%s | Status: %s",
                account.buying_power,
                account.status,
            )
        except ImportError:
            logger.warning("alpaca-py not installed. Install with: pip install alpaca-py")
            self.api = None
        except Exception as e:
            logger.error("Alpaca init failed: %s", e)
            self.api = None

    def place_order(self, symbol: str, side: str, qty: int = 1) -> Order:
        if self.api is None:
            logger.warning("Alpaca not connected. Order not placed: %s %s", side, symbol)
            return Order(symbol=symbol, side=side, qty=qty, status="failed")

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            side_enum = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_enum,
                time_in_force=TimeInForce.DAY,
            )

            response = self.api.submit_order(order_request)
            logger.info(
                "Order placed: %s %s x%d @ $%s [%s]",
                side, symbol, qty, response.filled_avg_price or "market",
                response.id,
            )

            return Order(
                symbol=symbol,
                side=side,
                qty=qty,
                price=float(response.filled_avg_price) if response.filled_avg_price else None,
                status=response.status,
            )

        except Exception as e:
            logger.error("Order failed: %s %s — %s", side, symbol, e)
            return Order(symbol=symbol, side=side, qty=qty, status="failed")

    def get_positions(self) -> list[dict]:
        if self.api is None:
            return []
        try:
            positions = self.api.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "market_value": p.market_value,
                    "unrealized_pl": p.unrealized_pl,
                }
                for p in positions
            ]
        except Exception as e:
            logger.error("Failed to get positions: %s", e)
            return []

    def get_account(self) -> dict:
        if self.api is None:
            return {}
        try:
            account = self.api.get_account()
            return {
                "cash": account.cash,
                "buying_power": account.buying_power,
                "equity": account.equity,
                "status": account.status,
            }
        except Exception as e:
            logger.error("Failed to get account: %s", e)
            return {}


class NoOpBroker(Broker):
    def place_order(self, symbol: str, side: str, qty: int = 1) -> Order:
        logger.info("[NOOP] Would place: %s %s x%d", side, symbol, qty)
        return Order(symbol=symbol, side=side, qty=qty, status="noop")

    def get_positions(self) -> list[dict]:
        return []

    def get_account(self) -> dict:
        return {"status": "noop"}
