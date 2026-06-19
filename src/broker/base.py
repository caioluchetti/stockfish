from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:
    symbol: str
    side: str
    qty: int
    price: Optional[float] = None
    status: str = "pending"


class Broker(ABC):
    @abstractmethod
    def place_order(self, symbol: str, side: str, qty: int = 1) -> Order:
        ...

    @abstractmethod
    def get_positions(self) -> list[dict]:
        ...

    @abstractmethod
    def get_account(self) -> dict:
        ...
