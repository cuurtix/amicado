from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AppConfig:
    symbol: str = "BTCUSDT"
    prediction_interval_seconds: int = 10
    forecast_horizon: int = 5
    sync_lookback: int = 200

    endpoints: Dict[str, str] = field(default_factory=lambda: {
        "1s": "ws://localhost:8761",
        "1m": "ws://localhost:8762",
        "5m": "ws://localhost:8763",
        "15m": "ws://localhost:8764",
        "1h": "ws://localhost:8765",
        "orderbook": "ws://localhost:8766",
        "trades": "ws://localhost:8767",
    })
