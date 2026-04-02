import asyncio
import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import websockets


class DataEngine:
    """Ingestion multi-source avec contrôle qualité strict."""

    def __init__(self, symbol: str, buffer_size: int = 10000):
        self.symbol = symbol

        self.ohlcv_buffers = {tf: deque(maxlen=buffer_size) for tf in ["1s", "1m", "5m", "15m", "1h"]}
        self.orderbook_buffer = deque(maxlen=1000)
        self.trades_buffer = deque(maxlen=5000)

        self.quality_metrics = {
            "latency_ms": deque(maxlen=100),
            "gaps_detected": deque(maxlen=100),
            "outlier_scores": deque(maxlen=100),
        }

    async def ingest_websocket_multi_stream(self, endpoints: Dict[str, str]):
        tasks = [asyncio.create_task(self._stream_handler(stream_type, uri)) for stream_type, uri in endpoints.items()]
        await asyncio.gather(*tasks)

    async def _stream_handler(self, stream_type: str, uri: str):
        while True:
            try:
                async with websockets.connect(uri) as ws:
                    async for message in ws:
                        await self._process_message(stream_type, message)
            except Exception as e:
                logging.error("Stream %s error: %s", stream_type, e)
                await asyncio.sleep(2)

    async def _process_message(self, stream_type: str, message: str):
        payload = json.loads(message)
        recv_ts = datetime.now(timezone.utc).timestamp() * 1000
        evt_ts = payload.get("timestamp", recv_ts)
        latency_ms = max(0.0, recv_ts - evt_ts)
        self.quality_metrics["latency_ms"].append(latency_ms)

        if stream_type in self.ohlcv_buffers:
            self.ohlcv_buffers[stream_type].append(payload)
        elif stream_type == "orderbook":
            self.orderbook_buffer.append(payload)
        elif stream_type == "trades":
            self.trades_buffer.append(payload)

        quality = self._compute_data_quality_score(payload, latency_ms)
        self.quality_metrics["outlier_scores"].append(quality)

    def _compute_data_quality_score(self, data_point: dict, latency_ms: float) -> float:
        score = 1.0

        if latency_ms > 100:
            score -= 0.2

        if "ohlcv" in data_point:
            o, h, l, c = data_point["ohlcv"]
            if not (l <= min(o, c) <= max(o, c) <= h):
                score -= 0.4

        z = data_point.get("return_zscore")
        if z is not None and abs(z) > 5:
            score -= 0.3

        return max(0.0, score)

    def get_synchronized_data(self, lookback: int = 500) -> Optional[Dict]:
        if not all(len(buf) > lookback for buf in self.ohlcv_buffers.values()):
            return None

        quality_slice = list(self.quality_metrics["outlier_scores"])[-20:]
        quality_score = float(np.mean(quality_slice)) if quality_slice else 1.0

        return {
            "ohlcv": {tf: list(buf)[-lookback:] for tf, buf in self.ohlcv_buffers.items()},
            "orderbook": list(self.orderbook_buffer)[-100:],
            "trades": list(self.trades_buffer)[-1000:],
            "quality_score": quality_score,
        }
