from typing import Dict, Optional

import numpy as np
import pandas as pd
from numba import jit

from core.base_types import StateVector


class FeatureEngine:
    """Transformation des données brutes en StateVector orthogonal."""

    def __init__(self):
        self.feature_cache = {}
        self.rolling_stats = {}

    def build_state_vector(self, synchronized_data: Dict) -> Optional[StateVector]:
        if not self._validate_data_sufficiency(synchronized_data):
            return None

        sv = StateVector(timestamp=pd.Timestamp.now())
        sv = self._add_price_action_features(sv, synchronized_data)
        sv = self._add_microstructure_features(sv, synchronized_data)
        sv = self._add_market_structure_features(sv, synchronized_data)
        sv = self._add_statistical_features(sv, synchronized_data)
        sv = self._add_quality_context_features(sv, synchronized_data)
        return sv

    def _validate_data_sufficiency(self, data: Dict) -> bool:
        return bool(data and "ohlcv" in data and "1m" in data["ohlcv"] and len(data["ohlcv"]["1m"]) >= 50)

    def _add_price_action_features(self, sv: StateVector, data: Dict) -> StateVector:
        ohlcv_1m = pd.DataFrame(data["ohlcv"]["1m"])
        closes = ohlcv_1m["close"].values
        highs = ohlcv_1m["high"].values
        lows = ohlcv_1m["low"].values

        returns_1 = np.log(closes[-1] / closes[-2]) if len(closes) > 1 else 0.0
        returns_5 = np.log(closes[-1] / closes[-6]) if len(closes) > 5 else 0.0
        returns_15 = np.log(closes[-1] / closes[-16]) if len(closes) > 15 else 0.0
        sv.returns_multi_horizon = np.array([returns_1, returns_5, returns_15])

        if len(highs) > 20:
            hl_ratios = np.log(highs[-20:] / lows[-20:])
            sv.realized_vol = float(np.sqrt(np.mean(hl_ratios**2) / (4 * np.log(2))) * np.sqrt(1440))

        if len(closes) > 14:
            sv.atr_normalized = float(self._compute_atr_normalized(highs, lows, closes))

        if len(closes) > 10:
            momentum_5 = (closes[-1] - closes[-6]) / closes[-6]
            momentum_10 = (closes[-6] - closes[-11]) / closes[-11]
            sv.momentum_acceleration = float(momentum_5 - momentum_10)

        return sv

    @staticmethod
    @jit(nopython=True)
    def _compute_atr_normalized(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        n = min(14, len(closes) - 1)
        tr_sum = 0.0
        for i in range(-n, 0):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            tr_sum += tr
        return (tr_sum / n) / closes[-1]

    def _add_microstructure_features(self, sv: StateVector, data: Dict) -> StateVector:
        if not data.get("orderbook") or not data.get("trades"):
            return sv

        latest_ob = data["orderbook"][-1]
        recent_trades = data["trades"][-100:]

        bids = np.array([level["quantity"] for level in latest_ob["bids"][:5]])
        asks = np.array([level["quantity"] for level in latest_ob["asks"][:5]])
        total_volume = np.sum(bids) + np.sum(asks)
        if total_volume > 0:
            sv.order_book_imbalance = float((np.sum(bids) - np.sum(asks)) / total_volume)

        buy_volume = sum(t["quantity"] for t in recent_trades if t["side"] == "buy")
        sell_volume = sum(t["quantity"] for t in recent_trades if t["side"] == "sell")
        total_trade_vol = buy_volume + sell_volume
        if total_trade_vol > 0:
            sv.signed_volume_ratio = float((buy_volume - sell_volume) / total_trade_vol)

        best_bid = latest_ob["bids"][0]["price"]
        best_ask = latest_ob["asks"][0]["price"]
        mid_price = (best_bid + best_ask) / 2
        sv.spread_normalized = float((best_ask - best_bid) / mid_price)

        depth_bid = sum(level["quantity"] for level in latest_ob["bids"][:10])
        depth_ask = sum(level["quantity"] for level in latest_ob["asks"][:10])
        if depth_ask > 0:
            sv.depth_asymmetry = float(depth_bid / depth_ask)

        return sv

    def _add_market_structure_features(self, sv: StateVector, data: Dict) -> StateVector:
        ohlcv_1m = pd.DataFrame(data["ohlcv"]["1m"])
        closes = ohlcv_1m["close"].values
        highs = ohlcv_1m["high"].values
        lows = ohlcv_1m["low"].values

        if len(closes) >= 20:
            rolling_high = np.max(highs[-20:])
            rolling_low = np.min(lows[-20:])
            current = closes[-1]
            span = max(rolling_high - rolling_low, 1e-8)
            sv.distance_to_liquidity_pools = float(min(abs(current - rolling_high), abs(current - rolling_low)) / span)

            if current > np.max(highs[-6:-1]):
                sv.break_of_structure_score = 1
            elif current < np.min(lows[-6:-1]):
                sv.break_of_structure_score = -1
            else:
                sv.break_of_structure_score = 0

            atr = float(self._compute_atr_normalized(highs, lows, closes))
            sv.compression_ratio = float(atr / (np.std(closes[-20:]) / max(current, 1e-8) + 1e-8))

        return sv

    def _add_statistical_features(self, sv: StateVector, data: Dict) -> StateVector:
        ohlcv_1m = pd.DataFrame(data["ohlcv"]["1m"])
        closes = ohlcv_1m["close"].values
        log_returns = np.diff(np.log(closes[-50:]))

        sv.hurst_exponent = self._compute_hurst_rs(log_returns)
        sv.local_entropy = self._compute_local_entropy(log_returns)

        if len(log_returns) > 20:
            recent_mean = np.mean(log_returns[-20:])
            recent_std = np.std(log_returns[-20:])
            if recent_std > 1e-8:
                sv.zscore_return = float((log_returns[-1] - recent_mean) / recent_std)

        autocorrs = [pd.Series(log_returns).autocorr(lag=i) for i in range(1, 6)]
        valid = [ac for ac in autocorrs if not np.isnan(ac)]
        sv.autocorrelation_signature = float(np.mean(valid)) if valid else 0.0
        return sv

    def _compute_hurst_rs(self, returns: np.ndarray) -> float:
        if len(returns) < 20:
            return 0.5
        try:
            n = len(returns)
            deviations = np.cumsum(returns - np.mean(returns))
            r_val = np.max(deviations) - np.min(deviations)
            s_val = np.std(returns)
            if s_val == 0 or n < 2:
                return 0.5
            return float(np.clip(np.log(r_val / s_val) / np.log(n), 0.0, 1.0))
        except Exception:
            return 0.5

    def _compute_local_entropy(self, returns: np.ndarray, bins: int = 10) -> float:
        if len(returns) < bins:
            return 1.0
        try:
            hist, _ = np.histogram(returns, bins=bins)
            probs = hist / np.sum(hist)
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            return float(entropy / np.log2(bins))
        except Exception:
            return 1.0

    def _add_quality_context_features(self, sv: StateVector, data: Dict) -> StateVector:
        sv.data_quality_score = float(data.get("quality_score", 1.0))
        sv.session_liquidity_score = float(max(0.0, 1.0 - sv.spread_normalized * 200))
        sv.event_proximity = 1.0
        sv.confidence = float(np.clip((sv.data_quality_score + sv.session_liquidity_score) / 2, 0.0, 1.0))
        return sv
