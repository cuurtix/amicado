from dataclasses import dataclass
from typing import Tuple

from core.base_types import ForecastDistribution, RegimeState, StateVector


@dataclass
class TradingDecision:
    action: str
    size_fraction: float
    entry_price: float
    target_price: float
    stop_price: float
    edge_gross: float
    edge_net: float
    total_costs: float
    confidence: float
    risk_reward_ratio: float
    reason: str
    is_valid: bool = False


class PolicyEngine:
    """Convertit les prévisions en décisions avec calcul d'edge rigoureux."""

    def __init__(
        self,
        min_edge_threshold: float = 0.0003,
        min_confidence: float = 0.45,
        min_risk_reward: float = 1.5,
        commission_pct: float = 0.0005,
        slippage_atr_fraction: float = 0.05,
    ):
        self.min_edge_threshold = min_edge_threshold
        self.min_confidence = min_confidence
        self.min_risk_reward = min_risk_reward
        self.commission_pct = commission_pct
        self.slippage_atr_fraction = slippage_atr_fraction

    def make_decision(
        self,
        forecast: ForecastDistribution,
        state_vector: StateVector,
        regime: RegimeState,
        current_price: float,
        current_position: float = 0.0,
    ) -> TradingDecision:
        decision = TradingDecision(
            action="FLAT",
            size_fraction=0.0,
            entry_price=current_price,
            target_price=current_price,
            stop_price=current_price,
            edge_gross=0.0,
            edge_net=0.0,
            total_costs=0.0,
            confidence=forecast.confidence,
            risk_reward_ratio=0.0,
            reason="",
            is_valid=False,
        )

        if not regime.is_tradeable:
            decision.reason = f"Régime non-tradeable: {regime.regime_type.value}"
            return decision

        if forecast.confidence < self.min_confidence:
            decision.reason = f"Confiance insuffisante: {forecast.confidence:.3f}"
            return decision

        if forecast.p_up > max(forecast.p_down, forecast.p_flat):
            direction, action = 1, "LONG"
        elif forecast.p_down > max(forecast.p_up, forecast.p_flat):
            direction, action = -1, "SHORT"
        else:
            decision.reason = "Signal directionnel insuffisant"
            return decision

        edge_gross = abs(forecast.expected_return)
        total_costs = self._compute_total_costs(current_price, state_vector.atr_normalized, state_vector.spread_normalized)
        edge_net = edge_gross - total_costs

        if edge_net < self.min_edge_threshold:
            decision.reason = f"Edge net insuffisant: {edge_net:.5f} < {self.min_edge_threshold}"
            return decision

        target_price, stop_price = self._compute_levels(current_price, direction, forecast)
        gross_profit = abs(target_price - current_price)
        gross_loss = abs(current_price - stop_price)

        if gross_loss < 1e-8:
            decision.reason = "Stop trop proche"
            return decision

        risk_reward = gross_profit / gross_loss
        if risk_reward < self.min_risk_reward:
            decision.reason = f"R/R insuffisant: {risk_reward:.2f} < {self.min_risk_reward}"
            return decision

        decision.action = action
        decision.target_price = target_price
        decision.stop_price = stop_price
        decision.edge_gross = edge_gross
        decision.edge_net = edge_net
        decision.total_costs = total_costs
        decision.risk_reward_ratio = risk_reward
        decision.is_valid = True
        decision.reason = f"Edge={edge_net:.5f}, R/R={risk_reward:.2f}, P({action.lower()})={max(forecast.p_up, forecast.p_down):.3f}"
        return decision

    def _compute_total_costs(self, price: float, atr_normalized: float, spread_normalized: float) -> float:
        commission = self.commission_pct * 2
        slippage = self.slippage_atr_fraction * atr_normalized
        spread_cost = spread_normalized * 0.5
        adverse_selection = 0.0001
        return commission + slippage + spread_cost + adverse_selection

    def _compute_levels(self, price: float, direction: int, forecast: ForecastDistribution) -> Tuple[float, float]:
        if direction == 1:
            target = price * (1 + forecast.return_quantiles[0.75])
            stop = price * (1 + forecast.return_quantiles[0.1])
        else:
            target = price * (1 + forecast.return_quantiles[0.25])
            stop = price * (1 + forecast.return_quantiles[0.9])
        return target, stop
