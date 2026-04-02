from typing import List

import numpy as np
import pandas as pd
from hmmlearn import hmm

from core.base_types import RegimeState, RegimeType, StateVector


class RegimeEngine:
    """Détection de régimes via HMM et règles heuristiques."""

    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        self.hmm_model = None
        self.regime_history: List[RegimeType] = []
        self.stability_window = 20

    def fit_regime_model(self, feature_matrix: np.ndarray, returns: np.ndarray):
        vol_rolling = pd.Series(returns).rolling(20).std().fillna(0).values
        X = np.column_stack([returns, vol_rolling, np.abs(returns)])
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]

        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        self.hmm_model.fit(X_clean)
        return self

    def detect_regime(self, sv: StateVector) -> RegimeState:
        heuristic_regime = self._detect_heuristic_regime(sv)
        _ = self._get_hmm_state(sv) if self.hmm_model else 0
        stability = self._compute_regime_stability()
        is_tradeable = self._assess_tradeability(sv, stability)

        regime_state = RegimeState(
            regime_type=heuristic_regime,
            regime_strength=self._compute_regime_strength(sv),
            stability_score=stability,
            is_tradeable=is_tradeable,
            expected_persistence_bars=self._estimate_persistence(),
        )

        self.regime_history.append(regime_state.regime_type)
        if len(self.regime_history) > 100:
            self.regime_history.pop(0)
        return regime_state

    def _get_hmm_state(self, sv: StateVector) -> int:
        X = np.array([[sv.returns_multi_horizon[0], sv.realized_vol, abs(sv.returns_multi_horizon[0])]])
        return int(self.hmm_model.predict(X)[0])

    def _detect_heuristic_regime(self, sv: StateVector) -> RegimeType:
        if sv.momentum_acceleration > 0.001 and sv.hurst_exponent > 0.6 and sv.signed_volume_ratio > 0.2:
            return RegimeType.TRENDING_BULL
        if sv.momentum_acceleration < -0.001 and sv.hurst_exponent > 0.6 and sv.signed_volume_ratio < -0.2:
            return RegimeType.TRENDING_BEAR
        if sv.realized_vol > 0.03 and sv.local_entropy > 0.8 and abs(sv.zscore_return) > 2:
            return RegimeType.STRESS
        if sv.hurst_exponent < 0.4 and abs(sv.autocorrelation_signature) > 0.3:
            return RegimeType.MEAN_REVERTING
        if sv.spread_normalized > 0.002 and sv.depth_asymmetry > 3.0:
            return RegimeType.LOW_LIQUIDITY
        return RegimeType.MEAN_REVERTING

    def _compute_regime_stability(self) -> float:
        if len(self.regime_history) < self.stability_window:
            return 0.5
        recent = self.regime_history[-self.stability_window :]
        most_common = max(set(recent), key=recent.count)
        return recent.count(most_common) / len(recent)

    def _assess_tradeability(self, sv: StateVector, stability: float) -> bool:
        if sv.data_quality_score < 0.6:
            return False
        if sv.spread_normalized > 0.005:
            return False
        if stability < 0.4:
            return False
        if sv.local_entropy > 0.9:
            return False
        return True

    def _compute_regime_strength(self, sv: StateVector) -> float:
        components = [
            min(abs(sv.momentum_acceleration) * 100, 1.0),
            min(abs(sv.signed_volume_ratio), 1.0),
            min(abs(sv.autocorrelation_signature), 1.0),
        ]
        return float(np.mean(components))

    def _estimate_persistence(self) -> int:
        stability = self._compute_regime_stability()
        return int(np.clip(round(stability * 50), 5, 50))
