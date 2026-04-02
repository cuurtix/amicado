from typing import Dict, List

import lightgbm as lgb
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from core.base_types import ForecastDistribution, RegimeState


class ForecastEngine:
    """Prévisions probabilistes multi-horizons avec calibration."""

    def __init__(self, horizons: List[int] = None):
        self.horizons = horizons or [1, 3, 5, 15]
        self.direction_models = {}
        self.return_models = {}
        self.quantile_models = {}
        self.is_fitted = False

    def fit(self, X: np.ndarray, y_data: Dict[int, Dict]):
        for h in self.horizons:
            if h not in y_data:
                continue

            y_dir = y_data[h]["direction"]
            y_ret = y_data[h]["returns"]

            base_clf = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                min_child_samples=50,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            )

            tscv = TimeSeriesSplit(n_splits=5)
            self.direction_models[h] = CalibratedClassifierCV(base_clf, cv=tscv, method="isotonic")
            self.direction_models[h].fit(X, y_dir)

            self.return_models[h] = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=25,
                min_child_samples=50,
                random_state=42,
                verbose=-1,
            )
            self.return_models[h].fit(X, y_ret)

            self.quantile_models[h] = {}
            for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
                model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=42,
                    verbose=-1,
                )
                model.fit(X, y_ret)
                self.quantile_models[h][q] = model

        self.is_fitted = True
        return self

    def predict_distribution(self, state_vector: np.ndarray, regime: RegimeState, horizon: int = 1) -> ForecastDistribution:
        if not self.is_fitted or horizon not in self.direction_models:
            return self._default_forecast(horizon)

        X = state_vector.reshape(1, -1)
        dir_probs = self.direction_models[horizon].predict_proba(X)[0]
        classes = self.direction_models[horizon].classes_

        prob_map = dict(zip(classes, dir_probs))
        p_up = float(prob_map.get(1, 0.33))
        p_down = float(prob_map.get(-1, 0.33))
        p_flat = float(prob_map.get(0, 0.34))

        expected_return = float(self.return_models[horizon].predict(X)[0])
        quantiles = {q: float(model.predict(X)[0]) for q, model in self.quantile_models[horizon].items()}

        model_confidence = self._compute_model_confidence(p_up, p_down, p_flat, quantiles, regime)
        prob_touch_levels = self._estimate_touch_probabilities(expected_return, quantiles)

        return ForecastDistribution(
            p_up=p_up,
            p_down=p_down,
            p_flat=p_flat,
            expected_return=expected_return,
            return_quantiles=quantiles,
            expected_range=quantiles[0.9] - quantiles[0.1],
            prob_touch_levels=prob_touch_levels,
            confidence=model_confidence,
            model_agreement=self._compute_model_agreement(p_up, expected_return),
            horizon_bars=horizon,
        )

    def _default_forecast(self, horizon: int) -> ForecastDistribution:
        return ForecastDistribution(
            p_up=0.33,
            p_down=0.33,
            p_flat=0.34,
            expected_return=0.0,
            return_quantiles={0.1: -0.002, 0.25: -0.001, 0.5: 0.0, 0.75: 0.001, 0.9: 0.002},
            expected_range=0.004,
            prob_touch_levels={"target": 0.25, "stop": 0.1},
            confidence=0.0,
            model_agreement=0.0,
            horizon_bars=horizon,
        )

    def _compute_model_confidence(
        self,
        p_up: float,
        p_down: float,
        p_flat: float,
        quantiles: Dict[float, float],
        regime: RegimeState,
    ) -> float:
        directional_clarity = max(p_up, p_down, p_flat) - (1 / 3)
        quantile_coherence = 1.0
        q_values = [quantiles[q] for q in sorted(quantiles.keys())]
        for i in range(len(q_values) - 1):
            if q_values[i] > q_values[i + 1]:
                quantile_coherence -= 0.2
        regime_penalty = 1.0 - (1.0 - regime.stability_score) * 0.5
        return float(np.clip(directional_clarity * quantile_coherence * regime_penalty, 0.0, 1.0))

    def _estimate_touch_probabilities(self, expected_return: float, quantiles: Dict[float, float]) -> Dict[str, float]:
        if expected_return > 0:
            _ = quantiles[0.75]
            prob_target = 0.25
        else:
            _ = quantiles[0.25]
            prob_target = 0.25
        _ = quantiles[0.1] if expected_return > 0 else quantiles[0.9]
        return {"target": prob_target, "stop": 0.1}

    def _compute_model_agreement(self, p_up: float, expected_return: float) -> float:
        directional_sign = 1 if p_up >= 0.5 else -1
        return_sign = 1 if expected_return >= 0 else -1
        return 1.0 if directional_sign == return_sign else 0.0
