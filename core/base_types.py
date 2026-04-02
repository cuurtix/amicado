from dataclasses import dataclass, field
from typing import Dict
from enum import Enum
import numpy as np
from datetime import datetime


@dataclass
class StateVector:
    """Représentation unifiée de l'état du marché à l'instant t."""

    timestamp: datetime

    # Price action quantifié
    returns_multi_horizon: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    realized_vol: float = 0.0
    atr_normalized: float = 0.0
    momentum_acceleration: float = 0.0

    # Microstructure
    order_book_imbalance: float = 0.0
    signed_volume_ratio: float = 0.0
    spread_normalized: float = 0.0
    depth_asymmetry: float = 1.0

    # Structure de marché
    distance_to_liquidity_pools: float = 0.0
    break_of_structure_score: int = 0  # -1, 0, 1
    compression_ratio: float = 0.0

    # Statistiques avancées
    hurst_exponent: float = 0.5
    local_entropy: float = 1.0
    zscore_return: float = 0.0
    autocorrelation_signature: float = 0.0

    # Contexte et qualité
    session_liquidity_score: float = 0.0
    event_proximity: float = 1.0
    data_quality_score: float = 1.0
    confidence: float = 0.0


@dataclass
class ForecastDistribution:
    """Distribution complète des scénarios futurs."""

    p_up: float
    p_down: float
    p_flat: float

    expected_return: float
    return_quantiles: Dict[float, float]

    expected_range: float
    prob_touch_levels: Dict[str, float]

    confidence: float
    model_agreement: float
    horizon_bars: int


class RegimeType(Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    STRESS = "stress"


@dataclass
class RegimeState:
    regime_type: RegimeType
    regime_strength: float  # [0, 1]
    stability_score: float  # [0, 1]
    is_tradeable: bool
    expected_persistence_bars: int
