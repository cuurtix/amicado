import numpy as np
from core.base_types import StateVector


def state_vector_to_numpy(sv: StateVector) -> np.ndarray:
    """
    Convertit le StateVector en vecteur numérique exploitable par ForecastEngine.
    L'ordre des features doit rester stable.
    """
    return np.array([
        sv.returns_multi_horizon[0],
        sv.returns_multi_horizon[1],
        sv.returns_multi_horizon[2],
        sv.realized_vol,
        sv.atr_normalized,
        sv.momentum_acceleration,
        sv.order_book_imbalance,
        sv.signed_volume_ratio,
        sv.spread_normalized,
        sv.depth_asymmetry,
        sv.distance_to_liquidity_pools,
        float(sv.break_of_structure_score),
        sv.compression_ratio,
        sv.hurst_exponent,
        sv.local_entropy,
        sv.zscore_return,
        sv.autocorrelation_signature,
        sv.session_liquidity_score,
        sv.event_proximity,
        sv.data_quality_score,
        sv.confidence,
    ], dtype=float)