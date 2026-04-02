import asyncio
import logging
from datetime import datetime

from config import AppConfig
from utils import state_vector_to_numpy

from core.data_engine import DataEngine
from core.feature_engine import FeatureEngine
from core.regime_engine import RegimeEngine
from core.forecast_engine import ForecastEngine
from core.policy_engine import PolicyEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


class TradingPredictionApp:
    def __init__(self, config: AppConfig):
        self.config = config

        self.data_engine = DataEngine(symbol=config.symbol)
        self.feature_engine = FeatureEngine()
        self.regime_engine = RegimeEngine()
        self.forecast_engine = ForecastEngine(horizons=[1, 3, 5, 15])
        self.policy_engine = PolicyEngine()

        self._stream_task = None

    async def start(self):
        """
        Lance les streams puis la boucle principale de prédiction.
        """
        logging.info("Initialisation de l'application pour %s", self.config.symbol)

        self._stream_task = asyncio.create_task(
            self.data_engine.ingest_websocket_multi_stream(self.config.endpoints)
        )

        await self.run_prediction_loop()

    async def run_prediction_loop(self):
        """
        Boucle principale : récupère les données, construit les features,
        détecte le régime, prédit, puis logge le résultat.
        """
        while True:
            try:
                synchronized_data = self.data_engine.get_synchronized_data(
                    lookback=self.config.sync_lookback
                )

                if synchronized_data is None:
                    logging.info("Données insuffisantes pour prédire, attente...")
                    await asyncio.sleep(self.config.prediction_interval_seconds)
                    continue

                state_vector = self.feature_engine.build_state_vector(synchronized_data)
                if state_vector is None:
                    logging.warning("StateVector non construit : données insuffisantes.")
                    await asyncio.sleep(self.config.prediction_interval_seconds)
                    continue

                regime = self.regime_engine.detect_regime(state_vector)
                x = state_vector_to_numpy(state_vector)

                forecast = self.forecast_engine.predict_distribution(
                    state_vector=x,
                    regime=regime,
                    horizon=self.config.forecast_horizon,
                )

                current_price = self._extract_current_price(synchronized_data)
                decision = self.policy_engine.make_decision(
                    forecast=forecast,
                    state_vector=state_vector,
                    regime=regime,
                    current_price=current_price,
                )

                self._print_prediction(
                    current_price=current_price,
                    regime=regime,
                    forecast=forecast,
                    decision=decision,
                )

            except Exception as e:
                logging.exception("Erreur dans la boucle principale: %s", e)

            await asyncio.sleep(self.config.prediction_interval_seconds)

    def _extract_current_price(self, synchronized_data: dict) -> float:
        """
        Extrait le dernier prix depuis les données 1m.
        Adapte cette logique si ton payload réel a un autre schéma.
        """
        candles_1m = synchronized_data["ohlcv"]["1m"]
        last_candle = candles_1m[-1]

        if isinstance(last_candle, dict):
            return float(last_candle["close"])

        raise ValueError("Format de bougie non supporté pour extraire le prix courant.")

    def _print_prediction(self, current_price, regime, forecast, decision):
        """
        Affichage lisible console. Peut être remplacé par export JSON, DB, API, etc.
        """
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        logging.info("=" * 80)
        logging.info("Heure                : %s", now_str)
        logging.info("Symbole              : %s", self.config.symbol)
        logging.info("Prix courant         : %.2f", current_price)

        logging.info(
            "Régime               : %s | strength=%.3f | stabilité=%.3f | tradeable=%s",
            regime.regime_type.value,
            regime.regime_strength,
            regime.stability_score,
            regime.is_tradeable,
        )

        logging.info(
            "Prévision h=%s       : p_up=%.3f | p_down=%.3f | p_flat=%.3f",
            forecast.horizon_bars,
            forecast.p_up,
            forecast.p_down,
            forecast.p_flat,
        )
        logging.info(
            "Expected return      : %.6f | expected range=%.6f | confidence=%.3f | agreement=%.3f",
            forecast.expected_return,
            forecast.expected_range,
            forecast.confidence,
            forecast.model_agreement,
        )
        logging.info(
            "Quantiles            : q10=%.6f | q25=%.6f | q50=%.6f | q75=%.6f | q90=%.6f",
            forecast.return_quantiles.get(0.1, 0.0),
            forecast.return_quantiles.get(0.25, 0.0),
            forecast.return_quantiles.get(0.5, 0.0),
            forecast.return_quantiles.get(0.75, 0.0),
            forecast.return_quantiles.get(0.9, 0.0),
        )

        logging.info(
            "Décision             : %s | valid=%s | confidence=%.3f | edge_net=%.6f | R/R=%.3f",
            decision.action,
            decision.is_valid,
            decision.confidence,
            decision.edge_net,
            decision.risk_reward_ratio,
        )
        logging.info("Raison               : %s", decision.reason)
        logging.info("=" * 80)


async def main():
    config = AppConfig()
    app = TradingPredictionApp(config)
    await app.start()


if __name__ == "__main__":
    asyncio.run(main())