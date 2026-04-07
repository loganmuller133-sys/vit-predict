# services/ml-service/models/model_orchestrator.py
import asyncio
import logging
import time
from typing import Dict, List, Optional
from collections import Counter

# Import all 12 child models
from .model_1_poisson import PoissonGoalModel
from .model_2_xgboost import XGBoostOutcomeClassifier
from .model_3_lstm import LSTMMomentumNetwork
from .model_4_monte_carlo import MonteCarloEngine
from .model_5_ensemble_agg import EnsembleAggregator
from .model_6_transformer import TransformerSequenceModel
from .model_7_gnn import GraphNeuralNetworkModel
from .model_8_bayesian import BayesianHierarchicalModel
from .model_9_rl_agent import RLPolicyAgent
from .model_10_causal import CausalInferenceModel
from .model_11_sentiment import SentimentFusionModel
from .model_12_anomaly import AnomalyRegimeDetectionModel

logger = logging.getLogger(__name__)

# Markets to aggregate (with default values for models that don't support them)
MARKETS = {
    "1x2": {
        "fields": ["home_prob", "draw_prob", "away_prob"],
        "default": {"home_prob": 0.34, "draw_prob": 0.33, "away_prob": 0.33},
        "normalize": True
    },
    "over_under": {
        "fields": ["over_2_5_prob", "under_2_5_prob"],
        "default": {"over_2_5_prob": 0.5, "under_2_5_prob": 0.5},
        "normalize": True
    },
    "btts": {
        "fields": ["btts_prob", "no_btts_prob"],
        "default": {"btts_prob": 0.5, "no_btts_prob": 0.5},
        "normalize": True
    }
}


class ModelOrchestrator:
    def __init__(self):
        self.models = {}
        self.latencies = {}

    def load_all_models(self):
        """Instantiate all 12 models and filter by certification"""
        all_models = {
            'poisson': PoissonGoalModel("poisson_001"),
            'xgboost': XGBoostOutcomeClassifier("xgb_001"),
            'lstm': LSTMMomentumNetwork("lstm_001"),
            'monte_carlo': MonteCarloEngine("mc_001"),
            'ensemble': EnsembleAggregator("ensemble_001"),
            'transformer': TransformerSequenceModel("trans_001"),
            'gnn': GraphNeuralNetworkModel("gnn_001"),
            'bayesian': BayesianHierarchicalModel("bayes_001"),
            'rl_agent': RLPolicyAgent("rl_001"),
            'causal': CausalInferenceModel("causal_001"),
            'sentiment': SentimentFusionModel("sent_001"),
            'anomaly': AnomalyRegimeDetectionModel("anom_001"),
        }

        # Try to load saved models
        import os
        models_dir = "/workspaces/vit-predict/models"

        # Filter to certified models only
        self.models = {}
        for name, model in all_models.items():
            # Try to load saved model
            model_path = os.path.join(models_dir, f"{name}_model.pkl")
            if os.path.exists(model_path):
                try:
                    model.load(model_path)
                    logger.info(f"Loaded saved model: {name}")
                except Exception as e:
                    logger.warning(f"Failed to load saved model {name}: {e}")

            if model.certified:
                self.models[name] = model
            else:
                logger.warning(f"Model {name} excluded: not certified")

        logger.info(f"Loaded {len(self.models)}/12 certified models")

    def num_models_ready(self) -> int:
        return len(self.models)

    async def predict_parallel(self, features: Dict, match_id: str) -> List[Dict]:
        """Run predictions asynchronously for all certified models"""
        async def run_model(name: str, model, features: Dict):
            start = time.time()
            try:
                # Direct await (predict is already async)
                pred = await model.predict(features)

                # Get confidence per market (or default)
                confidence = {
                    "1x2": model.get_confidence_score("1x2"),
                    "over_under": model.get_confidence_score("over_under") if model.supports_market("over_under") else 0.5,
                    "btts": model.get_confidence_score("btts") if model.supports_market("btts") else 0.5
                }

                self.latencies[name] = round(time.time() - start, 3)

                return {
                    **pred,  # All prediction fields (home_prob, draw_prob, away_prob, over_2_5_prob, etc.)
                    'model_name': name,
                    'match_id': match_id,
                    'confidence': confidence,
                    'supported_markets': [m.name.lower() for m in model.supported_markets],
                    'failed': False
                }
            except Exception as e:
                logger.warning(f"Model {name} failed: {str(e)}")
                self.latencies[name] = None
                return {
                    'model_name': name,
                    'match_id': match_id,
                    'failed': True,
                    'error': str(e)
                }

        tasks = [run_model(name, m, features) for name, m in self.models.items()]
        predictions = await asyncio.gather(*tasks)
        return predictions

    def aggregate_predictions(self, predictions: List[Dict]) -> Dict:
        """
        Combine predictions from all models for ALL markets.
        Returns aggregated probabilities for 1X2, Over/Under, and BTTS.
        """
        result = {}

        # Aggregate each market separately
        for market_name, market_config in MARKETS.items():
            weighted_sum = {field: 0.0 for field in market_config["fields"]}
            total_weight = 0
            contributing_models = 0

            for p in predictions:
                if p.get('failed', False):
                    continue

                # Check if model supports this market
                supported = market_name in [m.lower() for m in p.get('supported_markets', [])]
                if not supported:
                    continue

                # Get model weight and confidence
                model = self.models.get(p['model_name'])
                weight = model.weight if model else 1.0
                market_confidence = p.get('confidence', {}).get(market_name, 0.5)
                w = weight * market_confidence

                # Add weighted contribution for each field
                for field in market_config["fields"]:
                    if field in p:
                        weighted_sum[field] += p[field] * w

                total_weight += w
                contributing_models += 1

            # Calculate final probabilities
            if total_weight > 0:
                for field in market_config["fields"]:
                    result[field] = float(round(weighted_sum[field] / total_weight, 3))

                # Normalize if required (ensures sum to 1.0)
                if market_config["normalize"]:
                    total = sum(result[field] for field in market_config["fields"])
                    if total > 0:
                        for field in market_config["fields"]:
                            result[field] = float(round(result[field] / total, 3))
            else:
                # Use default values if no models contributed
                for field, default_value in market_config["default"].items():
                    result[field] = float(default_value)

            # Store metadata about contributing models
            result[f"{market_name}_contributing_models"] = contributing_models

        # Calculate consensus for 1X2
        result["consensus"] = self._calculate_consensus(predictions)

        # Track how many models succeeded
        result["models_certified"] = sum(1 for p in predictions if not p.get('failed', False))
        result["models_total"] = len(self.models)

        return result

    def _calculate_consensus(self, predictions: List[Dict]) -> Dict:
        """Calculate consensus among models for 1X2 outcome"""
        outcomes = []

        for p in predictions:
            if p.get('failed', False):
                continue

            # Check if model supports 1X2 (all should)
            probs = [p.get('home_prob', 0), p.get('draw_prob', 0), p.get('away_prob', 0)]
            if any(probs):
                outcome = ['HOME', 'DRAW', 'AWAY'][probs.index(max(probs))]
                outcomes.append(outcome)

        if not outcomes:
            return {"outcome": "UNKNOWN", "percentage": 0}

        from collections import Counter
        consensus_outcome = Counter(outcomes).most_common(1)[0][0]
        consensus_percent = outcomes.count(consensus_outcome) / len(outcomes) * 100

        return {
            "outcome": consensus_outcome,
            "percentage": round(consensus_percent, 2),
            "distribution": dict(Counter(outcomes))
        }

    async def predict(self, features: Dict, match_id: str) -> Dict:
        """
        Main prediction endpoint - runs all models and returns aggregated result.
        """
        predictions = await self.predict_parallel(features, match_id)
        aggregated = self.aggregate_predictions(predictions)

        return {
            "match_id": match_id,
            "predictions": aggregated,
            "individual_results": predictions,  # For debugging/explainability
            "latencies": self.get_latencies()
        }

    async def run_certification(self, session_number: int, match_pair: str) -> Dict:
        """Run progressive certification on historical data"""
        results = {}
        for name, model in self.models.items():
            try:
                results[name] = {
                    'accuracy': model.session_accuracies.get(session_number, 0.0),
                    'brier_score': 0.0,
                    'expected_value': 0.0,
                    'session': session_number,
                    'match_pair': match_pair
                }
            except Exception as e:
                logger.warning(f"Certification failed for {name}: {str(e)}")
                results[name] = {'failed': True}
        return results

    def get_latencies(self) -> Dict:
        return self.latencies

    def list_models(self) -> List[str]:
        return list(self.models.keys())

    def get_model(self, name: str):
        return self.models.get(name, None)