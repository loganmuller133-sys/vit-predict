#!/usr/bin/env python
# scripts/train_all_models.py
"""Train all 12 models on historical data"""

import json
import asyncio
import logging
import sys
sys.path.append('/workspaces/vit-predict')

from services.ml_service.models.model_1_poisson import PoissonGoalModel
from services.ml_service.models.model_2_xgboost import XGBoostOutcomeClassifier
from services.ml_service.models.model_3_lstm import LSTMMomentumNetwork
from services.ml_service.models.model_4_monte_carlo import MonteCarloEngine
from services.ml_service.models.model_5_ensemble_agg import EnsembleAggregator
from services.ml_service.models.model_6_transformer import TransformerSequenceModel
from services.ml_service.models.model_7_gnn import GraphNeuralNetworkModel
from services.ml_service.models.model_8_bayesian import BayesianHierarchicalModel
from services.ml_service.models.model_9_rl_agent import RLPolicyAgent
from services.ml_service.models.model_10_causal import CausalInferenceModel
from services.ml_service.models.model_11_sentiment import SentimentFusionModel
from services.ml_service.models.model_12_anomaly import AnomalyRegimeDetectionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train all models"""
    
    MODELS = {
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
    
    def __init__(self, data_path: str = "/workspaces/vit-predict/data/historical_matches.json"):
        self.data_path = data_path
        self.matches = []
        self.results = {}
    
    def load_data(self):
        """Load historical match data"""
        logger.info(f"Loading data from {self.data_path}")
        try:
            with open(self.data_path, 'r') as f:
                self.matches = json.load(f)
            logger.info(f"Loaded {len(self.matches)} matches")
            return True
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            return False
    
    def train_model(self, name: str, model) -> dict:
        """Train a single model"""
        logger.info(f"=== Training {name} ===")
        
        try:
            # Try training with different parameter sets based on model type
            if name == 'poisson':
                result = model.train(
                    self.matches,
                    validation_split=0.2,
                    use_time_weights=True
                )
            elif name in ['xgboost', 'lstm', 'transformer', 'gnn', 'bayesian']:
                result = model.train(
                    self.matches,
                    validation_split=0.2
                )
            elif name in ['monte_carlo', 'ensemble', 'rl_agent']:
                result = model.train(self.matches)
            elif name in ['causal', 'sentiment', 'anomaly']:
                result = model.train(self.matches, validation_split=0.2)
            else:
                result = model.train(self.matches)
            
            # Save model
            model_path = f"/workspaces/vit-predict/models/{name}_model.pkl"
            model.save(model_path)
            logger.info(f"✅ {name} trained and saved to {model_path}")
            
            return {
                'name': name,
                'status': 'success',
                'accuracy': result.get('accuracy', 0) if isinstance(result, dict) else 0,
                'message': 'Training completed successfully'
            }
        except Exception as e:
            logger.error(f"❌ {name} training failed: {e}")
            return {
                'name': name,
                'status': 'failed',
                'error': str(e)
            }
    
    def train_all(self):
        """Train all models sequentially"""
        logger.info("=" * 60)
        logger.info("Starting model training for all 12 models")
        logger.info("=" * 60)
        
        if not self.load_data():
            return False
        
        for name, model in self.MODELS.items():
            result = self.train_model(name, model)
            self.results[name] = result
            logger.info("")
        
        # Summary
        logger.info("=" * 60)
        logger.info("Training Summary")
        logger.info("=" * 60)
        
        successful = sum(1 for r in self.results.values() if r.get('status') == 'success')
        failed = sum(1 for r in self.results.values() if r.get('status') == 'failed')
        
        for name, result in self.results.items():
            status_icon = "✅" if result.get('status') == 'success' else "❌"
            logger.info(f"{status_icon} {name}: {result.get('status')}")
        
        logger.info("=" * 60)
        logger.info(f"Successfully trained: {successful}/{len(self.MODELS)}")
        logger.info(f"Failed: {failed}/{len(self.MODELS)}")
        logger.info("=" * 60)
        
        return successful > 0


async def main():
    """Main training workflow"""
    trainer = ModelTrainer()
    trainer.train_all()


if __name__ == "__main__":
    asyncio.run(main())