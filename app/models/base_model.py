# app/models/base_model.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4
from collections import deque
import asyncio
from enum import IntEnum
import numpy as np


class Session(IntEnum):
    PHASE_1 = 1
    PHASE_2 = 2
    PHASE_3 = 3


class MarketType(IntEnum):
    """Supported betting markets"""
    MATCH_ODDS = 1      # 1X2
    OVER_UNDER = 2      # Over/Under 2.5 goals
    BTTS = 3            # Both Teams to Score
    EXACT_SCORE = 4     # Exact score prediction
    HANDICAP = 5        # Asian Handicap (Phase 2)
    HALF_TIME = 6       # Half-time/Full-time (Phase 2)


class BaseModel(ABC):
    """
    Abstract base class for all VIT child models.

    Implements:
        - Model registry fields (id, type, version, params, status)
        - Certification session tracking
        - Core lifecycle methods (train, predict, save, load)
        - Multi-market prediction support (1X2, O/U, BTTS, Exact Score)
        - Confidence scoring per market
        - Batch prediction (with optional override for vectorised inference)
        - Diversity / error correlation for hybrid weighting
    """

    def __init__(
        self,
        model_name: str,
        model_type: str,
        weight: float = 1.0,
        version: int = 1,
        params: Optional[Dict[str, Any]] = None,
        supported_markets: Optional[List[MarketType]] = None,
    ):
        self.model_id: str = str(uuid4())
        self.model_name: str = model_name
        self.model_type: str = model_type
        self.version: int = version
        self.weight: float = weight
        self.params: Dict[str, Any] = params or {}
        self.status: str = "TRAINING"  # TRAINING, CERTIFIED, EXCLUDED, ACTIVE, ARCHIVED
        
        # Which markets this model supports
        self.supported_markets: List[MarketType] = supported_markets or [
            MarketType.MATCH_ODDS  # All models must support at least 1X2
        ]

        # Certification tracking per market
        self.session_accuracies: Dict[Session, Optional[float]] = {
            Session.PHASE_1: None,
            Session.PHASE_2: None,
            Session.PHASE_3: None
        }
        self.session_metrics: Dict[Session, Dict[str, float]] = {
            Session.PHASE_1: {},
            Session.PHASE_2: {},
            Session.PHASE_3: {}
        }
        self.final_score: Optional[float] = None
        self.certified: bool = False

        # Track recent errors for diversity calculation (per market)
        self._recent_errors: Dict[str, deque[float]] = {
            "1x2": deque(maxlen=100),
            "over_under": deque(maxlen=100),
            "btts": deque(maxlen=100)
        }

    # --------------------------------------------------------------------------
    # Core abstract methods (must be implemented by child)
    # --------------------------------------------------------------------------

    @abstractmethod
    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict multiple markets for a match.
        
        Returns:
        {
            # Required for all models
            "home_prob": float,      # 0-1
            "draw_prob": float,      # 0-1
            "away_prob": float,      # 0-1
            
            # Optional (if model supports Over/Under)
            "over_2_5_prob": float,  # 0-1
            "under_2_5_prob": float, # 0-1
            
            # Optional (if model supports BTTS)
            "btts_prob": float,      # 0-1
            "no_btts_prob": float,   # 0-1
            
            # Optional (if model supports Exact Score)
            "exact_score_probs": Dict[str, float],  # e.g., {"1-0": 0.12, "2-1": 0.08}
            
            # Expected goals (for models that calculate them)
            "home_goals_expectation": float,
            "away_goals_expectation": float,
            
            # Per-market confidence scores
            "confidence": {
                "1x2": float,        # 0-1
                "over_under": float, # 0-1 (if supported)
                "btts": float        # 0-1 (if supported)
            }
        }
        """
        pass

    @abstractmethod
    def get_confidence_score(self, market: str = "1x2") -> float:
        """
        Returns model's confidence for a specific market.
        
        Args:
            market: "1x2", "over_under", or "btts"
        
        Returns:
            Confidence score between 0 and 1
        """
        pass

    @abstractmethod
    def train(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the model on historical match data.
        
        Returns training metrics (loss, accuracy, etc.) per market:
        {
            "1x2_accuracy": float,
            "over_under_accuracy": float,  # if supported
            "btts_accuracy": float,        # if supported
            "log_loss": float,
            "brier_score": float
        }
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialize model to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass

    # --------------------------------------------------------------------------
    # Concrete methods
    # --------------------------------------------------------------------------

    async def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict outcomes for multiple matches concurrently.
        Returns full prediction dicts (all markets) for each match.
        """
        tasks = [self.predict(features) for features in features_list]
        return await asyncio.gather(*tasks)

    def get_error_correlation(self, other_errors: List[float], market: str = "1x2") -> float:
        """
        Compute Pearson correlation with another model's errors for a specific market.
        """
        errors = self._recent_errors.get(market, deque(maxlen=100))
        if not errors or not other_errors:
            return 0.0

        min_len = min(len(errors), len(other_errors))
        if min_len == 0:
            return 0.0

        corr = np.corrcoef(list(errors)[:min_len], other_errors[:min_len])[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def log_error(self, error: float, market: str = "1x2") -> None:
        """Record a prediction error for diversity tracking for a specific market."""
        if market in self._recent_errors:
            self._recent_errors[market].append(error)
        else:
            # Fallback to 1x2
            self._recent_errors["1x2"].append(error)

    def supports_market(self, market: Union[MarketType, str]) -> bool:
        """Check if model supports a specific market."""
        if isinstance(market, str):
            market_map = {
                "1x2": MarketType.MATCH_ODDS,
                "over_under": MarketType.OVER_UNDER,
                "btts": MarketType.BTTS,
                "exact_score": MarketType.EXACT_SCORE
            }
            market = market_map.get(market, MarketType.MATCH_ODDS)
        return market in self.supported_markets

    def get_diversity_multiplier(self, all_other_errors: List[List[float]], market: str = "1x2") -> float:
        """
        Calculate diversity multiplier for hybrid weighting.
        Formula: 1.0 + (0.5 × (1 - avg_correlation))
        """
        if not all_other_errors:
            return 1.0
        
        correlations = []
        for other_errors in all_other_errors:
            corr = self.get_error_correlation(other_errors, market)
            correlations.append(corr)
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        return 1.0 + (0.5 * (1 - avg_correlation))

    # --------------------------------------------------------------------------
    # Registry helpers
    # --------------------------------------------------------------------------

    def to_registry_entry(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "params": self.params,
            "status": self.status,
            "supported_markets": [m.value for m in self.supported_markets],
        }

    def certification_summary(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "session_accuracies": {k.name: v for k, v in self.session_accuracies.items()},
            "session_metrics": {k.name: v for k, v in self.session_metrics.items()},
            "final_score": self.final_score,
            "certified": self.certified,
            "supported_markets": [m.name for m in self.supported_markets],
        }

    # --------------------------------------------------------------------------
    # Helper methods for child models
    # --------------------------------------------------------------------------

    def normalize_probabilities(self, probs: Dict[str, float]) -> Dict[str, float]:
        """Ensure probability distributions sum to 1.0."""
        result = probs.copy()
        
        # Normalize 1X2
        if all(k in probs for k in ["home_prob", "draw_prob", "away_prob"]):
            total = probs["home_prob"] + probs["draw_prob"] + probs["away_prob"]
            if total > 0:
                result["home_prob"] = probs["home_prob"] / total
                result["draw_prob"] = probs["draw_prob"] / total
                result["away_prob"] = probs["away_prob"] / total
        
        # Normalize Over/Under
        if all(k in probs for k in ["over_2_5_prob", "under_2_5_prob"]):
            total = probs["over_2_5_prob"] + probs["under_2_5_prob"]
            if total > 0:
                result["over_2_5_prob"] = probs["over_2_5_prob"] / total
                result["under_2_5_prob"] = probs["under_2_5_prob"] / total
        
        # Normalize BTTS
        if all(k in probs for k in ["btts_prob", "no_btts_prob"]):
            total = probs["btts_prob"] + probs["no_btts_prob"]
            if total > 0:
                result["btts_prob"] = probs["btts_prob"] / total
                result["no_btts_prob"] = probs["no_btts_prob"] / total
        
        return result