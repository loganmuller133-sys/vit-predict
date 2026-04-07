# services/ml-service/models/model_12_anomaly.py
import numpy as np
import pandas as pd
import logging
import pickle
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

from app.models.base_model import BaseModel, MarketType, Session

logger = logging.getLogger(__name__)


@dataclass
class AnomalyConfig:
    """Configuration for Anomaly Detection Model."""
    contamination: float = 0.1
    n_estimators: int = 100
    max_samples: str = 'auto'
    random_state: int = 42
    window_size: int = 50
    threshold_zscore: float = 3.0
    threshold_iqr: float = 1.5
    regime_detection_window: int = 100
    retraining_threshold: float = 0.15
    drift_psi_threshold: float = 0.2
    regime_shift_persistence: int = 3


# Global outcome mapping for consistency
OUTCOME_MAP = {
    "H": "home",
    "D": "draw", 
    "A": "away",
    "home": "home",
    "draw": "draw",
    "away": "away",
    "1": "home",
    "X": "draw",
    "2": "away"
}


class AnomalyRegimeDetectionModel(BaseModel):
    """
    Anomaly & Regime Detection Model V2 - Fixed and production-ready.

    Fixes applied:
        - Fixed dataclass import typo
        - Fixed prediction stream format (numeric array, not dict)
        - Fixed outcome comparison logic
        - Normalized outcome mapping
        - Fixed Elliptic Envelope feature space
        - Removed unused OneClassSVM
        - Robust PSI with percentile bins
        - Capped drift impact on health score
        - Persistent regime detection
        - Conservative retraining triggers
    """

    def __init__(
        self,
        model_name: str,
        weight: float = 1.0,
        version: int = 2,
        params: Optional[Dict[str, Any]] = None,
        contamination: float = 0.1,
        window_size: int = 50,
        regime_window: int = 100,
        threshold_zscore: float = 3.0,
        retraining_threshold: float = 0.15
    ):
        super().__init__(
            model_name=model_name,
            model_type="Anomaly",
            weight=weight,
            version=version,
            params=params,
            supported_markets=[
                MarketType.MATCH_ODDS,
                MarketType.OVER_UNDER,
                MarketType.BTTS
            ]
        )

        # Configuration
        self.config = AnomalyConfig(
            contamination=contamination,
            n_estimators=100,
            max_samples='auto',
            random_state=42,
            window_size=window_size,
            threshold_zscore=threshold_zscore,
            threshold_iqr=1.5,
            regime_detection_window=regime_window,
            retraining_threshold=retraining_threshold,
            drift_psi_threshold=0.2,
            regime_shift_persistence=3
        )

        # Anomaly detectors
        self.isolation_forest: Optional[IsolationForest] = None
        self.elliptic_envelope: Optional[EllipticEnvelope] = None

        # Scalers
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)

        # Historical data streams (numeric arrays only)
        self.prediction_stream: deque = deque(maxlen=10000)  # Stores [home, draw, away] arrays
        self.outcome_stream: deque = deque(maxlen=10000)    # Stores 1 for correct, 0 for incorrect
        self.edge_stream: deque = deque(maxlen=10000)       # Stores edge values
        self.model_confidence_stream: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.market_odds_stream: deque = deque(maxlen=10000)

        # Performance metrics over time
        self.rolling_accuracy: deque = deque(maxlen=100)
        self.rolling_ev: deque = deque(maxlen=100)
        self.rolling_calibration: deque = deque(maxlen=100)

        # Regime state
        self.current_regime: str = "normal"
        self.regime_confidence: float = 1.0
        self.regime_history: deque = deque(maxlen=1000)
        self.regime_shift_detected: bool = False
        self.last_regime_shift: Optional[datetime] = None
        self.regime_shift_counter: int = 0

        # Anomaly flags
        self.recent_anomalies: deque = deque(maxlen=100)
        self.alert_history: deque = deque(maxlen=1000)

        # Drift metrics
        self.psi_scores: deque = deque(maxlen=100)
        self.drift_detected: bool = False

        # Training metadata
        self.trained_matches_count: int = 0
        self.baseline_statistics: Dict[str, float] = {}
        self.feature_columns: List[str] = []

        # Retraining trigger
        self.retraining_needed: bool = False
        self.retraining_reason: Optional[str] = None

        # PCA state
        self.pca_fitted: bool = False

        # Always certified (uses sklearn which is core)
        self.certified = True

    def _calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI) using percentile bins.
        Robust against outliers.
        """
        # Use percentiles for bin edges (robust to outliers)
        percentiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(np.concatenate([expected, actual]), percentiles)
        bin_edges = np.unique(bin_edges)  # Remove duplicates

        if len(bin_edges) < 2:
            return 0.0

        # Calculate distributions
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        # Add small epsilon to avoid division by zero
        expected_pct = (expected_counts + 1e-8) / (len(expected) + 1e-8)
        actual_pct = (actual_counts + 1e-8) / (len(actual) + 1e-8)

        # Calculate PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

        return float(psi)

    def _calculate_cusum(
        self,
        series: np.ndarray,
        threshold: float = 1.0
    ) -> List[int]:
        """
        Cumulative Sum (CUSUM) for change point detection.
        Detects when a process has shifted.
        """
        mean = np.mean(series)
        cusum_pos = np.zeros(len(series))
        cusum_neg = np.zeros(len(series))
        change_points = []

        for i in range(1, len(series)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + (series[i] - mean - threshold))
            cusum_neg[i] = min(0, cusum_neg[i-1] + (series[i] - mean + threshold))

            if cusum_pos[i] > threshold * 5:
                change_points.append(i)
                cusum_pos[i] = 0
            if cusum_neg[i] < -threshold * 5:
                change_points.append(i)
                cusum_neg[i] = 0

        return change_points

    def _detect_regime_shift(self) -> Tuple[str, float, bool]:
        """
        Detect regime shifts using multiple signals with persistence.
        Requires multiple consecutive signals to trigger.
        """
        if len(self.rolling_accuracy) < self.config.regime_detection_window:
            return "normal", 1.0, False

        # Convert to arrays
        acc_array = np.array(list(self.rolling_accuracy))
        ev_array = np.array(list(self.rolling_ev))

        # Calculate rolling statistics
        acc_mean = np.mean(acc_array[-50:])
        acc_std = np.std(acc_array[-50:])
        acc_recent = np.mean(acc_array[-10:])

        ev_mean = np.mean(ev_array[-50:])
        ev_std = np.std(ev_array[-50:])
        ev_recent = np.mean(ev_array[-10:])

        # Detect degradation with z-score
        acc_degradation_z = (acc_mean - acc_recent) / (acc_std + 1e-8)
        ev_degradation_z = (ev_mean - ev_recent) / (ev_std + 1e-8)

        # CUSUM change point detection
        change_points = self._calculate_cusum(acc_array)
        recent_change = any(cp > len(acc_array) - 20 for cp in change_points)

        # Persistent regime detection
        is_degrading = acc_degradation_z > 2.0 or ev_degradation_z > 2.0

        if is_degrading:
            self.regime_shift_counter += 1
        else:
            self.regime_shift_counter = max(0, self.regime_shift_counter - 1)

        # Require persistence before triggering
        shift_detected = self.regime_shift_counter >= self.config.regime_shift_persistence

        # Regime classification
        if shift_detected:
            regime = "degrading"
            confidence = min(0.95, 0.5 + acc_degradation_z / 10)
        elif recent_change:
            regime = "transition"
            confidence = 0.7
            shift_detected = False
        elif acc_std > 0.15:
            regime = "volatile"
            confidence = 0.8
            shift_detected = False
        else:
            regime = "normal"
            confidence = 0.9
            shift_detected = False
            self.regime_shift_counter = 0

        return regime, confidence, shift_detected

    def _detect_data_drift(self) -> Tuple[bool, float, str]:
        """
        Detect data drift using PSI on key features.
        """
        if len(self.prediction_stream) < 200:
            return False, 0.0, "insufficient_data"

        # Convert prediction stream to array (flatten probabilities)
        pred_array = np.array(list(self.prediction_stream))

        # Split into baseline and recent
        baseline = pred_array[:100]
        recent = pred_array[-100:]

        # Flatten for PSI calculation
        baseline_flat = baseline.flatten()
        recent_flat = recent.flatten()

        # Calculate PSI
        psi = self._calculate_psi(baseline_flat, recent_flat)

        self.psi_scores.append(psi)

        if psi > 0.25:
            return True, psi, "significant_drift"
        elif psi > self.config.drift_psi_threshold:
            return True, psi, "moderate_drift"
        else:
            return False, psi, "stable"

    def _detect_outliers(
        self,
        features_scaled: np.ndarray,
        features_pca: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect outliers using multiple methods.
        Now receives both scaled and PCA features.
        """
        if features_scaled.shape[0] < 10:
            return {"is_anomaly": False, "methods": {}, "score": 0.0, "votes": 0, "total_methods": 0}

        results = {}
        anomaly_votes = 0
        total_methods = 0

        # Isolation Forest (uses scaled features)
        if self.isolation_forest is not None:
            pred = self.isolation_forest.predict(features_scaled)
            results['isolation_forest'] = pred == -1
            anomaly_votes += results['isolation_forest'].sum()
            total_methods += 1

        # Elliptic Envelope (uses PCA features)
        if self.elliptic_envelope is not None:
            pred = self.elliptic_envelope.predict(features_pca)
            results['elliptic_envelope'] = pred == -1
            anomaly_votes += results['elliptic_envelope'].sum()
            total_methods += 1

        # Z-score method
        if features_scaled.shape[1] > 0:
            z_scores = np.abs(stats.zscore(features_scaled, axis=0))
            results['zscore'] = (z_scores > self.config.threshold_zscore).any(axis=1)
            anomaly_votes += results['zscore'].sum()
            total_methods += 1

        # IQR method
        Q1 = np.percentile(features_scaled, 25, axis=0)
        Q3 = np.percentile(features_scaled, 75, axis=0)
        IQR = Q3 - Q1
        results['iqr'] = ((features_scaled < Q1 - self.config.threshold_iqr * IQR) | 
                          (features_scaled > Q3 + self.config.threshold_iqr * IQR)).any(axis=1)
        anomaly_votes += results['iqr'].sum()
        total_methods += 1

        # Consensus score (proportion of methods flagging anomaly)
        consensus_score = anomaly_votes / (total_methods * features_scaled.shape[0]) if total_methods > 0 else 0

        return {
            "is_anomaly": consensus_score > 0.5,
            "methods": {k: bool(v[0]) if len(v) > 0 else False for k, v in results.items()},
            "score": float(consensus_score),
            "votes": int(anomaly_votes),
            "total_methods": total_methods
        }

    def _detect_performance_anomaly(self) -> Dict[str, Any]:
        """
        Detect anomalies in model performance.
        """
        if len(self.rolling_accuracy) < 20:
            return {"has_anomaly": False, "reason": "insufficient_data"}

        recent_acc = np.mean(list(self.rolling_accuracy)[-10:])
        historical_acc = np.mean(list(self.rolling_accuracy)[-50:-10]) if len(self.rolling_accuracy) >= 50 else recent_acc
        acc_drop = historical_acc - recent_acc

        recent_ev = np.mean(list(self.rolling_ev)[-10:])
        historical_ev = np.mean(list(self.rolling_ev)[-50:-10]) if len(self.rolling_ev) >= 50 else recent_ev
        ev_drop = historical_ev - recent_ev

        anomalies = []

        if acc_drop > 0.08:
            anomalies.append(f"accuracy_dropped_{acc_drop:.1%}")
        if ev_drop > 0.04:
            anomalies.append(f"ev_dropped_{ev_drop:.1%}")

        # Consecutive losses
        if len(self.outcome_stream) > 0:
            recent_outcomes = list(self.outcome_stream)[-20:]
            consecutive_losses = 0
            for o in reversed(recent_outcomes):
                if o == 0:
                    consecutive_losses += 1
                else:
                    break

            if consecutive_losses > 5:
                anomalies.append(f"consecutive_losses_{consecutive_losses}")

        return {
            "has_anomaly": len(anomalies) > 0,
            "anomalies": anomalies,
            "acc_drop": float(acc_drop),
            "ev_drop": float(ev_drop)
        }

    def train(
        self,
        matches: List[Dict[str, Any]],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train anomaly detection models on historical data.
        """
        if not matches:
            return {"error": "No training data"}

        logger.info(f"Training anomaly detection on {len(matches)} matches")

        # Prepare training features
        X = []

        for match in matches:
            features = []

            # Model predictions
            for model_name in ['poisson', 'xgboost', 'lstm', 'transformer', 'gnn', 'bayesian']:
                features.append(match.get(f'{model_name}_home_prob', 0.33))
                features.append(match.get(f'{model_name}_draw_prob', 0.33))
                features.append(match.get(f'{model_name}_away_prob', 0.33))
                features.append(match.get(f'{model_name}_confidence', 0.5))

            # Market features
            features.append(match.get('home_odds', 2.0))
            features.append(match.get('draw_odds', 3.2))
            features.append(match.get('away_odds', 2.0))
            features.append(match.get('odds_movement', 0))

            # Match features
            features.append(match.get('home_goals', 0))
            features.append(match.get('away_goals', 0))
            features.append(match.get('total_goals', 0))

            X.append(features)

        X = np.array(X)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # PCA for Elliptic Envelope
        if X_scaled.shape[1] > 10:
            X_pca = self.pca.fit_transform(X_scaled)
            self.pca_fitted = True
        else:
            X_pca = X_scaled
            self.pca_fitted = False

        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.config.contamination,
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state
        )
        self.isolation_forest.fit(X_scaled)

        # Train Elliptic Envelope
        self.elliptic_envelope = EllipticEnvelope(
            contamination=self.config.contamination,
            random_state=self.config.random_state
        )
        self.elliptic_envelope.fit(X_pca)

        # Calculate baseline statistics
        self.baseline_statistics = {
            'mean_accuracy': float(np.mean([m.get('accuracy', 0.5) for m in matches])),
            'std_accuracy': float(np.std([m.get('accuracy', 0.5) for m in matches])),
            'mean_ev': float(np.mean([m.get('edge', 0) for m in matches])),
            'std_ev': float(np.std([m.get('edge', 0) for m in matches])),
            'mean_confidence': float(np.mean([m.get('confidence', 0.5) for m in matches])),
            'std_confidence': float(np.std([m.get('confidence', 0.5) for m in matches]))
        }

        self.trained_matches_count = len(matches)
        self.feature_columns = [f'feature_{i}' for i in range(X.shape[1])]

        logger.info(f"Training complete. Baseline accuracy: {self.baseline_statistics['mean_accuracy']:.3f}")

        return {
            "model_type": self.model_type,
            "version": self.version,
            "matches_trained": self.trained_matches_count,
            "baseline_statistics": self.baseline_statistics,
            "contamination": self.config.contamination,
            "window_size": self.config.window_size,
            "feature_dimension": X.shape[1],
            "pca_fitted": self.pca_fitted
        }

    def update_stream(
        self,
        prediction: Dict[str, float],
        actual_outcome: str,
        actual_home_goals: int,
        actual_away_goals: int,
        market_odds: Dict[str, float],
        model_name: str = "ensemble"
    ):
        """
        Update data streams with new match result.
        """
        # Normalize outcome
        actual_outcome_norm = OUTCOME_MAP.get(actual_outcome, actual_outcome)

        # Prediction probabilities as numeric array (FIXED)
        self.prediction_stream.append([
            prediction.get('home_prob', 0.33),
            prediction.get('draw_prob', 0.33),
            prediction.get('away_prob', 0.33)
        ])

        # Determine predicted outcome (FIXED)
        outcome_probs = {
            "home": prediction.get('home_prob', 0),
            "draw": prediction.get('draw_prob', 0),
            "away": prediction.get('away_prob', 0)
        }
        predicted_outcome = max(outcome_probs, key=outcome_probs.get)

        # Check correctness
        is_correct = (predicted_outcome == actual_outcome_norm)
        self.outcome_stream.append(1 if is_correct else 0)

        # Calculate edge (FIXED with normalized outcome)
        if actual_outcome_norm in market_odds:
            odds = market_odds[actual_outcome_norm]
            prob = outcome_probs.get(actual_outcome_norm, 0.33)
            edge = (prob * odds - 1) if is_correct else -1
            self.edge_stream.append(edge)
        else:
            self.edge_stream.append(-1 if not is_correct else 0.1)

        # Model confidence
        self.model_confidence_stream[model_name].append(prediction.get('confidence', 0.5))

        # Market odds
        self.market_odds_stream.append(market_odds)

        # Update rolling metrics
        if len(self.outcome_stream) >= 10:
            recent_outcomes = list(self.outcome_stream)[-50:]
            self.rolling_accuracy.append(np.mean(recent_outcomes))

        if len(self.edge_stream) >= 10:
            recent_edges = list(self.edge_stream)[-50:]
            self.rolling_ev.append(np.mean(recent_edges))

        if len(self.model_confidence_stream[model_name]) >= 10:
            recent_conf = list(self.model_confidence_stream[model_name])[-50:]
            predicted_probs = [p[0] for p in list(self.prediction_stream)[-50:]]  # home prob
            actual_outcomes_binary = [1 if o == 1 else 0 for o in list(self.outcome_stream)[-50:]]
            calibration_error = np.mean((np.array(predicted_probs) - np.array(actual_outcomes_binary)) ** 2)
            self.rolling_calibration.append(calibration_error)

    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies and regime shifts in current state.
        """
        # Build feature vector for current state
        current_features = []

        # Model predictions
        for model_name in ['poisson', 'xgboost', 'lstm', 'transformer', 'gnn', 'bayesian']:
            current_features.append(features.get(f'{model_name}_home_prob', 0.33))
            current_features.append(features.get(f'{model_name}_draw_prob', 0.33))
            current_features.append(features.get(f'{model_name}_away_prob', 0.33))
            current_features.append(features.get(f'{model_name}_confidence', 0.5))

        # Market features
        market_odds = features.get('market_odds', {})
        current_features.append(market_odds.get('home', 2.0))
        current_features.append(market_odds.get('draw', 3.2))
        current_features.append(market_odds.get('away', 2.0))
        current_features.append(features.get('odds_movement', 0))

        current_features = np.array(current_features).reshape(1, -1)

        # Scale features
        if hasattr(self.scaler, 'mean_'):
            current_features_scaled = self.scaler.transform(current_features)
        else:
            current_features_scaled = current_features

        # Apply PCA if fitted
        if self.pca_fitted and hasattr(self.pca, 'components_'):
            current_features_pca = self.pca.transform(current_features_scaled)
        else:
            current_features_pca = current_features_scaled

        # Detect anomalies (FIXED: pass both scaled and PCA)
        anomaly_result = self._detect_outliers(current_features_scaled, current_features_pca)

        # Detect regime shift
        regime, regime_confidence, shift_detected = self._detect_regime_shift()

        # Detect data drift
        drift_detected, drift_psi, drift_level = self._detect_data_drift()

        # Detect performance anomaly
        performance_anomaly = self._detect_performance_anomaly()

        # Update state
        self.current_regime = regime
        self.regime_confidence = regime_confidence
        self.drift_detected = drift_detected

        # Conservative retraining trigger (FIXED)
        self.retraining_needed = (
            (shift_detected and regime_confidence > 0.7) or
            (drift_detected and drift_psi > 0.2) or
            (performance_anomaly['has_anomaly'] and len(performance_anomaly['anomalies']) >= 2)
        )

        if shift_detected and regime_confidence > 0.7:
            self.last_regime_shift = datetime.now()
            self.regime_history.append({
                'timestamp': self.last_regime_shift,
                'regime': regime,
                'confidence': regime_confidence
            })
            self.retraining_reason = f"regime_shift_to_{regime}"
        elif drift_detected and drift_psi > 0.2:
            self.retraining_reason = f"data_drift_psi_{drift_psi:.2f}"
        elif performance_anomaly['has_anomaly'] and len(performance_anomaly['anomalies']) >= 2:
            self.retraining_reason = ", ".join(performance_anomaly['anomalies'][:2])
        else:
            self.retraining_reason = None

        # Calculate overall health score (capped drift impact)
        health_score = 1.0
        if anomaly_result['is_anomaly']:
            health_score -= 0.3
        if shift_detected:
            health_score -= 0.2
        if drift_detected:
            health_score -= min(drift_psi, 0.3)  # Capped at 0.3
        if performance_anomaly['has_anomaly']:
            health_score -= min(len(performance_anomaly['anomalies']) * 0.1, 0.3)

        health_score = max(0, min(1, health_score))

        # Determine alert level
        if health_score < 0.5:
            alert_level = "CRITICAL"
        elif health_score < 0.7:
            alert_level = "WARNING"
        else:
            alert_level = "NORMAL"

        return {
            "is_anomaly": anomaly_result['is_anomaly'],
            "anomaly_score": anomaly_result['score'],
            "anomaly_votes": anomaly_result['votes'],
            "anomaly_methods": list(anomaly_result['methods'].keys()),

            "current_regime": self.current_regime,
            "regime_confidence": self.regime_confidence,
            "regime_shift_detected": shift_detected,
            "last_regime_shift": self.last_regime_shift.isoformat() if self.last_regime_shift else None,

            "data_drift_detected": drift_detected,
            "data_drift_psi": drift_psi,
            "data_drift_level": drift_level,

            "performance_anomaly": performance_anomaly['has_anomaly'],
            "performance_anomalies": performance_anomaly.get('anomalies', []),
            "accuracy_drop": performance_anomaly.get('acc_drop', 0),
            "ev_drop": performance_anomaly.get('ev_drop', 0),

            "retraining_needed": self.retraining_needed,
            "retraining_reason": self.retraining_reason,

            "health_score": health_score,
            "alert_level": alert_level,

            "statistics": {
                "rolling_accuracy": float(np.mean(list(self.rolling_accuracy))) if self.rolling_accuracy else 0.5,
                "rolling_ev": float(np.mean(list(self.rolling_ev))) if self.rolling_ev else 0,
                "baseline_accuracy": self.baseline_statistics.get('mean_accuracy', 0.5),
                "samples_since_retrain": self.trained_matches_count
            },

            "confidence": {
                "1x2": health_score,
                "over_under": health_score,
                "btts": health_score
            }
        }

    def should_retrain(self) -> Tuple[bool, str]:
        """
        Determine if models should be retrained based on anomaly detection.
        """
        return self.retraining_needed, self.retraining_reason

    def get_regime_summary(self) -> Dict[str, Any]:
        """
        Get summary of recent regime history.
        """
        if not self.regime_history:
            return {"current_regime": self.current_regime, "recent_shifts": []}

        recent_shifts = list(self.regime_history)[-10:]

        return {
            "current_regime": self.current_regime,
            "regime_confidence": self.regime_confidence,
            "recent_shifts": [
                {
                    "timestamp": s['timestamp'].isoformat(),
                    "regime": s['regime'],
                    "confidence": s['confidence']
                }
                for s in recent_shifts
            ],
            "total_shifts": len(self.regime_history)
        }

    def get_confidence_score(self, market: str = "1x2") -> float:
        """Return confidence based on health score."""
        return self.get_health_score()

    def get_health_score(self) -> float:
        """Get overall system health score."""
        score = 1.0

        if self.current_regime == "degrading":
            score -= 0.3
        elif self.current_regime == "volatile":
            score -= 0.15

        if self.drift_detected:
            score -= 0.2

        if self.rolling_accuracy:
            recent_acc = np.mean(list(self.rolling_accuracy)[-10:])
            if recent_acc < 0.5:
                score -= 0.2

        return max(0, min(1, score))

    def save(self, path: str) -> None:
        """Save model to disk."""
        save_data = {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'version': self.version,
            'weight': self.weight,
            'params': self.params,
            'status': self.status,
            'config': self.config,
            'isolation_forest': self.isolation_forest,
            'elliptic_envelope': self.elliptic_envelope,
            'scaler': self.scaler,
            'pca': self.pca,
            'pca_fitted': self.pca_fitted,
            'baseline_statistics': self.baseline_statistics,
            'feature_columns': self.feature_columns,
            'trained_matches_count': self.trained_matches_count,
            'prediction_stream': list(self.prediction_stream),
            'outcome_stream': list(self.outcome_stream),
            'edge_stream': list(self.edge_stream),
            'rolling_accuracy': list(self.rolling_accuracy),
            'rolling_ev': list(self.rolling_ev),
            'rolling_calibration': list(self.rolling_calibration),
            'regime_history': list(self.regime_history),
            'current_regime': self.current_regime,
            'regime_confidence': self.regime_confidence,
            'session_accuracies': {k.value: v for k, v in self.session_accuracies.items()},
            'final_score': self.final_score,
            'certified': self.certified
        }

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"Anomaly Detection Model V{self.version} saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model_id = data['model_id']
        self.model_name = data['model_name']
        self.model_type = data['model_type']
        self.version = data.get('version', 2)
        self.weight = data['weight']
        self.params = data['params']
        self.status = data['status']
        self.config = data['config']
        self.isolation_forest = data['isolation_forest']
        self.elliptic_envelope = data['elliptic_envelope']
        self.scaler = data['scaler']
        self.pca = data['pca']
        self.pca_fitted = data.get('pca_fitted', False)
        self.baseline_statistics = data['baseline_statistics']
        self.feature_columns = data['feature_columns']
        self.trained_matches_count = data['trained_matches_count']
        self.prediction_stream = deque(data['prediction_stream'], maxlen=10000)
        self.outcome_stream = deque(data['outcome_stream'], maxlen=10000)
        self.edge_stream = deque(data['edge_stream'], maxlen=10000)
        self.rolling_accuracy = deque(data['rolling_accuracy'], maxlen=100)
        self.rolling_ev = deque(data['rolling_ev'], maxlen=100)
        self.rolling_calibration = deque(data['rolling_calibration'], maxlen=100)
        self.regime_history = deque(data['regime_history'], maxlen=1000)
        self.current_regime = data['current_regime']
        self.regime_confidence = data['regime_confidence']

        # Restore certification data
        for session_val, accuracy in data.get('session_accuracies', {}).items():
            self.session_accuracies[Session(session_val)] = accuracy
        self.final_score = data.get('final_score')
        self.certified = data.get('certified', False)

        logger.info(f"Anomaly Detection Model V{self.version} loaded from {path}")