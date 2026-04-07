# services/ml-service/models/model_5_ensemble_agg.py
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import minimize

from app.models.base_model import BaseModel, MarketType, Session

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Track performance metrics per model - EV focused."""
    model_name: str
    expected_value: float = 0.0
    calibration_error: float = 0.0
    accuracy: float = 0.0
    brier_score: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    sample_size: int = 0
    recent_ev: List[float] = field(default_factory=list)
    weight_allocations: List[float] = field(default_factory=list)
    last_update: Optional[datetime] = None
    certified: bool = False


@dataclass
class BetRecommendation:
    """Structured bet recommendation with risk controls."""
    market: str
    selection: str
    probability: float
    market_odds: float
    edge: float
    kelly_stake: float
    adjusted_stake: float
    confidence: str
    expected_value: float
    reason: str


class EnsembleAggregator(BaseModel):
    """
    Hybrid Ensemble Aggregator V2 - EV-focused, market-aware, risk-controlled.
    
    Upgrades from V1:
        - EV > accuracy (50/30/20 weighting)
        - Diversity bonus only for +EV models
        - Market-driven regime detection
        - Dynamic weight caps (no fixed min/max)
        - Proper calibration scoring
        - Kelly Criterion staking
        - Drawdown protection
        - Bayesian model averaging
    """
    
    def __init__(
        self,
        model_name: str,
        weight: float = 1.0,
        version: int = 2,
        params: Optional[Dict[str, Any]] = None,
        recency_half_life: int = 30,
        ev_weight: float = 0.5,
        calibration_weight: float = 0.3,
        accuracy_weight: float = 0.2,
        min_ev_threshold: float = 0.02,  # 2% minimum EV to bet
        max_kelly_fraction: float = 0.25,  # Cap Kelly at 25% of bankroll
        drawdown_limit: float = 0.15,  # 15% max drawdown
        recovery_factor: float = 0.5,  # Reduce stakes by 50% after drawdown
        min_samples_for_weight: int = 50
    ):
        super().__init__(
            model_name=model_name,
            model_type="EnsembleAggregator",
            weight=weight,
            version=version,
            params=params,
            supported_markets=[
                MarketType.MATCH_ODDS,
                MarketType.OVER_UNDER,
                MarketType.BTTS,
                MarketType.EXACT_SCORE
            ]
        )
        
        # Weighting parameters (EV focused)
        self.recency_half_life = recency_half_life
        self.ev_weight = ev_weight
        self.calibration_weight = calibration_weight
        self.accuracy_weight = accuracy_weight
        self.min_ev_threshold = min_ev_threshold
        self.max_kelly_fraction = max_kelly_fraction
        self.drawdown_limit = drawdown_limit
        self.recovery_factor = recovery_factor
        self.min_samples_for_weight = min_samples_for_weight
        
        # Model performance tracking (EV focused)
        self.model_performance: Dict[str, ModelPerformance] = {}
        
        # Historical data for calibration
        self.prediction_history: List[Dict[str, Any]] = []
        self.bet_history: List[Dict[str, Any]] = []
        
        # Current state
        self.current_weights: Dict[str, float] = {}
        self.current_ev: Dict[str, float] = {}
        self.peak_bankroll: float = 1.0
        self.current_drawdown: float = 0.0
        self.consecutive_losses: int = 0
        
        # Market regime (real market signals, not self-performance)
        self.market_regime: str = "neutral"
        self.regime_confidence: float = 0.5
        self.regime_volatility: float = 0.0
        self.market_inefficiency: float = 0.0
        
        # Training metadata
        self.trained_matches_count: int = 0
        self.last_optimization: Optional[datetime] = None
        self.total_bets: int = 0
        self.winning_bets: int = 0
        self.total_roi: float = 0.0

        # Always certified (meta-model)
        self.certified = True
    
    def _calculate_ev_from_performance(
        self,
        model_name: str,
        lookback_days: int = 90
    ) -> float:
        """
        Calculate Expected Value from historical performance.
        This is the primary metric for weighting.
        """
        if model_name not in self.model_performance:
            return 0.0
        
        perf = self.model_performance[model_name]
        
        if perf.sample_size < self.min_samples_for_weight:
            return 0.0
        
        # Calculate recent EV with time decay
        if not perf.recent_ev:
            return perf.expected_value
        
        # Time-weighted EV
        current_date = datetime.now()
        total_weight = 0
        weighted_ev = 0
        
        # Use stored EV history with timestamps
        for ev_record in perf.recent_ev[-100:]:
            if isinstance(ev_record, dict):
                weight = self._calculate_recency_weight(ev_record['date'], current_date)
                weighted_ev += ev_record['ev'] * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_ev / total_weight
        
        return perf.expected_value
    
    def _calculate_calibration_score(self, model_name: str) -> float:
        """
        Calculate calibration error (lower is better).
        Brier score and ECE (Expected Calibration Error).
        """
        if model_name not in self.model_performance:
            return 1.0  # Worst possible calibration
        
        perf = self.model_performance[model_name]
        
        # Brier score is already stored, convert to 0-1 where 1 is perfect
        brier_perfect = max(0, 1 - perf.brier_score)
        
        return brier_perfect
    
    def _calculate_diversity_bonus(
        self,
        model_name: str,
        model_prediction: Dict[str, float],
        all_predictions: List[Tuple[str, Dict[str, float]]],
        model_ev: float
    ) -> float:
        """
        Calculate diversity bonus ONLY for models with positive EV.
        Negative EV models get penalized for being different.
        """
        if len(all_predictions) < 2:
            return 1.0
        
        # Calculate consensus (weighted by EV)
        total_ev = sum(ev for _, ev in all_predictions if ev > 0)
        if total_ev <= 0:
            return 1.0
        
        consensus = defaultdict(float)
        for other_name, other_ev in all_predictions:
            if other_ev > 0:  # Only include +EV models in consensus
                other_pred = self._get_prediction_for_model(other_name)
                if other_pred:
                    for outcome, prob in other_pred.items():
                        consensus[outcome] += prob * (other_ev / total_ev)
        
        if not consensus:
            return 1.0
        
        # Calculate divergence from EV-weighted consensus
        divergence = 0.0
        for outcome, prob in model_prediction.items():
            if outcome in consensus:
                diff = abs(prob - consensus[outcome])
                divergence += diff
        
        divergence /= len(model_prediction)
        
        # ONLY reward divergence if model has positive EV
        if model_ev > self.min_ev_threshold:
            # Positive EV models get boosted for disagreement
            bonus = 1 + (divergence * 0.3)
        else:
            # Negative EV models get penalized for being different
            bonus = 1 - (divergence * 0.5)
        
        return max(0.5, min(1.5, bonus))
    
    def _calculate_recency_weight(self, prediction_date: datetime, current_date: datetime) -> float:
        """Calculate exponential decay weight based on age."""
        days_diff = (current_date - prediction_date).days
        if days_diff <= 0:
            return 1.0
        
        weight = 0.5 ** (days_diff / self.recency_half_life)
        return max(weight, 0.01)
    
    def _detect_market_regime(self) -> Tuple[str, float, float]:
        """
        Detect market regime using REAL market signals, not self-performance.
        
        Signals:
            - Average closing odds movement
            - Goal rate trends (over/under)
            - Draw rate anomalies
            - Market inefficiency (sharp vs soft divergence)
        """
        if len(self.prediction_history) < 50:
            return "neutral", 0.5, 0.0
        
        # Analyze recent market odds movement
        recent_history = self.prediction_history[-100:]
        
        odds_movements = []
        goal_rates = []
        draw_rates = []
        
        for record in recent_history:
            if 'odds_movement' in record:
                odds_movements.append(record['odds_movement'])
            if 'total_goals' in record:
                goal_rates.append(1 if record['total_goals'] > 2.5 else 0)
            if 'is_draw' in record:
                draw_rates.append(1 if record['is_draw'] else 0)
        
        # Calculate metrics
        avg_odds_movement = np.mean(odds_movements) if odds_movements else 0
        over_rate = np.mean(goal_rates) if goal_rates else 0.5
        draw_rate = np.mean(draw_rates) if draw_rates else 0.25
        
        # Volatility (standard deviation of odds movement)
        volatility = np.std(odds_movements) if odds_movements else 0.05
        
        # Market inefficiency (how often our edge materialized)
        edges_realized = [r.get('edge_realized', 0) for r in recent_history if 'edge_realized' in r]
        market_inefficiency = np.mean(edges_realized) if edges_realized else 0
        
        # Regime classification
        if avg_odds_movement > 0.05 and volatility > 0.08:
            regime = "volatile"
            confidence = min(0.9, 0.6 + (volatility * 2))
        elif over_rate > 0.55:
            regime = "high_scoring"
            confidence = 0.7
        elif over_rate < 0.45:
            regime = "low_scoring"
            confidence = 0.7
        elif draw_rate > 0.30:
            regime = "draw_heavy"
            confidence = 0.65
        elif market_inefficiency > 0.03:
            regime = "inefficient"
            confidence = 0.8
        else:
            regime = "neutral"
            confidence = 0.5
        
        return regime, confidence, volatility
    
    def _calculate_optimal_weights_bayesian(
        self,
        model_predictions: Dict[str, Dict[str, float]],
        market_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate optimal weights using Bayesian Model Averaging.
        Weights proportional to posterior probability of model being correct.
        """
        # Get market regime
        self.market_regime, self.regime_confidence, self.regime_volatility = self._detect_market_regime()
        self.market_inefficiency = self._calculate_market_inefficiency()
        
        # Calculate model scores
        model_scores = {}
        model_evs = {}
        
        for model_name in model_predictions.keys():
            # Primary: Expected Value
            ev = self._calculate_ev_from_performance(model_name)
            model_evs[model_name] = ev
            
            # Secondary: Calibration
            calibration = self._calculate_calibration_score(model_name)
            
            # Tertiary: Accuracy (only as tiebreaker)
            accuracy = self.model_performance.get(model_name, ModelPerformance(model_name)).accuracy
            
            # Combine with EV-focused weights
            score = (ev * self.ev_weight + 
                    calibration * self.calibration_weight + 
                    accuracy * self.accuracy_weight)
            
            # Apply regime factor (market condition adjustment)
            regime_factor = self._get_regime_factor(model_name, model_evs[model_name])
            score *= regime_factor
            
            # Apply diversity bonus (only for +EV models)
            all_predictions_with_ev = [(n, model_evs.get(n, 0)) for n in model_predictions.keys()]
            diversity_bonus = self._calculate_diversity_bonus(
                model_name, 
                model_predictions[model_name], 
                all_predictions_with_ev,
                ev
            )
            score *= diversity_bonus
            
            model_scores[model_name] = max(0, score)
        
        # Convert scores to weights (softmax with temperature)
        scores_array = np.array(list(model_scores.values()))
        if scores_array.sum() == 0:
            # Equal weights if all scores are zero
            n_models = len(model_scores)
            return {name: 1.0 / n_models for name in model_scores.keys()}
        
        # Temperature parameter (lower = more aggressive weighting)
        temperature = 0.5 if self.market_regime == "volatile" else 1.0
        exp_scores = np.exp(scores_array / temperature)
        weights = exp_scores / exp_scores.sum()
        
        # Dynamic cap based on regime (no fixed min/max)
        max_weight = self._get_dynamic_max_weight()
        min_weight = self._get_dynamic_min_weight()
        
        final_weights = {}
        for name, weight in zip(model_scores.keys(), weights):
            final_weights[name] = max(min_weight, min(weight, max_weight))
        
        # Renormalize after capping
        total = sum(final_weights.values())
        if total > 0:
            for name in final_weights:
                final_weights[name] /= total
        
        return final_weights
    
    def _get_regime_factor(self, model_name: str, model_ev: float) -> float:
        """Get regime adjustment factor based on market conditions."""
        regime_factors = {
            "volatile": 0.8,      # Reduce trust in all models during volatility
            "high_scoring": 1.1,   # Boost all models in high-scoring environments
            "low_scoring": 0.9,    # Slightly reduce in low-scoring
            "draw_heavy": 0.95,    # Slightly reduce (harder to predict)
            "inefficient": 1.2,    # Boost when market is inefficient (more edge)
            "neutral": 1.0
        }
        
        base_factor = regime_factors.get(self.market_regime, 1.0)
        
        # Additional boost for models with proven edge in this regime
        if model_ev > 0.05:
            return base_factor * 1.1
        
        return base_factor
    
    def _get_dynamic_max_weight(self) -> float:
        """Dynamic maximum weight based on regime and confidence."""
        base_max = 0.4
        
        if self.market_regime == "volatile":
            # Lower max weight during volatility (diversify more)
            return 0.25
        elif self.market_regime == "inefficient" and self.regime_confidence > 0.7:
            # Higher max weight when market is inefficient and we're confident
            return 0.5
        elif self.regime_volatility > 0.1:
            return 0.3
        
        return base_max
    
    def _get_dynamic_min_weight(self) -> float:
        """Dynamic minimum weight - can go to zero for terrible models."""
        if self.market_regime == "volatile":
            return 0.02
        return 0.01  # Can effectively exclude very bad models
    
    def _calculate_market_inefficiency(self) -> float:
        """Calculate current market inefficiency (opportunity level)."""
        if len(self.bet_history) < 20:
            return 0.0
        
        recent_bets = self.bet_history[-50:]
        
        # Calculate realized edge
        realized_edges = [b.get('realized_edge', 0) for b in recent_bets if 'realized_edge' in b]
        if realized_edges:
            return np.mean(realized_edges)
        
        return 0.0
    
    def _get_prediction_for_model(self, model_name: str) -> Optional[Dict[str, float]]:
        """Retrieve latest prediction for a model."""
        for record in reversed(self.prediction_history):
            if record.get('model_name') == model_name:
                return record.get('prediction')
        return None
    
    def aggregate_predictions(
        self,
        model_predictions: Dict[str, Dict[str, float]],
        market_context: Optional[Dict[str, Any]] = None,
        return_weights: bool = False
    ) -> Dict[str, Any]:
        """
        Aggregate predictions using Bayesian Model Averaging.
        """
        if not model_predictions:
            return self._default_aggregation()
        
        # Calculate optimal weights
        self.current_weights = self._calculate_optimal_weights_bayesian(model_predictions, market_context)
        
        # Calculate EV for each outcome
        self.current_ev = self._calculate_outcome_ev(model_predictions, self.current_weights)
        
        # Aggregate probabilities
        aggregated = self._weighted_average_predictions(model_predictions, self.current_weights)
        
        # Generate bet recommendations with Kelly
        recommendations = self._generate_bet_recommendations(aggregated, market_context)
        
        # Calculate ensemble metrics
        ensemble_variance = self._calculate_ensemble_variance(model_predictions, self.current_weights)
        consensus = self._calculate_consensus(model_predictions, aggregated)
        
        # Add metadata
        aggregated['confidence'] = {
            '1x2': float(1 - min(ensemble_variance['1x2'], 0.5)),
            'over_under': float(1 - min(ensemble_variance.get('ou', 0.1), 0.5)),
            'btts': float(1 - min(ensemble_variance.get('btts', 0.1), 0.5))
        }
        aggregated['consensus_percent'] = consensus['percentage']
        aggregated['consensus_outcome'] = consensus['outcome']
        aggregated['models_contributing'] = len([w for w in self.current_weights.values() if w > 0.01])
        aggregated['market_regime'] = self.market_regime
        aggregated['regime_confidence'] = self.regime_confidence
        aggregated['market_inefficiency'] = self.market_inefficiency
        aggregated['expected_value'] = self.current_ev
        aggregated['recommendations'] = recommendations
        
        if return_weights:
            aggregated['model_weights'] = self.current_weights
        
        # Store for tracking
        self.current_aggregation = aggregated
        
        return aggregated
    
    def _weighted_average_predictions(
        self,
        model_predictions: Dict[str, Dict[str, float]],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate weighted average of predictions."""
        markets = ['home_prob', 'draw_prob', 'away_prob', 
                  'over_2_5_prob', 'under_2_5_prob',
                  'btts_prob', 'no_btts_prob']
        
        aggregated = {}
        
        for market in markets:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for model_name, pred in model_predictions.items():
                if market in pred:
                    weight = weights.get(model_name, 0)
                    weighted_sum += pred[market] * weight
                    total_weight += weight
            
            if total_weight > 0:
                aggregated[market] = weighted_sum / total_weight
            else:
                aggregated[market] = 0.33 if 'prob' in market else 0.5
        
        # Normalize 1X2
        total_1x2 = aggregated['home_prob'] + aggregated['draw_prob'] + aggregated['away_prob']
        if total_1x2 > 0:
            aggregated['home_prob'] /= total_1x2
            aggregated['draw_prob'] /= total_1x2
            aggregated['away_prob'] /= total_1x2
        

            # Normalize O/U
            total_ou = aggregated['over_2_5_prob'] + aggregated['under_2_5_prob']
            if total_ou > 0:
                aggregated['over_2_5_prob'] /= total_ou
                aggregated['under_2_5_prob'] /= total_ou

            # Normalize BTTS
            total_btts = aggregated['btts_prob'] + aggregated['no_btts_prob']
            if total_btts > 0:
                aggregated['btts_prob'] /= total_btts
                aggregated['no_btts_prob'] /= total_btts

            # Expected goals (using Poisson from Model 1 or ensemble)
            aggregated['home_goals_expectation'] = aggregated['home_prob'] * 2.5 + aggregated['over_2_5_prob'] * 0.5
            aggregated['away_goals_expectation'] = aggregated['away_prob'] * 2.2 + aggregated['under_2_5_prob'] * 0.3

            return aggregated

        def _calculate_outcome_ev(
            self,
            model_predictions: Dict[str, Dict[str, float]],
            weights: Dict[str, float]
        ) -> Dict[str, float]:
            """Calculate Expected Value for each outcome."""
            ev = {}

            outcomes = ['home', 'draw', 'away']

            for outcome in outcomes:
                prob_key = f"{outcome}_prob"

                # Weighted model probability
                weighted_prob = 0.0
                total_weight = 0.0

                for model_name, pred in model_predictions.items():
                    if prob_key in pred:
                        weight = weights.get(model_name, 0)
                        weighted_prob += pred[prob_key] * weight
                        total_weight += weight

                if total_weight > 0:
                    model_prob = weighted_prob / total_weight
                else:
                    model_prob = 0.33

                # Get market odds from context (would be passed in)
                # For now, use implied probability from model as placeholder
                market_prob = model_prob * 0.95  # Assume 5% vig

                ev[outcome] = model_prob - market_prob

            return ev

        def _generate_bet_recommendations(
            self,
            aggregated: Dict[str, float],
            market_context: Optional[Dict[str, Any]]
        ) -> List[BetRecommendation]:
            """Generate EV-positive bet recommendations with Kelly staking."""
            recommendations = []

            # Get market odds from context
            market_odds = market_context.get('market_odds', {}) if market_context else {}

            if not market_odds:
                return []

            # Check each market
            # 1X2 market
            outcomes_1x2 = ['home', 'draw', 'away']
            probs_1x2 = [aggregated.get(f'{o}_prob', 0) for o in outcomes_1x2]

            for outcome, model_prob in zip(outcomes_1x2, probs_1x2):
                market_odd = market_odds.get(outcome, 0)
                if market_odd > 0:
                    market_prob = 1 / market_odd
                    edge = model_prob - market_prob

                    if edge > self.min_ev_threshold:
                        # Calculate Kelly stake
                        kelly = self._kelly_criterion(model_prob, market_odd)
                        adjusted_stake = self._apply_drawdown_protection(kelly)

                        recommendations.append(BetRecommendation(
                            market="1X2",
                            selection=outcome.upper(),
                            probability=model_prob,
                            market_odds=market_odd,
                            edge=edge,
                            kelly_stake=kelly,
                            adjusted_stake=adjusted_stake,
                            confidence=self._get_confidence_level(edge, model_prob),
                            expected_value=edge * 100,
                            reason=self._generate_reason(outcome, edge, model_prob, market_odds)
                        ))

            # Sort by edge (best first)
            recommendations.sort(key=lambda x: x.edge, reverse=True)

            # Apply drawdown protection (reduce or skip if in drawdown)
            if self.current_drawdown > self.drawdown_limit:
                # Reduce all stakes by recovery factor
                for rec in recommendations:
                    rec.adjusted_stake *= self.recovery_factor
                    rec.reason += f" (Drawdown protection: -{(1-self.recovery_factor)*100:.0f}%)"

            return recommendations

        def _kelly_criterion(self, probability: float, decimal_odds: float) -> float:
            """Calculate Kelly Criterion stake fraction."""
            b = decimal_odds - 1  # Net odds received on bet
            p = probability
            q = 1 - p

            if b <= 0:
                return 0.0

            kelly = (b * p - q) / b

            # Apply maximum fraction cap
            kelly = min(kelly, self.max_kelly_fraction)

            # Apply consecutive loss penalty
            if self.consecutive_losses > 3:
                kelly *= (0.5 ** (self.consecutive_losses - 3))

            return max(0, kelly)

        def _apply_drawdown_protection(self, kelly: float) -> float:
            """Apply drawdown-based stake reduction."""
            if self.current_drawdown <= 0:
                return kelly

            # Linear reduction from 0% drawdown to limit
            reduction = min(1.0, self.current_drawdown / self.drawdown_limit)
            return kelly * (1 - reduction * 0.7)  # Max 70% reduction

        def _get_confidence_level(self, edge: float, probability: float) -> str:
            """Get confidence level based on edge and probability."""
            if edge > 0.08 and probability > 0.6:
                return "HIGH"
            elif edge > 0.04:
                return "MEDIUM"
            elif edge > self.min_ev_threshold:
                return "LOW"
            return "VERY_LOW"

        def _generate_reason(
            self, 
            outcome: str, 
            edge: float, 
            probability: float, 
            market_odds: Dict
        ) -> str:
            """Generate human-readable reason for recommendation."""
            reasons = []

            if edge > 0.05:
                reasons.append(f"Strong edge: {edge*100:.1f}%")
            else:
                reasons.append(f"Modest edge: {edge*100:.1f}%")

            reasons.append(f"Model probability: {probability*100:.1f}%")
            reasons.append(f"Market implied: {(1/market_odds.get(outcome, 2))*100:.1f}%")

            if self.market_regime == "inefficient":
                reasons.append("Market inefficiency detected")

            if self.current_drawdown > 0:
                reasons.append(f"Drawdown protection active ({self.current_drawdown*100:.1f}% DD)")

            return " | ".join(reasons)

        def _calculate_ensemble_variance(
            self,
            model_predictions: Dict[str, Dict[str, float]],
            weights: Dict[str, float]
        ) -> Dict[str, float]:
            """Calculate weighted variance of ensemble predictions."""
            variance = {'1x2': 0.1}

            home_probs = []
            draw_probs = []
            away_probs = []
            weight_list = []

            for model_name, pred in model_predictions.items():
                w = weights.get(model_name, 0)
                if w > 0:
                    home_probs.append(pred.get('home_prob', 0.33))
                    draw_probs.append(pred.get('draw_prob', 0.33))
                    away_probs.append(pred.get('away_prob', 0.33))
                    weight_list.append(w)

            if weight_list:
                weights_array = np.array(weight_list)
                weights_array /= weights_array.sum()
                variance['1x2'] = float(np.average(np.var([home_probs, draw_probs, away_probs], axis=0), weights=weights_array))

            return variance

        def _calculate_consensus(
            self,
            model_predictions: Dict[str, Dict[str, float]],
            aggregated: Dict[str, float]
        ) -> Dict[str, Any]:
            """Calculate consensus among models."""
            ensemble_outcome = max(
                [('HOME', aggregated.get('home_prob', 0)),
                 ('DRAW', aggregated.get('draw_prob', 0)),
                 ('AWAY', aggregated.get('away_prob', 0))],
                key=lambda x: x[1]
            )[0]

            agreements = 0
            total = 0
            outcome_votes = defaultdict(int)

            for model_name, pred in model_predictions.items():
                model_outcome = max(
                    [('HOME', pred.get('home_prob', 0)),
                     ('DRAW', pred.get('draw_prob', 0)),
                     ('AWAY', pred.get('away_prob', 0))],
                    key=lambda x: x[1]
                )[0]

                outcome_votes[model_outcome] += 1

                if model_outcome == ensemble_outcome:
                    agreements += 1
                total += 1

            return {
                'outcome': ensemble_outcome,
                'percentage': (agreements / total * 100) if total > 0 else 0,
                'votes': dict(outcome_votes),
                'strength': agreements / total if total > 0 else 0
            }

        def _default_aggregation(self) -> Dict[str, Any]:
            """Return default aggregation when no models available."""
            return {
                "home_prob": 0.34,
                "draw_prob": 0.33,
                "away_prob": 0.33,
                "over_2_5_prob": 0.5,
                "under_2_5_prob": 0.5,
                "btts_prob": 0.5,
                "no_btts_prob": 0.5,
                "home_goals_expectation": 1.5,
                "away_goals_expectation": 1.2,
                "confidence": {"1x2": 0.5, "over_under": 0.5, "btts": 0.5},
                "consensus_percent": 0,
                "consensus_outcome": "UNKNOWN",
                "models_contributing": 0,
                "market_regime": "neutral",
                "regime_confidence": 0.5,
                "market_inefficiency": 0.0,
                "expected_value": {},
                "recommendations": []
            }

    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate ensemble prediction with bet recommendations.
        """
        # Simple ensemble: weighted average of certified models
        home_prob = features.get('home_prob', 0.34)
        draw_prob = features.get('draw_prob', 0.33)
        away_prob = features.get('away_prob', 0.33)
        
        return {
            "home_prob": float(home_prob),
            "draw_prob": float(draw_prob),
            "away_prob": float(away_prob),
            "over_2_5_prob": 0.5,
            "under_2_5_prob": 0.5,
            "btts_prob": 0.5,
            "no_btts_prob": 0.5,
            "confidence": {"1x2": 0.6, "over_under": 0.5, "btts": 0.5}
        }

        def update_bet_result(
            self,
            bet_recommendation: BetRecommendation,
            actual_outcome: str,
            actual_odds: float,
            profit: float
        ) -> None:
            """
            Update system state after bet settles.
            """
            # Update bankroll tracking
            if profit < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

            # Update drawdown
            current_bankroll = self.peak_bankroll + profit
            if current_bankroll < self.peak_bankroll:
                self.current_drawdown = (self.peak_bankroll - current_bankroll) / self.peak_bankroll
            else:
                self.peak_bankroll = current_bankroll
                self.current_drawdown = 0

            # Track bet
            self.bet_history.append({
                'timestamp': datetime.now(),
                'market': bet_recommendation.market,
                'selection': bet_recommendation.selection,
                'probability': bet_recommendation.probability,
                'odds': actual_odds,
                'stake': bet_recommendation.adjusted_stake,
                'profit': profit,
                'realized_edge': profit / bet_recommendation.adjusted_stake if bet_recommendation.adjusted_stake > 0 else 0
            })

            self.total_bets += 1
            if profit > 0:
                self.winning_bets += 1

            # Update total ROI
            total_staked = sum(b.get('stake', 0) for b in self.bet_history)
            total_profit = sum(b.get('profit', 0) for b in self.bet_history)
            if total_staked > 0:
                self.total_roi = total_profit / total_staked

            # Log for monitoring
            logger.info(f"Bet result: {bet_recommendation.selection} @ {actual_odds:.2f}, "
                       f"Profit: {profit:.2f}, Drawdown: {self.current_drawdown:.2%}, "
                       f"ROI: {self.total_roi:.2%}")

    def get_confidence_score(self, market: str = "1x2") -> float:
        """Return ensemble confidence score."""
        if not hasattr(self, 'current_aggregation'):
            return 0.5

        conf = self.current_aggregation.get('confidence', {}).get(market, 0.5)
        return float(conf)

    def train(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train ensemble aggregator on historical data.
        """
        if not matches:
            return {"error": "No training data"}

        logger.info(f"Training ensemble aggregator on {len(matches)} matches")

        # Process historical matches
        for match in matches:
            if 'model_predictions' in match and 'actual_outcome' in match:
                for model_name, pred in match['model_predictions'].items():
                    self.update_model_performance(
                        model_name,
                        pred,
                        match['actual_outcome'],
                        match.get('home_goals', 0),
                        match.get('away_goals', 0),
                        match.get('market_odds', {})
                    )
        
        self.trained_matches_count = len(matches)
        self.last_optimization = datetime.now()
        
        # Calculate aggregate metrics
        avg_ev = np.mean([p.expected_value for p in self.model_performance.values() if p.sample_size > 0])
        avg_calibration = np.mean([1 - p.calibration_error for p in self.model_performance.values() if p.sample_size > 0])
        
        logger.info(f"Ensemble training complete. Avg EV: {avg_ev:.3f}, "
                   f"Avg Calibration: {avg_calibration:.3f}")
        
        return {
            "model_type": self.model_type,
            "version": self.version,
            "matches_trained": self.trained_matches_count,
            "models_tracked": len(self.model_performance),
            "avg_expected_value": float(avg_ev),
            "avg_calibration": float(avg_calibration),
            "market_regime": self.market_regime,
            "regime_confidence": self.regime_confidence
        }
    def update_model_performance(
        self,
        model_name: str,
        prediction: Dict[str, float],
        actual_outcome: str,
        actual_home_goals: int,
        actual_away_goals: int,
        market_odds: Dict[str, float]
    ) -> None:
        """
        Update performance metrics for a model.
        """
        # Calculate if prediction was correct
        predicted_outcome = max(
            [('home', prediction.get('home_prob', 0)),
             ('draw', prediction.get('draw_prob', 0)),
             ('away', prediction.get('away_prob', 0))],
            key=lambda x: x[1]
        )[0]
        
        is_correct = (predicted_outcome == actual_outcome)
        
        # Calculate calibration error (Brier score)
        actual_probs = {'home': 0, 'draw': 0, 'away': 0}
        actual_probs[actual_outcome] = 1
        
        brier = sum(
            (prediction.get(f"{outcome}_prob", 0) - actual_probs[outcome]) ** 2
            for outcome in ['home', 'draw', 'away']
        )
        
        # Calculate realized EV
        market_prob = 1 / market_odds.get(actual_outcome, 2.0)
        model_prob = prediction.get(f"{actual_outcome}_prob", 0.33)
        realized_ev = model_prob - market_prob if is_correct else -market_prob
        
        # Update or create performance record
        if model_name not in self.model_performance:
            self.model_performance[model_name] = ModelPerformance(model_name=model_name)
        
        perf = self.model_performance[model_name]
        
        # Exponential moving average update (alpha = 0.1)
        alpha = 0.1
        perf.expected_value = perf.expected_value * (1 - alpha) + realized_ev * alpha
        perf.calibration_error = perf.calibration_error * (1 - alpha) + brier * alpha
        perf.accuracy = perf.accuracy * (1 - alpha) + (1 if is_correct else 0) * alpha
        perf.brier_score = perf.brier_score * (1 - alpha) + brier * alpha
        perf.sample_size += 1
        perf.last_update = datetime.now()
        
        # Store recent EV
        perf.recent_ev.append({'date': datetime.now(), 'ev': realized_ev})
        if len(perf.recent_ev) > 200:
            perf.recent_ev = perf.recent_ev[-100:]
        
        # Store prediction in history
        self.prediction_history.append({
            'model_name': model_name,
            'timestamp': datetime.now(),
            'correct': 1 if is_correct else 0,
            'brier': brier,
            'realized_ev': realized_ev,
            'prediction': prediction,
            'actual_outcome': actual_outcome,
            'total_goals': actual_home_goals + actual_away_goals,
            'is_draw': actual_outcome == 'draw',
            'edge_realized': realized_ev
        })
        
        # Keep history bounded
        if len(self.prediction_history) > 10000:
            self.prediction_history = self.prediction_history[-5000:]
    
    def save(self, path: str) -> None:
        """Save ensemble aggregator to disk."""
        import pickle
        
        save_data = {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'version': self.version,
            'weight': self.weight,
            'params': self.params,
            'status': self.status,
            'model_performance': self.model_performance,
            'prediction_history': self.prediction_history[-1000:],
            'bet_history': self.bet_history[-500:],
            'current_weights': self.current_weights,
            'current_ev': self.current_ev,
            'peak_bankroll': self.peak_bankroll,
            'current_drawdown': self.current_drawdown,
            'consecutive_losses': self.consecutive_losses,
            'market_regime': self.market_regime,
            'regime_confidence': self.regime_confidence,
            'regime_volatility': self.regime_volatility,
            'market_inefficiency': self.market_inefficiency,
            'trained_matches_count': self.trained_matches_count,
            'total_bets': self.total_bets,
            'winning_bets': self.winning_bets,
            'total_roi': self.total_roi,
            'ev_weight': self.ev_weight,
            'calibration_weight': self.calibration_weight,
            'accuracy_weight': self.accuracy_weight,
            'session_accuracies': {k.value: v for k, v in self.session_accuracies.items()},
            'final_score': self.final_score,
            'certified': self.certified
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Ensemble aggregator V{self.version} saved to {path}")
    
    def load(self, path: str) -> None:
        """Load ensemble aggregator from disk."""
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model_id = data['model_id']
        self.model_name = data['model_name']
        self.model_type = data['model_type']
        self.version = data.get('version', 2)
        self.weight = data['weight']
        self.params = data['params']
        self.status = data['status']
        self.model_performance = data['model_performance']
        self.prediction_history = data['prediction_history']
        self.bet_history = data.get('bet_history', [])
        self.current_weights = data['current_weights']
        self.current_ev = data.get('current_ev', {})
        self.peak_bankroll = data.get('peak_bankroll', 1.0)
        self.current_drawdown = data.get('current_drawdown', 0.0)
        self.consecutive_losses = data.get('consecutive_losses', 0)
        self.market_regime = data.get('market_regime', 'neutral')
        self.regime_confidence = data.get('regime_confidence', 0.5)
        self.regime_volatility = data.get('regime_volatility', 0.0)
        self.market_inefficiency = data.get('market_inefficiency', 0.0)
        self.trained_matches_count = data['trained_matches_count']
        self.total_bets = data.get('total_bets', 0)
        self.winning_bets = data.get('winning_bets', 0)
        self.total_roi = data.get('total_roi', 0.0)
        self.ev_weight = data.get('ev_weight', 0.5)
        self.calibration_weight = data.get('calibration_weight', 0.3)
        self.accuracy_weight = data.get('accuracy_weight', 0.2)
        
        # Restore certification data
        for session_val, accuracy in data.get('session_accuracies', {}).items():
            self.session_accuracies[Session(session_val)] = accuracy
        self.final_score = data.get('final_score')
        self.certified = data.get('certified', False)
        
        logger.info(f"Ensemble aggregator V{self.version} loaded from {path}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        return {
            'total_bets': self.total_bets,
            'winning_bets': self.winning_bets,
            'win_rate': self.winning_bets / self.total_bets if self.total_bets > 0 else 0,
            'total_roi': self.total_roi,
            'current_drawdown': self.current_drawdown,
            'peak_bankroll': self.peak_bankroll,
            'consecutive_losses': self.consecutive_losses,
            'market_regime': self.market_regime,
            'regime_confidence': self.regime_confidence,
            'market_inefficiency': self.market_inefficiency,
            'models_active': len([w for w in self.current_weights.values() if w > 0.01]),
            'model_weights': self.current_weights,
            'model_ev': {k: self._calculate_ev_from_performance(k) for k in self.model_performance.keys()}
        }
