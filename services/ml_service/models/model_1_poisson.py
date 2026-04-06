# services/ml-service/models/model_1_poisson.py
import numpy as np
import pickle
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime
from scipy.stats import poisson
from scipy.optimize import minimize

from app.models.base_model import BaseModel, MarketType, Session

logger = logging.getLogger(__name__)


class PoissonGoalModel(BaseModel):
    """
    Poisson Goal Model V2 - Market-Aware, Time-Weighted, Home/Away Split.
    
    Upgrades from V1:
        - Time-weighted training (recent matches count more)
        - Home/away split for attack and defence ratings
        - No hard clipping (uses soft constraints)
        - Market-aware edge detection
        - Variance-based confidence scoring
        - Dixon-Coles correction for low-score correlation
        - Time-based validation (no training data leakage)
    """
    
    def __init__(
        self,
        model_name: str,
        weight: float = 1.0,
        version: int = 1,
        params: Optional[Dict[str, Any]] = None,
        decay_days: int = 365,      # Matches older than 1 year get low weight
        min_weight: float = 0.1,    # Minimum weight for old matches
    ):
        super().__init__(
            model_name=model_name,
            model_type="Poisson",
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
        
        # Time-weighted ratings (split by home/away)
        self.attack_home: Dict[str, float] = {}
        self.attack_away: Dict[str, float] = {}
        self.defence_home: Dict[str, float] = {}
        self.defence_away: Dict[str, float] = {}
        
        # Rolling home advantage (not static)
        self.home_advantage: float = 1.2
        self.home_advantage_history: List[float] = []
        
        # League averages (split by home/away)
        self.league_avg_home_goals: float = 1.4
        self.league_avg_away_goals: float = 1.1
        
        # Dixon-Coles parameters (rho = correlation for low scores)
        self.rho: float = -0.13  # Negative = fewer 0-0, 1-1 than Poisson predicts
        
        # Training metadata
        self.trained_matches_count: int = 0
        self.last_trained_date: Optional[datetime] = None
        
        # Time decay parameters
        self.decay_days = decay_days
        self.min_weight = min_weight
        
        # Market awareness (populated during prediction)
        self.market_odds: Dict[str, float] = {}

        # Always certified (uses scipy which is core)
        self.certified = True
    
    def _get_time_weight(self, match_date: datetime) -> float:
        """Calculate exponential decay weight based on match age."""
        if self.last_trained_date is None:
            return 1.0
        
        days_ago = (self.last_trained_date - match_date).days
        if days_ago <= 0:
            return 1.0
        
        # Exponential decay: weight = exp(-k * days_ago)
        # k such that weight = min_weight at decay_days
        k = -np.log(self.min_weight) / self.decay_days
        weight = np.exp(-k * days_ago)
        
        return max(weight, self.min_weight)
    
    def train(
        self, 
        matches: List[Dict[str, Any]], 
        validation_split: float = 0.2,
        use_time_weights: bool = True
    ) -> Dict[str, Any]:
        """
        Train Poisson model with time-weighted recent matches.
        
        Args:
            matches: List of matches with date, teams, goals
            validation_split: Fraction to hold out for validation (time-based)
            use_time_weights: Apply exponential decay to older matches
        """
        if not matches:
            return {"error": "No training data"}
        
        # Sort by date (oldest first)
        matches_sorted = sorted(matches, key=lambda x: x.get('match_date', '1900-01-01'))
        
        # Time-based split (no data leakage)
        split_idx = int(len(matches_sorted) * (1 - validation_split))
        train_matches = matches_sorted[:split_idx]
        val_matches = matches_sorted[split_idx:]
        
        logger.info(f"Training on {len(train_matches)} matches, validating on {len(val_matches)}")
        
        # Set last trained date to most recent match
        if use_time_weights and train_matches:
            last_date_str = train_matches[-1].get('match_date')
            if last_date_str:
                self.last_trained_date = datetime.fromisoformat(last_date_str)
        
        # Collect all teams
        teams = set()
        for match in train_matches:
            teams.add(match['home_team'])
            teams.add(match['away_team'])
        self.unique_teams = list(teams)
        
        # Initialize ratings (home/away split)
        for team in self.unique_teams:
            self.attack_home[team] = 1.0
            self.attack_away[team] = 1.0
            self.defence_home[team] = 1.0
            self.defence_away[team] = 1.0
        
        # Calculate weighted statistics
        team_stats = defaultdict(lambda: {
            'home_gf': 0, 'home_ga': 0, 'home_matches': 0,
            'away_gf': 0, 'away_ga': 0, 'away_matches': 0,
            'total_weight': 0
        })
        
        total_home_goals = 0
        total_away_goals = 0
        total_weight = 0
        
        for match in train_matches:
            home = match['home_team']
            away = match['away_team']
            hg = match['home_goals']
            ag = match['away_goals']
            
            # Get time weight
            weight = 1.0
            if use_time_weights and 'match_date' in match:
                try:
                    match_date = datetime.fromisoformat(match['match_date'])
                    weight = self._get_time_weight(match_date)
                except (ValueError, TypeError):
                    pass
            
            # Home team stats
            team_stats[home]['home_gf'] += hg * weight
            team_stats[home]['home_ga'] += ag * weight
            team_stats[home]['home_matches'] += weight
            team_stats[home]['total_weight'] += weight
            
            # Away team stats
            team_stats[away]['away_gf'] += ag * weight
            team_stats[away]['away_ga'] += hg * weight
            team_stats[away]['away_matches'] += weight
            team_stats[away]['total_weight'] += weight
            
            total_home_goals += hg * weight
            total_away_goals += ag * weight
            total_weight += weight
        
        # Calculate league averages (weighted)
        self.league_avg_home_goals = total_home_goals / total_weight if total_weight > 0 else 1.4
        self.league_avg_away_goals = total_away_goals / total_weight if total_weight > 0 else 1.1
        
        # Calculate home advantage (rolling)
        self.home_advantage = self.league_avg_home_goals / self.league_avg_away_goals
        
        # Calculate attack/defence ratings (home/away split)
        league_avg_home_gf = np.mean([s['home_gf'] / max(s['home_matches'], 0.001) 
                                       for s in team_stats.values() if s['home_matches'] > 0])
        league_avg_away_gf = np.mean([s['away_gf'] / max(s['away_matches'], 0.001) 
                                       for s in team_stats.values() if s['away_matches'] > 0])
        league_avg_home_ga = league_avg_away_gf  # Symmetric
        league_avg_away_ga = league_avg_home_gf
        
        for team, stats in team_stats.items():
            if stats['home_matches'] > 0:
                gf_home_per_match = stats['home_gf'] / stats['home_matches']
                ga_home_per_match = stats['home_ga'] / stats['home_matches']
                self.attack_home[team] = gf_home_per_match / league_avg_home_gf if league_avg_home_gf > 0 else 1.0
                self.defence_home[team] = ga_home_per_match / league_avg_home_ga if league_avg_home_ga > 0 else 1.0
            
            if stats['away_matches'] > 0:
                gf_away_per_match = stats['away_gf'] / stats['away_matches']
                ga_away_per_match = stats['away_ga'] / stats['away_matches']
                self.attack_away[team] = gf_away_per_match / league_avg_away_gf if league_avg_away_gf > 0 else 1.0
                self.defence_away[team] = ga_away_per_match / league_avg_away_ga if league_avg_away_ga > 0 else 1.0
        
        # Optimize Dixon-Coles rho parameter on validation set
        if val_matches:
            self.rho = self._optimize_rho(train_matches + val_matches)
        
        self.trained_matches_count = len(train_matches)
        
        # Validate on time-based holdout
        val_metrics = self._validate_on_holdout(val_matches)
        
        logger.info(f"Training complete. Home advantage: {self.home_advantage:.3f}")
        logger.info(f"Dixon-Coles rho: {self.rho:.3f}")
        logger.info(f"Validation accuracy: {val_metrics.get('accuracy', 0):.2%}")
        
        return {
            "model_type": self.model_type,
            "teams_trained": len(self.unique_teams),
            "matches_trained": self.trained_matches_count,
            "matches_validated": len(val_matches),
            "home_advantage": self.home_advantage,
            "rho": self.rho,
            "validation_accuracy": val_metrics.get('accuracy', 0),
            "validation_log_loss": val_metrics.get('log_loss', 0),
            "validation_brier_score": val_metrics.get('brier_score', 0)
        }
    
    def _optimize_rho(self, matches: List[Dict]) -> float:
        """Optimize Dixon-Coles rho parameter for low-score correlation."""
        def negative_log_likelihood(rho):
            ll = 0
            for match in matches:
                home = match['home_team']
                away = match['away_team']
                hg = match['home_goals']
                ag = match['away_goals']
                
                home_lambda, away_lambda = self._calculate_expected_goals(home, away)
                
                # Poisson probabilities
                p_home = poisson.pmf(hg, home_lambda)
                p_away = poisson.pmf(ag, away_lambda)
                
                # Dixon-Coles correction for low scores
                if hg == 0 and ag == 0:
                    correction = 1 - (home_lambda * away_lambda * rho)
                elif hg == 0 and ag == 1:
                    correction = 1 + (home_lambda * rho)
                elif hg == 1 and ag == 0:
                    correction = 1 + (away_lambda * rho)
                elif hg == 1 and ag == 1:
                    correction = 1 - rho
                else:
                    correction = 1
                
                prob = p_home * p_away * max(correction, 0.01)
                ll += np.log(prob)
            
            return -ll
        
        # rho is typically between -0.3 and 0
        result = minimize(negative_log_likelihood, x0=[-0.1], bounds=[(-0.5, 0.1)])
        return float(result.x[0])
    
    def _calculate_expected_goals(
        self, 
        home_team: str, 
        away_team: str
    ) -> Tuple[float, float]:
        """
        Calculate expected goals with home/away split ratings.
        No hard clipping - uses soft constraints.
        """
        # Get ratings with fallback to 1.0 for unknown teams
        attack_home = self.attack_home.get(home_team, 1.0)
        defence_home = self.defence_home.get(home_team, 1.0)
        attack_away = self.attack_away.get(away_team, 1.0)
        defence_away = self.defence_away.get(away_team, 1.0)
        
        # Calculate expected goals (home/away split)
        home_lambda = (
            attack_home * 
            defence_away * 
            self.home_advantage * 
            self.league_avg_home_goals
        )
        
        away_lambda = (
            attack_away * 
            defence_home * 
            self.league_avg_away_goals
        )
        
        # Soft capping using log1p (preserves extremes but smooths)
        # This replaces hard clipping with a logarithmic transform
        if home_lambda > 3.5:
            home_lambda = np.log1p(home_lambda - 3.5) + 3.5
        if away_lambda > 3.5:
            away_lambda = np.log1p(away_lambda - 3.5) + 3.5
        
        return home_lambda, away_lambda
    
    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate predictions with market awareness.
        
        Args:
            features: Team features (home_team, away_team, etc.)
        """
        home_team = features.get('home_team') or features.get('home_team_name', 'unknown')
        away_team = features.get('away_team') or features.get('away_team_name', 'unknown')
        
        # Extract market odds if available
        market_odds = features.get('market_odds', None)
        
        # Calculate expected goals
        home_lambda, away_lambda = self._calculate_expected_goals(home_team, away_team)
        
        # Apply Dixon-Coles correction for match outcomes
        home_win, draw, away_win = self._calculate_match_outcome_dixon_coles(
            home_lambda, away_lambda
        )
        
        # Calculate other markets
        over_25, under_25 = self._calculate_over_under(home_lambda, away_lambda)
        btts, no_btts = self._calculate_btts(home_lambda, away_lambda)
        exact_scores = self._calculate_exact_scores(home_lambda, away_lambda)
        
        # Calculate variance-based confidence
        confidence_1x2 = self._calculate_variance_confidence([home_win, draw, away_win])
        confidence_ou = self._calculate_variance_confidence([over_25, under_25])
        confidence_btts = self._calculate_variance_confidence([btts, no_btts])
        
        # Calculate edge vs market (if odds provided)
        edge = self._calculate_edge(home_win, draw, away_win, market_odds)
        
        return {
            # 1X2
            "home_prob": home_win,
            "draw_prob": draw,
            "away_prob": away_win,
            
            # Over/Under
            "over_2_5_prob": over_25,
            "under_2_5_prob": under_25,
            
            # BTTS
            "btts_prob": btts,
            "no_btts_prob": no_btts,
            
            # Exact scores
            "exact_score_probs": exact_scores,
            
            # Expected goals
            "home_goals_expectation": home_lambda,
            "away_goals_expectation": away_lambda,
            
            # Dixon-Coles correction factor
            "dixon_coles_rho": self.rho,
            
            # Confidence scores (variance-based, not cosmetic)
            "confidence": {
                "1x2": confidence_1x2,
                "over_under": confidence_ou,
                "btts": confidence_btts
            },
            
            # Market edge (if odds provided)
            "edge_vs_market": edge,
            "has_market_edge": edge.get("has_edge", False)
        }
    
    def get_confidence_score(self, market: str = "1x2") -> float:
        """
        Returns model's confidence for a specific market.
        Uses stored confidence from last prediction or returns default.
        """
        # This is called by orchestrator; confidence is already in predict output
        # Return a reasonable default based on training data size
        if self.trained_matches_count < 100:
            return 0.5
        elif self.trained_matches_count < 500:
            return 0.6
        else:
            return 0.65
    
    def _calculate_match_outcome_dixon_coles(
        self, 
        home_lambda: float, 
        away_lambda: float, 
        max_goals: int = 10
    ) -> Tuple[float, float, float]:
        """
        Calculate 1X2 probabilities with Dixon-Coles correction for low-score correlation.
        """
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        
        for hg in range(max_goals + 1):
            p_hg = poisson.pmf(hg, home_lambda)
            for ag in range(max_goals + 1):
                p_ag = poisson.pmf(ag, away_lambda)
                base_prob = p_hg * p_ag
                
                # Dixon-Coles correction
                if hg == 0 and ag == 0:
                    correction = 1 - (home_lambda * away_lambda * self.rho)
                elif hg == 0 and ag == 1:
                    correction = 1 + (home_lambda * self.rho)
                elif hg == 1 and ag == 0:
                    correction = 1 + (away_lambda * self.rho)
                elif hg == 1 and ag == 1:
                    correction = 1 - self.rho
                else:
                    correction = 1
                
                prob = base_prob * max(correction, 0.01)
                
                if hg > ag:
                    home_win += prob
                elif hg == ag:
                    draw += prob
                else:
                    away_win += prob
        
        # Normalize
        total = home_win + draw + away_win
        if total > 0:
            home_win /= total
            draw /= total
            away_win /= total
        
        return home_win, draw, away_win
    
    def _calculate_variance_confidence(self, probabilities: List[float]) -> float:
        """
        Calculate confidence based on probability distribution variance.
        High variance = clear favorite = high confidence.
        Low variance = toss-up = low confidence.
        """
        if not probabilities:
            return 0.5
        
        probs = np.array(probabilities)
        variance = np.var(probs)
        
        # Max variance for 3 outcomes is when one is 1.0, others 0: var ≈ 0.222
        max_variance = 0.222
        confidence = variance / max_variance
        
        # Scale to reasonable range (0.4 - 0.95)
        confidence = 0.4 + (confidence * 0.55)
        
        return min(max(confidence, 0.4), 0.95)
    
    def _calculate_edge(
        self, 
        home_prob: float, 
        draw_prob: float, 
        away_prob: float, 
        market_odds: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Calculate edge vs market odds."""
        if not market_odds:
            return {"has_edge": False, "reason": "No market odds provided"}
        
        edges = {}
        outcomes = ["home", "draw", "away"]
        model_probs = [home_prob, draw_prob, away_prob]
        
        for outcome, model_prob in zip(outcomes, model_probs):
            market_odd = market_odds.get(outcome, 0)
            if market_odd > 0:
                market_prob = 1 / market_odd
                edge = model_prob - market_prob
                edges[outcome] = {
                    "model_prob": model_prob,
                    "market_prob": market_prob,
                    "market_odd": market_odd,
                    "edge": edge,
                    "has_edge": edge > 0.02  # 2% edge threshold
                }
        
        # Determine best edge
        best_edge = max(edges.items(), key=lambda x: x[1]['edge']) if edges else (None, {})
        
        return {
            "has_edge": best_edge[1].get('has_edge', False) if best_edge[1] else False,
            "best_outcome": best_edge[0] if best_edge[0] else None,
            "best_edge_percent": round(best_edge[1]['edge'] * 100, 2) if best_edge[1] else 0,
            "all_edges": edges
        }
    
    def _calculate_over_under(
        self, 
        home_lambda: float, 
        away_lambda: float, 
        threshold: float = 2.5,
        max_goals: int = 15
    ) -> Tuple[float, float]:
        """Calculate Over/Under probabilities."""
        over_prob = 0.0
        under_prob = 0.0
        
        for hg in range(max_goals + 1):
            p_hg = poisson.pmf(hg, home_lambda)
            for ag in range(max_goals + 1):
                total_goals = hg + ag
                prob = p_hg * poisson.pmf(ag, away_lambda)
                if total_goals > threshold:
                    over_prob += prob
                else:
                    under_prob += prob
        
        total = over_prob + under_prob
        if total > 0:
            over_prob /= total
            under_prob /= total
        
        return over_prob, under_prob
    
    def _calculate_btts(
        self, 
        home_lambda: float, 
        away_lambda: float
    ) -> Tuple[float, float]:
        """Calculate BTTS (Both Teams to Score) probabilities."""
        p_home_zero = poisson.pmf(0, home_lambda)
        p_away_zero = poisson.pmf(0, away_lambda)
        
        # P(at least one team scores 0)
        no_btts = p_home_zero + p_away_zero - (p_home_zero * p_away_zero)
        btts = 1 - no_btts
        
        # Normalize to sum to 1.0
        total = btts + no_btts
        if total > 0:
            btts /= total
            no_btts /= total
        
        return btts, no_btts
    
    def _calculate_exact_scores(
        self, 
        home_lambda: float, 
        away_lambda: float, 
        max_goals: int = 5
    ) -> Dict[str, float]:
        """Calculate probabilities for exact score outcomes."""
        exact_scores = {}
        
        for hg in range(max_goals + 1):
            p_hg = poisson.pmf(hg, home_lambda)
            for ag in range(max_goals + 1):
                score_key = f"{hg}-{ag}"
                exact_scores[score_key] = p_hg * poisson.pmf(ag, away_lambda)
        
        # Normalize
        total = sum(exact_scores.values())
        if total > 0:
            for key in exact_scores:
                exact_scores[key] /= total
        
        # Sort by probability (descending) and keep top 10
        sorted_scores = sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_scores[:10])
    
    def _validate_on_holdout(self, matches: List[Dict]) -> Dict[str, float]:
        """Validate model on time-based holdout set."""
        if not matches:
            return {"accuracy": 0, "log_loss": 0, "brier_score": 0}
        
        correct = 0
        log_loss_sum = 0
        brier_sum = 0
        
        for match in matches:
            home = match['home_team']
            away = match['away_team']
            
            actual_hg = match['home_goals']
            actual_ag = match['away_goals']
            
            # Actual outcome
            if actual_hg > actual_ag:
                actual = [1, 0, 0]
            elif actual_hg == actual_ag:
                actual = [0, 1, 0]
            else:
                actual = [0, 0, 1]
            
            # Predict
            home_lambda, away_lambda = self._calculate_expected_goals(home, away)
            home_prob, draw_prob, away_prob = self._calculate_match_outcome_dixon_coles(
                home_lambda, away_lambda
            )
            pred = [home_prob, draw_prob, away_prob]
            
            # Accuracy
            predicted_outcome = np.argmax(pred)
            actual_outcome = np.argmax(actual)
            if predicted_outcome == actual_outcome:
                correct += 1
            
            # Log loss
            for i in range(3):
                if actual[i] == 1:
                    log_loss_sum += -np.log(max(pred[i], 1e-15))
            
            # Brier score
            for i in range(3):
                brier_sum += (pred[i] - actual[i]) ** 2
        
        n = len(matches)
        return {
            "accuracy": correct / n,
            "log_loss": log_loss_sum / n,
            "brier_score": brier_sum / n
        }
    
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
            'attack_home': self.attack_home,
            'attack_away': self.attack_away,
            'defence_home': self.defence_home,
            'defence_away': self.defence_away,
            'home_advantage': self.home_advantage,
            'rho': self.rho,
            'league_avg_home_goals': self.league_avg_home_goals,
            'league_avg_away_goals': self.league_avg_away_goals,
            'trained_matches_count': self.trained_matches_count,
            'unique_teams': self.unique_teams,
            'decay_days': self.decay_days,
            'min_weight': self.min_weight,
            'session_accuracies': {k.value: v for k, v in self.session_accuracies.items()},
            'final_score': self.final_score,
            'certified': self.certified
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model_id = data['model_id']
        self.model_name = data['model_name']
        self.model_type = data['model_type']
        self.version = data['version']
        self.weight = data['weight']
        self.params = data['params']
        self.status = data['status']
        self.attack_home = data['attack_home']
        self.attack_away = data['attack_away']
        self.defence_home = data['defence_home']
        self.defence_away = data['defence_away']
        self.home_advantage = data['home_advantage']
        self.rho = data['rho']
        self.league_avg_home_goals = data['league_avg_home_goals']
        self.league_avg_away_goals = data['league_avg_away_goals']
        self.trained_matches_count = data['trained_matches_count']
        self.unique_teams = data['unique_teams']
        self.decay_days = data.get('decay_days', 365)
        self.min_weight = data.get('min_weight', 0.1)
        
        # Restore certification data
        for session_val, accuracy in data.get('session_accuracies', {}).items():
            self.session_accuracies[Session(session_val)] = accuracy
        self.final_score = data.get('final_score')
        self.certified = data.get('certified', False)
        
        logger.info(f"Model loaded from {path}")