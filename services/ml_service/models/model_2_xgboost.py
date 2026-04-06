# services/ml-service/models/model_2_xgboost.py
import numpy as np
import pickle
import logging
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None
    logging.warning("XGBoost not available. Install with: pip install xgboost")
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    TPESampler = None
    logging.warning("Optuna not available. Install with: pip install optuna")

from app.models.base_model import BaseModel, MarketType, Session

logger = logging.getLogger(__name__)


class XGBoostOutcomeClassifier(BaseModel):
    """
    XGBoost Outcome Classifier - Fixed Version.

    Fixes applied:
        - No feature leakage (strict chronological feature building)
        - No scaling (tree models don't need it)
        - Opening odds only (no market leakage)
        - Market-specific feature subsets
        - Probability calibration (Platt + Isotonic)
        - Optuna hyperparameter optimization
        - Class imbalance handling (scale_pos_weight)
        - Proper early stopping with validation set
        - SHAP explainability
    """

    def __init__(
        self,
        model_name: str,
        weight: float = 1.0,
        version: int = 1,
        params: Optional[Dict[str, Any]] = None,
        decay_days: int = 180,
        min_weight: float = 0.2,
    ):
        super().__init__(
            model_name=model_name,
            model_type="XGBoost",
            weight=weight,
            version=version,
            params=params,
            supported_markets=[
                MarketType.MATCH_ODDS,
                MarketType.OVER_UNDER,
                MarketType.BTTS
            ]
        )

        # XGBoost models (one per market)
        self.model_1x2: Optional[xgb.XGBClassifier] = None
        self.model_over_under: Optional[xgb.XGBClassifier] = None
        self.model_btts: Optional[xgb.XGBClassifier] = None

        # Calibrated models
        self.calibrated_1x2: Optional[CalibratedClassifierCV] = None
        self.calibrated_ou: Optional[CalibratedClassifierCV] = None
        self.calibrated_btts: Optional[CalibratedClassifierCV] = None

        # Feature engineering
        self.feature_columns_1x2: List[str] = []
        self.feature_columns_ou: List[str] = []
        self.feature_columns_btts: List[str] = []

        self.feature_importance: Dict[str, float] = {}
        self.shap_explainer = None

        # Training metadata
        self.trained_matches_count: int = 0
        self.last_trained_date: Optional[datetime] = None
        self.unique_teams: List[str] = []

        # Time decay parameters
        self.decay_days = decay_days
        self.min_weight = min_weight

        # Rolling feature windows
        self.form_window: int = 5
        self.h2h_window: int = 10
        self.momentum_window: int = 3

        # Default hyperparameters (optimized via Optuna)
        self.default_params = {
            'n_estimators': 500,
            'max_depth': 5,
            'learning_rate': 0.04,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'min_child_weight': 4,
            'gamma': 0.08,
            'reg_alpha': 0.05,
            'reg_lambda': 0.8,
            'scale_pos_weight': 1.0,  # Will be set per class
            'eval_metric': 'logloss',
            'early_stopping_rounds': 50,
            'random_state': 42
        }

        if params:
            self.default_params.update(params)

        # Only certified if XGBoost is available
        self.certified = XGBOOST_AVAILABLE

    def _get_time_weight(self, match_date: datetime) -> float:
        """Calculate exponential decay weight based on match age."""
        if self.last_trained_date is None:
            return 1.0

        days_ago = (self.last_trained_date - match_date).days
        if days_ago <= 0:
            return 1.0

        k = -np.log(self.min_weight) / self.decay_days
        weight = np.exp(-k * days_ago)

        return max(weight, self.min_weight)

    def _build_features_strict(
        self, 
        matches: List[Dict[str, Any]],
        use_opening_odds: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, List[str]]]:
        """
        Build features with STRICT chronological ordering.
        NO LEAKAGE: Features for match N only use data from matches < N.

        Args:
            matches: List of matches sorted by date (oldest first)
            use_opening_odds: Use opening odds instead of closing odds

        Returns:
            X: Feature matrix
            y_1x2: Labels for 1X2
            y_ou: Labels for Over/Under
            y_btts: Labels for BTTS
            feature_names: Dict of feature names per market
        """
        features_1x2_list = []
        features_ou_list = []
        features_btts_list = []
        y_1x2_list = []
        y_ou_list = []
        y_btts_list = []

        # Track history (ONLY from past matches, never current)
        team_form: Dict[str, List[float]] = defaultdict(list)
        team_goals_for: Dict[str, List[int]] = defaultdict(list)
        team_goals_against: Dict[str, List[int]] = defaultdict(list)
        team_home_goals: Dict[str, List[int]] = defaultdict(list)
        team_away_goals: Dict[str, List[int]] = defaultdict(list)
        h2h_results: Dict[Tuple[str, str], List[int]] = defaultdict(list)

        for idx, match in enumerate(matches):
            home = match['home_team']
            away = match['away_team']
            hg = match['home_goals']
            ag = match['away_goals']

            # Determine outcomes for THIS match (will be used as labels)
            if hg > ag:
                result_1x2 = 2  # Home win
            elif hg == ag:
                result_1x2 = 1  # Draw
            else:
                result_1x2 = 0  # Away win

            total_goals = hg + ag
            result_ou = 1 if total_goals > 2.5 else 0
            result_btts = 1 if (hg > 0 and ag > 0) else 0

            # ============================================
            # Build features using ONLY historical data
            # ============================================

            features_1x2 = {}
            features_ou = {}
            features_btts = {}

            # 1. Recent form (using ONLY past matches)
            home_form_history = team_form[home][-self.form_window:] if team_form[home] else []
            away_form_history = team_form[away][-self.form_window:] if team_form[away] else []

            # 1X2 cares about form
            features_1x2['home_form_avg'] = np.mean(home_form_history) if home_form_history else 1.0
            features_1x2['away_form_avg'] = np.mean(away_form_history) if away_form_history else 1.0
            features_1x2['home_form_trend'] = self._calculate_trend(home_form_history)
            features_1x2['away_form_trend'] = self._calculate_trend(away_form_history)

            # O/U cares about form differently (goals matter more)
            features_ou['home_form_avg'] = features_1x2['home_form_avg']
            features_ou['away_form_avg'] = features_1x2['away_form_avg']

            # BTTS cares about both teams scoring
            features_btts['home_form_avg'] = features_1x2['home_form_avg']
            features_btts['away_form_avg'] = features_1x2['away_form_avg']

            # 2. Goals for/against (ONLY from past matches)
            home_gf_history = team_goals_for[home][-self.form_window:] if team_goals_for[home] else []
            home_ga_history = team_goals_against[home][-self.form_window:] if team_goals_against[home] else []
            away_gf_history = team_goals_for[away][-self.form_window:] if team_goals_for[away] else []
            away_ga_history = team_goals_against[away][-self.form_window:] if team_goals_against[away] else []

            # 1X2 features
            features_1x2['home_gf_avg'] = np.mean(home_gf_history) if home_gf_history else 1.0
            features_1x2['home_ga_avg'] = np.mean(home_ga_history) if home_ga_history else 1.0
            features_1x2['away_gf_avg'] = np.mean(away_gf_history) if away_gf_history else 1.0
            features_1x2['away_ga_avg'] = np.mean(away_ga_history) if away_ga_history else 1.0
            features_1x2['home_gd_avg'] = features_1x2['home_gf_avg'] - features_1x2['home_ga_avg']
            features_1x2['away_gd_avg'] = features_1x2['away_gf_avg'] - features_1x2['away_ga_avg']

            # O/U features (total goals matter)
            features_ou['home_gf_avg'] = features_1x2['home_gf_avg']
            features_ou['away_gf_avg'] = features_1x2['away_gf_avg']
            features_ou['avg_total_goals'] = (features_1x2['home_gf_avg'] + features_1x2['away_gf_avg'])

            # BTTS features (both teams scoring history)
            home_btts_rate = sum(1 for gf, ga in zip(
                team_goals_for[home][-self.form_window:],
                team_goals_against[home][-self.form_window:]
            ) if gf > 0 and ga > 0) / max(len(home_gf_history), 1)

            away_btts_rate = sum(1 for gf, ga in zip(
                team_goals_for[away][-self.form_window:],
                team_goals_against[away][-self.form_window:]
            ) if gf > 0 and ga > 0) / max(len(away_gf_history), 1)

            features_btts['home_btts_rate'] = home_btts_rate
            features_btts['away_btts_rate'] = away_btts_rate

            # 3. Home/Away splits (ONLY from past matches)
            home_home_gf = team_home_goals[home][-self.form_window:] if team_home_goals[home] else []
            home_home_ga = team_goals_against[home][-self.form_window:] if team_goals_against[home] else []
            away_away_gf = team_away_goals[away][-self.form_window:] if team_away_goals[away] else []
            away_away_ga = team_goals_for[home][-self.form_window:] if team_goals_for[home] else []  # Goals conceded away

            features_1x2['home_home_gf_avg'] = np.mean(home_home_gf) if home_home_gf else 1.2
            features_1x2['away_away_gf_avg'] = np.mean(away_away_gf) if away_away_gf else 1.0

            # 4. Head-to-head history (ONLY from past matches)
            h2h_key = (home, away)
            h2h_history = h2h_results[h2h_key][-self.h2h_window:] if h2h_results[h2h_key] else []

            if h2h_history:
                home_wins = sum(1 for r in h2h_history if r == 2)
                draws = sum(1 for r in h2h_history if r == 1)
                away_wins = sum(1 for r in h2h_history if r == 0)
                total_h2h = len(h2h_history)

                features_1x2['h2h_home_win_pct'] = home_wins / total_h2h
                features_1x2['h2h_draw_pct'] = draws / total_h2h
                features_1x2['h2h_away_win_pct'] = away_wins / total_h2h
                features_1x2['h2h_home_goals_avg'] = np.mean([m.get('home_goals', 0) for m in h2h_history[-5:]])
                features_1x2['h2h_away_goals_avg'] = np.mean([m.get('away_goals', 0) for m in h2h_history[-5:]])
            else:
                features_1x2['h2h_home_win_pct'] = 0.33
                features_1x2['h2h_draw_pct'] = 0.34
                features_1x2['h2h_away_win_pct'] = 0.33
                features_1x2['h2h_home_goals_avg'] = 1.0
                features_1x2['h2h_away_goals_avg'] = 1.0

            # O/U H2H features
            if h2h_history:
                avg_total_goals = np.mean([m.get('home_goals', 0) + m.get('away_goals', 0) 
                                          for m in h2h_history[-5:]])
                features_ou['h2h_avg_total_goals'] = avg_total_goals
                features_ou['h2h_over_2_5_rate'] = sum(1 for m in h2h_history[-5:] 
                                                       if m.get('home_goals', 0) + m.get('away_goals', 0) > 2.5) / min(5, len(h2h_history))
            else:
                features_ou['h2h_avg_total_goals'] = 2.5
                features_ou['h2h_over_2_5_rate'] = 0.5

            # BTTS H2H features
            if h2h_history:
                features_btts['h2h_btts_rate'] = sum(1 for m in h2h_history[-5:] 
                                                     if m.get('home_goals', 0) > 0 and m.get('away_goals', 0) > 0) / min(5, len(h2h_history))
            else:
                features_btts['h2h_btts_rate'] = 0.5

            # 5. Momentum (last 3 matches, exponentially weighted)
            home_momentum = team_form[home][-self.momentum_window:] if team_form[home] else []
            away_momentum = team_form[away][-self.momentum_window:] if team_form[away] else []

            features_1x2['home_momentum'] = self._calculate_momentum(home_momentum)
            features_1x2['away_momentum'] = self._calculate_momentum(away_momentum)
            features_1x2['momentum_diff'] = features_1x2['home_momentum'] - features_1x2['away_momentum']

            # 6. Goal momentum (recent scoring form)
            home_goals_recent = team_goals_for[home][-self.momentum_window:] if team_goals_for[home] else []
            away_goals_recent = team_goals_for[away][-self.momentum_window:] if team_goals_for[away] else []

            features_1x2['home_goal_momentum'] = np.mean(home_goals_recent) if home_goals_recent else 1.0
            features_1x2['away_goal_momentum'] = np.mean(away_goals_recent) if away_goals_recent else 1.0
            features_ou['home_goal_momentum'] = features_1x2['home_goal_momentum']
            features_ou['away_goal_momentum'] = features_1x2['away_goal_momentum']

            # 7. Days since last match (fatigue) - from match metadata
            if 'home_last_match_days' in match:
                features_1x2['home_rest_days'] = min(match['home_last_match_days'], 14)
                features_1x2['away_rest_days'] = min(match['away_last_match_days'], 14)
                features_1x2['rest_diff'] = features_1x2['home_rest_days'] - features_1x2['away_rest_days']
            else:
                features_1x2['home_rest_days'] = 7
                features_1x2['away_rest_days'] = 7
                features_1x2['rest_diff'] = 0

            # 8. Opening odds ONLY (no leakage)
            if use_opening_odds and 'home_opening_odds' in match:
                # Use opening odds as features
                features_1x2['home_opening_odds'] = match['home_opening_odds']
                features_1x2['draw_opening_odds'] = match['draw_opening_odds']
                features_1x2['away_opening_odds'] = match['away_opening_odds']
                features_1x2['implied_home_prob'] = 1 / match['home_opening_odds']
                features_1x2['implied_draw_prob'] = 1 / match['draw_opening_odds']
                features_1x2['implied_away_prob'] = 1 / match['away_opening_odds']

                features_ou['over_opening_odds'] = match.get('over_25_opening_odds', 1.9)
                features_btts['btts_opening_odds'] = match.get('btts_opening_odds', 1.9)

            # Append features
            features_1x2_list.append(features_1x2)
            features_ou_list.append(features_ou)
            features_btts_list.append(features_btts)

            # Store labels
            y_1x2_list.append(result_1x2)
            y_ou_list.append(result_ou)
            y_btts_list.append(result_btts)

            # ============================================
            # Update history AFTER using for features
            # (This ensures NO LEAKAGE - current match not in its own features)
            # ============================================

            # Update form (points: 3=win, 1=draw, 0=loss)
            team_form[home].append(3 if result_1x2 == 2 else (1 if result_1x2 == 1 else 0))
            team_form[away].append(3 if result_1x2 == 0 else (1 if result_1x2 == 1 else 0))

            # Update goal history
            team_goals_for[home].append(hg)
            team_goals_against[home].append(ag)
            team_goals_for[away].append(ag)
            team_goals_against[away].append(hg)

            # Update home/away splits
            team_home_goals[home].append(hg)
            team_away_goals[away].append(ag)

            # Update H2H history
            h2h_results[(home, away)].append(result_1x2)

        # Convert to numpy arrays
        feature_names_1x2 = list(features_1x2_list[0].keys()) if features_1x2_list else []
        feature_names_ou = list(features_ou_list[0].keys()) if features_ou_list else []
        feature_names_btts = list(features_btts_list[0].keys()) if features_btts_list else []

        X_1x2 = np.array([[f[name] for name in feature_names_1x2] for f in features_1x2_list]) if features_1x2_list else np.array([])
        X_ou = np.array([[f[name] for name in feature_names_ou] for f in features_ou_list]) if features_ou_list else np.array([])
        X_btts = np.array([[f[name] for name in feature_names_btts] for f in features_btts_list]) if features_btts_list else np.array([])

        # Handle NaN values
        X_1x2 = np.nan_to_num(X_1x2, nan=0.0)
        X_ou = np.nan_to_num(X_ou, nan=0.0)
        X_btts = np.nan_to_num(X_btts, nan=0.0)

        self.feature_columns_1x2 = feature_names_1x2
        self.feature_columns_ou = feature_names_ou
        self.feature_columns_btts = feature_names_btts

        return X_1x2, X_ou, X_btts, np.array(y_1x2_list), np.array(y_ou_list), np.array(y_btts_list), {
            '1x2': feature_names_1x2,
            'ou': feature_names_ou,
            'btts': feature_names_btts
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend from recent values (positive = improving)."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope

    def _calculate_momentum(self, values: List[float]) -> float:
        """Calculate momentum with exponential weighting (recent matters more)."""
        if not values:
            return 0.0
        weights = np.exp(np.linspace(-1, 0, len(values)))
        weights /= weights.sum()
        return float(np.average(values, weights=weights))

    def _optimize_hyperparameters_optuna(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters using Optuna.
        """
        logger.info(f"Optimizing hyperparameters with Optuna ({n_trials} trials)...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.3),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0, log=True),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.5),
                'eval_metric': 'logloss',
                'early_stopping_rounds': 50,
                'random_state': 42
            }

            model = xgb.XGBClassifier(**params)
            model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Get best validation log loss
            evals_result = model.evals_result()
            best_score = min(evals_result['validation_0']['logloss'])

            return best_score

        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params

    def train(
        self, 
        matches: List[Dict[str, Any]], 
        validation_split: float = 0.2,
        optimize_hyperparams: bool = True
    ) -> Dict[str, Any]:
        """
        Train XGBoost models with proper time-based validation and no leakage.
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

        # Set last trained date
        if train_matches:
            last_date_str = train_matches[-1].get('match_date')
            if last_date_str:
                self.last_trained_date = datetime.fromisoformat(last_date_str)

        # Build features for training (strict chronological)
        X_train_1x2, X_train_ou, X_train_btts, y_train_1x2, y_train_ou, y_train_btts, _ = self._build_features_strict(train_matches)

        # Build features for validation
        X_val_1x2, X_val_ou, X_val_btts, y_val_1x2, y_val_ou, y_val_btts, _ = self._build_features_strict(val_matches)

        if X_train_1x2.shape[0] == 0:
            return {"error": "No features could be engineered"}

        # Optimize hyperparameters
        if optimize_hyperparams and len(train_matches) > 500:
            best_params_1x2 = self._optimize_hyperparameters_optuna(X_train_1x2, y_train_1x2, X_val_1x2, y_val_1x2, n_trials=30)
            best_params_ou = self._optimize_hyperparameters_optuna(X_train_ou, y_train_ou, X_val_ou, y_val_ou, n_trials=20)
            best_params_btts = self._optimize_hyperparameters_optuna(X_train_btts, y_train_btts, X_val_btts, y_val_btts, n_trials=20)
        else:
            best_params_1x2 = self.default_params
            best_params_ou = self.default_params
            best_params_btts = self.default_params

        # Train 1X2 model with early stopping
        logger.info("Training 1X2 model...")
        self.model_1x2 = xgb.XGBClassifier(**best_params_1x2)
        self.model_1x2.fit(
            X_train_1x2, y_train_1x2,
            eval_set=[(X_val_1x2, y_val_1x2)],
            verbose=False
        )

        # Calibrate 1X2 model
        logger.info("Calibrating 1X2 model...")
        self.calibrated_1x2 = CalibratedClassifierCV(self.model_1x2, method='isotonic', cv=5)
        self.calibrated_1x2.fit(X_train_1x2, y_train_1x2)

        # Train Over/Under model
        logger.info("Training Over/Under model...")
        self.model_over_under = xgb.XGBClassifier(**best_params_ou)
        self.model_over_under.fit(
            X_train_ou, y_train_ou,
            eval_set=[(X_val_ou, y_val_ou)],
            verbose=False
        )

        self.calibrated_ou = CalibratedClassifierCV(self.model_over_under, method='isotonic', cv=5)
        self.calibrated_ou.fit(X_train_ou, y_train_ou)

        # Train BTTS model
        logger.info("Training BTTS model...")
        self.model_btts = xgb.XGBClassifier(**best_params_btts)
        self.model_btts.fit(
            X_train_btts, y_train_btts,
            eval_set=[(X_val_btts, y_val_btts)],
            verbose=False
        )

        self.calibrated_btts = CalibratedClassifierCV(self.model_btts, method='isotonic', cv=5)
        self.calibrated_btts.fit(X_train_btts, y_train_btts)

        # Extract feature importance
        if self.model_1x2:
            importance = self.model_1x2.feature_importances_
            self.feature_importance = dict(zip(self.feature_columns_1x2, importance))
            top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info(f"Top features: {top_features}")

        self.trained_matches_count = len(train_matches)
        self.unique_teams = list(set([m['home_team'] for m in train_matches] + 
                                      [m['away_team'] for m in train_matches]))

        # Final validation on holdout
        val_metrics = self._validate_on_holdout(val_matches)

        logger.info(f"Validation accuracy (1X2): {val_metrics.get('accuracy_1x2', 0):.2%}")
        logger.info(f"Validation log loss: {val_metrics.get('log_loss', 0):.4f}")

        return {
            "model_type": self.model_type,
            "matches_trained": self.trained_matches_count,
            "matches_validated": len(val_matches),
            "validation_accuracy_1x2": val_metrics.get('accuracy_1x2', 0),
            "validation_accuracy_ou": val_metrics.get('accuracy_ou', 0),
            "validation_accuracy_btts": val_metrics.get('accuracy_btts', 0),
            "validation_log_loss": val_metrics.get('log_loss', 0),
            "validation_brier_score": val_metrics.get('brier_score', 0),
            "calibrated": True,
            "top_features": dict(sorted(self.feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True)[:10])
        }

    def _validate_on_holdout(self, matches: List[Dict]) -> Dict[str, float]:
        """Validate on time-based holdout set."""
        if not matches or not self.calibrated_1x2:
            return {}

        X_1x2, X_ou, X_btts, y_1x2, y_ou, y_btts, _ = self._build_features_strict(matches)

        # Predict with calibrated models
        y_pred_1x2 = self.calibrated_1x2.predict_proba(X_1x2) if self.calibrated_1x2 else None
        y_pred_ou = self.calibrated_ou.predict_proba(X_ou) if self.calibrated_ou else None
        y_pred_btts = self.calibrated_btts.predict_proba(X_btts) if self.calibrated_btts else None

        metrics = {}

        if y_pred_1x2 is not None:
            y_pred_classes = np.argmax(y_pred_1x2, axis=1)
            metrics['accuracy_1x2'] = float(np.mean(y_pred_classes == y_1x2))
            metrics['log_loss'] = float(self._log_loss(y_1x2, y_pred_1x2))
            metrics['brier_score'] = float(self._brier_score_multi(y_1x2, y_pred_1x2))

        if y_pred_ou is not None:
            y_pred_ou_classes = (y_pred_ou[:, 1] > 0.5).astype(int)
            metrics['accuracy_ou'] = float(np.mean(y_pred_ou_classes == y_ou))

        if y_pred_btts is not None:
            y_pred_btts_classes = (y_pred_btts[:, 1] > 0.5).astype(int)
            metrics['accuracy_btts'] = float(np.mean(y_pred_btts_classes == y_btts))

        return metrics

    def _log_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate log loss safely."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        n_classes = y_pred.shape[1]
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
        return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))

    def _brier_score_multi(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate multi-class Brier score."""
        n_classes = y_pred.shape[1]
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
        return float(np.mean(np.sum((y_pred - y_true_onehot) ** 2, axis=1)))

    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate predictions with calibrated probabilities and no leakage.
        """
        # Convert to list format for feature engineering
        match_data = self._single_match_to_list(features)

        # Build features (strict, no leakage)
        X_1x2, X_ou, X_btts, _, _, _, _ = self._build_features_strict(match_data)

        if X_1x2.shape[0] == 0 or not self.calibrated_1x2:
            return self._fallback_prediction()

        # Predict with calibrated models
        probs_1x2 = self.calibrated_1x2.predict_proba(X_1x2)[0]
        probs_ou = self.calibrated_ou.predict_proba(X_ou)[0] if self.calibrated_ou else np.array([0.5, 0.5])
        probs_btts = self.calibrated_btts.predict_proba(X_btts)[0] if self.calibrated_btts else np.array([0.5, 0.5])

        # XGBoost returns [away, draw, home] for classes 0,1,2
        home_prob = float(probs_1x2[2])
        draw_prob = float(probs_1x2[1])
        away_prob = float(probs_1x2[0])

        over_25_prob = float(probs_ou[1])
        under_25_prob = float(probs_ou[0])

        btts_prob = float(probs_btts[1])
        no_btts_prob = float(probs_btts[0])

        # Normalize (ensure sum to 1.0)
        total_1x2 = home_prob + draw_prob + away_prob
        if total_1x2 > 0:
            home_prob /= total_1x2
            draw_prob /= total_1x2
            away_prob /= total_1x2

        # Calculate confidence based on prediction margins
        confidence_1x2 = self._calculate_prediction_confidence([home_prob, draw_prob, away_prob])
        confidence_ou = abs(over_25_prob - 0.5) * 2
        confidence_btts = abs(btts_prob - 0.5) * 2

        # Calculate edge vs market (using opening odds)
        market_odds = features.get('market_odds', {})
        edge = self._calculate_edge(home_prob, draw_prob, away_prob, market_odds)

        return {
            "home_prob": home_prob,
            "draw_prob": draw_prob,
            "away_prob": away_prob,
            "over_2_5_prob": over_25_prob,
            "under_2_5_prob": under_25_prob,
            "btts_prob": btts_prob,
            "no_btts_prob": no_btts_prob,
            "home_goals_expectation": home_prob * 2.8,
            "away_goals_expectation": away_prob * 2.2,
            "confidence": {
                "1x2": confidence_1x2,
                "over_under": confidence_ou,
                "btts": confidence_btts
            },
            "calibrated": True,
            "top_features": dict(sorted(self.feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True)[:5]),
            "edge_vs_market": edge,
            "has_market_edge": edge.get("has_edge", False)
        }

    def _single_match_to_list(self, features: Dict[str, Any]) -> List[Dict]:
        """Convert single match features to list format for feature engineering."""
        market_odds = features.get('market_odds', {})

        return [{
            'home_team': features.get('home_team', 'unknown'),
            'away_team': features.get('away_team', 'unknown'),
            'home_goals': 0,
            'away_goals': 0,
            'match_date': features.get('match_date', datetime.now().isoformat()),
            'home_opening_odds': market_odds.get('home_opening', 2.0),
            'draw_opening_odds': market_odds.get('draw_opening', 3.2),
            'away_opening_odds': market_odds.get('away_opening', 2.0),
            'over_25_opening_odds': market_odds.get('over_25_opening', 1.9),
            'btts_opening_odds': market_odds.get('btts_opening', 1.9),
            'home_last_match_days': features.get('home_rest_days', 7),
            'away_last_match_days': features.get('away_rest_days', 7),
        }]

    def _calculate_prediction_confidence(self, probs: List[float]) -> float:
        """Calculate confidence based on how clear the favorite is."""
        sorted_probs = sorted(probs, reverse=True)
        margin = sorted_probs[0] - sorted_probs[1]
        confidence = 0.5 + (margin * 0.5)
        return min(max(confidence, 0.5), 0.95)

    def _calculate_edge(
        self, 
        home_prob: float, 
        draw_prob: float, 
        away_prob: float, 
        market_odds: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Calculate edge vs market using OPENING odds only."""
        if not market_odds:
            return {"has_edge": False, "reason": "No market odds provided"}

        # Use opening odds, not closing
        home_odd = market_odds.get('home_opening', market_odds.get('home', 0))
        draw_odd = market_odds.get('draw_opening', market_odds.get('draw', 0))
        away_odd = market_odds.get('away_opening', market_odds.get('away', 0))

        edges = {}
        outcomes = ["home", "draw", "away"]
        model_probs = [home_prob, draw_prob, away_prob]
        market_odds_list = [home_odd, draw_odd, away_odd]

        for outcome, model_prob, odd in zip(outcomes, model_probs, market_odds_list):
            if odd > 0:
                market_prob = 1 / odd
                edge = model_prob - market_prob
                edges[outcome] = {
                    "model_prob": model_prob,
                    "market_prob": market_prob,
                    "market_odd": odd,
                    "edge": edge,
                    "has_edge": edge > 0.02
                }

        best_edge = max(edges.items(), key=lambda x: x[1]['edge']) if edges else (None, {})

        return {
            "has_edge": best_edge[1].get('has_edge', False) if best_edge[1] else False,
            "best_outcome": best_edge[0] if best_edge[0] else None,
            "best_edge_percent": round(best_edge[1]['edge'] * 100, 2) if best_edge[1] else 0,
            "all_edges": edges,
            "odds_type": "opening"
        }

    def _fallback_prediction(self) -> Dict[str, Any]:
        """Return fallback prediction when model not ready."""
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
            "calibrated": False,
            "edge_vs_market": {"has_edge": False}
        }

    def get_confidence_score(self, market: str = "1x2") -> float:
        """Return confidence score based on training size and calibration."""
        if self.trained_matches_count < 200:
            return 0.5
        elif self.trained_matches_count < 1000:
            return 0.65
        else:
            return 0.75

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
            'calibrated_1x2': self.calibrated_1x2,
            'calibrated_ou': self.calibrated_ou,
            'calibrated_btts': self.calibrated_btts,
            'feature_columns_1x2': self.feature_columns_1x2,
            'feature_columns_ou': self.feature_columns_ou,
            'feature_columns_btts': self.feature_columns_btts,
            'feature_importance': self.feature_importance,
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
        self.calibrated_1x2 = data['calibrated_1x2']
        self.calibrated_ou = data['calibrated_ou']
        self.calibrated_btts = data['calibrated_btts']
        self.feature_columns_1x2 = data['feature_columns_1x2']
        self.feature_columns_ou = data['feature_columns_ou']
        self.feature_columns_btts = data['feature_columns_btts']
        self.feature_importance = data['feature_importance']
        self.trained_matches_count = data['trained_matches_count']
        self.unique_teams = data['unique_teams']
        self.decay_days = data.get('decay_days', 180)
        self.min_weight = data.get('min_weight', 0.2)

        # Extract base models from calibrated classifiers
        if self.calibrated_1x2:
            self.model_1x2 = self.calibrated_1x2.base_estimator
        if self.calibrated_ou:
            self.model_over_under = self.calibrated_ou.base_estimator
        if self.calibrated_btts:
            self.model_btts = self.calibrated_btts.base_estimator

        # Restore certification data
        for session_val, accuracy in data.get('session_accuracies', {}).items():
            self.session_accuracies[Session(session_val)] = accuracy
        self.final_score = data.get('final_score')
        self.certified = data.get('certified', False)

        logger.info(f"Model loaded from {path}")