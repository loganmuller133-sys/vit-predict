# app/services/market_utils.py
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class MarketUtils:
    """
    Market utility functions for vig removal and edge calculation.

    Bookmakers build margin (overround) into their odds.
    Raw edge calculations are misleading without removing vig first.
    """

    @staticmethod
    def calculate_implied_probabilities(
        home_odds: float,
        draw_odds: float,
        away_odds: float
    ) -> Dict[str, float]:
        """Calculate implied probabilities from odds"""
        return {
            "home": 1 / home_odds if home_odds > 0 else 0.33,
            "draw": 1 / draw_odds if draw_odds > 0 else 0.33,
            "away": 1 / away_odds if away_odds > 0 else 0.33
        }

    @staticmethod
    def calculate_overround(
        home_odds: float,
        draw_odds: float,
        away_odds: float
    ) -> float:
        """Calculate bookmaker margin (overround)"""
        implied_sum = (1 / home_odds) + (1 / draw_odds) + (1 / away_odds)
        return implied_sum - 1.0

    @staticmethod
    def remove_vig(
        home_odds: float,
        draw_odds: float,
        away_odds: float
    ) -> Dict[str, float]:
        """
        Remove vig from odds to get true market probabilities.

        Formula: true_prob = (1/odds) / sum(1/odds)
        """
        home_implied = 1 / home_odds if home_odds > 0 else 0
        draw_implied = 1 / draw_odds if draw_odds > 0 else 0
        away_implied = 1 / away_odds if away_odds > 0 else 0

        total = home_implied + draw_implied + away_implied

        if total == 0:
            return {"home": 0.33, "draw": 0.33, "away": 0.33}

        return {
            "home": home_implied / total,
            "draw": draw_implied / total,
            "away": away_implied / total
        }

    @staticmethod
    def calculate_true_edge(
        model_prob: float,
        market_odds: float,
        home_odds: float,
        draw_odds: float,
        away_odds: float,
        bet_side: str
    ) -> Tuple[float, float, float]:
        """
        Calculate true edge after removing bookmaker vig.

        Returns:
            raw_edge: model_prob - implied_prob (misleading)
            vig_free_edge: model_prob - vig_free_prob (accurate)
            normalized_edge: edge normalized by market probability
        """
        # Raw implied probability (includes vig)
        raw_implied = 1 / market_odds if market_odds > 0 else 0.33

        # Vig-free probabilities
        vig_free_probs = MarketUtils.remove_vig(home_odds, draw_odds, away_odds)
        vig_free_prob = vig_free_probs.get(bet_side, 0.33)

        # Calculate edges
        raw_edge = model_prob - raw_implied
        vig_free_edge = model_prob - vig_free_prob

        # Normalized edge (as percentage of market probability)
        if vig_free_prob > 0:
            normalized_edge = vig_free_edge / vig_free_prob
        else:
            normalized_edge = 0

        return raw_edge, vig_free_edge, normalized_edge

    @staticmethod
    def calculate_clv(entry_odds: float, closing_odds: float) -> float:
        """Calculate Closing Line Value"""
        if closing_odds <= 0:
            return 0.0
        return (entry_odds - closing_odds) / closing_odds

    @staticmethod
    def determine_best_bet(
        home_prob: float,
        draw_prob: float,
        away_prob: float,
        home_odds: float,
        draw_odds: float,
        away_odds: float
    ) -> Dict[str, any]:
        """
        Determine which bet (if any) has positive edge after vig removal.

        Returns:
            {
                "has_edge": bool,
                "best_side": str,
                "edge": float,
                "vig_free_edge": float,
                "kelly_stake": float
            }
        """
        vig_free_probs = MarketUtils.remove_vig(home_odds, draw_odds, away_odds)

        candidates = []

        # Check home
        home_vig_free = vig_free_probs["home"]
        home_raw_edge = home_prob - (1 / home_odds if home_odds > 0 else 0.33)
        home_true_edge = home_prob - home_vig_free

        candidates.append({
            "side": "home",
            "model_prob": home_prob,
            "vig_free_prob": home_vig_free,
            "true_edge": home_true_edge,
            "raw_edge": home_raw_edge,
            "odds": home_odds
        })

        # Check draw
        draw_vig_free = vig_free_probs["draw"]
        draw_raw_edge = draw_prob - (1 / draw_odds if draw_odds > 0 else 0.33)
        draw_true_edge = draw_prob - draw_vig_free

        candidates.append({
            "side": "draw",
            "model_prob": draw_prob,
            "vig_free_prob": draw_vig_free,
            "true_edge": draw_true_edge,
            "raw_edge": draw_raw_edge,
            "odds": draw_odds
        })

        # Check away
        away_vig_free = vig_free_probs["away"]
        away_raw_edge = away_prob - (1 / away_odds if away_odds > 0 else 0.33)
        away_true_edge = away_prob - away_vig_free

        candidates.append({
            "side": "away",
            "model_prob": away_prob,
            "vig_free_prob": away_vig_free,
            "true_edge": away_true_edge,
            "raw_edge": away_raw_edge,
            "odds": away_odds
        })

        # Find best positive edge
        best = None
        for c in candidates:
            if c["true_edge"] > 0.02:  # 2% edge threshold
                if best is None or c["true_edge"] > best["true_edge"]:
                    best = c

        if best:
            # Calculate Kelly stake
            b = best["odds"] - 1
            p = best["model_prob"]
            q = 1 - p
            kelly = (b * p - q) / b if b > 0 else 0
            kelly = max(0, min(kelly, 0.10))  # Cap at 10%

            return {
                "has_edge": True,
                "best_side": best["side"],
                "edge": best["true_edge"],
                "raw_edge": best["raw_edge"],
                "vig_free_prob": best["vig_free_prob"],
                "odds": best["odds"],
                "kelly_stake": kelly
            }

        return {
            "has_edge": False,
            "best_side": None,
            "edge": 0,
            "raw_edge": 0,
            "vig_free_prob": 0,
            "odds": 0,
            "kelly_stake": 0
        }