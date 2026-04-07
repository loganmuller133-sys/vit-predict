#!/usr/bin/env python
# scripts/train_poisson_model.py
"""Train the Poisson goal model with synthetic data"""

import sys
import os
sys.path.append('/workspaces/vit-predict')

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

from services.ml_service.models.model_1_poisson import PoissonGoalModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_matches(num_matches: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic match data for training"""
    logger.info(f"Generating {num_matches} synthetic matches...")

    teams = [
        "Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United",
        "Tottenham", "Newcastle", "Aston Villa", "West Ham", "Crystal Palace",
        "Brighton", "Fulham", "Wolverhampton", "Southampton", "Everton"
    ]

    matches = []
    base_date = datetime.now() - timedelta(days=365)

    for i in range(num_matches):
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])

        # Generate realistic goals (Poisson-like distribution)
        home_goals = np.random.poisson(1.4)  # Home advantage
        away_goals = np.random.poisson(1.1)

        match_date = base_date + timedelta(days=i*2)  # Spread over time

        match = {
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': int(home_goals),
            'away_goals': int(away_goals),
            'match_date': match_date.isoformat(),
            'league': 'premier_league'
        }

        matches.append(match)

    logger.info(f"Generated {len(matches)} matches")
    return matches

def train_poisson_model():
    """Train the Poisson model with synthetic data"""
    logger.info("Starting Poisson model training...")

    # Generate training data
    matches = generate_synthetic_matches(2000)

    # Initialize model
    model = PoissonGoalModel("poisson_001")

    # Train the model
    logger.info("Training model...")
    result = model.train(matches, validation_split=0.2, use_time_weights=True)

    if 'error' in result:
        logger.error(f"Training failed: {result['error']}")
        return None

    logger.info("Training completed successfully!")
    logger.info(f"Trained on {result.get('trained_matches', 0)} matches")
    logger.info(".3f")
    logger.info(".3f")

    # Test prediction
    test_features = {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "league": "premier_league"
    }

    logger.info("Testing prediction...")
    prediction = model.predict(test_features)

    logger.info("Sample prediction:")
    logger.info(".3f")
    logger.info(".3f")
    logger.info(".3f")

    # Save model
    model_path = "/workspaces/vit-predict/models/poisson_model.pkl"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    return model

if __name__ == "__main__":
    model = train_poisson_model()
    if model:
        logger.info("Poisson model training completed successfully!")
    else:
        logger.error("Model training failed")