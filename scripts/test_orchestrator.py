#!/usr/bin/env python
# scripts/test_orchestrator.py
"""Test the model orchestrator"""

import sys
import asyncio
sys.path.append('/workspaces/vit-predict')

from services.ml_service.models.model_orchestrator import ModelOrchestrator

async def test_orchestrator():
    """Test the orchestrator prediction"""
    print("Testing orchestrator...")

    orch = ModelOrchestrator()
    orch.load_all_models()

    print(f"Loaded {len(orch.models)} models")

    features = {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "league": "premier_league"
    }

    try:
        result = await orch.predict(features, "test_match_123")
        print("Prediction successful!")
        print(f"Predictions: {result.get('predictions', {})}")
        return result
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_orchestrator())