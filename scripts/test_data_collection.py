#!/usr/bin/env python
# scripts/test_data_collection.py
"""Test script for data collection pipeline"""

import asyncio
import logging
from datetime import datetime
from app.pipelines.data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_data_collection():
    """Test the data collection pipeline"""
    logger.info("Testing data collection pipeline...")

    # Initialize data loader with demo keys
    data_loader = DataLoader(
        api_key="demo_key",  # Will fail but shows structure
        odds_api_key="demo_key",
        enable_scraping=False,  # Disable scraping for now
        enable_odds=False  # Disable odds for now
    )

    try:
        # Test fetching fixtures
        logger.info("Fetching fixtures for Premier League...")
        context = await data_loader.fetch_all_context(
            competition="premier_league",
            days_ahead=7,
            include_recent_form=False,  # Simplify for testing
            include_h2h=False,
            include_odds=False
        )

        logger.info(f"Found {len(context.fixtures)} fixtures")
        logger.info(f"Found {len(context.injuries)} injuries")

        # Show sample fixture
        if context.fixtures:
            fixture = context.fixtures[0]
            logger.info(f"Sample fixture: {fixture.get('home_team', {}).get('name')} vs {fixture.get('away_team', {}).get('name')}")

        return context

    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return None
    finally:
        await data_loader.close()

if __name__ == "__main__":
    result = asyncio.run(test_data_collection())
    if result:
        logger.info("Data collection test completed successfully")
    else:
        logger.error("Data collection test failed")