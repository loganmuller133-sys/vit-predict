# app/tasks/odds.py
import logging
from datetime import datetime
from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(name="fetch_odds_task")
def fetch_odds_task(match_id: int):
    """Fetch latest odds for a match"""
    logger.info(f"Fetching odds for match {match_id}")
    # TODO: Implement odds fetching logic
    return {"match_id": match_id, "status": "completed", "timestamp": datetime.now().isoformat()}


@shared_task(name="fetch_batch_odds_task")
def fetch_batch_odds_task(match_ids: list):
    """Fetch odds for multiple matches"""
    results = []
    for match_id in match_ids:
        result = fetch_odds_task.delay(match_id)
        results.append(result.id)
    return {"batch_size": len(match_ids), "task_ids": results}