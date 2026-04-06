# app/tasks/edges.py
import logging
from celery import shared_task
from app.services.edge_database import EdgeDatabase
from app.db.database import AsyncSessionLocal

logger = logging.getLogger(__name__)


@shared_task(name="recalculate_edges_task")
def recalculate_edges_task(bet_side: str, edge_value: float):
    """Recalculate edges asynchronously"""
    import asyncio

    async def _recalculate():
        async with AsyncSessionLocal() as db:
            edge_db = EdgeDatabase(db)
            # Update edge performance
            await edge_db.update_edge_performance(
                edge_id=f"{bet_side}_edge",
                new_roi=edge_value,
                new_edge_value=edge_value
            )
            await db.commit()

    asyncio.run(_recalculate())
    logger.info(f"Edges recalculated for {bet_side}")