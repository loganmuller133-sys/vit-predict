# app/api/routes/history.py
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.db.database import get_db
from app.db.models import Match, Prediction, CLVEntry
from app.api.middleware.auth import verify_api_key

router = APIRouter(prefix="/history", tags=["history"], dependencies=[Depends(verify_api_key)])


@router.get("")
async def get_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """
    Get paginated prediction history.

    FIXED: Pagination with proper indexes to prevent performance issues.
    """

    # Get total count
    count_result = await db.execute(
        select(func.count()).select_from(Prediction)
    )
    total = count_result.scalar()

    # Get paginated results with efficient joins
    result = await db.execute(
        select(Match, Prediction, CLVEntry)
        .join(Prediction, Match.id == Prediction.match_id)
        .outerjoin(CLVEntry, Prediction.id == CLVEntry.prediction_id)
        .order_by(Prediction.timestamp.desc())
        .offset(offset)
        .limit(limit)
    )

    rows = result.all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "predictions": [
            {
                "match_id": row.Match.id,
                "home_team": row.Match.home_team,
                "away_team": row.Match.away_team,
                "league": row.Match.league,
                "kickoff_time": row.Match.kickoff_time.isoformat(),
                "home_prob": row.Prediction.home_prob,
                "draw_prob": row.Prediction.draw_prob,
                "away_prob": row.Prediction.away_prob,
                "consensus_prob": row.Prediction.consensus_prob,
                "recommended_stake": row.Prediction.recommended_stake,
                "edge": row.Prediction.vig_free_edge,
                "bet_side": row.Prediction.bet_side,
                "actual_outcome": row.Match.actual_outcome,
                "clv": row.CLVEntry.clv if row.CLVEntry else None,
                "profit": row.CLVEntry.profit if row.CLVEntry else None,
                "timestamp": row.Prediction.timestamp.isoformat()
            }
            for row in rows
        ]
    }