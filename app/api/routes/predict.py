# app/api/routes/predict.py
import hashlib
import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone

from app.db.database import get_db
from app.db.models import Match, Prediction
from app.schemas.schemas import MatchRequest, PredictionResponse
from app.services.clv_tracker import CLVTracker
from app.services.market_utils import MarketUtils
from app.api.middleware.auth import verify_api_key
from app.services.alerts import BetAlert

# Celery tasks
from app.tasks.clv import update_clv_task
from app.tasks.edges import recalculate_edges_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predictions"], dependencies=[Depends(verify_api_key)])

orchestrator = None
telegram_alerts = None
MAX_STAKE = 0.05
MIN_EDGE_THRESHOLD = 0.02


def set_orchestrator(orch):
    global orchestrator
    orchestrator = orch


def set_telegram_alerts(alerts):
    """Set telegram alerts instance"""
    global telegram_alerts
    telegram_alerts = alerts


def to_naive_utc(dt_input) -> datetime:
    """
    Convert any datetime to naive UTC for storage in TIMESTAMP WITHOUT TIME ZONE.

    Args:
        dt_input: datetime object or ISO string

    Returns:
        Naive datetime object (timezone-aware datetimes converted to UTC, then stripped)
    """
    if isinstance(dt_input, str):
        # Parse ISO string
        try:
            parsed = datetime.fromisoformat(dt_input.replace('Z', '+00:00'))
            # Strip timezone info - assume it's UTC
            return parsed.replace(tzinfo=None)
        except Exception as e:
            logger.warning(f"Failed to parse kickoff_time string '{dt_input}': {e}")
            return datetime.now()
    elif isinstance(dt_input, datetime):
        # If it has timezone info, convert to UTC then strip
        if dt_input.tzinfo is not None:
            utc_dt = dt_input.astimezone(timezone.utc)
            return utc_dt.replace(tzinfo=None)
        # If already naive, assume it's UTC and return as-is
        return dt_input
    else:
        return datetime.now()


def create_idempotency_key(match: MatchRequest) -> str:
    """Create unique idempotency key from match data"""
    content = {
        "home_team": match.home_team,
        "away_team": match.away_team,
        "kickoff_time": match.kickoff_time.isoformat(),
        "league": match.league
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:32]


def validate_prediction_response(result: dict) -> dict:
    """Validate orchestrator response - fail fast if missing required fields"""
    required_fields = ["home_prob", "draw_prob", "away_prob"]

    for field in required_fields:
        if field not in result:
            raise ValueError(f"Orchestrator response missing required field: {field}")

    # CRITICAL: Validate probabilities sum to 1
    total = result["home_prob"] + result["draw_prob"] + result["away_prob"]
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Probabilities sum to {total}, not 1.0")

    return result


def build_prediction_response(prediction: Prediction, match: Match) -> PredictionResponse:
    """Build prediction response from database objects"""
    return PredictionResponse(
        match_id=match.id,
        home_prob=prediction.home_prob,
        draw_prob=prediction.draw_prob,
        away_prob=prediction.away_prob,
        over_25_prob=prediction.over_25_prob,
        under_25_prob=prediction.under_25_prob,
        btts_prob=prediction.btts_prob,
        consensus_prob=prediction.consensus_prob,
        final_ev=prediction.final_ev,
        recommended_stake=prediction.recommended_stake,
        edge=prediction.vig_free_edge,
        confidence=prediction.confidence,
        timestamp=prediction.timestamp
    )


@router.post("", response_model=PredictionResponse)
async def predict(
    match: MatchRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate prediction for a match with idempotency and timezone safety"""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    idempotency_key = create_idempotency_key(match)

    try:
        # Check for existing prediction
        existing = await db.execute(
            select(Prediction).where(Prediction.request_hash == idempotency_key)
        )
        existing_pred = existing.scalar_one_or_none()

        if existing_pred:
            logger.info(f"Returning cached prediction for {idempotency_key[:8]}...")
            match_result = await db.execute(select(Match).where(Match.id == existing_pred.match_id))
            db_match = match_result.scalar_one_or_none()
            if db_match:
                return build_prediction_response(existing_pred, db_match)

        # ✅ FIX: Convert kickoff_time to naive UTC before saving
        naive_kickoff = to_naive_utc(match.kickoff_time)

        logger.debug(f"Kickoff time conversion: {match.kickoff_time} -> {naive_kickoff}")

        # Save match to database
        db_match = Match(
            home_team=match.home_team,
            away_team=match.away_team,
            league=match.league,
            kickoff_time=naive_kickoff,  # ✅ Use naive UTC
            opening_odds_home=match.market_odds.get("home"),
            opening_odds_draw=match.market_odds.get("draw"),
            opening_odds_away=match.market_odds.get("away")
        )
        db.add(db_match)
        await db.flush()

        logger.info(f"Match saved: {match.home_team} vs {match.away_team} at {naive_kickoff}")

        # Run orchestrator
        features = {
            "home_team": match.home_team,
            "away_team": match.away_team,
            "league": match.league,
            "market_odds": match.market_odds
        }

        raw_result = await orchestrator.predict(features, idempotency_key)
        result = validate_prediction_response(raw_result.get("predictions", raw_result))

        # Extract probabilities
        home_prob = float(result.get("home_prob", 0.33))
        draw_prob = float(result.get("draw_prob", 0.33))
        away_prob = float(result.get("away_prob", 0.33))

        # Determine best bet
        home_odds = match.market_odds.get("home", 2.0)
        draw_odds = match.market_odds.get("draw", 3.2)
        away_odds = match.market_odds.get("away", 2.0)

        best_bet = MarketUtils.determine_best_bet(
            home_prob, draw_prob, away_prob,
            home_odds, draw_odds, away_odds
        )

        # Clamp stake
        recommended_stake = best_bet.get("kelly_stake", 0)
        if recommended_stake > MAX_STAKE:
            recommended_stake = MAX_STAKE

        probs = {"home": home_prob, "draw": draw_prob, "away": away_prob}
        consensus_prob = max(probs.values())

        # Save prediction
        prediction = Prediction(
            request_hash=idempotency_key,
            match_id=db_match.id,
            home_prob=home_prob,
            draw_prob=draw_prob,
            away_prob=away_prob,
            over_25_prob=result.get("over_2_5_prob"),
            under_25_prob=result.get("under_2_5_prob"),
            btts_prob=result.get("btts_prob"),
            no_btts_prob=result.get("no_btts_prob"),
            consensus_prob=consensus_prob,
            final_ev=best_bet.get("edge", 0),
            recommended_stake=recommended_stake,
            model_weights=result.get("model_weights", {}),
            confidence=result.get("confidence", {}).get("1x2", 0.5),
            bet_side=best_bet.get("best_side"),
            entry_odds=best_bet.get("odds", 2.0),
            raw_edge=best_bet.get("raw_edge", 0),
            normalized_edge=best_bet.get("edge", 0),
            vig_free_edge=best_bet.get("edge", 0)
        )
        db.add(prediction)
        await db.flush()
        await db.commit()

        logger.info(f"Prediction saved: match_id={db_match.id}, side={best_bet.get('best_side')}, edge={best_bet.get('edge', 0):.4f}")

        # ✅ Send Telegram alert if edge > 3%
        edge_value = best_bet.get("edge", 0)
        if edge_value > 0.03 and telegram_alerts and telegram_alerts.enabled:
            try:
                alert = BetAlert(
                    match_id=db_match.id,
                    home_team=match.home_team,
                    away_team=match.away_team,
                    prediction=best_bet.get("best_side", "N/A"),
                    probability=consensus_prob,
                    edge=edge_value,
                    stake=recommended_stake,
                    odds=best_bet.get("odds", 2.0),
                    confidence=result.get("confidence", {}).get("1x2", 0.5),
                    kickoff_time=naive_kickoff
                )
                await telegram_alerts.send_bet_alert(alert)
                logger.info(f"Edge alert sent for {match.home_team} vs {match.away_team}")
            except Exception as e:
                logger.warning(f"Failed to send Telegram alert: {e}")

        return build_prediction_response(prediction, db_match)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))