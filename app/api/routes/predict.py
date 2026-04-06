# app/api/routes/predict.py
import hashlib
import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

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
telegram_alerts = None  # Add this global variable
MAX_STAKE = 0.05
MIN_EDGE_THRESHOLD = 0.02


def set_orchestrator(orch):
    global orchestrator
    orchestrator = orch


def set_telegram_alerts(alerts):
    """Set telegram alerts instance"""
    global telegram_alerts
    telegram_alerts = alerts


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
    """Generate prediction for a match with idempotency"""
    global telegram_alerts  # Declare global to access the variable

    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    idempotency_key = create_idempotency_key(match)

    # Check for existing prediction using request_hash column
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
        else:
            logger.warning(f"Match not found for prediction {existing_pred.id}")

    # Use transaction block for data consistency
    async with db.begin():
        try:
            # Save match to database
            db_match = Match(
                home_team=match.home_team,
                away_team=match.away_team,
                league=match.league,
                kickoff_time=match.kickoff_time,
                opening_odds_home=match.market_odds.get("home"),
                opening_odds_draw=match.market_odds.get("draw"),
                opening_odds_away=match.market_odds.get("away")
            )
            db.add(db_match)
            await db.flush()

            # Run orchestrator
            features = {
                "home_team": match.home_team,
                "away_team": match.away_team,
                "league": match.league,
                "market_odds": match.market_odds
            }

            result = await orchestrator.predict(features)
            result = validate_prediction_response(result)

            # Determine best bet
            home_prob = result.get("home_prob", 0.33)
            draw_prob = result.get("draw_prob", 0.33)
            away_prob = result.get("away_prob", 0.33)

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
                logger.warning(f"Stake {recommended_stake:.2%} clamped to {MAX_STAKE:.2%}")
                recommended_stake = MAX_STAKE

            probs = {"home": home_prob, "draw": draw_prob, "away": away_prob}
            consensus_prob = max(probs.values())

            # Save prediction with idempotency key
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

            # Record CLV entry (ONLY ONE)
            if best_bet.get("has_edge") and best_bet.get("best_side"):
                await CLVTracker.record_entry(
                    db, 
                    db_match.id, 
                    prediction.id,
                    best_bet["best_side"],
                    best_bet["odds"]
                )

            await db.commit()
            await db.refresh(db_match)
            await db.refresh(prediction)

            # Dispatch async tasks AFTER commit (so data is persisted)
            if best_bet.get("has_edge"):
                update_clv_task.delay(db_match.id, prediction.id, best_bet["odds"])
                recalculate_edges_task.delay(best_bet["best_side"], best_bet["edge"])

            # Send Telegram alert if edge is significant (AFTER commit)
            if telegram_alerts and best_bet.get("has_edge") and best_bet.get("edge", 0) > MIN_EDGE_THRESHOLD:
                try:
                    alert = BetAlert(
                        match_id=db_match.id,
                        home_team=match.home_team,
                        away_team=match.away_team,
                        prediction=best_bet.get("best_side", "unknown"),
                        probability=best_bet.get("edge", 0) + 0.5,
                        edge=best_bet.get("edge", 0),
                        stake=recommended_stake,
                        odds=best_bet.get("odds", 2.0),
                        confidence=result.get("confidence", {}).get("1x2", 0.5),
                        kickoff_time=match.kickoff_time
                    )
                    await telegram_alerts.send_bet_alert(alert)
                    logger.info(f"Telegram alert sent for match {db_match.id}")
                except Exception as e:
                    logger.error(f"Failed to send Telegram alert: {e}")

            logger.info(f"Prediction saved: match_id={db_match.id}, side={best_bet.get('best_side')}, edge={best_bet.get('edge', 0):.4f}")

            return build_prediction_response(prediction, db_match)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))