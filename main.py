# main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from dotenv import load_dotenv

from app.db.database import engine, Base, get_db
from app.api.routes import predict, result, history
from app.api.middleware.auth import APIKeyMiddleware
from app.api.middleware.logging import LoggingMiddleware
from app.schemas.schemas import HealthResponse
from app.pipelines.data_loader import DataLoader
from app.services.alerts import TelegramAlert, AlertPriority

# ML Orchestrator
from services.ml_service.models.model_orchestrator import ModelOrchestrator

load_dotenv()

orchestrator = None
data_loader = None
telegram_alerts = None


# --- HELPER FUNCTIONS ---
async def _check_db_connection() -> bool:
    """Check database connection"""
    try:
        async for session in get_db():
            await session.execute(select(1))
            return True
    except:
        pass
    return False


async def fetch_and_predict(competition: str, days_ahead: int = 7):
    """Background task to fetch data and generate predictions"""
    global data_loader, orchestrator, telegram_alerts

    if not data_loader or not orchestrator:
        print("Data loader or orchestrator not initialized")
        return

    try:
        print(f"\n📡 Fetching data for {competition}...")

        # Fetch context
        context = await data_loader.fetch_all_context(
            competition=competition,
            days_ahead=days_ahead,
            include_recent_form=True,
            include_h2h=True,
            include_odds=True
        )

        print(f"   ✅ Fetched {len(context.fixtures)} fixtures")
        print(f"   ✅ Found {len(context.injuries)} injuries")
        print(f"   ✅ Got odds for {len(context.odds)} matches")

        # Generate predictions for each fixture
        predictions = []
        for fixture in context.fixtures:
            features = {
                "home_team": fixture["home_team"]["name"],
                "away_team": fixture["away_team"]["name"],
                "league": competition,
                "injury_impact": context.injuries,
                "standings": context.standings,
                "market_odds": fixture.get("odds", {}),
                "recent_form": context.recent_form.get(fixture["home_team"]["external_id"], []),
                "away_recent_form": context.recent_form.get(fixture["away_team"]["external_id"], [])
            }

            prediction = await orchestrator.predict(features)
            predictions.append(prediction)

            # Send alert for high-value bets
            edge_value = prediction.get("edge_vs_market", {}).get("best_edge_percent", 0)
            if prediction.get("has_market_edge") and edge_value > 3:
                if telegram_alerts and telegram_alerts.enabled:
                    from app.services.alerts import BetAlert
                    from datetime import datetime

                    alert = BetAlert(
                        match_id=fixture.get("external_id", 0),
                        home_team=fixture["home_team"]["name"],
                        away_team=fixture["away_team"]["name"],
                        prediction=prediction.get("consensus_outcome", "N/A"),
                        probability=prediction.get("consensus_prob", 0.5),
                        edge=edge_value / 100,
                        stake=prediction.get("recommended_stake", 0),
                        odds=prediction.get("edge_vs_market", {}).get("best_odds", 2.0),
                        confidence=prediction.get("confidence", {}).get("1x2", 0.5),
                        kickoff_time=datetime.fromisoformat(fixture.get("kickoff_time", datetime.now().isoformat()))
                    )
                    await telegram_alerts.send_bet_alert(alert)

        print(f"   ✅ Generated {len(predictions)} predictions")
        return predictions

    except Exception as e:
        print(f"❌ Error in background task: {e}")
        if telegram_alerts and telegram_alerts.enabled:
            await telegram_alerts.send_anomaly_alert(
                "Data Fetch Error",
                {"competition": competition, "error": str(e)},
                "warning"
            )
        return []


# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, data_loader, telegram_alerts

    print("🚀 Starting VIT Sports Intelligence Network...")
    print("=" * 50)

    # Load models
    print("📦 Loading ML models...")
    orchestrator = ModelOrchestrator()
    await orchestrator.load_all_models()
    print(f"   ✅ Loaded {orchestrator.num_models_ready()} models")

    # Initialize data loader
    print("🔌 Initializing data sources...")
    football_api_key = os.getenv("FOOTBALL_DATA_API_KEY", "")
    odds_api_key = os.getenv("ODDS_API_KEY", "")

    data_loader = DataLoader(
        api_key=football_api_key,
        odds_api_key=odds_api_key,
        enable_scraping=os.getenv("ENABLE_SCRAPING", "true").lower() == "true",
        enable_odds=os.getenv("ENABLE_ODDS", "true").lower() == "true"
    )
    print(f"   ✅ Football API: {'ENABLED' if football_api_key else 'DISABLED'}")
    print(f"   ✅ Odds API: {'ENABLED' if odds_api_key else 'DISABLED'}")
    print(f"   ✅ Scraping: {'ENABLED' if data_loader.enable_scraping else 'DISABLED'}")

    # Initialize Telegram alerts
    print("📱 Setting up notifications...")
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if bot_token and chat_id:
        telegram_alerts = TelegramAlert(bot_token, chat_id, enabled=True)
        await telegram_alerts.send_startup_message()
        print("   ✅ Telegram Alerts: ENABLED")
    else:
        telegram_alerts = TelegramAlert("", "", enabled=False)
        print("   ⚠️ Telegram Alerts: DISABLED (missing credentials)")

    # Set orchestrator for routes
    predict.set_orchestrator(orchestrator)
    if telegram_alerts and telegram_alerts.enabled:
        predict.set_telegram_alerts(telegram_alerts)

    # Create database tables
    print("🗄️ Initializing database...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("   ✅ Database ready")

    print("=" * 50)
    print("✅ VIT Network is OPERATIONAL")
    print(f"📍 API: http://localhost:8000")
    print(f"📊 Health: http://localhost:8000/health")
    print("=" * 50)

    yield

    # Cleanup
    print("\n🛑 VIT Network shutting down...")
    if telegram_alerts and telegram_alerts.enabled:
        await telegram_alerts.send_shutdown_message()
    if data_loader:
        await data_loader.close()
    orchestrator = None
    print("✅ Cleanup complete")


# --- APP ---
app = FastAPI(
    title="VIT Sports Intelligence Network",
    description="12-Model Ensemble for Football Prediction with CLV Tracking",
    version="2.0",
    lifespan=lifespan
)

# Add middleware (order matters)
app.add_middleware(LoggingMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router)
app.include_router(result.router)
app.include_router(history.router)


# --- HEALTH ---
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint (no auth required)"""
    db_status = False
    try:
        async for session in get_db():
            await session.execute(select(1))
            db_status = True
            break
    except:
        pass

    return HealthResponse(
        status="ok",
        models_loaded=orchestrator.num_models_ready() if orchestrator else 0,
        db_connected=db_status,
        clv_tracking_enabled=True
    )


# --- DATA ENDPOINTS ---
@app.get("/fetch")
async def fetch_fixtures(
    competition: str = "premier_league",
    days_ahead: int = 7,
    background_tasks: BackgroundTasks = None
):
    """
    Fetch fixtures and generate predictions.

    Args:
        competition: League name (premier_league, la_liga, bundesliga, etc.)
        days_ahead: Number of days to look ahead
        background_tasks: FastAPI background tasks
    """
    if not data_loader:
        return {"error": "Data loader not initialized"}

    if background_tasks:
        background_tasks.add_task(fetch_and_predict, competition, days_ahead)
        return {
            "message": f"Background task started for {competition}",
            "status": "processing",
            "competition": competition,
            "days_ahead": days_ahead
        }

    # Synchronous fetch (for testing)
    context = await data_loader.fetch_all_context(
        competition=competition,
        days_ahead=days_ahead,
        include_recent_form=True,
        include_h2h=True,
        include_odds=True
    )

    return {
        "competition": competition,
        "fixtures_count": len(context.fixtures),
        "injuries_count": len(context.injuries),
        "odds_count": len(context.odds),
        "fixtures": [
            {
                "home_team": f["home_team"]["name"],
                "away_team": f["away_team"]["name"],
                "kickoff_time": f["kickoff_time"],
                "odds": f.get("odds", {})
            }
            for f in context.fixtures[:10]
        ]
    }


@app.get("/fetch/historical")
async def fetch_historical(
    competition: str = "premier_league",
    days_back: int = 90,
    limit: int = 100
):
    """Fetch historical matches for training"""
    if not data_loader:
        return {"error": "Data loader not initialized"}

    matches = await data_loader.fetch_historical_matches(
        competition=competition,
        days_back=days_back,
        limit=limit
    )

    return {
        "competition": competition,
        "matches_count": len(matches),
        "matches": matches[:20]
    }


# --- ODDS ENDPOINTS ---
@app.get("/odds")
async def get_odds(
    competition: str = "premier_league",
    days_ahead: int = 3
):
    """Get current odds for matches"""
    if not data_loader:
        return {"error": "Data loader not initialized"}

    if not data_loader.odds_client:
        return {"error": "Odds client not initialized. Check ODDS_API_KEY in .env"}

    odds = await data_loader.fetch_odds_only(
        competition=competition,
        days_ahead=days_ahead
    )

    return {
        "competition": competition,
        "matches": [
            {
                "match_id": o.match_id,
                "home_odds": o.home_odds,
                "draw_odds": o.draw_odds,
                "away_odds": o.away_odds,
                "over_25": o.over_25_odds,
                "under_25": o.under_25_odds,
                "btts_yes": o.btts_yes_odds,
                "btts_no": o.btts_no_odds,
                "vig_free_home": o.vig_free_probabilities()["home"],
                "vig_free_draw": o.vig_free_probabilities()["draw"],
                "vig_free_away": o.vig_free_probabilities()["away"],
                "overround": o.overround(),
                "bookmaker": o.bookmaker,
                "timestamp": o.timestamp.isoformat() if o.timestamp else None
            }
            for o in odds
        ]
    }


@app.get("/odds/sharp")
async def get_sharp_odds(
    competition: str = "premier_league"
):
    """Get odds from sharp books only (Pinnacle)"""
    if not data_loader:
        return {"error": "Data loader not initialized"}

    if not data_loader.odds_client:
        return {"error": "Odds client not initialized"}

    odds = await data_loader.fetch_sharp_odds_only(competition)

    return {
        "competition": competition,
        "sharp_odds": [
            {
                "match_id": o.match_id,
                "home_odds": o.home_odds,
                "draw_odds": o.draw_odds,
                "away_odds": o.away_odds,
                "implied_home": o.implied_probabilities()["home"],
                "implied_draw": o.implied_probabilities()["draw"],
                "implied_away": o.implied_probabilities()["away"],
                "vig_free_home": o.vig_free_probabilities()["home"],
                "vig_free_draw": o.vig_free_probabilities()["draw"],
                "vig_free_away": o.vig_free_probabilities()["away"]
            }
            for o in odds
        ]
    }


# --- ALERTS ENDPOINTS ---
@app.get("/alerts/test")
async def test_alerts():
    """Test Telegram alerts"""
    if not telegram_alerts or not telegram_alerts.enabled:
        return {"error": "Telegram alerts not enabled"}

    await telegram_alerts.send_message("✅ Test message from VIT Network", AlertPriority.SUCCESS)
    return {"message": "Test alert sent"}


@app.get("/alerts/status")
async def alerts_status():
    """Get alerts system status"""
    return {
        "telegram_enabled": telegram_alerts.enabled if telegram_alerts else False,
        "bot_token_configured": bool(os.getenv("TELEGRAM_BOT_TOKEN", "")),
        "chat_id_configured": bool(os.getenv("TELEGRAM_CHAT_ID", ""))
    }


# --- SYSTEM ENDPOINTS ---
@app.get("/system/status")
async def system_status():
    """Get complete system status"""
    db_connected = await _check_db_connection()

    return {
        "orchestrator": {
            "models_loaded": orchestrator.num_models_ready() if orchestrator else 0,
            "status": "ready" if orchestrator else "not_initialized"
        },
        "data_loader": {
            "scraping_enabled": data_loader.enable_scraping if data_loader else False,
            "odds_enabled": data_loader.enable_odds if data_loader else False,
            "status": "ready" if data_loader else "not_initialized"
        },
        "alerts": {
            "telegram_enabled": telegram_alerts.enabled if telegram_alerts else False,
            "status": "ready" if (telegram_alerts and telegram_alerts.enabled) else "disabled"
        },
        "database": {
            "connected": db_connected,
            "type": "postgresql"
        }
    }


# --- ROOT ---
@app.get("/")
async def root():
    return {
        "name": "VIT Sports Intelligence Network",
        "version": "2.0",
        "status": "operational",
        "endpoints": {
            "predict": "POST /predict - Generate prediction",
            "results": "POST /results/{match_id} - Update match result",
            "history": "GET /history - View prediction history",
            "health": "GET /health - System health",
            "fetch": "GET /fetch - Fetch fixtures",
            "historical": "GET /fetch/historical - Fetch historical matches",
            "odds": "GET /odds - Get current odds",
            "sharp_odds": "GET /odds/sharp - Get sharp odds",
            "alerts_test": "GET /alerts/test - Test alerts",
            "alerts_status": "GET /alerts/status - Alerts status",
            "system_status": "GET /system/status - Complete system status"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )