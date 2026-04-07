# main.py
import sys
import os
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_env
from app.db.database import engine, Base, get_db
from app.api.routes import predict, result, history
from app.api.middleware.auth import APIKeyMiddleware
from app.api.middleware.logging import LoggingMiddleware
from app.schemas.schemas import HealthResponse
from app.pipelines.data_loader import DataLoader
from app.services.alerts import TelegramAlert, AlertPriority

# ML Orchestrator
from services.ml_service.models.model_orchestrator import ModelOrchestrator

orchestrator = None
data_loader = None
telegram_alerts = None


# --- HELPER FUNCTIONS ---
async def _check_db_connection() -> bool:
    """Check database connection with actual query"""
    try:
        async for session in get_db():
            await session.execute(select(1))
            return True
    except Exception as e:
        print(f"❌ DB Connection Check Failed: {e}")
        return False


async def fetch_and_predict(competition: str, days_ahead: int = 7):
    """Background task to fetch data and generate predictions"""
    global data_loader, orchestrator, telegram_alerts

    if not data_loader or not orchestrator:
        print("❌ Data loader or orchestrator not initialized")
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
            try:
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

                prediction = await orchestrator.predict(features, str(fixture.get("external_id", "")))
                predictions.append(prediction)

                # Send alert for high-value bets (>3% edge)
                edge_value = prediction.get("edge_vs_market", {}).get("best_edge_percent", 0)
                if prediction.get("has_market_edge") and edge_value > 3:
                    if telegram_alerts and telegram_alerts.enabled:
                        from app.services.alerts import BetAlert

                        # FIX: Ensure kickoff_time is timezone-naive UTC
                        kickoff_str = fixture.get("kickoff_time", datetime.now(timezone.utc).isoformat())
                        try:
                            # Parse ISO string and strip timezone info (store as naive UTC)
                            if isinstance(kickoff_str, str):
                                parsed = datetime.fromisoformat(kickoff_str.replace('Z', '+00:00'))
                                kickoff_time = parsed.replace(tzinfo=None)  # Make timezone-naive
                            else:
                                kickoff_time = kickoff_str.replace(tzinfo=None)  # Remove timezone info
                        except Exception as e:
                            print(f"⚠️ Kickoff time parse error: {e}")
                            kickoff_time = datetime.now()

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
                            kickoff_time=kickoff_time
                        )
                        await telegram_alerts.send_bet_alert(alert)
                        print(f"   📱 Edge alert sent for {alert.home_team} vs {alert.away_team}")

            except Exception as e:
                print(f"⚠️ Error processing fixture: {e}")
                continue

        print(f"   ✅ Generated {len(predictions)} predictions")
        return predictions

    except Exception as e:
        print(f"❌ Error in background task: {e}")
        if telegram_alerts and telegram_alerts.enabled:
            try:
                await telegram_alerts.send_anomaly_alert(
                    "Data Fetch Error",
                    {"competition": competition, "error": str(e)},
                    "warning"
                )
            except Exception as alert_err:
                print(f"⚠️ Failed to send anomaly alert: {alert_err}")
        return []


# --- LIFESPAN (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, data_loader, telegram_alerts

    print("\n" + "=" * 60)
    print("🚀 VIT Sports Intelligence Network - Initializing")
    print("=" * 60)

    # 1. VALIDATE API CREDENTIALS
    print("\n🔑 Checking API Credentials...")
    odds_api_key = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY")
    football_api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    print(f"   {'✅' if odds_api_key else '❌'} THE_ODDS_API_KEY / ODDS_API_KEY: {'LOADED' if odds_api_key else 'MISSING'}")
    print(f"   {'✅' if football_api_key else '❌'} FOOTBALL_DATA_API_KEY: {'LOADED' if football_api_key else 'MISSING'}")
    print(f"   {'✅' if bot_token else '⚠️'} TELEGRAM_BOT_TOKEN: {'CONFIGURED' if bot_token else 'NOT SET'}")
    print(f"   {'✅' if chat_id else '⚠️'} TELEGRAM_CHAT_ID: {'CONFIGURED' if chat_id else 'NOT SET'}")

    # 2. LOAD ML MODELS
    print("\n📦 Loading ML Models...")
    try:
        orchestrator = ModelOrchestrator()
        orchestrator.load_all_models()
        num_models = orchestrator.num_models_ready() if orchestrator else 0
        print(f"   ✅ Loaded {num_models} models successfully")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        orchestrator = None

    # 3. INITIALIZE DATA LOADER
    print("\n🔌 Initializing Data Sources...")
    try:
        data_loader = DataLoader(
            api_key=football_api_key or "",
            odds_api_key=odds_api_key or "",
            enable_scraping=get_env("ENABLE_SCRAPING", "true").lower() == "true",
            enable_odds=get_env("ENABLE_ODDS", "true").lower() == "true"
        )
        print(f"   ✅ Football API: {'ENABLED' if football_api_key else 'DISABLED'}")
        print(f"   ✅ Odds API: {'ENABLED' if odds_api_key else 'DISABLED'}")
        print(f"   ✅ Scraping: {'ENABLED' if data_loader.enable_scraping else 'DISABLED'}")
    except Exception as e:
        print(f"   ❌ Data loader initialization failed: {e}")
        data_loader = None

    # 4. SETUP TELEGRAM ALERTS
    print("\n📱 Setting Up Notifications...")
    if bot_token and chat_id:
        try:
            telegram_alerts = TelegramAlert(bot_token, chat_id, enabled=True)
            await telegram_alerts.send_startup_message()
            print("   ✅ Telegram Alerts: ENABLED")
        except Exception as e:
            print(f"   ⚠️ Telegram initialization failed: {e}")
            telegram_alerts = TelegramAlert("", "", enabled=False)
    else:
        telegram_alerts = TelegramAlert("", "", enabled=False)
        print("   ⚠️ Telegram Alerts: DISABLED (missing credentials)")

    # 5. SET ORCHESTRATOR FOR ROUTES
    print("\n🔗 Linking Components to Routes...")
    try:
        predict.set_orchestrator(orchestrator)
        if telegram_alerts and telegram_alerts.enabled:
            predict.set_telegram_alerts(telegram_alerts)
        print("   ✅ Routes configured")
    except Exception as e:
        print(f"   ⚠️ Route configuration partial: {e}")

    # 6. INITIALIZE DATABASE
    print("\n🗄️ Initializing Database...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Verify connection
        db_ok = await _check_db_connection()
        if db_ok:
            print("   ✅ Database ready and connected")
        else:
            print("   ⚠️ Database created but connection check failed")
    except Exception as e:
        print(f"   ❌ Database initialization failed: {e}")

    # 7. STARTUP COMPLETE
    port = int(os.getenv("PORT", "5000"))
    print("\n" + "=" * 60)
    print("✅ VIT Network: All Systems Operational")
    print(f"📍 API:     http://localhost:{port}/api")
    print(f"📊 Health:  http://localhost:{port}/health")
    print(f"📈 Status:  http://localhost:{port}/system/status")
    print("=" * 60 + "\n")

    yield

    # SHUTDOWN
    print("\n🛑 VIT Network Shutting Down...")
    if telegram_alerts and telegram_alerts.enabled:
        try:
            await telegram_alerts.send_shutdown_message()
        except Exception as e:
            print(f"⚠️ Shutdown message failed: {e}")

    if data_loader:
        try:
            await data_loader.close()
        except Exception as e:
            print(f"⚠️ Data loader cleanup failed: {e}")

    print("✅ Cleanup Complete\n")


# --- FASTAPI APP ---
app = FastAPI(
    title="VIT Sports Intelligence Network",
    description="12-Model Ensemble for Football Prediction with CLV Tracking",
    version="2.0",
    lifespan=lifespan
)

# MIDDLEWARE (order matters!)
app.add_middleware(LoggingMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ROUTERS
app.include_router(predict.router)
app.include_router(result.router)
app.include_router(history.router)


# --- HEALTH CHECK (with real DB verification) ---
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint - verifies all systems"""
    db_status = await _check_db_connection()

    return HealthResponse(
        status="ok",
        models_loaded=orchestrator.num_models_ready() if orchestrator else 0,
        db_connected=db_status,
        clv_tracking_enabled=True
    )


# --- DATA FETCH ENDPOINTS ---
@app.get("/fetch")
async def fetch_fixtures(
    competition: str = "premier_league",
    days_ahead: int = 7,
    background_tasks: BackgroundTasks = None
):
    """
    Fetch fixtures and optionally run predictions in background.

    Args:
        competition: League name (premier_league, la_liga, bundesliga, serie_a, ligue_1)
        days_ahead: Number of days to look ahead
        background_tasks: If True, runs predictions async
    """
    if not data_loader:
        return {"error": "Data loader not initialized", "status": "unavailable"}

    if background_tasks:
        background_tasks.add_task(fetch_and_predict, competition, days_ahead)
        return {
            "message": f"Background task started for {competition}",
            "status": "processing",
            "competition": competition,
            "days_ahead": days_ahead
        }

    # Synchronous fetch for testing
    try:
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
    except Exception as e:
        return {"error": str(e), "status": "failed", "type": type(e).__name__}


@app.get("/fetch/historical")
async def fetch_historical(
    competition: str = "premier_league",
    days_back: int = 90,
    limit: int = 100
):
    """
    Fetch historical matches for training/backtesting.

    Args:
        competition: League name
        days_back: How many days back to fetch
        limit: Maximum matches to return
    """
    if not data_loader:
        return {"error": "Data loader not initialized", "status": "unavailable"}

    try:
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
    except Exception as e:
        return {"error": str(e), "status": "failed", "type": type(e).__name__}


# --- ODDS ENDPOINTS ---
@app.get("/odds")
async def get_odds(
    competition: str = "premier_league",
    days_ahead: int = 3
):
    """
    Get current odds from all bookmakers.

    Args:
        competition: League name
        days_ahead: Look-ahead window
    """
    if not data_loader:
        return {"error": "Data loader not initialized", "status": "unavailable"}

    if not data_loader.odds_client:
        return {"error": "Odds client not initialized. Check THE_ODDS_API_KEY in .env", "status": "unavailable"}

    try:
        odds = await data_loader.fetch_odds_only(
            competition=competition,
            days_ahead=days_ahead
        )

        return {
            "competition": competition,
            "matches_count": len(odds),
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
    except Exception as e:
        return {"error": str(e), "status": "failed", "type": type(e).__name__}


@app.get("/odds/sharp")
async def get_sharp_odds(competition: str = "premier_league"):
    """
    Get odds from sharp books only (Pinnacle).
    Useful for edge calculation and line movement tracking.

    Args:
        competition: League name
    """
    if not data_loader:
        return {"error": "Data loader not initialized", "status": "unavailable"}

    if not data_loader.odds_client:
        return {"error": "Odds client not initialized", "status": "unavailable"}

    try:
        odds = await data_loader.fetch_sharp_odds_only(competition)

        return {
            "competition": competition,
            "sharp_count": len(odds),
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
    except Exception as e:
        return {"error": str(e), "status": "failed", "type": type(e).__name__}


# --- ALERTS ENDPOINTS ---
@app.get("/alerts/test")
async def test_alerts():
    """Test Telegram alert system"""
    if not telegram_alerts or not telegram_alerts.enabled:
        return {"error": "Telegram alerts not enabled", "status": "unavailable"}

    try:
        await telegram_alerts.send_message("✅ Test message from VIT Network", AlertPriority.SUCCESS)
        return {"message": "Test alert sent successfully", "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}


@app.get("/alerts/status")
async def alerts_status():
    """Get alerts system status"""
    return {
        "telegram_enabled": telegram_alerts.enabled if telegram_alerts else False,
        "bot_token_configured": bool(os.getenv("TELEGRAM_BOT_TOKEN", "")),
        "chat_id_configured": bool(os.getenv("TELEGRAM_CHAT_ID", "")),
        "status": "ready" if (telegram_alerts and telegram_alerts.enabled) else "disabled"
    }


# --- SYSTEM DIAGNOSTICS ---
@app.get("/system/status")
async def system_status():
    """Get complete system status across all components"""
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


# --- TESTING ENDPOINT ---
@app.post("/test-predict")
async def test_predict(match: dict):
    """
    Test prediction without database persistence.
    Useful for debugging model output.

    Example:
        {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "league": "premier_league"
        }
    """
    if orchestrator is None:
        return {"error": "Orchestrator not initialized", "status": "unavailable"}

    features = {
        "home_team": match.get("home_team"),
        "away_team": match.get("away_team"),
        "league": match.get("league", "premier_league")
    }

    try:
        result = await orchestrator.predict(features, "test")
        return {
            "status": "success",
            "raw_result": result,
            "predictions": result.get("predictions", {})
        }
    except Exception as e:
        return {
            "error": str(e),
            "type": type(e).__name__,
            "status": "failed"
        }


# --- API INFO ---
@app.get("/api")
async def root():
    """API endpoint discovery and documentation"""
    return {
        "name": "VIT Sports Intelligence Network",
        "version": "2.0",
        "status": "operational",
        "components": {
            "orchestrator": f"{'✅ Ready' if orchestrator else '❌ Not initialized'} ({orchestrator.num_models_ready() if orchestrator else 0} models)",
            "data_loader": f"{'✅ Ready' if data_loader else '❌ Not initialized'}",
            "alerts": f"{'✅ Ready' if (telegram_alerts and telegram_alerts.enabled) else '⚠️ Disabled'}"
        },
        "endpoints": {
            "core": {
                "POST /api/predict": "Generate prediction for a match",
                "POST /api/result/{match_id}": "Update match result",
                "GET /api/history": "View prediction history"
            },
            "data": {
                "GET /fetch": "Fetch upcoming fixtures",
                "GET /fetch/historical": "Fetch historical matches",
                "GET /odds": "Get current odds (all bookmakers)",
                "GET /odds/sharp": "Get sharp odds (Pinnacle)"
            },
            "testing": {
                "POST /test-predict": "Test prediction without DB",
                "GET /health": "Health check",
                "GET /system/status": "System diagnostics"
            },
            "alerts": {
                "GET /alerts/test": "Test Telegram alerts",
                "GET /alerts/status": "Alerts system status"
            }
        }
    }


# --- STATIC FRONTEND (mount last - catches unmatched routes) ---
if os.path.isdir("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")


# --- EXECUTION ---
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "5000"))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )