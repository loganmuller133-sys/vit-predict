# tests/conftest.py
import pytest
import asyncio
import uuid
from typing import AsyncGenerator
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import app
from app.db.database import Base, get_db
from app.api.middleware.auth import AUTH_ENABLED

# CRITICAL FIX: Use unique test database to prevent production accidents
TEST_DB_ID = str(uuid.uuid4())[:8]
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", 
    f"postgresql+asyncpg://vit_user:vit_password@localhost:5432/vit_db_test_{TEST_DB_ID}"
)

# Disable auth for tests
os.environ["AUTH_ENABLED"] = "false"

# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    poolclass=NullPool,
)

TestSessionLocal = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    """Override database dependency for tests with transaction rollback"""
    async with TestSessionLocal() as session:
        # Start transaction
        async with session.begin():
            yield session
            # Rollback after each test
            await session.rollback()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True, scope="session")
async def setup_database():
    """Create schema once before all tests"""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    # Drop after all tests
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    # Clean up test database
    await test_engine.dispose()


@pytest.fixture
async def client() -> AsyncGenerator:
    """HTTP client for testing"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def db_session() -> AsyncGenerator:
    """Database session for testing with rollback isolation"""
    async with TestSessionLocal() as session:
        async with session.begin():
            yield session
            await session.rollback()


@pytest.fixture
def sample_match_request():
    """Sample match request data"""
    return {
        "home_team": "Manchester City",
        "away_team": "Liverpool",
        "league": "Premier League",
        "kickoff_time": "2024-12-15T15:00:00",
        "market_odds": {
            "home": 1.85,
            "draw": 3.60,
            "away": 4.20
        }
    }


@pytest.fixture
def sample_result_update():
    """Sample result update data"""
    return {
        "home_goals": 2,
        "away_goals": 1,
        "closing_odds_home": 1.80,
        "closing_odds_draw": 3.70,
        "closing_odds_away": 4.50
    }