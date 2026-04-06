# app/db/database.py
import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from dotenv import load_dotenv

load_dotenv()

_raw_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/vit_db")

# Guarantee asyncpg driver — replace any sync postgres scheme
def _make_async_url(url: str) -> str:
    for sync_prefix in ("postgresql://", "postgres://", "postgresql+psycopg2://"):
        if url.startswith(sync_prefix):
            return url.replace(sync_prefix, "postgresql+asyncpg://", 1)
    return url

DATABASE_URL = _make_async_url(_raw_url)

# Async Engine with connection pooling
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


# Dependency for FastAPI
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
