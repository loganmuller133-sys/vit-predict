# app/__init__.py
"""VIT Sports Intelligence Network - Main Application Package"""

__version__ = "2.0.0"
__author__ = "VIT Sports Intelligence"
__description__ = "12-Model Ensemble for Football Prediction with CLV Tracking"

from app.db.database import engine, AsyncSessionLocal, get_db
from app.db.models import Base

__all__ = [
    "engine",
    "AsyncSessionLocal", 
    "get_db",
    "Base"
]