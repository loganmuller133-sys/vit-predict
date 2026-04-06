# app/api/routes/__init__.py
"""API Routes Package"""

from app.api.routes import predict
from app.api.routes import result
from app.api.routes import history

__all__ = [
    "predict",
    "result",
    "history"
]