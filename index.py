"""Vercel serverless entrypoint.

Vercel's Python runtime looks for a top-level FastAPI ``app`` symbol in ``index.py``,
``app.py``, ``server.py``, ``app/server.py``, etc. This module re-exports the app defined
in ``app/main.py``.
"""

from app.main import app

__all__ = ["app"]
