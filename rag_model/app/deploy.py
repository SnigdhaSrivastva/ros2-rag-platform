"""Compatibility wrapper for legacy imports.
Use main.py FastAPI service and /ask endpoint for production queries.
"""

from main import app  # noqa: F401
