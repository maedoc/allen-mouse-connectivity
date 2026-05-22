"""
Allen Mouse Connectivity CLI — build a mouse structural connectivity matrix
from Allen Institute tracer experiments without requiring The Virtual Brain.

Usage::

    python -m allen_mouse_connectivity_cli --output-dir ./results
"""

from .connectivity import build_connectivity
from .cli import main

__all__ = ["build_connectivity", "main"]
