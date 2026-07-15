"""Ensures the project root is on sys.path so `from src.xxx import yyy` works
regardless of the pytest invocation directory or import-mode defaults."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
