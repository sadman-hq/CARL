"""Lets the scripts run from a plain clone (without `pip install -e .`)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
