"""Run the same tests from pytest or the public check script."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'orda_sim'))
