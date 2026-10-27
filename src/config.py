"""Shared settings for the pipeline (assumptions follow Table 3.2 of the report)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
OUTPUTS = ROOT / "outputs"
for _p in (DATA, FIGURES, OUTPUTS):
    _p.mkdir(exist_ok=True)

SEED = 42
START_DATE = "2024-01-01"
N_DAYS = 547                 # 1 Jan 2024 - 30 Jun 2025
HORIZON = 14                 # forecast / prediction horizon in days
TRAIN_END_DAY = 366          # 2024 -> training, Jan-Jun 2025 -> test
VALID_START_DAY = 280        # Oct-Dec 2024 -> alert-threshold tuning
SERVICE_Z = 2.33             # 99 % service level for the dynamic reorder point
