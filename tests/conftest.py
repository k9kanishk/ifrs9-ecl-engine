"""Pytest configuration and shared fixtures."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_accounts():
    """Minimal account dataset for testing."""
    return pd.DataFrame(
        {
            "account_id": ["A0001", "A0002", "A0003"],
            "segment": ["Retail_Mortgage", "Corp_TermLoan", "Revolving"],
            "ttc_pd_annual": [0.01, 0.02, 0.05],
            "origination_date": pd.to_datetime([
                "2020-01-31",
                "2019-06-30",
                "2021-03-31",
            ]),
            "maturity_date": pd.to_datetime([
                "2040-01-31",
                "2024-06-30",
                "2023-03-31",
            ]),
            "eir": [0.03, 0.05, 0.15],
            "limit_amount": [0, 0, 10000],
        }
    )


@pytest.fixture
def sample_macro():
    """Minimal macro dataset."""
    dates = pd.date_range("2024-01-31", periods=12, freq="ME")
    return pd.DataFrame(
        {
            "date": list(dates) * 3,
            "scenario": ["Base"] * 12 + ["Upside"] * 12 + ["Downside"] * 12,
            "unemployment": [5.5] * 12 + [4.8] * 12 + [6.7] * 12,
            "gdp_yoy": [2.0] * 12 + [2.8] * 12 + [0.5] * 12,
            "policy_rate": [3.0] * 12 + [2.7] * 12 + [3.6] * 12,
        }
    )


@pytest.fixture
def latest_asof():
    """Get latest ASOF from output files."""
    paths = sorted(Path("data/curated").glob("ecl_output_asof_*.parquet"))
    if not paths:
        pytest.skip("No ECL output files found. Run pipeline first.")
    return paths[-1].stem.replace("ecl_output_asof_", "")
