"""Test core ECL engine invariants."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_asof_date_consistency(latest_asof):
    """Test 1: All output files have matching ASOF dates."""
    asof_dt = pd.Timestamp(latest_asof).normalize()

    files = [
        f"data/curated/ecl_output_asof_{latest_asof}.parquet",
        "data/curated/ecl_with_overlays.parquet",
    ]

    for fpath in files:
        if not Path(fpath).exists():
            continue
        df = pd.read_parquet(fpath)
        assert "asof_date" in df.columns, f"{fpath} missing asof_date column"
        file_asofs = pd.to_datetime(df["asof_date"]).dt.normalize().unique()
        assert len(file_asofs) == 1, f"{fpath} has multiple ASOF dates"
        assert file_asofs[0] == asof_dt, f"{fpath} ASOF mismatch"


def test_scenario_weights(latest_asof):
    """Test 2: Weighted ECL = 0.6*Base + 0.2*Upside + 0.2*Downside."""
    ecl = pd.read_parquet(f"data/curated/ecl_output_asof_{latest_asof}.parquet")

    # Scenario weights from config
    wB, wU, wD = 0.6, 0.2, 0.2

    # Verify selected ECL matches weighted formula
    calc = (
        wB * ecl["ecl_selected_base"]
        + wU * ecl["ecl_selected_upside"]
        + wD * ecl["ecl_selected_downside"]
    )

    diff = (calc - ecl["ecl_selected"]).abs()
    assert diff.max() < 1e-6, f"Scenario weighting incorrect, max diff={diff.max()}"


def test_stage_horizon_rule(latest_asof):
    """Test 3: Stage 1 uses 12m ECL, Stage 2/3 use lifetime ECL."""
    ecl = pd.read_parquet(f"data/curated/ecl_output_asof_{latest_asof}.parquet")

    # Stage 1 should select 12m
    s1 = ecl[ecl["stage"] == 1].copy()
    if len(s1) > 0:
        diff_12m = (s1["ecl_selected_base"] - s1["ecl12_base"]).abs()
        assert diff_12m.max() < 1e-6, "Stage 1 not using 12m ECL"

    # Stage 2/3 should select lifetime
    s23 = ecl[ecl["stage"].isin([2, 3])].copy()
    if len(s23) > 0:
        # Stage 3 gets overridden by workout, so only check Stage 2
        s2 = s23[s23["stage"] == 2]
        if len(s2) > 0:
            diff_lt_s2 = (s2["ecl_selected_base"] - s2["ecllt_base"]).abs()
            assert diff_lt_s2.max() < 1e-6, "Stage 2 not using lifetime ECL"


def test_stage3_workout_override(latest_asof):
    """Test 4: Stage 3 ECL equals workout ECL (scenario-specific)."""
    ecl = pd.read_parquet(f"data/curated/ecl_output_asof_{latest_asof}.parquet")

    s3 = ecl[ecl["stage"] == 3].copy()
    if len(s3) == 0:
        pytest.skip("No Stage 3 accounts in this ASOF")

    # Check that Stage 3 selected ECL matches workout ECL for each scenario
    for scen in ["base", "upside", "downside"]:
        diff = (s3[f"ecl_selected_{scen}"] - s3[f"ecl_stage3_{scen}"]).abs()
        assert diff.max() < 1e-6, f"Stage 3 {scen} ECL doesn't match workout"


def test_overlay_additivity(latest_asof):
    """Test 5: ecl_post_overlay = ecl_pre_overlay + overlay_amount."""
    ov = pd.read_parquet("data/curated/ecl_with_overlays.parquet")

    calc = ov["ecl_pre_overlay"] + ov["overlay_amount"]
    diff = (calc - ov["ecl_post_overlay"]).abs()

    assert diff.max() < 1e-6, f"Overlay additivity violated, max diff={diff.max()}"

    # Also check segment-level
    seg_pre = ov.groupby("segment")["ecl_pre_overlay"].sum()
    seg_overlay = ov.groupby("segment")["overlay_amount"].sum()
    seg_post = ov.groupby("segment")["ecl_post_overlay"].sum()

    seg_calc = seg_pre + seg_overlay
    seg_diff = (seg_calc - seg_post).abs()
    assert seg_diff.max() < 1e-6, "Segment-level overlay additivity violated"


def test_macro_extraction_no_nans(sample_macro):
    """Test 6: Macro extraction handles missing dates without NaNs."""
    from ecl_engine.ecl import _macro_matrix
    from ecl_engine.utils.macro import prepare_macro_for_ecl

    mz = prepare_macro_for_ecl(sample_macro)

    # Request dates beyond available range
    future_dates = pd.date_range("2025-01-31", periods=6, freq="ME")
    cols = ["unemployment_z", "gdp_yoy_z", "policy_rate_z"]

    result = _macro_matrix(mz, "Base", future_dates, cols)

    assert not np.isnan(result).any(), "Macro extraction produced NaNs"
    assert result.shape == (6, 3), f"Wrong shape: {result.shape}"


def test_stage_rules_basic(latest_asof):
    """Test 7: Basic staging rules (DPD thresholds)."""
    staging = pd.read_parquet("data/curated/staging_output.parquet")

    # Filter to latest snapshot
    asof_dt = pd.Timestamp(latest_asof)
    staging["snapshot_date"] = pd.to_datetime(staging["snapshot_date"])
    latest = staging[staging["snapshot_date"] == asof_dt].copy()

    if len(latest) == 0:
        pytest.skip("No staging data for this ASOF")

    # Stage 3: DPD >= 90 or default_flag = 1
    if "default_flag" in latest.columns:
        s3_actual = latest["stage"] == 3
        s3_accounts = latest[s3_actual]
        if len(s3_accounts) > 0:
            # At least check that most Stage 3 have high DPD
            high_dpd_pct = (s3_accounts["dpd"] >= 90).mean()
            assert high_dpd_pct > 0.5, "Most Stage 3 accounts should have DPD >= 90"


def test_dcf_stage3_equals_workout(latest_asof):
    """Test 8: DCF Stage 3 ECL matches workout ECL exactly."""
    dcf_path = Path(f"data/curated/ecl_dcf_asof_{latest_asof}.parquet")

    if not dcf_path.exists():
        pytest.skip("DCF output not found. Run: python -m ecl_engine.dcf_ecl")

    dcf = pd.read_parquet(dcf_path)
    s3 = dcf[dcf["stage"] == 3].copy()

    if len(s3) == 0:
        pytest.skip("No Stage 3 accounts in DCF output")

    diff = (s3["dcf_ecl_selected"] - s3["ecl_stage3_workout"]).abs()
    assert diff.max() < 1e-6, f"DCF Stage 3 doesn't match workout, max diff={diff.max()}"
