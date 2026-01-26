from __future__ import annotations

from pathlib import Path
import pandas as pd


def _latest_asof() -> str:
    paths = sorted(Path("data/curated").glob("ecl_output_asof_*.parquet"))
    assert paths, "No ecl_output_asof_*.parquet found. Run the pipeline first."
    return paths[-1].stem.replace("ecl_output_asof_", "")


def test_asof_not_nat_and_single_value() -> None:
    asof = _latest_asof()
    ecl = pd.read_parquet(f"data/curated/ecl_output_asof_{asof}.parquet")
    assert "asof_date" in ecl.columns
    assert ecl["asof_date"].isna().mean() == 0.0
    assert ecl["asof_date"].nunique() == 1


def test_overlay_asof_matches_ecl() -> None:
    asof = _latest_asof()
    ecl = pd.read_parquet(f"data/curated/ecl_output_asof_{asof}.parquet")
    ov = pd.read_parquet("data/curated/ecl_with_overlays.parquet")

    # both should be the same date
    e_asof = str(pd.to_datetime(ecl["asof_date"].iloc[0]).date())
    o_asof = str(pd.to_datetime(ov["asof_date"].iloc[0]).date())
    assert e_asof == o_asof


def test_scenario_weight_identity_model() -> None:
    asof = _latest_asof()
    ecl = pd.read_parquet(f"data/curated/ecl_output_asof_{asof}.parquet")

    # If weights live in config, you can load them here instead.
    wB, wU, wD = 0.6, 0.2, 0.2

    calc = (
        wB * ecl["ecl_selected_base"]
        + wU * ecl["ecl_selected_upside"]
        + wD * ecl["ecl_selected_downside"]
    )
    diff = (calc - ecl["ecl_selected"]).abs()
    assert float(diff.max()) == 0.0


def test_dcf_stage3_equals_workout() -> None:
    asof = _latest_asof()
    dcf_path = Path(f"data/curated/ecl_dcf_asof_{asof}.parquet")
    assert dcf_path.exists(), f"{dcf_path} missing. Run: python -m ecl_engine.dcf_ecl --asof {asof}"

    d = pd.read_parquet(dcf_path)
    s3 = d[d["stage"] == 3].copy()
    assert len(s3) > 0, "No Stage 3 accounts found; cannot validate Stage 3 logic."

    # After your fix, these must match exactly (you already observed this in console)
    diff = (s3["dcf_ecl_selected"] - s3["ecl_stage3_workout"]).abs()
    assert float(diff.max()) == 0.0
