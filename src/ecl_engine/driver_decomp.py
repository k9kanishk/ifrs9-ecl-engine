from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from ecl_engine.ecl import compute_ecl_asof, load_yml
from ecl_engine.overlay import apply_overlays

ECL_WITH_OV = Path("data/curated/ecl_with_overlays.parquet")
ECL_ASOF = Path("data/curated/ecl_output_asof_2024-12-31.parquet")  # fallback only


def load_ecl_for_decomp() -> tuple[pd.Timestamp, pd.DataFrame]:
    if ECL_WITH_OV.exists():
        df = pd.read_parquet(ECL_WITH_OV)
    else:
        df = pd.read_parquet(ECL_ASOF)

    df["asof_date"] = pd.to_datetime(df["asof_date"])
    asof = df["asof_date"].max()
    df = df[df["asof_date"] == asof].copy()
    return asof, df


def scenario_contribution(ecl_raw: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Uses scenario ECLs already in ecl_output:
      ecl_selected is stage-selected weighted;
    We'll compute weighted contributions from each scenario for the selected horizon rule.
    """
    w = weights["weights"]
    out = ecl_raw.copy()

    # Stage-selected per scenario
    out["sel_base"] = out["ecl12_base"].where(out["stage"] == 1, out["ecllt_base"])
    out["sel_upside"] = out["ecl12_upside"].where(
        out["stage"] == 1, out["ecllt_upside"]
    )
    out["sel_downside"] = out["ecl12_downside"].where(
        out["stage"] == 1, out["ecllt_downside"]
    )

    # Note: Stage 3 proxy was applied to ecl_selected only, so we keep scenario view pre-proxy for narrative.
    contrib = pd.DataFrame(
        {
            "scenario": ["Base", "Upside", "Downside"],
            "weight": [float(w["Base"]), float(w["Upside"]), float(w["Downside"])],
            "unweighted_sum": [
                out["sel_base"].sum(),
                out["sel_upside"].sum(),
                out["sel_downside"].sum(),
            ],
        }
    )
    contrib["weighted_contribution"] = (
        contrib["weight"] * contrib["unweighted_sum"]
    )
    contrib["weighted_share"] = (
        contrib["weighted_contribution"] / contrib["weighted_contribution"].sum()
    )
    return contrib


def run_decomposition(
    asof: str | None = None,
    staging_path: str = "data/curated/staging_output.parquet",
    accounts_path: str = "data/curated/accounts.parquet",
    macro_path: str = "data/curated/macro_scenarios_monthly.parquet",
    policy_path: str = "configs/policy.yml",
    weights_path: str = "configs/scenario_weights.yml",
    params_path: str = "configs/portfolio_params.yml",
    overlays_path: str = "data/curated/overlays.csv",
    outdir: str = "data/curated",
) -> None:
    asof_loaded, df = load_ecl_for_decomp()
    if asof is None:
        asof = asof_loaded

    staged = pd.read_parquet(staging_path)
    accounts = pd.read_parquet(accounts_path)
    macro = pd.read_parquet(macro_path)

    policy = load_yml(policy_path)
    weights = load_yml(weights_path)
    params = load_yml(params_path)

    staged["snapshot_date"] = pd.to_datetime(staged["snapshot_date"])
    asof_dt = pd.to_datetime(asof) if asof else staged["snapshot_date"].max()

    # Base run (with your current params)
    base = compute_ecl_asof(staged, accounts, macro, policy, weights, params, asof_dt)
    base_ov = apply_overlays(base, overlays_path)

    # Scenario contribution (pre-overlay, because overlay is not scenario-specific)
    scen_tbl = scenario_contribution(base, weights)

    # Driver toggles:
    # 1) PD effect: force all scenarios to use Base macro (i.e., remove scenario variation)
    # Implemented by setting weights 100% Base (keeps PIT level but kills scenario weighting).
    w_pd = {"weights": {"Base": 1.0, "Upside": 0.0, "Downside": 0.0}}
    pd_only = compute_ecl_asof(staged, accounts, macro, policy, w_pd, params, asof_dt)
    pd_only_ov = apply_overlays(pd_only, overlays_path)

    # 2) LGD effect: set scenario multipliers to 1 across scenarios
    params_lgd_flat = yaml.safe_load(yaml.safe_dump(params))
    params_lgd_flat["lgd_scenario_multiplier"] = {
        "Base": 1.0,
        "Upside": 1.0,
        "Downside": 1.0,
    }
    lgd_flat = compute_ecl_asof(
        staged, accounts, macro, policy, weights, params_lgd_flat, asof_dt
    )
    lgd_flat_ov = apply_overlays(lgd_flat, overlays_path)

    # 3) EAD effect (revolving): set CCF to 0 (so EAD = balance for revolving)
    params_ccf0 = yaml.safe_load(yaml.safe_dump(params))
    params_ccf0["ccf_base"] = {"Revolving": 0.0}
    ccf0 = compute_ecl_asof(
        staged, accounts, macro, policy, weights, params_ccf0, asof_dt
    )
    ccf0_ov = apply_overlays(ccf0, overlays_path)

    # Summaries (post-overlay, because business cares about reported number)
    def seg_sum(df: pd.DataFrame) -> pd.Series:
        return df.groupby("segment")["ecl_post_overlay"].sum()

    base_seg = seg_sum(base_ov)
    pd_seg = seg_sum(pd_only_ov)
    lgd_seg = seg_sum(lgd_flat_ov)
    ead_seg = seg_sum(ccf0_ov)

    decomp = pd.DataFrame(
        {
            "ecl_reported": base_seg,
            "pd_scenario_component": base_seg - pd_seg,
            "lgd_downturn_component": base_seg - lgd_seg,
            "ead_ccf_component": base_seg - ead_seg,
        }
    ).fillna(0.0)

    decomp = decomp.sort_values("ecl_reported", ascending=False)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scen_path = outdir / f"scenario_contribution_{asof_dt.date().isoformat()}.csv"
    decomp_path = outdir / f"driver_decomposition_{asof_dt.date().isoformat()}.csv"

    scen_tbl.to_csv(scen_path, index=False)
    decomp.to_csv(decomp_path)

    print(f"Wrote: {scen_path}")
    print(f"Wrote: {decomp_path}")

    print("\nScenario contribution:")
    print(scen_tbl.round(4))

    print("\nDriver decomposition (top 10 segments):")
    reported_col = (
        "ecl_post_overlay" if "ecl_post_overlay" in df.columns else "ecl_selected"
    )
    seg_tot = (
        df.groupby("segment")[reported_col].sum().sort_values(ascending=False).head(10)
    )
    print(seg_tot)


if __name__ == "__main__":
    run_decomposition()
