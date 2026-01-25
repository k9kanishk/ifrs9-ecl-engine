from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ecl_engine.utils.io import load_yml


def as_month_end(x) -> pd.Timestamp:
    return (pd.to_datetime(x) + pd.offsets.MonthEnd(0)).normalize()


def month_ends_forward(asof_dt: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    start_me = as_month_end(asof_dt)
    return pd.date_range(start=start_me + pd.offsets.MonthEnd(1), periods=periods, freq="ME")


def _to_float(a: pd.Series | np.ndarray | float, default: float = 0.0) -> np.ndarray:
    if isinstance(a, (float, int, np.floating, np.integer)):
        return np.array([float(a)], dtype=np.float64)
    if isinstance(a, np.ndarray):
        return a.astype(np.float64)
    s = pd.to_numeric(a, errors="coerce").fillna(default)
    return s.to_numpy(dtype=np.float64)


def _safe_div(a: np.ndarray, b: np.ndarray, default: float = 0.0) -> np.ndarray:
    out = np.full_like(a, default, dtype=np.float64)
    mask = b != 0
    out[mask] = a[mask] / b[mask]
    return out


@dataclass
class DCFParams:
    horizon_months: int
    eir_fallback_annual: float
    # Stage selection rule: stage1->12m; stage2/3->lt
    stage1_horizon_months: int
    # optional EAD curve controls
    ead_decay_half_life_m: float
    # discounting controls
    min_eir_annual: float
    max_eir_annual: float


def load_dcf_params(path: str | Path = "configs/dcf_params.yml") -> DCFParams:
    cfg = load_yml(path)
    d = cfg.get("dcf", cfg)
    return DCFParams(
        horizon_months=int(d.get("horizon_months", 360)),
        eir_fallback_annual=float(d.get("eir_fallback_annual", 0.05)),
        stage1_horizon_months=int(d.get("stage1_horizon_months", 12)),
        ead_decay_half_life_m=float(d.get("ead_decay_half_life_m", 0.0)),
        min_eir_annual=float(d.get("min_eir_annual", -0.5)),
        max_eir_annual=float(d.get("max_eir_annual", 2.0)),
    )


def _discount_factors(eir_annual: np.ndarray, horizon: int, p: DCFParams) -> np.ndarray:
    e = np.clip(eir_annual.astype(np.float64), p.min_eir_annual, p.max_eir_annual)
    r_m = e / 12.0
    t = np.arange(1, horizon + 1, dtype=np.float64)[None, :]
    return 1.0 / np.power(1.0 + r_m[:, None], t)


def _ead_curve(
    ead0: np.ndarray,
    limit_amount: np.ndarray,
    segment: np.ndarray,
    horizon: int,
    p: DCFParams,
) -> np.ndarray:
    """
    Build a simple EAD curve (month 1..H) per account.

    - For non-revolving: flat EAD = ead0
    - For revolving: ead0 already includes CCF uplift in your core model; keep it flat by default.
    - Optional decay via half-life (ead_decay_half_life_m > 0): exponential decay to 0 over time.
      (This gives you a DCF-style cashflow projection even without contractual amortization schedules.)
    """
    n = len(ead0)
    E = np.tile(ead0[:, None], (1, horizon)).astype(np.float64)

    hl = float(p.ead_decay_half_life_m)
    if hl and hl > 0:
        t = np.arange(horizon, dtype=np.float64)[None, :]
        w = np.exp(-np.log(2.0) * t / hl)
        E = E * w

    return np.maximum(E, 0.0)


def _pd_hazard_from_cum(pd_cum_12m: np.ndarray) -> np.ndarray:
    """
    Convert 12m cumulative PD to constant monthly hazard.
    """
    pd_cum_12m = np.clip(pd_cum_12m.astype(np.float64), 1e-8, 0.95)
    return 1.0 - np.power(1.0 - pd_cum_12m, 1.0 / 12.0)


def _cum_from_hazard(h: np.ndarray, months: np.ndarray) -> np.ndarray:
    return 1.0 - np.power(1.0 - h, months.astype(np.float64))


def compute_dcf_ecl_for_accounts(
    out: pd.DataFrame,
    dcf_params: DCFParams,
    scenario_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Add a DCF-style ECL projection layer on top of the existing account-level ECL output `out`.

    Expects `out` to already include:
      - stage, segment, balance, limit_amount (optional), eir (optional)
      - pd_12m_base / pd_12m_upside / pd_12m_downside  (cumulative 12m)
      - lgd_base / lgd_upside / lgd_downside
      - months_to_maturity (optional; if absent will approximate using config defaults in your core model)

    Produces:
      - dcf_ecl12_{scenario}, dcf_ecllt_{scenario}
      - dcf_ecl_selected_{scenario}, dcf_ecl_selected_weighted, dcf_ecl_selected
      - (optional) debug columns: dcf_hazard_{scenario}
    """
    df = out.copy()

    scen_list = ["Base", "Upside", "Downside"]
    if scenario_weights is None:
        scenario_weights = {"Base": 0.6, "Upside": 0.2, "Downside": 0.2}
    w_base = float(scenario_weights.get("Base", 0.6))
    w_up = float(scenario_weights.get("Upside", 0.2))
    w_dn = float(scenario_weights.get("Downside", 0.2))
    w_sum = w_base + w_up + w_dn
    if w_sum <= 0:
        w_base, w_up, w_dn = 0.6, 0.2, 0.2
        w_sum = 1.0
    w_base, w_up, w_dn = w_base / w_sum, w_up / w_sum, w_dn / w_sum

    stage = df["stage"].astype(int).to_numpy()
    seg = df["segment"].astype(str).to_numpy()

    # EAD0 selection:
    # Prefer an explicit 'ead' if present; else compute from balance + limit_amount if present; else use balance.
    if "ead" in df.columns:
        ead0 = _to_float(df["ead"], default=0.0)

        # Stage 3 is defaulted exposure: prefer ead_default if present
        if "ead_default" in df.columns:
            ead_def = _to_float(df["ead_default"], default=0.0)
            ead0 = np.where(stage == 3, ead_def, ead0)
    else:
        bal = _to_float(df.get("balance", 0.0), default=0.0)
        lim = df.get("limit_amount", 0.0)
        if isinstance(lim, (float, int, np.floating, np.integer)):
            lim_arr = np.full_like(bal, float(lim), dtype=np.float64)
        else:
            lim_arr = _to_float(lim, default=0.0)
        # If it's revolving, use same rule as core model (75% CCF by default if present already? keep simple)
        is_rev = seg == "Revolving"
        undrawn = np.maximum(lim_arr - bal, 0.0)
        ead0 = bal.copy()
        ead0[is_rev] = bal[is_rev] + 0.75 * undrawn[is_rev]

    lim2 = df.get("limit_amount", 0.0)
    if isinstance(lim2, (float, int, np.floating, np.integer)):
        limit_amount = np.full_like(ead0, float(lim2), dtype=np.float64)
    else:
        limit_amount = _to_float(lim2, default=0.0)

    # Maturity months
    if "months_to_maturity" in df.columns:
        mat_m = pd.to_numeric(df["months_to_maturity"], errors="coerce").fillna(36).to_numpy(dtype=np.float64)
    else:
        mat_m = np.full(len(df), 36.0, dtype=np.float64)
    mat_m = np.clip(mat_m, 1.0, float(dcf_params.horizon_months))

    # EIR discount rate
    eir = df.get("eir", np.nan)
    if isinstance(eir, (float, int, np.floating, np.integer)):
        eir_annual = np.full(len(df), float(eir), dtype=np.float64)
    else:
        eir_annual = pd.to_numeric(eir, errors="coerce").fillna(dcf_params.eir_fallback_annual).to_numpy(dtype=np.float64)

    H = int(dcf_params.horizon_months)
    DF = _discount_factors(eir_annual, H, dcf_params)
    E = _ead_curve(ead0, limit_amount, seg, H, dcf_params)

    # Month index 1..H
    months = np.arange(1, H + 1, dtype=np.float64)[None, :]

    def scenario_cols(s: str) -> tuple[str, str]:
        return f"pd_12m_{s.lower()}", f"lgd_{s.lower()}"

    dcf_ecl12 = {}
    dcf_ecllt = {}
    hazards = {}

    for s in scen_list:
        pd_col, lgd_col = scenario_cols(s)
        if pd_col not in df.columns:
            # fallback: try pd_12m_hat or pd_12m_anchor (treated as base)
            base_pd = df.get("pd_12m_hat", df.get("pd_12m_anchor", 0.02))
            pd12 = _to_float(base_pd, default=0.02)
        else:
            pd12 = _to_float(df[pd_col], default=0.02)

        if lgd_col not in df.columns:
            lgd = np.full(len(df), 0.45, dtype=np.float64)
        else:
            lgd = _to_float(df[lgd_col], default=0.45)

        h = _pd_hazard_from_cum(pd12)
        hazards[s] = h

        # survival to start of each month (t-1), with hazard h
        t = months  # 1..H
        surv_prev = np.power(1.0 - h[:, None], (t - 1.0))
        p_default_t = h[:, None] * surv_prev

        # discounted loss in month t
        loss_t = (E * lgd[:, None] * p_default_t) * DF
        pv_loss = loss_t.cumsum(axis=1)

        # 12m ECL = PV loss through month 12
        ecl12_s = pv_loss[:, min(12, H) - 1]
        # LT ECL = PV loss through maturity (min(mat_m, H))
        idx = np.clip(np.round(mat_m).astype(int), 1, H) - 1
        ecllt_s = pv_loss[np.arange(len(df)), idx]

        dcf_ecl12[s] = ecl12_s
        dcf_ecllt[s] = ecllt_s

        df[f"dcf_hazard_{s.lower()}"] = h
        df[f"dcf_ecl12_{s.lower()}"] = ecl12_s
        df[f"dcf_ecllt_{s.lower()}"] = ecllt_s

    # ----------------------------
    # Stage 3 override (critical)
    # ----------------------------
    s3 = stage == 3
    have_stage3_scen = all(
        c in df.columns
        for c in ["ecl_stage3_base", "ecl_stage3_upside", "ecl_stage3_downside"]
    )

    if s3.any() and have_stage3_scen:
        # Force DCF scenario ECLs to match workout scenario ECLs for Stage 3
        df.loc[s3, "dcf_ecl12_base"] = df.loc[s3, "ecl_stage3_base"]
        df.loc[s3, "dcf_ecllt_base"] = df.loc[s3, "ecl_stage3_base"]

        df.loc[s3, "dcf_ecl12_upside"] = df.loc[s3, "ecl_stage3_upside"]
        df.loc[s3, "dcf_ecllt_upside"] = df.loc[s3, "ecl_stage3_upside"]

        df.loc[s3, "dcf_ecl12_downside"] = df.loc[s3, "ecl_stage3_downside"]
        df.loc[s3, "dcf_ecllt_downside"] = df.loc[s3, "ecl_stage3_downside"]

        # Optional: make hazards explicit (not used after override, but nice for debugging)
        df.loc[s3, "dcf_hazard_base"] = 1.0
        df.loc[s3, "dcf_hazard_upside"] = 1.0
        df.loc[s3, "dcf_hazard_downside"] = 1.0

    # selected horizon rule per stage
    df["dcf_ecl_selected_base"] = np.where(stage == 1, df["dcf_ecl12_base"], df["dcf_ecllt_base"])
    df["dcf_ecl_selected_upside"] = np.where(stage == 1, df["dcf_ecl12_upside"], df["dcf_ecllt_upside"])
    df["dcf_ecl_selected_downside"] = np.where(stage == 1, df["dcf_ecl12_downside"], df["dcf_ecllt_downside"])

    df["dcf_ecl_selected_weighted"] = (
        w_base * df["dcf_ecl_selected_base"]
        + w_up * df["dcf_ecl_selected_upside"]
        + w_dn * df["dcf_ecl_selected_downside"]
    )
    df["dcf_ecl_selected"] = df["dcf_ecl_selected_weighted"].astype("float64")

    # Convenience: delta vs reported (if present)
    if "ecl_selected" in df.columns:
        df["dcf_vs_model_delta"] = df["dcf_ecl_selected"] - df["ecl_selected"]

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="ASOF date (YYYY-MM-DD). If omitted, infer from latest ecl output.")
    ap.add_argument("--infile", default=None, help="Input parquet. Default: data/curated/ecl_output_asof_<asof>.parquet")
    ap.add_argument("--outfile", default=None, help="Output parquet. Default: data/curated/ecl_dcf_asof_<asof>.parquet")
    ap.add_argument("--params", default="configs/dcf_params.yml", help="DCF params yaml")
    args = ap.parse_args()

    dcfp = load_dcf_params(args.params)

    if args.infile:
        in_path = Path(args.infile)
        if args.asof is None:
            # attempt parse
            stem = in_path.stem
            if "asof_" in stem:
                args.asof = stem.split("asof_")[-1]
    else:
        if args.asof is None:
            # pick latest ecl_output
            paths = sorted(Path("data/curated").glob("ecl_output_asof_*.parquet"))
            if not paths:
                raise FileNotFoundError("No ecl_output_asof_*.parquet found. Run: python -m ecl_engine.ecl")
            in_path = paths[-1]
            args.asof = in_path.stem.replace("ecl_output_asof_", "")
        else:
            in_path = Path(f"data/curated/ecl_output_asof_{args.asof}.parquet")

    out_path = Path(args.outfile) if args.outfile else Path(f"data/curated/ecl_dcf_asof_{args.asof}.parquet")

    df = pd.read_parquet(in_path)
    df2 = compute_dcf_ecl_for_accounts(df, dcfp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df2.to_parquet(out_path, index=False)

    print(f"Read: {in_path}")
    print(f"Wrote: {out_path}")
    print("DCF selected total:", float(df2["dcf_ecl_selected"].sum()))


if __name__ == "__main__":
    main()
