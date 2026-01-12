from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def month_ends(start: str, months: int) -> pd.DatetimeIndex:
    # Month-end frequency; start should be a month-end date string like "2018-01-31"
    idx = pd.date_range(start=start, periods=months, freq="M")
    return idx


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# -----------------------------
# Synthetic macro scenarios
# -----------------------------
def generate_macro(start: str, months: int, seed: int = 7) -> pd.DataFrame:
    """
    Produces monthly macro paths for Base/Upside/Downside.
    Variables are stylized, not "forecasts" (that's fine for a portfolio project).
    """
    rng = np.random.default_rng(seed)
    dates = month_ends(start, months)

    # Base path (random walk-ish around reasonable levels)
    unemp = 5.5 + np.cumsum(rng.normal(0, 0.03, size=months))          # %
    gdp_yoy = 1.5 + np.cumsum(rng.normal(0, 0.05, size=months))        # %
    rates = 2.0 + np.cumsum(rng.normal(0, 0.02, size=months))          # %

    base = pd.DataFrame(
        {"date": dates, "unemployment": unemp, "gdp_yoy": gdp_yoy, "policy_rate": rates}
    )

    # Scenario shocks (simple and transparent)
    # Upside: lower unemployment, higher GDP, slightly lower rates
    # Downside: higher unemployment, lower GDP, higher rates
    def scenario_df(name: str, du: float, dg: float, dr: float) -> pd.DataFrame:
        df = base.copy()
        df["scenario"] = name
        df["unemployment"] = df["unemployment"] + du
        df["gdp_yoy"] = df["gdp_yoy"] + dg
        df["policy_rate"] = df["policy_rate"] + dr
        return df

    out = pd.concat(
        [
            scenario_df("Base", 0.0, 0.0, 0.0),
            scenario_df("Upside", -0.7, +0.8, -0.3),
            scenario_df("Downside", +1.2, -1.5, +0.6),
        ],
        ignore_index=True,
    )

    return out


# -----------------------------
# Synthetic accounts
# -----------------------------
def generate_accounts(n: int, asof_start: str, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    segments = np.array(["Retail_Mortgage", "Retail_PL", "SME_TermLoan", "Corp_TermLoan", "Revolving"])
    weights = np.array([0.35, 0.20, 0.20, 0.15, 0.10])
    seg = rng.choice(segments, size=n, p=weights)

    # Origination/maturity
    asof = pd.Timestamp(asof_start)
    orig_months_ago = rng.integers(1, 84, size=n)  # up to 7y seasoned
    origination = (asof - pd.to_timedelta(orig_months_ago * 30, unit="D")).normalize()

    # Term by segment
    term_months = np.where(
        np.isin(seg, ["Retail_Mortgage"]),
        rng.integers(120, 360, size=n),
        np.where(np.isin(seg, ["Retail_PL"]), rng.integers(24, 84, size=n),
                 np.where(np.isin(seg, ["SME_TermLoan", "Corp_TermLoan"]), rng.integers(36, 120, size=n),
                          rng.integers(12, 60, size=n)))
    )
    maturity = origination + pd.to_timedelta(term_months * 30, unit="D")

    # EIR proxy
    eir = np.where(seg == "Retail_Mortgage", rng.normal(0.028, 0.006, size=n),
          np.where(seg == "Retail_PL", rng.normal(0.085, 0.020, size=n),
          np.where(seg == "Revolving", rng.normal(0.160, 0.030, size=n),
                   rng.normal(0.050, 0.015, size=n))))
    eir = np.clip(eir, 0.01, 0.35)

    # Limits for revolving only
    limit = np.where(seg == "Revolving", rng.integers(1000, 25000, size=n), 0)

    # Simple risk drivers
    ltv = np.where(seg == "Retail_Mortgage", rng.uniform(0.40, 1.05, size=n), np.nan)
    rating_grade = np.where(
        np.isin(seg, ["SME_TermLoan", "Corp_TermLoan"]),
        rng.integers(1, 9, size=n),  # 1 best .. 8 worst
        np.nan
    )

    # TTC PD proxy (annual) by segment, then add idiosyncratic noise
    base_ttc = np.select(
        [
            seg == "Retail_Mortgage",
            seg == "Retail_PL",
            seg == "SME_TermLoan",
            seg == "Corp_TermLoan",
            seg == "Revolving",
        ],
        [0.006, 0.035, 0.025, 0.015, 0.045],
        default=0.02,
    )
    ttc_pd_annual = np.clip(base_ttc * np.exp(rng.normal(0, 0.35, size=n)), 0.0005, 0.35)

    accounts = pd.DataFrame(
        {
            "account_id": [f"A{str(i).zfill(7)}" for i in range(n)],
            "segment": seg,
            "origination_date": origination,
            "maturity_date": maturity,
            "eir": eir,
            "limit_amount": limit,
            "ltv": ltv,
            "rating_grade": rating_grade,
            "ttc_pd_annual": ttc_pd_annual,
        }
    )
    return accounts


# -----------------------------
# Synthetic monthly performance
# -----------------------------
def generate_performance(
    accounts: pd.DataFrame, macro_base: pd.DataFrame, months: int, seed: int = 7
) -> pd.DataFrame:
    """
    Generates monthly balances, utilization, dpd, default_flag.
    Default process: monthly hazard derived from TTC PD and macro stress.
    """
    rng = np.random.default_rng(seed)
    dates = macro_base.loc[macro_base["scenario"] == "Base", "date"].sort_values().unique()
    if len(dates) < months:
        raise ValueError("Macro table shorter than requested months.")
    dates = pd.to_datetime(dates[:months])

    # Macro scalar for stress (higher unemployment / rates -> higher hazard; higher GDP -> lower hazard)
    mac = macro_base[macro_base["scenario"] == "Base"].set_index("date").loc[dates]
    stress = (
        0.35 * (mac["unemployment"].values - mac["unemployment"].values.mean())
        - 0.25 * (mac["gdp_yoy"].values - mac["gdp_yoy"].values.mean())
        + 0.20 * (mac["policy_rate"].values - mac["policy_rate"].values.mean())
    )  # roughly centered

    n = len(accounts)
    seg = accounts["segment"].values
    ttc_pd_annual = accounts["ttc_pd_annual"].values

    # Convert annual TTC PD to baseline monthly hazard approx
    # h ≈ 1 - (1 - PD_annual)^(1/12)
    h0 = 1.0 - np.power(1.0 - ttc_pd_annual, 1.0 / 12.0)

    # Segment multipliers
    seg_mult = np.select(
        [seg == "Retail_Mortgage", seg == "Retail_PL", seg == "SME_TermLoan", seg == "Corp_TermLoan", seg == "Revolving"],
        [0.85, 1.10, 1.00, 0.90, 1.20],
        default=1.0,
    )

    # Start balances
    start_bal = np.select(
        [seg == "Retail_Mortgage", seg == "Retail_PL", seg == "SME_TermLoan", seg == "Corp_TermLoan", seg == "Revolving"],
        [
            rng.integers(80000, 400000, size=n),
            rng.integers(2000, 25000, size=n),
            rng.integers(20000, 250000, size=n),
            rng.integers(50000, 500000, size=n),
            rng.integers(500, 12000, size=n),
        ],
        default=rng.integers(5000, 50000, size=n),
    ).astype(float)

    # Revolving utilization process
    limit = accounts["limit_amount"].values.astype(float)
    util = np.where(seg == "Revolving", rng.uniform(0.10, 0.95, size=n), np.nan)

    alive = np.ones(n, dtype=bool)
    ever_defaulted = np.zeros(n, dtype=bool)

    rows = []
    bal = start_bal.copy()

    for t, d in enumerate(dates):
        # macro-adjust hazard
        # amplify stress modestly, keep hazards bounded
        ht = np.clip(h0 * seg_mult * np.exp(0.8 * stress[t]), 0.00001, 0.20)

        u = rng.random(n)
        new_default = (u < ht) & alive  # default arrival
        ever_defaulted |= new_default
        alive &= ~new_default

        # balances evolve: amortising for term loans, stable-ish for revolving
        amort_rate = np.select(
            [seg == "Retail_Mortgage", seg == "Retail_PL", np.isin(seg, ["SME_TermLoan", "Corp_TermLoan"]), seg == "Revolving"],
            [0.0025, 0.020, 0.010, 0.0],
            default=0.01,
        )
        bal = np.where(alive, np.maximum(bal * (1.0 - amort_rate), 0.0), bal)

        # revolving utilization random walk
        if np.any(seg == "Revolving"):
            util_rw = np.clip(util + rng.normal(0, 0.04, size=n), 0.0, 1.0)
            util = np.where(seg == "Revolving", util_rw, util)
            bal = np.where(seg == "Revolving", util * np.maximum(limit, 1.0), bal)

        # dpd: mostly 0, some delinquency, default -> 90+
        delinquent = (rng.random(n) < 0.03) & alive
        dpd = np.zeros(n, dtype=int)
        dpd = np.where(delinquent, rng.choice([30, 60], size=n, p=[0.8, 0.2]), dpd)
        dpd = np.where(ever_defaulted, 90, dpd)

        rows.append(
            pd.DataFrame(
                {
                    "snapshot_date": d,
                    "account_id": accounts["account_id"].values,
                    "balance": bal,
                    "utilization": np.where(seg == "Revolving", util, np.nan),
                    "dpd": dpd,
                    "default_flag": ever_defaulted.astype(int),
                }
            )
        )

    perf = pd.concat(rows, ignore_index=True)
    return perf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_accounts", type=int, default=8000)
    ap.add_argument("--start_month_end", type=str, default="2018-01-31")
    ap.add_argument("--months", type=int, default=84)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", type=str, default="data/curated")
    args = ap.parse_args()

    macro = generate_macro(args.start_month_end, args.months, seed=args.seed)
    accounts = generate_accounts(args.n_accounts, args.start_month_end, seed=args.seed)
    perf = generate_performance(accounts, macro, args.months, seed=args.seed)

    outdir = args.outdir
    accounts.to_parquet(f"{outdir}/accounts.parquet", index=False)
    perf.to_parquet(f"{outdir}/performance_monthly.parquet", index=False)
    macro.to_parquet(f"{outdir}/macro_scenarios_monthly.parquet", index=False)

    print("Wrote:")
    print(f" - {outdir}/accounts.parquet")
    print(f" - {outdir}/performance_monthly.parquet")
    print(f" - {outdir}/macro_scenarios_monthly.parquet")


if __name__ == "__main__":
    main()
