from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

def load_yml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _exp_half_life_weights(horizon: int, half_life_months: float) -> np.ndarray:
    """
    weights[t] ∝ exp(-ln(2)*(t)/half_life) for t=0..H-1
    Sum(weights)=1
    """
    half_life_months = max(float(half_life_months), 0.1)
    t = np.arange(horizon, dtype=np.float64)
    w = np.exp(-np.log(2.0) * t / half_life_months)
    w = w / w.sum()
    return w


def _discount_factors_monthly(eir_annual: np.ndarray, horizon: int) -> np.ndarray:
    """
    eir_annual: (N,)
    returns DF: (N,H) with DF_t = 1/(1+eir/12)^t for t=1..H
    """
    e = np.clip(eir_annual.astype(np.float64), -0.5, 2.0)
    r_m = e / 12.0
    t = np.arange(1, horizon + 1, dtype=np.float64)[None, :]
    return 1.0 / np.power(1.0 + r_m[:, None], t)


def stage3_workout_table(
    asof_dt: pd.Timestamp,
    accounts: pd.DataFrame,
    perf_asof: pd.DataFrame,
    staging_asof: pd.DataFrame,
    portfolio_params: dict,
    workout_cfg_path: str = "configs/workout_lgd.yml",
) -> pd.DataFrame:
    """
    Returns account-level Stage 3 workout fields:
      - ead_default
      - pv_recoveries
      - workout_lgd
      - ecl_stage3_workout

    Inputs expected columns:
      accounts: account_id, segment, eir, limit_amount
      perf_asof: account_id, balance
      staging_asof: account_id, stage
    """
    cfg = load_yml(workout_cfg_path)
    wcfg = cfg["workout"]
    H = int(wcfg["horizon_months"])
    seg_params = wcfg["segment_params"]
    recovery_cap = float(wcfg.get("recovery_cap", 0.95))
    recovery_floor = float(wcfg.get("recovery_floor", 0.0))
    fallback_eir = float(wcfg.get("fallback_eir_annual", 0.05))

    # Join base
    base = (
        staging_asof[["account_id", "stage"]]
        .merge(perf_asof[["account_id", "balance"]], on="account_id", how="left")
        .merge(accounts[["account_id", "segment", "eir", "limit_amount"]], on="account_id", how="left")
    )
    base["stage"] = base["stage"].fillna(1).astype(int)

    # Keep Stage 3 only
    s3 = base[base["stage"] == 3].copy()
    if s3.empty:
        return pd.DataFrame(
            columns=["account_id", "ead_default", "pv_recoveries", "workout_lgd", "ecl_stage3_workout"]
        )

    # Inputs
    seg = s3["segment"].astype(str).to_numpy()
    bal = s3["balance"].fillna(0.0).astype(np.float64).to_numpy()
    limit_amt = s3["limit_amount"].fillna(0.0).astype(np.float64).to_numpy()

    eir = s3["eir"].fillna(fallback_eir).astype(np.float64).to_numpy()
    # avoid 0/NaN weirdness
    eir = np.where(np.isfinite(eir), eir, fallback_eir)

    # EAD at default
    # If revolving: balance + CCF*(undrawn). Otherwise: balance
    ccf_map = portfolio_params.get("ccf_base", {})
    ccf_rev = float(ccf_map.get("Revolving", 0.75))
    is_rev = seg == "Revolving"
    undrawn = np.maximum(limit_amt - bal, 0.0)
    ead = bal.copy()
    ead[is_rev] = bal[is_rev] + ccf_rev * undrawn[is_rev]
    ead = np.maximum(ead, 0.0)

    # Segment recovery settings
    total_recovery = np.array(
        [float(seg_params.get(s, {}).get("total_recovery", 0.35)) for s in seg], dtype=np.float64
    )
    half_life = np.array(
        [float(seg_params.get(s, {}).get("half_life_months", 12)) for s in seg], dtype=np.float64
    )
    cost = np.array(
        [float(seg_params.get(s, {}).get("collection_cost", 0.10)) for s in seg], dtype=np.float64
    )

    total_recovery = np.clip(total_recovery, recovery_floor, recovery_cap)
    cost = np.clip(cost, 0.0, 0.50)

    # Discount factors (N,H)
    DF = _discount_factors_monthly(eir, H)

    # PV recoveries per account: sum_t [ EAD * total_recovery * w_t * (1-cost) * DF_t ]
    # Build weights per account (fast enough; N~50k max, H=36)
    pv = np.zeros(len(s3), dtype=np.float64)
    for i in range(len(s3)):
        w = _exp_half_life_weights(H, half_life[i])
        rec_cf = ead[i] * total_recovery[i] * w * (1.0 - cost[i])
        pv[i] = float(np.sum(rec_cf * DF[i, :]))

    # Workout LGD and ECL
    pv = np.minimum(pv, ead)  # PV recoveries cannot exceed EAD
    lgd = 1.0 - np.where(ead > 0, pv / ead, 0.0)
    lgd = np.clip(lgd, 0.0, 1.0)
    ecl3 = ead * lgd

    out = pd.DataFrame(
        {
            "account_id": s3["account_id"].values,
            "ead_default": ead,
            "pv_recoveries": pv,
            "workout_lgd": lgd,
            "ecl_stage3_workout": ecl3,
            "workout_horizon_m": H,
            "workout_total_recovery_assumed": total_recovery,
            "workout_half_life_months_assumed": half_life,
            "workout_collection_cost_assumed": cost,
        }
    )
    out["asof_date"] = asof_dt
    return out
