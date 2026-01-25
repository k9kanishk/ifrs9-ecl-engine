from __future__ import annotations

import numpy as np
import pandas as pd

from ecl_engine.utils.io import load_yml


def _exp_half_life_weights(horizon: int, half_life_months: float) -> np.ndarray:
    half_life_months = max(float(half_life_months), 0.1)
    t = np.arange(horizon, dtype=np.float64)
    w = np.exp(-np.log(2.0) * t / half_life_months)
    w = w / w.sum()
    return w


def _discount_factors_monthly(eir_annual: np.ndarray, horizon: int) -> np.ndarray:
    e = np.clip(eir_annual.astype(np.float64), -0.5, 2.0)
    r_m = e / 12.0
    t = np.arange(1, horizon + 1, dtype=np.float64)[None, :]
    return 1.0 / np.power(1.0 + r_m[:, None], t)


def stage3_workout_table_scenarios(
    asof_dt: pd.Timestamp,
    accounts: pd.DataFrame,
    perf_asof: pd.DataFrame,
    staging_asof: pd.DataFrame,
    portfolio_params: dict,
    stress_by_scenario: dict[str, float],
    scenario_weights: dict[str, float],
    workout_cfg_path: str = "configs/workout_lgd.yml",
) -> pd.DataFrame:
    """
    Stage 3: scenario-linked workout ECL.

    Returns account-level columns:
      - ead_default
      - pv_recoveries_{base/upside/downside}, workout_lgd_{base/upside/downside}, ecl_stage3_{base/upside/downside}
      - pv_recoveries (weighted), workout_lgd (weighted), ecl_stage3_workout (weighted)
      - audit fields: workout_total_recovery_assumed, workout_half_life_months_assumed, workout_collection_cost_assumed
    """
    cfg = load_yml(workout_cfg_path)
    wcfg = cfg["workout"]
    H = int(wcfg["horizon_months"])
    seg_params = wcfg["segment_params"]
    recovery_cap = float(wcfg.get("recovery_cap", 0.95))
    recovery_floor = float(wcfg.get("recovery_floor", 0.0))
    fallback_eir = float(wcfg.get("fallback_eir_annual", 0.05))

    link = wcfg.get("scenario_linkage", {})
    rec_beta_map = link.get("recovery_beta_by_segment", {})
    half_life_beta = float(link.get("half_life_beta", 0.0))
    cost_beta = float(link.get("cost_beta", 0.0))

    # Join base
    base = (
        staging_asof[["account_id", "stage"]]
        .merge(perf_asof[["account_id", "balance"]], on="account_id", how="left")
        .merge(accounts[["account_id", "segment", "eir", "limit_amount"]], on="account_id", how="left")
    )
    base["stage"] = base["stage"].fillna(1).astype(int)

    s3 = base[base["stage"] == 3].copy()
    if s3.empty:
        return pd.DataFrame(columns=["account_id"])

    seg = s3["segment"].astype(str).to_numpy()
    bal = s3["balance"].fillna(0.0).astype(np.float64).to_numpy()
    limit_amt = s3["limit_amount"].fillna(0.0).astype(np.float64).to_numpy()

    eir = s3["eir"].fillna(fallback_eir).astype(np.float64).to_numpy()
    eir = np.where(np.isfinite(eir), eir, fallback_eir)

    # EAD@default
    ccf_map = portfolio_params.get("ccf_base", {})
    ccf_rev = float(ccf_map.get("Revolving", 0.75))
    is_rev = seg == "Revolving"
    undrawn = np.maximum(limit_amt - bal, 0.0)
    ead = bal.copy()
    ead[is_rev] = bal[is_rev] + ccf_rev * undrawn[is_rev]
    ead = np.maximum(ead, 0.0)

    # Base segment assumptions (audit baseline)
    total_recovery_base = np.array(
        [float(seg_params.get(s, {}).get("total_recovery", 0.35)) for s in seg], dtype=np.float64
    )
    half_life_base = np.array(
        [float(seg_params.get(s, {}).get("half_life_months", 12)) for s in seg], dtype=np.float64
    )
    cost_base = np.array(
        [float(seg_params.get(s, {}).get("collection_cost", 0.10)) for s in seg], dtype=np.float64
    )

    total_recovery_base = np.clip(total_recovery_base, recovery_floor, recovery_cap)
    cost_base = np.clip(cost_base, 0.0, 0.50)

    DF = _discount_factors_monthly(eir, H)

    def compute_for_scenario(stress: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # stress>0 => worse macro
        stress_pos = max(float(stress), 0.0)

        rec_beta = np.array([float(rec_beta_map.get(s, 0.15)) for s in seg], dtype=np.float64)
        rec_mult = np.exp(-rec_beta * stress)  # downside => lower recovery
        total_recovery = np.clip(total_recovery_base * rec_mult, recovery_floor, recovery_cap)

        half_life = half_life_base * (1.0 + half_life_beta * stress_pos)
        half_life = np.clip(half_life, 1.0, 120.0)

        cost = cost_base * (1.0 + cost_beta * stress_pos)
        cost = np.clip(cost, 0.0, 0.50)

        pv = np.zeros(len(s3), dtype=np.float64)
        for i in range(len(s3)):
            w = _exp_half_life_weights(H, half_life[i])
            rec_cf = ead[i] * total_recovery[i] * w * (1.0 - cost[i])
            pv[i] = float(np.sum(rec_cf * DF[i, :]))

        pv = np.minimum(pv, ead)
        ratio = np.zeros_like(pv, dtype=np.float64)
        np.divide(pv, ead, out=ratio, where=ead > 0)
        lgd = 1.0 - ratio
        lgd = np.clip(lgd, 0.0, 1.0)
        ecl = ead * lgd
        return pv, lgd, ecl

    scen_list = ["Base", "Upside", "Downside"]
    out = pd.DataFrame({"account_id": s3["account_id"].values})
    out["ead_default"] = ead
    out["workout_horizon_m"] = H

    # scenario outputs
    pv_s, lgd_s, ecl_s = {}, {}, {}
    for scen in scen_list:
        stress = float(stress_by_scenario.get(scen, 0.0))
        pv, lgd, ecl = compute_for_scenario(stress)
        pv_s[scen], lgd_s[scen], ecl_s[scen] = pv, lgd, ecl
        out[f"pv_recoveries_{scen.lower()}"] = pv
        out[f"workout_lgd_{scen.lower()}"] = lgd
        out[f"ecl_stage3_{scen.lower()}"] = ecl
        out[f"stress_index_{scen.lower()}"] = stress

    # weighted
    w_base = float(scenario_weights.get("Base", 0.6))
    w_up = float(scenario_weights.get("Upside", 0.2))
    w_dn = float(scenario_weights.get("Downside", 0.2))
    w_sum = w_base + w_up + w_dn
    if w_sum <= 0:
        w_base, w_up, w_dn = 0.6, 0.2, 0.2
        w_sum = 1.0
    w_base, w_up, w_dn = w_base / w_sum, w_up / w_sum, w_dn / w_sum

    out["pv_recoveries"] = w_base * pv_s["Base"] + w_up * pv_s["Upside"] + w_dn * pv_s["Downside"]
    pvw = out["pv_recoveries"].to_numpy(dtype=np.float64)
    ratio = np.zeros_like(pvw, dtype=np.float64)
    np.divide(pvw, ead, out=ratio, where=ead > 0)
    out["workout_lgd"] = np.clip(1.0 - ratio, 0.0, 1.0)
    out["ecl_stage3_workout"] = w_base * ecl_s["Base"] + w_up * ecl_s["Upside"] + w_dn * ecl_s["Downside"]

    # audit baseline assumptions
    out["workout_total_recovery_assumed"] = total_recovery_base
    out["workout_half_life_months_assumed"] = half_life_base
    out["workout_collection_cost_assumed"] = cost_base

    out["asof_date"] = asof_dt
    return out
