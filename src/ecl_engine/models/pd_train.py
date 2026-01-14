from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _pick_col(cols: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _month_end(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x) + pd.offsets.MonthEnd(0)


def build_labels_from_staging(staging: pd.DataFrame) -> pd.DataFrame:
    """
    staging: must include snapshot_date, account_id, stage
    Returns table with default_date per account (first date stage==3)
    """
    s = staging.copy()
    s["snapshot_date"] = _month_end(s["snapshot_date"])
    s["stage"] = s["stage"].astype(int)

    default_dates = (
        s[s["stage"] == 3]
        .groupby("account_id", as_index=False)["snapshot_date"]
        .min()
        .rename(columns={"snapshot_date": "default_date"})
    )
    return default_dates


def make_pd_panel(
    accounts_path: str = "data/curated/accounts.parquet",
    perf_path: str = "data/curated/performance_monthly.parquet",
    staging_path: str = "data/curated/staging_output.parquet",
    out_sample_path: str = "data/curated/pd_training_sample.parquet",
    horizon_months: int = 12,
) -> pd.DataFrame:
    acc = pd.read_parquet(accounts_path)
    perf = pd.read_parquet(perf_path)
    stg = pd.read_parquet(staging_path)

    # Normalize dates
    stg["snapshot_date"] = _month_end(stg["snapshot_date"])
    perf["snapshot_date"] = _month_end(perf["snapshot_date"])

    # Column detection (robust)
    acc_cols = acc.columns.tolist()
    perf_cols = perf.columns.tolist()

    seg_acc = _pick_col(acc_cols, ["segment", "product_segment", "portfolio", "seg"])
    seg_perf = _pick_col(perf_cols, ["segment", "product_segment", "portfolio", "seg"])

    bal_col = _pick_col(perf_cols, ["balance", "outstanding_balance", "bal"])
    dpd_col = _pick_col(perf_cols, ["dpd", "days_past_due", "dpd_days", "delinquency_days"])
    limit_col = _pick_col(acc_cols, ["limit_amount", "credit_limit", "limit"])
    eir_col = _pick_col(acc_cols, ["eir", "effective_interest_rate"])
    ttc_col = _pick_col(acc_cols, ["ttc_pd_annual", "ttc_pd", "pd_ttc"])

    if seg_acc is None:
        raise ValueError("accounts.parquet must contain a segment column (e.g., 'segment').")
    if bal_col is None:
        raise ValueError("performance_monthly.parquet must contain a balance column (e.g., 'balance').")

    # Stage at each snapshot
    stg_min = stg[["account_id", "snapshot_date", "stage"]].copy()

    # Default date per account
    dd = build_labels_from_staging(stg_min)

    # Build base panel at each snapshot (t)
    # Join perf with stage by (account_id, snapshot_date)
    base = perf.merge(stg_min, on=["account_id", "snapshot_date"], how="left")

    # Attach segment: prefer perf.segment if exists; else accounts.segment
    if seg_perf and seg_perf in base.columns:
        base = base.rename(columns={seg_perf: "segment"})
        base["segment"] = base["segment"].fillna("UNKNOWN")
    else:
        base = base.merge(
            acc[["account_id", seg_acc]].rename(columns={seg_acc: "segment"}),
            on="account_id",
            how="left",
        )

    # Attach account-level vars
    # IMPORTANT: do NOT merge the segment column again, or you'll get segment_x/segment_y
    acc_keep = ["account_id"]
    if limit_col:
        acc_keep.append(limit_col)
    if eir_col:
        acc_keep.append(eir_col)
    if ttc_col:
        acc_keep.append(ttc_col)

    base = base.merge(acc[acc_keep].copy(), on="account_id", how="left")

    # Safety: if segment got duplicated earlier, normalize it back
    if "segment" not in base.columns:
        if "segment_x" in base.columns or "segment_y" in base.columns:
            base["segment"] = base["segment_x"] if "segment_x" in base.columns else np.nan
            if "segment_y" in base.columns:
                base["segment"] = base["segment"].fillna(base["segment_y"])
            base = base.drop(columns=[c for c in ["segment_x", "segment_y"] if c in base.columns])

    # Standardize names
    base = base.rename(columns={bal_col: "balance"})
    if dpd_col:
        base = base.rename(columns={dpd_col: "dpd"})
    else:
        base["dpd"] = 0.0

    if limit_col:
        base = base.rename(columns={limit_col: "limit_amount"})
    else:
        base["limit_amount"] = np.nan

    if eir_col:
        base = base.rename(columns={eir_col: "eir"})
    else:
        base["eir"] = np.nan

    if ttc_col:
        base = base.rename(columns={ttc_col: "ttc_pd_annual"})
    else:
        base["ttc_pd_annual"] = np.nan

    # Label: default occurs within next 12 months
    base = base.merge(dd, on="account_id", how="left")
    base["default_date"] = pd.to_datetime(base["default_date"])

    base["horizon_end"] = base["snapshot_date"] + pd.offsets.MonthEnd(horizon_months)
    base["y_default_12m"] = (
        base["default_date"].notna()
        & (base["default_date"] > base["snapshot_date"])
        & (base["default_date"] <= base["horizon_end"])
    ).astype(int)

    # Remove already-defaulted at time t (stage 3 at snapshot)
    base = base[base["stage"].fillna(1).astype(int) != 3].copy()

    # Feature engineering (simple but bank-standard)
    base["utilization"] = np.where(
        base["limit_amount"].notna() & (base["limit_amount"] > 0),
        base["balance"] / base["limit_amount"],
        np.nan,
    )
    base["log_balance"] = np.log1p(np.maximum(base["balance"], 0.0))
    base["dpd_bucket"] = pd.cut(
        base["dpd"].fillna(0.0),
        bins=[-0.1, 0.1, 30, 60, 90, 10_000],
        labels=["0", "1_30", "31_60", "61_90", "90p"],
        include_lowest=True,
    ).astype(str)

    # Keep minimal training set
    panel = base[
        [
            "account_id",
            "snapshot_date",
            "segment",
            "balance",
            "log_balance",
            "dpd",
            "dpd_bucket",
            "utilization",
            "ttc_pd_annual",
            "y_default_12m",
        ]
    ].copy()

    # Save sample for debugging (optional)
    Path(out_sample_path).parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_sample_path, index=False)
    print(f"Wrote: {out_sample_path} ({panel.shape[0]:,} rows)")

    return panel


def fit_pd_model(
    panel: pd.DataFrame,
    train_end: str = "2022-12-31",
    model_path: str = "models/pd_logit.joblib",
    schema_path: str = "models/pd_feature_schema.json",
    metrics_path: str = "reports/pd_validation_metrics.csv",
) -> None:
    panel = panel.copy()
    panel["snapshot_date"] = pd.to_datetime(panel["snapshot_date"])
    train_end_dt = pd.to_datetime(train_end) + pd.offsets.MonthEnd(0)

    # Split by time (bank-typical; avoids leakage)
    train = panel[panel["snapshot_date"] <= train_end_dt].copy()
    test = panel[panel["snapshot_date"] > train_end_dt].copy()

    # Features
    cat_cols = ["segment", "dpd_bucket"]
    num_cols = ["log_balance", "dpd", "utilization", "ttc_pd_annual"]

    # Fill missing numerics conservatively
    for c in num_cols:
        if c in train.columns:
            train[c] = train[c].astype(float)
            test[c] = test[c].astype(float)
    train[num_cols] = train[num_cols].fillna(train[num_cols].median(numeric_only=True))
    test[num_cols] = test[num_cols].fillna(train[num_cols].median(numeric_only=True))

    X_train = train[cat_cols + num_cols]
    y_train = train["y_default_12m"].astype(int)

    X_test = test[cat_cols + num_cols]
    y_test = test["y_default_12m"].astype(int)

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
        ]
    )

    clf = LogisticRegression(
        max_iter=4000,
        class_weight=None,  # IMPORTANT: stop inflating base rate
        solver="lbfgs",
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    # Predict
    p_train = pipe.predict_proba(X_train)[:, 1]
    p_test = pipe.predict_proba(X_test)[:, 1]

    # --- Calibration-in-the-large (CITL): shift logits so mean predicted matches observed ---
    def logit(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    obs_rate = float(y_train.mean())
    pred_rate = float(p_train.mean())

    # Shift in logit space
    delta = logit(obs_rate) - logit(pred_rate)

    p_train = sigmoid(logit(p_train) + delta)
    p_test = sigmoid(logit(p_test) + delta)

    # Metrics
    auc_train = roc_auc_score(y_train, p_train) if y_train.nunique() > 1 else np.nan
    auc_test = roc_auc_score(y_test, p_test) if y_test.nunique() > 1 else np.nan

    def ks(y, p):
        if pd.Series(y).nunique() <= 1:
            return np.nan
        fpr, tpr, _ = roc_curve(y, p)
        return float(np.max(np.abs(tpr - fpr)))

    ks_train = ks(y_train, p_train)
    ks_test = ks(y_test, p_test)

    # Save metrics
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(
        [
            {
                "split": "train",
                "auc": auc_train,
                "ks": ks_train,
                "n": len(train),
                "defaults": int(y_train.sum()),
                "citl_delta": delta,
            },
            {
                "split": "test",
                "auc": auc_test,
                "ks": ks_test,
                "n": len(test),
                "defaults": int(y_test.sum()),
                "citl_delta": delta,
            },
        ]
    )
    m.to_csv(metrics_path, index=False)
    print(f"Wrote: {metrics_path}")

    # ROC plot (test)
    if pd.Series(y_test).nunique() > 1:
        fpr, tpr, _ = roc_curve(y_test, p_test)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("PD Model ROC (Test)")
        plt.savefig("reports/pd_roc.png", bbox_inches="tight")
        plt.close()
        print("Wrote: reports/pd_roc.png")

    # Calibration plot (test)
    if pd.Series(y_test).nunique() > 1:
        frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
        plt.figure()
        plt.plot(mean_pred, frac_pos, marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Mean predicted PD")
        plt.ylabel("Observed default rate")
        plt.title("PD Calibration (Test)")
        plt.savefig("reports/pd_calibration.png", bbox_inches="tight")
        plt.close()
        print("Wrote: reports/pd_calibration.png")

    # Score distribution
    plt.figure()
    plt.hist(p_test, bins=50)
    plt.title("PD Score Distribution (Test)")
    plt.xlabel("Predicted 12M PD")
    plt.ylabel("Count")
    plt.savefig("reports/pd_score_hist.png", bbox_inches="tight")
    plt.close()
    print("Wrote: reports/pd_score_hist.png")

    # Save model + schema
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"Wrote: {model_path}")

    schema = {"cat_cols": cat_cols, "num_cols": num_cols, "citl_delta": float(delta)}
    Path(schema_path).parent.mkdir(parents=True, exist_ok=True)
    Path(schema_path).write_text(json.dumps(schema, indent=2))
    print(f"Wrote: {schema_path}")


def main() -> None:
    panel = make_pd_panel()
    fit_pd_model(panel)


if __name__ == "__main__":
    main()
