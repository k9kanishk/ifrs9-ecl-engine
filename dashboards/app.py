from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="IFRS 9 ECL Dashboard", layout="wide")

st.title("IFRS 9 ECL Engine — Dashboard")
st.caption("Stage, ECL, Scenarios, Overlays — built from reproducible pipeline outputs.")

# ASOF Selector
st.sidebar.header("📅 ASOF Selection")
available_asofs = sorted(
    [p.stem.replace("ecl_output_asof_", "") for p in Path("data/curated").glob("ecl_output_asof_*.parquet")],
    reverse=True,
)

if not available_asofs:
    st.error("No ECL outputs found. Run: python -m ecl_engine.pipeline --asof 2024-12-31")
    st.stop()

selected_asof = st.sidebar.selectbox("Select ASOF", available_asofs, index=0)


@st.cache_data
def load_asof_outputs(asof: str):
    ecl_path = f"data/curated/ecl_output_asof_{asof}.parquet"
    ecl = pd.read_parquet(ecl_path)
    ecl_ov = pd.read_parquet("data/curated/ecl_with_overlays.parquet")

    ecl_ov["asof_date"] = pd.to_datetime(ecl_ov["asof_date"])
    ecl_ov = ecl_ov[ecl_ov["asof_date"] == pd.Timestamp(asof)].copy()

    scen_path = f"data/curated/scenario_contribution_{asof}.csv"
    scen = pd.read_csv(scen_path) if Path(scen_path).exists() else None

    dcf_path = f"data/curated/ecl_dcf_asof_{asof}.parquet"
    dcf = pd.read_parquet(dcf_path) if Path(dcf_path).exists() else None

    return asof, ecl, ecl_ov, scen, dcf


@st.cache_data
def load_additional_outputs(asof: str):
    drv_path = f"data/curated/driver_decomposition_{asof}.csv"
    qc_path = "data/curated/staging_qc_summary.parquet"

    drv = pd.read_csv(drv_path) if Path(drv_path).exists() else None
    qc = pd.read_parquet(qc_path) if Path(qc_path).exists() else None

    mig_path = "data/curated/stage_migration.parquet"
    mig = pd.read_parquet(mig_path) if Path(mig_path).exists() else None

    audit_path = "data/curated/overlay_audit.parquet"
    audit = pd.read_parquet(audit_path) if Path(audit_path).exists() else None

    exp_path = f"data/curated/account_explain_asof_{asof}.parquet"
    explain = pd.read_parquet(exp_path) if Path(exp_path).exists() else None

    return drv, qc, mig, audit, explain


asof, ecl, ecl_ov, scen, dcf = load_asof_outputs(selected_asof)
drv, qc, mig, audit, explain = load_additional_outputs(asof)

if dcf is not None:
    show_dcf = st.sidebar.checkbox("Show DCF comparison", value=False)
else:
    show_dcf = False

st.subheader(f"ASOF: {asof}")

# Sidebar filters
st.sidebar.header("Filters")
segments = sorted(ecl_ov["segment"].dropna().unique().tolist())
sel_segments = st.sidebar.multiselect("Segment", segments, default=segments)

stages = sorted(ecl_ov["stage"].dropna().unique().tolist())
sel_stages = st.sidebar.multiselect("Stage", stages, default=stages)

f = ecl_ov[(ecl_ov["segment"].isin(sel_segments)) & (ecl_ov["stage"].isin(sel_stages))].copy()

# Top KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Reported ECL (post-overlay)", f"{f['ecl_post_overlay'].sum():,.2f}")
col2.metric("Overlay impact", f"{f['overlay_amount'].sum():,.2f}")
col3.metric("Pre-overlay ECL", f"{f['ecl_pre_overlay'].sum():,.2f}")
col4.metric("Accounts in view", f"{f.shape[0]:,}")

# Top Movers Section
st.divider()
st.markdown("## 🔍 Top Movers")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Highest ECL Accounts")
    top_ecl = ecl_ov.nlargest(10, "ecl_post_overlay")[
        ["account_id", "segment", "stage", "balance", "ecl_post_overlay"]
    ].copy()
    top_ecl["ecl_post_overlay"] = top_ecl["ecl_post_overlay"].apply(lambda x: f"{x:,.2f}")
    st.dataframe(top_ecl, width="stretch", height=300)

with col2:
    st.markdown("### Highest Overlay Impact")
    top_ov = ecl_ov.nlargest(10, "overlay_amount")[
        ["account_id", "segment", "ecl_pre_overlay", "overlay_amount"]
    ].copy()
    top_ov["overlay_amount"] = top_ov["overlay_amount"].apply(lambda x: f"{x:,.2f}")
    st.dataframe(top_ov, width="stretch", height=300)

with col3:
    if show_dcf and dcf is not None:
        st.markdown("### Biggest DCF-Model Deltas")
        merged = dcf.merge(ecl_ov[["account_id", "ecl_post_overlay"]], on="account_id", how="left")
        merged["delta"] = (merged["dcf_ecl_selected"] - merged["ecl_post_overlay"]).abs()
        top_dcf = merged.nlargest(10, "delta")[
            ["account_id", "segment", "ecl_post_overlay", "dcf_ecl_selected", "delta"]
        ].copy()
        top_dcf["delta"] = top_dcf["delta"].apply(lambda x: f"{x:,.2f}")
        st.dataframe(top_dcf, width="stretch", height=300)
    else:
        st.info("Enable DCF comparison to see DCF-Model deltas")

st.divider()

st.markdown("## ECL Waterfall (Governance)")

pre = float(f["ecl_pre_overlay"].sum())
ov = float(f["overlay_amount"].sum())
post = float(f["ecl_post_overlay"].sum())

wf1, wf2 = st.columns(2)

with wf1:
    st.markdown("### Overall waterfall")
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["Pre-overlay", "Overlay", "Post-overlay"],
            y=[pre, ov, post],
        )
    )
    st.plotly_chart(fig, width="stretch")

with wf2:
    st.markdown("### Segment waterfall")
    seg_choice = st.selectbox("Pick a segment", sorted(f["segment"].unique()))
    sf = f[f["segment"] == seg_choice]
    pre_s = float(sf["ecl_pre_overlay"].sum())
    ov_s = float(sf["overlay_amount"].sum())
    post_s = float(sf["ecl_post_overlay"].sum())

    fig2 = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["Pre-overlay", "Overlay", "Post-overlay"],
            y=[pre_s, ov_s, post_s],
        )
    )
    st.plotly_chart(fig2, width="stretch")

st.markdown(
    "Pre-overlay is the model output. "
    "Overlay is management judgement. "
    "Post-overlay is the reported number. "
    "This is the cleanest “audit trail” view."
)

st.divider()

# Row 1: ECL by stage and by segment
c1, c2 = st.columns(2)

with c1:
    st.markdown("### ECL by Stage (post-overlay)")
    stage_tbl = (
        f.groupby("stage")[["ecl_pre_overlay", "overlay_amount", "ecl_post_overlay"]]
        .sum()
        .sort_index()
        .reset_index()
    )
    st.dataframe(stage_tbl, width="stretch")

with c2:
    st.markdown("### ECL by Segment (post-overlay)")
    seg_tbl = (
        f.groupby("segment")[["ecl_pre_overlay", "overlay_amount", "ecl_post_overlay"]]
        .sum()
        .sort_values("ecl_post_overlay", ascending=False)
        .reset_index()
    )
    st.dataframe(seg_tbl, width="stretch", height=360)

st.divider()

# Row 2: scenario contribution + driver sensitivities
c3, c4 = st.columns(2)

with c3:
    st.markdown("### Scenario Contribution (selected horizon rule, pre-overlay)")
    if scen is None:
        st.info("Scenario contribution file not found. Run: python -m ecl_engine.driver_decomp")
    else:
        scen_view = scen.copy()
        scen_view["unweighted_sum"] = scen_view["unweighted_sum"].map(lambda x: f"{x:,.2f}")
        scen_view["weighted_contribution"] = scen["weighted_contribution"].map(lambda x: f"{x:,.2f}")
        scen_view["weighted_share"] = scen["weighted_share"].map(lambda x: f"{x:.2%}")
        st.dataframe(scen_view, width="stretch")

with c4:
    st.markdown("### Driver Sensitivities by Segment (post-overlay)")
    if drv is None:
        st.info("Driver decomposition file not found. Run: python -m ecl_engine.driver_decomp")
    else:
        drv_view = drv.copy()
        # keep the biggest segments only
        drv_view = drv_view.sort_values("ecl_reported", ascending=False).head(15)
        st.dataframe(drv_view, width="stretch", height=360)

st.markdown("## Scenario QC (Governance)")

sev_path = f"data/curated/scenario_severity_asof_{asof}.csv"
pdq_path = f"data/curated/scenario_pd_summary_asof_{asof}.csv"

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Macro severity (z-scores)")
    if os.path.exists(sev_path):
        st.dataframe(pd.read_csv(sev_path), width="stretch")
    else:
        st.info("Run: python -m ecl_engine.scenario_qc")

with c2:
    st.markdown("### Implied PIT PD summary (Stage 1+2)")
    if os.path.exists(pdq_path):
        st.dataframe(pd.read_csv(pdq_path), width="stretch")
    else:
        st.info("Run: python -m ecl_engine.scenario_qc")

st.divider()

# Stage 3 workout summary
st.markdown("## Stage 3 Workout Summary (Waterfall)")

asof_str = str(asof.date()) if hasattr(asof, "date") else str(asof)
path_w = f"data/curated/stage3_workout_summary_asof_{asof_str}.csv"
path_s = f"data/curated/stage3_workout_summary_scenarios_asof_{asof_str}.csv"

if not os.path.exists(path_w):
    st.info("Run: python -m ecl_engine.stage3_summary")
else:
    df_w = pd.read_csv(path_w)

    # Scenario file optional (Phase 5)
    df_s = pd.read_csv(path_s) if os.path.exists(path_s) else None

    segs = ["All"] + sorted(df_w["segment"].unique().tolist())
    seg_pick = st.selectbox("Segment (Stage 3)", segs, index=0)

    scen_pick = "Weighted"
    if df_s is not None:
        scen_pick = st.selectbox("Scenario view", ["Weighted", "Base", "Upside", "Downside"], index=0)

    def get_vals(seg_name: str):
        if seg_name == "All":
            if df_s is None or scen_pick == "Weighted":
                ead = float(df_w["ead_default_sum"].sum())
                pv = float(df_w["pv_recoveries_sum"].sum())
                ecl = float(df_w["ecl_sum"].sum())
                lgd = 1 - (pv / ead) if ead > 0 else 0.0
                return ead, pv, ecl, lgd

            # scenario
            tmp = df_s.copy()
        else:
            if df_s is None or scen_pick == "Weighted":
                tmp = df_w[df_w["segment"] == seg_name]
                ead = float(tmp["ead_default_sum"].iloc[0])
                pv = float(tmp["pv_recoveries_sum"].iloc[0])
                ecl = float(tmp["ecl_sum"].iloc[0])
                lgd = float(tmp["implied_lgd"].iloc[0])
                return ead, pv, ecl, lgd

            tmp = df_s[df_s["segment"] == seg_name]

        # scenario values from df_s
        if seg_name == "All":
            ead = float(tmp["ead_default_sum"].sum())
            if scen_pick == "Base":
                pv = float(tmp["pv_base"].sum())
                ecl = float(tmp["ecl_base"].sum())
                lgd = float((tmp["lgd_base"] * tmp["ead_default_sum"]).sum() / ead)
            elif scen_pick == "Upside":
                pv = float(tmp["pv_up"].sum())
                ecl = float(tmp["ecl_up"].sum())
                lgd = float((tmp["lgd_up"] * tmp["ead_default_sum"]).sum() / ead)
            elif scen_pick == "Downside":
                pv = float(tmp["pv_dn"].sum())
                ecl = float(tmp["ecl_dn"].sum())
                lgd = float((tmp["lgd_dn"] * tmp["ead_default_sum"]).sum() / ead)
            else:
                pv = float(tmp["pv_w"].sum())
                ecl = float(tmp["ecl_w"].sum())
                lgd = float((tmp["lgd_w"] * tmp["ead_default_sum"]).sum() / ead)
            return ead, pv, ecl, lgd

        row = tmp.iloc[0]
        ead = float(row["ead_default_sum"])
        if scen_pick == "Base":
            pv = float(row["pv_base"])
            ecl = float(row["ecl_base"])
            lgd = float(row["lgd_base"])
        elif scen_pick == "Upside":
            pv = float(row["pv_up"])
            ecl = float(row["ecl_up"])
            lgd = float(row["lgd_up"])
        elif scen_pick == "Downside":
            pv = float(row["pv_dn"])
            ecl = float(row["ecl_dn"])
            lgd = float(row["lgd_dn"])
        else:
            pv = float(row["pv_w"])
            ecl = float(row["ecl_w"])
            lgd = float(row["lgd_w"])
        return ead, pv, ecl, lgd

    ead, pv, ecl, lgd = get_vals(seg_pick)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stage 3 EAD@Default", f"{ead:,.2f}")
    c2.metric("PV Recoveries", f"{pv:,.2f}")
    c3.metric("Stage 3 ECL", f"{ecl:,.2f}")
    c4.metric("Implied Workout LGD", f"{lgd:.2%}")

    title = f"Stage 3 Waterfall — {seg_pick} — {scen_pick}"
    fig = go.Figure(
        go.Waterfall(
            name="Stage3",
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["EAD@Default", "PV Recoveries", "ECL"],
            y=[ead, -pv, ecl],
            text=[f"{ead:,.0f}", f"{pv:,.0f}", f"{ecl:,.0f}"],
            textposition="outside",
        )
    )
    fig.update_layout(title=title, showlegend=False, waterfallgap=0.3, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Table")
    st.dataframe(df_w, width="stretch")
    if df_s is not None:
        st.markdown("### Scenario table (Phase 5)")
        st.dataframe(df_s, width="stretch")

st.divider()

# Row 3: Stage mix over time (from staging QC summary)
st.markdown("### Stage Counts Over Time (QC)")
if qc is None:
    st.info("Staging QC summary not found. Run: python src/ecl_engine/staging.py")
else:
    qc2 = qc[qc["segment"].isin(sel_segments)].copy()
    # aggregate across segments for a clean view
    qc_total = qc2.groupby(["snapshot_date", "stage"], as_index=False)["n_accounts"].sum()
    fig = px.line(
        qc_total,
        x="snapshot_date",
        y="n_accounts",
        color="stage",
        title="Stage Counts Over Time (QC)",
    )
    st.plotly_chart(fig, width="stretch")

st.divider()

st.markdown("## Stage Migration Matrix (t-1 → t)")

if mig is None:
    st.info("Migration table not found. Run: python src/ecl_engine/stage_migration.py")
else:
    mig["from_date"] = pd.to_datetime(mig["from_date"])
    mig["to_date"] = pd.to_datetime(mig["to_date"])

    # pick a month to view (use 'to_date')
    dates = sorted(mig["to_date"].unique())
    to_dt = st.selectbox("Select month (to_date)", dates, index=len(dates) - 1)

    view = mig[(mig["to_date"] == to_dt) & (mig["segment"].isin(sel_segments))].copy()

    # aggregate across selected segments
    agg = view.groupby(["stage_from", "stage_to"], as_index=False)["n_accounts"].sum()

    pivot = agg.pivot(index="stage_from", columns="stage_to", values="n_accounts").fillna(0.0)
    pivot = pivot.reindex(index=[1, 2, 3], columns=[1, 2, 3], fill_value=0.0)

    mode = st.radio("Display", ["Counts", "Row % (conditional on from_stage)"], horizontal=True)

    show = pivot.copy()
    if mode.startswith("Row %"):
        show = show.div(show.sum(axis=1).replace(0, 1), axis=0) * 100.0

    c1, c2 = st.columns([1, 1.2])

    with c1:
        st.markdown("### Matrix table")
        st.dataframe(show.round(2), width="stretch")

    with c2:
        st.markdown("### Heatmap")
        fig = go.Figure(
            data=go.Heatmap(
                z=show.values,
                x=[str(c) for c in show.columns],
                y=[str(i) for i in show.index],
                hoverongaps=False,
            )
        )
        fig.update_layout(xaxis_title="Stage to", yaxis_title="Stage from")
        st.plotly_chart(fig, width="stretch")

st.markdown(
    "Rows = stage last month (t-1), columns = stage this month (t). "
    "Diagonal = stable accounts. "
    "1→2 = SICR migration, 2→1 = cure (improvement). "
    "Any 1/2→3 = defaults. "
    "This is a standard governance monitoring view for staging stability."
)

st.divider()

st.markdown("## Stage Migration Trend (Governance Monitoring)")

if mig is None:
    st.info("Migration table not found. Run: python src/ecl_engine/stage_migration.py")
else:
    m = mig.copy()
    m["from_date"] = pd.to_datetime(m["from_date"])
    m["to_date"] = pd.to_datetime(m["to_date"])

    # apply segment filter
    m = m[m["segment"].isin(sel_segments)].copy()

    # aggregate counts by month + transition
    agg = (
        m.groupby(["to_date", "stage_from", "stage_to"], as_index=False)["n_accounts"]
        .sum()
        .sort_values("to_date")
    )

    # total accounts "from" each month (denominator)
    denom = (
        agg.groupby(["to_date", "stage_from"], as_index=False)["n_accounts"]
        .sum()
        .rename(columns={"n_accounts": "from_total"})
    )

    agg = agg.merge(denom, on=["to_date", "stage_from"], how="left")
    agg["pct_of_from_stage"] = 100.0 * agg["n_accounts"] / agg["from_total"].replace(0, 1)

    # build key transition series
    def series(stage_from: int, stage_to: int, label: str) -> pd.DataFrame:
        x = agg[
            (agg["stage_from"] == stage_from) & (agg["stage_to"] == stage_to)
        ][["to_date", "pct_of_from_stage"]].copy()
        x = x.rename(columns={"pct_of_from_stage": label})
        return x

    s12 = series(1, 2, "1→2 (SICR)")
    s21 = series(2, 1, "2→1 (Cure)")
    s13 = series(1, 3, "1→3 (Default)")
    s23 = series(2, 3, "2→3 (Default)")

    # merge on to_date
    trend = None
    for s in [s12, s21, s13, s23]:
        trend = s if trend is None else trend.merge(s, on="to_date", how="outer")

    trend = trend.fillna(0.0).sort_values("to_date")
    fig = px.line(
        trend,
        x="to_date",
        y=["1→2 (SICR)", "2→1 (Cure)", "1→3 (Default)", "2→3 (Default)"],
        title="Stage Migration Trend",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Percentages are conditional on the 'from stage' population each month.")

st.divider()

st.markdown("## Overlay Audit (Governance Evidence)")

if audit is None:
    st.info("Overlay audit not found. Run: python -m ecl_engine.overlay_audit")
else:
    st.markdown("### Overlay register (rules + allocated impact)")
    st.dataframe(audit, width="stretch")

    oid = st.selectbox("Select overlay_id to view impacted accounts", audit["overlay_id"].tolist())
    top_path = f"data/curated/overlay_top_accounts_{oid}.parquet"
    if glob.glob(top_path):
        top = pd.read_parquet(top_path)
        st.markdown("### Top accounts impacted by overlay allocation")
        st.dataframe(top, width="stretch", height=360)
    else:
        st.warning("Top accounts file not found for this overlay.")

st.divider()

st.markdown("## PD Model Validation (Phase 2)")

if os.path.exists("reports/pd_validation_metrics.csv"):
    met = pd.read_csv("reports/pd_validation_metrics.csv")
    st.dataframe(met, width="stretch")
else:
    st.info("No PD validation metrics found. Run: python -m ecl_engine.models.pd_train")

for img in ["reports/pd_roc.png", "reports/pd_calibration.png", "reports/pd_score_hist.png"]:
    if os.path.exists(img):
        st.image(img, caption=img)

st.divider()

# Drilldown
st.markdown("### Account Drilldown")
acct = st.text_input("Enter account_id (e.g., A0000123...)")
if acct:
    # base row from reported ECL file
    sub = ecl_ov[ecl_ov["account_id"] == acct].copy()

    if sub.empty:
        st.warning("No account found in current ASOF output.")
    else:
        st.markdown("#### Reported numbers (post-overlay)")
        row = sub.iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stage", int(row["stage"]))
        c2.metric("Balance", f"{float(row['balance']):,.2f}")
        c3.metric("Overlay", f"{float(row['overlay_amount']):,.2f}")
        c4.metric("Reported ECL", f"{float(row['ecl_post_overlay']):,.2f}")

        st.dataframe(sub, width="stretch")

        if int(row["stage"]) == 3 and "pv_recoveries" in sub.columns:
            st.markdown("#### Stage 3 Workout (recoveries-based)")
            st.write(f"EAD@Default: {float(row.get('ead_default', 0.0)):,.2f}")
            st.write(f"PV Recoveries: {float(row.get('pv_recoveries', 0.0)):,.2f}")
            st.write(f"Workout LGD: {float(row.get('workout_lgd', 0.0)):.2%}")
            st.write(f"Stage 3 ECL (workout): {float(row.get('ecl_stage3_workout', 0.0)):,.2f}")

        st.markdown("#### Explanation (drivers + assumptions)")
        if explain is None:
            st.info("Explain file not found. Run: python -m ecl_engine.explain")
        else:
            ex = explain[explain["account_id"] == acct].copy()
            if ex.empty:
                st.warning("No explanation row found for this account.")
            else:
                exr = ex.iloc[0]

                left, right = st.columns([1.2, 1])

                with left:
                    st.markdown("**Staging rationale**")
                    st.write(exr.get("stage_reason", "N/A"))
                    if "dpd" in ex.columns:
                        st.write(f"DPD: {exr.get('dpd', 'N/A')}")
                    # Original problematic code:
                    # st.write(f"Months to maturity: {int(exr.get('months_to_maturity', 0))}")

                    # Fixed version:
                    mtm = exr.get('months_to_maturity')
                    if pd.isna(mtm) or mtm < 0:
                        st.write("Months to maturity: **Matured** (past maturity date)")
                    elif mtm == 0:
                        st.write("Months to maturity: **< 1 month**")
                    else:
                        st.write(f"Months to maturity: **{int(mtm)} months**")

                    st.markdown("**EAD / LGD / Discounting**")
                    st.write(f"EAD rule: {exr.get('ead_rule','N/A')}")
                    st.write(f"EAD0: {float(exr.get('ead0', 0.0)):,.2f}")
                    st.write(f"Avg EAD (next 12m): {float(exr.get('ead12_avg', 0.0)):,.2f}")
                    st.write(f"LGD base: {float(exr.get('lgd_base', 0.0)):.2%}")
                    st.write(f"EIR (discount rate): {float(exr.get('eir', 0.0)):.2%}")

                with right:
                    st.markdown("**PD (PIT) summary**")
                    st.write(f"TTC PD (annual): {float(exr.get('ttc_pd_annual', 0.0)):.2%}")
                    st.write(f"PIT PD m1 Base: {float(exr.get('pit_pd_m1_base', 0.0)):.2%}")
                    st.write(f"PIT PD m1 Downside: {float(exr.get('pit_pd_m1_downside', 0.0)):.2%}")
                    st.write(f"Cumulative PD 12m Base: {float(exr.get('pit_cum_pd12_base', 0.0)):.2%}")
                    st.write(
                        f"Cumulative PD 12m Downside: {float(exr.get('pit_cum_pd12_downside', 0.0)):.2%}"
                    )

                    if "overlay_audit" in ex.columns:
                        st.markdown("**Overlay tags**")
                        st.code(str(exr.get("overlay_audit", "")))

                st.markdown("#### Full explanation row")
                st.dataframe(ex.T, width="stretch")
