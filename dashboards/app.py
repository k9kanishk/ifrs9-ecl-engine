from __future__ import annotations

import glob

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="IFRS 9 ECL Dashboard", layout="wide")

st.title("IFRS 9 ECL Engine — Dashboard")
st.caption("Stage, ECL, Scenarios, Overlays — built from reproducible pipeline outputs.")


@st.cache_data
def load_latest_outputs():
    ecl_paths = sorted(glob.glob("data/curated/ecl_output_asof_*.parquet"))
    if not ecl_paths:
        raise FileNotFoundError("No ecl_output_asof_*.parquet found. Run: python src/ecl_engine/ecl.py")

    latest_ecl_path = ecl_paths[-1]
    asof = latest_ecl_path.split("_")[-1].replace(".parquet", "")

    ecl = pd.read_parquet(latest_ecl_path)
    ecl_ov = pd.read_parquet("data/curated/ecl_with_overlays.parquet")

    scen_path = f"data/curated/scenario_contribution_{asof}.csv"
    drv_path = f"data/curated/driver_decomposition_{asof}.csv"
    qc_path = "data/curated/staging_qc_summary.parquet"

    scen = pd.read_csv(scen_path) if glob.glob(scen_path) else None
    drv = pd.read_csv(drv_path) if glob.glob(drv_path) else None
    qc = pd.read_parquet(qc_path) if glob.glob(qc_path) else None

    mig_path = "data/curated/stage_migration.parquet"
    mig = pd.read_parquet(mig_path) if glob.glob(mig_path) else None

    return asof, ecl, ecl_ov, scen, drv, qc, mig


asof, ecl, ecl_ov, scen, drv, qc, mig = load_latest_outputs()

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
    st.plotly_chart(fig, use_container_width=True)

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
    st.plotly_chart(fig2, use_container_width=True)

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
    st.dataframe(stage_tbl, use_container_width=True)

with c2:
    st.markdown("### ECL by Segment (post-overlay)")
    seg_tbl = (
        f.groupby("segment")[["ecl_pre_overlay", "overlay_amount", "ecl_post_overlay"]]
        .sum()
        .sort_values("ecl_post_overlay", ascending=False)
        .reset_index()
    )
    st.dataframe(seg_tbl, use_container_width=True, height=360)

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
        st.dataframe(scen_view, use_container_width=True)

with c4:
    st.markdown("### Driver Sensitivities by Segment (post-overlay)")
    if drv is None:
        st.info("Driver decomposition file not found. Run: python -m ecl_engine.driver_decomp")
    else:
        drv_view = drv.copy()
        # keep the biggest segments only
        drv_view = drv_view.sort_values("ecl_reported", ascending=False).head(15)
        st.dataframe(drv_view, use_container_width=True, height=360)

st.divider()

# Row 3: Stage mix over time (from staging QC summary)
st.markdown("### Stage Counts Over Time (QC)")
if qc is None:
    st.info("Staging QC summary not found. Run: python src/ecl_engine/staging.py")
else:
    qc2 = qc[qc["segment"].isin(sel_segments)].copy()
    # aggregate across segments for a clean view
    qc_total = qc2.groupby(["snapshot_date", "stage"], as_index=False)["n_accounts"].sum()
    # pivot for chart
    pivot = qc_total.pivot(index="snapshot_date", columns="stage", values="n_accounts").fillna(0).sort_index()
    st.line_chart(pivot)

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
        st.dataframe(show.round(2), use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "Rows = stage last month (t-1), columns = stage this month (t). "
    "Diagonal = stable accounts. "
    "1→2 = SICR migration, 2→1 = cure (improvement). "
    "Any 1/2→3 = defaults. "
    "This is a standard governance monitoring view for staging stability."
)

st.divider()

# Drilldown
st.markdown("### Account Drilldown")
acct = st.text_input("Enter account_id (e.g., A0000123...)")
if acct:
    sub = ecl_ov[ecl_ov["account_id"] == acct].copy()
    if sub.empty:
        st.warning("No account found in current ASOF output.")
    else:
        st.dataframe(sub.T, use_container_width=True)
