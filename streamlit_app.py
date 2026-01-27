"""
IFRS 9 ECL Engine - Streamlit Cloud Entry Point

This file handles two scenarios:
1. If data exists (committed to repo): Load dashboard directly
2. If data missing: Show setup wizard to generate

For fastest Streamlit Cloud deployment, commit your data/ folder.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

# ============================================================
# PAGE CONFIG - Must be first Streamlit command
# ============================================================
st.set_page_config(
    page_title="IFRS 9 ECL Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DATA CHECK
# ============================================================
DATA_DIR = Path("data/curated")
REQUIRED_FILES = [
    "ecl_output_asof_2024-12-31.parquet",
    "ecl_with_overlays.parquet",
    "staging_output.parquet",
    "accounts.parquet",
]


def data_exists() -> bool:
    """Check if required data files exist."""
    return all((DATA_DIR / f).exists() for f in REQUIRED_FILES)


def run_pipeline_step(cmd: list[str], description: str) -> bool:
    """Run a pipeline step with error handling."""
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=120
        )
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"❌ {description} failed:\n```\n{e.stderr[-500:]}\n```")
        return False
    except subprocess.TimeoutExpired:
        st.error(f"❌ {description} timed out")
        return False


def generate_data():
    """Generate all required data."""
    steps = [
        ("📊 Generating synthetic accounts...", 
         [sys.executable, "-m", "ecl_engine.data.make_synthetic", "--n_accounts", "1000", "--months", "84"]),
        ("🏗️ Running staging...", 
         [sys.executable, "-m", "ecl_engine.staging"]),
        ("🧠 Training PD model...", 
         [sys.executable, "-m", "ecl_engine.models.pd_train"]),
        ("📈 Scoring PD...", 
         [sys.executable, "-m", "ecl_engine.models.pd_score"]),
        ("⚓ Calibrating PD...", 
         [sys.executable, "-m", "ecl_engine.models.pd_anchor", "--level", "segment"]),
        ("💰 Calculating ECL...", 
         [sys.executable, "-m", "ecl_engine.ecl", "--asof", "2024-12-31"]),
        ("🎯 Applying overlays...", 
         [sys.executable, "-m", "ecl_engine.run_ecl_with_overlays", "--asof", "2024-12-31"]),
        ("🔍 Driver decomposition...", 
         [sys.executable, "-m", "ecl_engine.driver_decomp", "--asof", "2024-12-31"]),
        ("📝 Generating explanations...", 
         [sys.executable, "-m", "ecl_engine.explain", "--asof", "2024-12-31", "--n", "0"]),
        ("🔄 Stage migration...", 
         [sys.executable, "-m", "ecl_engine.stage_migration"]),
        ("🔐 Overlay audit...", 
         [sys.executable, "-m", "ecl_engine.overlay_audit"]),
        ("💵 DCF ECL...", 
         [sys.executable, "-m", "ecl_engine.dcf_ecl", "--asof", "2024-12-31"]),
    ]
    
    progress = st.progress(0)
    status = st.empty()
    
    for i, (desc, cmd) in enumerate(steps):
        status.markdown(f"**{desc}**")
        if not run_pipeline_step(cmd, desc):
            return False
        progress.progress((i + 1) / len(steps))
    
    status.markdown("**✅ All steps complete!**")
    return True


def show_setup_wizard():
    """Show setup UI when data is missing."""
    st.title("🚀 IFRS 9 ECL Engine")
    st.subheader("First-Time Setup")
    
    st.info("""
    **Welcome!** This dashboard needs to generate synthetic data before it can run.
    
    This will create:
    - 1,000 loan accounts across 5 segments
    - 84 months of performance history
    - Full ECL calculations with 3 scenarios
    
    **⏱️ Estimated time: ~45 seconds**
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🎬 Generate Data & Launch", type="primary", use_container_width=True):
            with st.spinner("Running ECL pipeline..."):
                success = generate_data()
            
            if success:
                st.success("✅ Setup complete!")
                st.balloons()
                st.cache_data.clear()
                st.rerun()
    
    with col2:
        st.markdown("""
        **What gets generated:**
        - Synthetic loan portfolio
        - PD/LGD/EAD models  
        - ECL by scenario (Base/Upside/Downside)
        - Management overlays
        - Governance reports
        """)


def run_dashboard():
    """Run the main dashboard."""
    app_path = Path("dashboards/app.py")
    
    if not app_path.exists():
        st.error("❌ Dashboard file not found: dashboards/app.py")
        return
    
    # Read the dashboard code
    code = app_path.read_text()
    
    # Remove set_page_config to avoid duplicate call
    lines = code.split('\n')
    filtered_lines = []
    in_page_config = False
    paren_depth = 0
    
    for line in lines:
        if 'st.set_page_config' in line:
            in_page_config = True
            paren_depth = line.count('(') - line.count(')')
            if paren_depth == 0:
                in_page_config = False
            continue
        
        if in_page_config:
            paren_depth += line.count('(') - line.count(')')
            if paren_depth <= 0:
                in_page_config = False
            continue
        
        filtered_lines.append(line)
    
    exec('\n'.join(filtered_lines), globals())


# ============================================================
# MAIN
# ============================================================
if data_exists():
    run_dashboard()
else:
    show_setup_wizard()
