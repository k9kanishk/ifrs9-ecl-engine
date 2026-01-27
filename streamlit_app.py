"""
IFRS 9 ECL Engine - Streamlit Cloud Entry Point

Handles:
1. Python path setup (critical for Streamlit Cloud)
2. Data check and generation
3. Dashboard loading
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ============================================================
# CRITICAL: Add src/ to Python path BEFORE any imports
# ============================================================
ROOT_DIR = Path(__file__).parent.absolute()
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
DATA_DIR = ROOT_DIR / "data" / "curated"
REQUIRED_FILES = [
    "ecl_output_asof_2024-12-31.parquet",
    "ecl_with_overlays.parquet",
    "staging_output.parquet",
    "accounts.parquet",
]


def data_exists() -> bool:
    """Check if required data files exist."""
    return all((DATA_DIR / f).exists() for f in REQUIRED_FILES)


def run_pipeline_step(cmd: list[str], description: str) -> tuple[bool, str]:
    """Run a pipeline step with error handling."""
    try:
        # Set PYTHONPATH for subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC_DIR) + ":" + env.get("PYTHONPATH", "")
        
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=180,  # 3 minute timeout per step
            cwd=str(ROOT_DIR),
            env=env,
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        return False, error_msg[-1000:]
    except subprocess.TimeoutExpired:
        return False, "Timeout after 180 seconds"
    except Exception as e:
        return False, str(e)


def generate_data():
    """Generate all required data."""
    py = sys.executable
    
    steps = [
        ("📊 Generating synthetic accounts...", 
         [py, "-m", "ecl_engine.data.make_synthetic", "--n_accounts", "1000", "--months", "84"]),
        ("🏗️ Running staging...", 
         [py, "-m", "ecl_engine.staging"]),
        ("🧠 Training PD model...", 
         [py, "-m", "ecl_engine.models.pd_train"]),
        ("📈 Scoring PD...", 
         [py, "-m", "ecl_engine.models.pd_score"]),
        ("⚓ Calibrating PD...", 
         [py, "-m", "ecl_engine.models.pd_anchor", "--level", "segment"]),
        ("💰 Calculating ECL...", 
         [py, "-m", "ecl_engine.ecl", "--asof", "2024-12-31"]),
        ("🎯 Applying overlays...", 
         [py, "-m", "ecl_engine.run_ecl_with_overlays", "--asof", "2024-12-31"]),
        ("🔍 Driver decomposition...", 
         [py, "-m", "ecl_engine.driver_decomp", "--asof", "2024-12-31"]),
        ("📝 Generating explanations...", 
         [py, "-m", "ecl_engine.explain", "--asof", "2024-12-31", "--n", "0"]),
        ("🔄 Stage migration...", 
         [py, "-m", "ecl_engine.stage_migration"]),
        ("🔐 Overlay audit...", 
         [py, "-m", "ecl_engine.overlay_audit"]),
        ("💵 DCF ECL...", 
         [py, "-m", "ecl_engine.dcf_ecl", "--asof", "2024-12-31"]),
    ]
    
    progress = st.progress(0)
    status = st.empty()
    error_container = st.empty()
    
    for i, (desc, cmd) in enumerate(steps):
        status.markdown(f"**{desc}**")
        success, output = run_pipeline_step(cmd, desc)
        
        if not success:
            error_container.error(f"❌ {desc} failed:\n```\n{output}\n```")
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
    
    **⏱️ Estimated time: ~60 seconds**
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🎬 Generate Data & Launch", type="primary", use_container_width=True):
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
    
    # Debug info (helpful for troubleshooting)
    with st.expander("🔧 Debug Info"):
        st.code(f"""
Python: {sys.executable}
Version: {sys.version}
Working Dir: {ROOT_DIR}
Src Dir: {SRC_DIR}
Src Exists: {SRC_DIR.exists()}
Data Dir: {DATA_DIR}
sys.path[0:3]: {sys.path[0:3]}
        """)


def run_dashboard():
    """Run the main dashboard."""
    app_path = ROOT_DIR / "dashboards" / "app.py"
    
    if not app_path.exists():
        st.error(f"❌ Dashboard file not found: {app_path}")
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
    
    # Execute with proper globals
    exec_globals = globals().copy()
    exec_globals['__file__'] = str(app_path)
    exec('\n'.join(filtered_lines), exec_globals)


# ============================================================
# MAIN
# ============================================================
if data_exists():
    run_dashboard()
else:
    show_setup_wizard()
