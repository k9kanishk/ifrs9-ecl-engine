#!/bin/bash
set -e

# Install missing deps
pip install -q ruff black pytest pytest-cov

# Run just the core pipeline (no optional --asof on pd_anchor)
echo "Running core pipeline..."
python -m ecl_engine.models.pd_anchor --level segment  # No --asof!
python -m ecl_engine.ecl --asof 2024-12-31
python -m ecl_engine.run_ecl_with_overlays --asof 2024-12-31
python -m ecl_engine.driver_decomp --asof 2024-12-31
python -m ecl_engine.stage3_summary --asof 2024-12-31
python -m ecl_engine.explain --asof 2024-12-31 --n 0
python -m ecl_engine.stage_migration
python -m ecl_engine.overlay_audit

echo "Running future pipeline..."
python -m ecl_engine.dcf_ecl --asof 2024-12-31
python -m ecl_engine.validation.pd_monitoring --asof 2024-12-31
python -m ecl_engine.validation.ecl_backtest --asof 2024-12-31

echo "Verifying ASOF..."
python -m ecl_engine.verify_asof 2024-12-31

echo "Running tests..."
pytest -v

echo "✅ Done!"
