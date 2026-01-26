from __future__ import annotations

import argparse
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="ASOF date YYYY-MM-DD (optional)")
    ap.add_argument("--skip-pd-train", action="store_true")
    ap.add_argument("--skip-pd-anchor", action="store_true")
    ap.add_argument("--skip-synth-data", action="store_true", help="Skip synthetic data generation")
    args = ap.parse_args()

    py = sys.executable

    # Generate synthetic data if not skipped
    if not args.skip_synth_data:
        _run([py, "-m", "ecl_engine.data.make_synthetic"])

    # Staging (uses all performance history)
    _run([py, "-m", "ecl_engine.staging"])

    # Core pipeline
    if not args.skip_pd_train:
        _run([py, "-m", "ecl_engine.models.pd_train"])

    # PD scoring (pass ASOF if provided)
    if args.asof:
        _run([py, "-m", "ecl_engine.models.pd_score", "--asof", args.asof])
    else:
        _run([py, "-m", "ecl_engine.models.pd_score"])

    # PD anchor calibration (pass ASOF if provided)
    if not args.skip_pd_anchor:
        if args.asof:
            _run([py, "-m", "ecl_engine.models.pd_anchor", "--level", "segment", "--asof", args.asof])
        else:
            _run([py, "-m", "ecl_engine.models.pd_anchor", "--level", "segment"])

    # ECL calculation (CRITICAL: pass ASOF)
    if args.asof:
        _run([py, "-m", "ecl_engine.ecl", "--asof", args.asof])
    else:
        _run([py, "-m", "ecl_engine.ecl"])

    # Post-ECL steps (CRITICAL: pass ASOF through!)
    if args.asof:
        _run([py, "-m", "ecl_engine.run_ecl_with_overlays", "--asof", args.asof])
        _run([py, "-m", "ecl_engine.driver_decomp", "--asof", args.asof])
        _run([py, "-m", "ecl_engine.stage3_summary", "--asof", args.asof])
        _run([py, "-m", "ecl_engine.explain", "--asof", args.asof, "--n", "0"])
    else:
        _run([py, "-m", "ecl_engine.run_ecl_with_overlays"])
        _run([py, "-m", "ecl_engine.driver_decomp"])
        _run([py, "-m", "ecl_engine.stage3_summary"])
        _run([py, "-m", "ecl_engine.explain", "--n", "0"])

    # Stage migration (doesn't need ASOF - processes all history)
    _run([py, "-m", "ecl_engine.stage_migration"])

    # Overlay audit
    _run([py, "-m", "ecl_engine.overlay_audit"])

    print("\n✅ Core pipeline complete.")
    asof_msg = f" --asof {args.asof}" if args.asof else ""
    print(f"Next: python -m ecl_engine.pipeline_future{asof_msg}")
    print("Then: streamlit run dashboards/app.py")


if __name__ == "__main__":
    main()
