# IFRS 9 ECL Engine

A production-grade Expected Credit Loss (ECL) calculation engine implementing the IFRS 9 three-stage impairment model.

![Dashboard Screenshot](docs/dashboard_preview.png)

## 🎯 Problem Statement

IFRS 9 requires financial institutions to recognize credit losses on a **forward-looking basis** using Expected Credit Losses (ECL). This engine implements:

- **Three-stage impairment model** (Stage 1: 12-month ECL | Stage 2/3: Lifetime ECL)
- **Multi-scenario forecasting** (Base, Upside, Downside with probability weighting)
- **PD, LGD, and EAD modeling** with macroeconomic linkage
- **Management overlays** for expert judgment adjustments
- **DCF-based ECL projections** with discounting
- **Governance dashboards** with full audit trails

## 🏗️ Architecture
```
┌─────────────────┐
│  Synthetic Data │
│   Generation    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Staging      │ ◄── Policy Rules (DPD, SICR, LCR, Cure)
│   (3-Stage)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   PD Modeling   │ ◄── Logistic Regression + Calibration
│ (Train/Score/   │
│    Anchor)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ECL Engine    │ ◄── Macro Scenarios + LGD + EAD
│  (Core Calc)    │
└────────┬────────┘
         │
         ├──────────┐
         ▼          ▼
┌─────────────┐   ┌─────────────┐
│  Overlays   │   │  DCF ECL    │
│ (Mgmt Adj)  │   │  (Enhanced) │
└──────┬──────┘   └──────┬──────┘
       │                 │
       └────────┬────────┘
                ▼
┌────────────────────────┐
│  Explain + Validation  │
│    + Dashboards        │
└────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data
```bash
python -m ecl_engine.data.make_synthetic --n_accounts 8000 --months 84
```

This creates:
- `data/curated/accounts.parquet` (8,000 accounts)
- `data/curated/performance_monthly.parquet` (84 months history)
- `data/curated/macro_scenarios_monthly.parquet` (Base/Upside/Downside)

### 3. Run Core Pipeline
```bash
python -m ecl_engine.pipeline --asof 2024-12-31
```

This runs:
- Staging (SICR classification)
- PD model training & scoring
- PD anchor calibration
- ECL calculation
- Overlay application
- Diagnostics & explain

### 4. Run Enhanced Pipeline (Optional)
```bash
python -m ecl_engine.pipeline_future --asof 2024-12-31
```

This adds:
- DCF-based ECL projections
- PD monitoring
- ECL backtesting

### 5. Launch Dashboard
```bash
streamlit run dashboards/app.py
```

Navigate to `http://localhost:8501`

## 📊 Key Outputs

| File | Description |
|------|-------------|
| `ecl_output_asof_<date>.parquet` | Core ECL results by account |
| `ecl_with_overlays.parquet` | Post-overlay reported ECL |
| `ecl_dcf_asof_<date>.parquet` | DCF-enhanced ECL projections |
| `account_explain_asof_<date>.parquet` | Full explainability per account |
| `scenario_contribution_<date>.csv` | Scenario ECL breakdown |
| `stage_migration.parquet` | Month-over-month stage transitions |
| `pd_validation_metrics.csv` | PD model performance (AUC, KS) |

## 🧪 Testing
```bash
# Run all tests
pytest -v

# Run specific test
pytest tests/test_core_invariants.py::test_scenario_weights -v

# With coverage
pytest --cov=src/ecl_engine tests/
```

## ⚙️ Configuration

All parameters are in `configs/`:

- **`policy.yml`**: Staging rules (DPD thresholds, SICR triggers, cure periods)
- **`portfolio_params.yml`**: Risk parameters (LGD, scenario weights, macro sensitivity)
- **`dcf_params.yml`**: DCF projection settings (horizon, discounting)
- **`workout_lgd.yml`**: Stage 3 recovery assumptions

## 📈 Model Assumptions

See [MODEL_ASSUMPTIONS.md](MODEL_ASSUMPTIONS.md) for detailed technical documentation.

**Key Assumptions:**
- **PD**: Constant monthly hazard rate from 12m cumulative PD
- **LGD**: Base LGD × scenario multiplier (downturn adjustment)
- **Stage 3**: Exponential half-life recovery curves with scenario linkage
- **Macro**: Logit-space PD adjustment using z-scored macro variables
- **Scenario weights**: 60% Base, 20% Upside, 20% Downside (configurable)

## ⚠️ Limitations

See [LIMITATIONS.md](LIMITATIONS.md) for known constraints.

**Key Limitations:**
- Synthetic data only (not validated on real portfolios)
- Simplified PD model (logistic regression, no neural networks)
- No prepayment modeling
- No explicit correlation modeling (copulas)
- Single-currency (no FX risk)

## 📂 Project Structure
```
ifrs9-ecl-engine/
├── configs/              # YAML configuration files
├── dashboards/           # Streamlit dashboard
├── data/
│   └── curated/         # Generated datasets (gitignored)
├── docs/                # Screenshots & diagrams
├── models/              # Trained ML models (gitignored)
├── reports/             # Validation outputs (gitignored)
├── src/ecl_engine/      # Core engine code
│   ├── data/           # Synthetic data generation
│   ├── models/         # PD, LGD modeling
│   ├── utils/          # Shared utilities
│   └── validation/     # Backtesting & monitoring
├── tests/               # Pytest test suite
├── .github/workflows/   # CI/CD pipelines
├── requirements.txt
└── README.md
```

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest -v`)
4. Lint code (`ruff check . && black .`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built with:
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [scikit-learn](https://scikit-learn.org/) - PD modeling
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Streamlit](https://streamlit.io/) - Dashboard framework

---

**Author**: [@k9kanishk](https://github.com/k9kanishk)  
**Project**: IFRS 9 ECL Engine  
**Status**: ✅ Production-ready portfolio project
