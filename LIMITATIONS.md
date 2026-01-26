# Known Limitations

This IFRS 9 ECL engine is a **portfolio demonstration project**. While production-ready in terms of code quality and architecture, it has the following known limitations:

## Data Limitations

1. **Synthetic Data Only**
   - All data is artificially generated
   - Not validated against real credit portfolios
   - Default patterns are stylized, not empirically calibrated

2. **No Historical Default Data**
   - TTC PDs are assigned based on segment averages
   - No backtesting against actual default history
   - Calibration assumes stylized default rates

3. **Limited Macro History**
   - Only 84 months of synthetic macro data
   - Scenarios are illustrative, not econometric forecasts
   - No stress-testing against historical crises (2008, COVID-19)

## Model Limitations

1. **Simplified PD Model**
   - Basic logistic regression (no ensemble methods, neural networks)
   - Limited feature engineering (no transaction patterns, behavioral variables)
   - No time-varying coefficients
   - No explicit credit scoring (just TTC PD + macro adjustment)

2. **Constant Hazard Rate Assumption**
   - Assumes default probability is constant over time
   - Reality: Default risk varies by loan age, season, portfolio vintage
   - More sophisticated models use term structures of PD

3. **No Prepayment Modeling**
   - Assumes loans run to maturity
   - Reality: Borrowers prepay, especially in declining rate environments
   - Affects EAD and effective maturity

4. **No Correlation Modeling**
   - Portfolio losses treated as independent across accounts
   - Reality: Defaults cluster during downturns (systemic risk)
   - Advanced models use copulas or factor models

5. **Single-Currency**
   - No foreign exchange risk
   - No cross-border exposures
   - No currency-specific macro scenarios

## Regulatory Limitations

1. **Not Audited**
   - No independent model validation
   - No regulatory approval
   - Not suitable for actual financial reporting without extensive review

2. **Missing Basel Alignment**
   - No PD/LGD term structures (Basel IRB requirements)
   - No economic capital calculations (Basel III)
   - No stress testing (CCAR, EBA)

3. **Simplified Governance**
   - Overlay process is illustrative only
   - No committee approval workflows
   - No challenger models

## Technical Limitations

1. **Performance**
   - Not optimized for millions of accounts
   - Matrix operations could be parallelized
   - No GPU acceleration for ML models

2. **Scalability**
   - Single-machine implementation
   - No distributed computing (Spark, Dask)
   - In-memory processing limits portfolio size

3. **Missing Features**
   - No collateral valuation engine
   - No forbearance tracking
   - No COVID-19 style policy interventions
   - No IFRS 17 integration (insurance contracts)

## What This Project IS

✅ **Architecture-complete**: All major IFRS 9 components implemented  
✅ **Code quality**: Production-grade structure, tests, CI/CD  
✅ **Governance-ready**: Audit trails, overlays, validation framework  
✅ **Demonstrable**: Working dashboard, explainability, documentation  

## What This Project IS NOT

❌ **Production ECL system**: Requires real data, validation, and regulatory approval  
❌ **Research-grade model**: Needs econometric rigor, backtesting, peer review  
❌ **Plug-and-play solution**: Must be adapted to specific bank's portfolio and policies  

---

**Recommendation for Production Use**:  
If adapting this engine for actual financial reporting:
1. Replace synthetic data with real portfolios
2. Validate PD/LGD models on historical default data
3. Engage external auditors and model validators
4. Add missing features (collateral, prepayment, correlation)
5. Implement full governance (committees, challenger models, documentation)

**This project demonstrates technical capability. Real-world IFRS 9 implementation requires domain expertise, regulatory engagement, and significant validation effort.**
