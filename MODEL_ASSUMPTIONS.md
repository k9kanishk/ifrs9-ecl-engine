# Model Assumptions

## PD (Probability of Default)

### 1. TTC to PIT Conversion
- **TTC PD** (Through-The-Cycle): Long-run average default rate by segment
- **PIT PD** (Point-In-Time): Macro-adjusted current default probability
- **Formula**: `logit(PD_PIT) = logit(PD_TTC) + β₀ + β₁·unemployment_z + β₂·GDP_z + β₃·rate_z`
- **Coefficients**: 
  - Intercept: -0.25
  - Unemployment sensitivity: +0.25 (↑ unemployment → ↑ PD)
  - GDP sensitivity: -0.18 (↑ GDP → ↓ PD)
  - Policy rate sensitivity: +0.12 (↑ rates → ↑ PD)

### 2. 12-Month to Lifetime PD
- **Assumption**: Constant monthly hazard rate
- **Monthly hazard**: `h = 1 - (1 - PD_12m)^(1/12)`
- **Lifetime PD**: `PD_LT = 1 - (1 - h)^months_to_maturity`

### 3. Scenario PDs
- **Base scenario**: Central macro forecast
- **Upside**: Unemployment -0.7pp, GDP +0.8pp, Rates -0.3pp
- **Downside**: Unemployment +1.2pp, GDP -1.5pp, Rates +0.6pp
- **PD adjustment**: `logit(PD_scenario) = logit(PD_anchor) + 0.8 × macro_stress`

## LGD (Loss Given Default)

### 1. Stage 1 & 2 LGD
- **Base LGD by segment**:
  - Retail Mortgage: 20%
  - Corp Term Loan: 45%
  - SME Term Loan: 50%
  - Retail PL: 75%
  - Revolving: 80%
- **Downturn adjustment**: `LGD_scenario = LGD_base × exp(β_segment × stress)`
- **Betas** range from 0.10 (mortgages) to 0.30 (revolving)

### 2. Stage 3 Workout LGD
- **Recovery curve**: Exponential half-life (e.g., 50% recovered in 12 months for mortgages)
- **Collection costs**: 5-15% of recovered amount
- **Scenario linkage**:
  - Recovery rate: `Recovery_scen = Recovery_base × exp(-β × stress)` (lower in downside)
  - Recovery speed: `Half_life_scen = Half_life_base × (1 + 0.08 × max(stress, 0))` (slower in downside)
- **Discounting**: Using account-level EIR (Effective Interest Rate)

## EAD (Exposure at Default)

### 1. Term Loans
- **EAD = Current Balance** (no undrawn)

### 2. Revolving Facilities
- **EAD = Balance + CCF × Undrawn**
- **CCF** (Credit Conversion Factor): 75% (default)
- **Assumption**: Borrowers draw down 75% of unused limit before default

## Staging Logic

### Stage 1 (Performing)
- DPD < 30 days
- PD ratio < 2.0 (or segment-specific threshold)
- Absolute PD increase < 100 bps
- OR: Low Credit Risk exemption (PIT PD < 1%)

### Stage 2 (SICR - Significant Increase in Credit Risk)
- DPD ≥ 30 days (backstop), OR
- PD ratio ≥ 2.0 (current PD / origination PD), OR
- Absolute PD increase ≥ 100 bps, OR
- Qualitative watchlist flag

### Stage 3 (Credit-Impaired)
- DPD ≥ 90 days, OR
- Default flag = 1

### Cure Logic
- **Stage 2 → 1**: 3 consecutive months with DPD < 30
- **Stage 3 → 2**: 6 consecutive months with DPD < 30 AND no default flag

## DCF ECL (Optional Enhancement)

### Approach
1. Project monthly EAD curve (with optional decay)
2. Convert 12m cumulative PD to constant monthly hazard
3. Calculate default probability each future month using survival function
4. Compute discounted expected loss: `ECL_t = EAD_t × LGD × P(default in month t) × DF_t`
5. Sum over projection horizon (12 months for Stage 1, lifetime for Stage 2/3)

### Key Difference from Simple Formula
- **Simple**: `ECL = EAD × LGD × PD` (single point-in-time)
- **DCF**: Month-by-month cashflow projection with discounting

## Scenario Weighting

- **Probability weights**: 60% Base, 20% Upside, 20% Downside
- **Formula**: `ECL_weighted = 0.6 × ECL_Base + 0.2 × ECL_Upside + 0.2 × ECL_Downside`
- **Applied at**: Scenario-selected ECL level (after horizon rule)

## Management Overlays

### Methods
1. **Multiplicative**: `ECL_adjusted = ECL_base × factor` (e.g., 1.05 = +5%)
2. **Additive**: Fixed amount allocated pro-rata across segment

### Governance
- Requires reason code, owner, and approver
- Documented in overlay register with full audit trail
- Allocated impact tracked per account

---

**Note**: All assumptions are configurable via YAML files in `configs/`.
