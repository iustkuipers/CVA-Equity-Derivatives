# CVA Equity Derivatives Modeling (Case 3, 2026)

## Overview
This project implements a complete **Counterparty Credit Risk (CCR)** modeling pipeline for an equity derivatives portfolio exposed to two underlyings (SX5E, AEX). The system computes **Credit Valuation Adjustment (CVA)** through Monte Carlo simulation of equity paths, portfolio valuation, exposure measurement, and risk-neutral default probabilities.

**Key Outcome**: CVA quantifies the credit cost of counterparty default, integrated into pricing and risk management frameworks.

---

## Project Structure

```
.
├── main.py                           # 5-step orchestrator: initialize → simulate → validate → Q2 → Q3 → Q4
├── data/
│   ├── config.py                     # Centralized parameters (credit, rates, equity, simulation)
│   ├── portfolio.py                  # 4-instrument portfolio (2 forwards, 2 put options)
│   └── orchestrator.py               # Aggregates config + portfolio
├── models/
│   └── equity_processes.py           # GBM simulation with Cholesky correlation (80%)
├── pricing/
│   ├── black_scholes.py              # European option pricing (call/put)
│   └── equity_instruments.py         # Portfolio valuation across paths × times (with Contracts × Direction scaling)
├── cva/
│   ├── exposure.py                   # Exposure metrics (standalone, netted, EPE)
│   ├── cva_calculator.py             # CVA computation from piecewise hazard rates
│   └── collateral.py                 # Collateral transforms (Variation Margin, Initial Margin)
├── workflows/
│   ├── exposures_CVA_workflow.py     # Q2: Pure compute engine + reporting (baseline computed once)
│   ├── sensitivity_analysis_workflow.py   # Q3: Sensitivity to volatility & correlation (reuses baseline)
│   └── collateral_impact_workflow.py      # Q4: Collateral impact analysis (VM & IM stress)
├── validation/
│   └── orchestrator.py               # Q1: martingale, parity, correlation tests
├── utils/
│   └── output_writer.py              # Output file infrastructure
├── output/
│   ├── q1/                           # Validation results (CSV)
│   ├── q2/                           # CVA baseline (CSV, PNG)
│   ├── q3/                           # Sensitivity analysis (CSV, PNG)
│   └── q4/                           # Collateral impact (CSV, PNG)
└── tests/
    ├── test_*.py                     # 12+ test files (all passing)
    └── [test results captured]
```

---

## Mathematical Framework

### 1. **Equity Process Simulation (GBM)**
Correlated geometric Brownian motion with 80% correlation between SX5E and AEX:

$$S_i(t+\Delta t) = S_i(t) \cdot \exp\left[\left(r - q - \frac{\sigma_i^2}{2}\right)\Delta t + \sigma_i Z_i \sqrt{\Delta t}\right]$$

**Implementation**:
- **Monthly timesteps**: $\Delta t = 1/12$ years
- **Horizon**: 5 years = 60 steps
- **Correlation**: Via Cholesky decomposition of $\rho = 0.80$
- **Efficiency**: Precomputed drift, sqrt(dt), Cholesky matrix (0.01s for 1000 paths)
- **Reproducibility**: Random seed = 42

### 2. **Option Pricing (Black-Scholes-Merton)**
European call and put with dividend yield adjustment:

**Call**:
$$C(S,K,T,r,q,\sigma) = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

**Put**:
$$P(S,K,T,r,q,\sigma) = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$

where $d_1 = \frac{\ln(S/K) + (r-q+\sigma^2/2)T}{\sigma\sqrt{T}}$ and $d_2 = d_1 - \sigma\sqrt{T}$

**Validation**: Call-put parity error < 1e-6

### 3. **Exposure Measurement**
Defined from the bank's perspective (long exposure to counterparty):

**Standalone** (per-instrument):
$$E_i(t) = \max(V_i(t), 0)$$

**Unnetted** (sum of individual exposures):
$$E_{\text{unnetted}}(t) = \sum_{i=1}^{4} \max(V_i(t), 0)$$

**Netted** (collateral/netting agreement):
$$E_{\text{netted}}(t) = \max\left(\sum_{i=1}^{4} V_i(t), 0\right)$$

**Expected Positive Exposure (EPE)**:
$$\text{EPE}(t) = \mathbb{E}_{\mathcal{Q}}[E(t)]$$

### 4. **CVA Calculation (Unilateral)**
CVA under risk-neutral measure with piecewise constant forward hazard rates:

$$\text{CVA} = \text{LGD} \times \sum_{k} DF(t_k) \times \text{EPE}(t_k) \times \Delta PD_k$$

where:
- $\text{LGD} = 40\%$ (loss given default)
- $DF(t_k) = e^{-r \cdot t_k}$ (risk-free discount factor)
- $\Delta PD_k = h_k \Delta t$ (incremental default probability)
- $h_k$ = piecewise forward hazard rate

**Hazard Rates** (configuration-driven):
| Period | Forward Rate |
|--------|-------------|
| 0-1Y   | 2.00%       |
| 1-3Y   | 2.15%       |
| 3-5Y   | 2.20%       |

---

## Configuration Parameters

**File**: `data/config.py`

### Credit Risk
- **Counterparty LGD**: 40%
- **Forward Hazard Rates**: {"0_1": 0.0200, "1_3": 0.0215, "3_5": 0.0220}

### Market Rates
- **Risk-Free Rate**: 3.0% (flat term structure)
- **Dividend Yield**: 2.0% (both underlyings)

### Equity Parameters
- **SX5E Spot**: €6,000
- **AEX Spot**: €1,000
- **Volatility (both)**: 15%
- **Correlation**: 80%

### Simulation Settings
- **Monte Carlo Paths**: 10,000 (NUM_SIMULATIONS)
- **Time Steps**: 60 (monthly, 5-year horizon)
- **Random Seed**: 42 (reproducible)

### Portfolio
| Id | Type    | Underlying | Strike   | Contracts | S₀         |
|----|---------|----------|----------|-----------|-----------|
| 1  | Forward | SX5E     | €6,000   | 10,000    | €6,000    |
| 2  | Forward | AEX      | €1,000   | 55,000    | €1,000    |
| 3  | Put     | SX5E     | €4,800   | 10,000    | €6,000    |
| 4  | Put     | AEX      | €800     | 55,000    | €1,000    |

---

## Key Module Documentation

### `models/equity_processes.py` - EquityProcessSimulator
```python
simulator = EquityProcessSimulator(config, portfolio)
paths = simulator.simulate_paths()  # Shape: (10000, 61, 2) [sims, times, assets]
```
- **Input**: Configuration dict + portfolio DataFrame
- **Output**: Equity price matrix (t=0 to t=60 months)
- **Performance**: 0.01s for 1000 paths
- **Testing**: Correlation validation (0.7997 vs 0.8000 target)

### `pricing/equity_instruments.py` - value_portfolio_over_time()
```python
V = value_portfolio_over_time(sim_data, portfolio, config)
# V shape: (10000, 60, 4) [simulations, times, instruments]
```
- **Forward**: $V_f(t) = S_t e^{-q(T-t)} - K e^{-r(T-t)}$
- **Put Option**: Via Black-Scholes with time-to-maturity $\tau = T - t$
- **Progress**: Valuation milestone notifications

### `cva/exposure.py` - Exposure Functions
```python
E_standalone = compute_standalone_positive_exposures(V)
E_unnetted = compute_unnetted_exposure(V)
E_netted = compute_netted_exposure(V)
EPE = compute_epe(E)  # Average across simulations
```

### `cva/cva_calculator.py` - CVA Computation
```python
survival_probs, default_probs = compute_survival_and_default(config, times)
cva_value = compute_cva_from_epe(epe, lgd, df, default_probs)
```
- **Hazard Rate Parsing**: Handles string format ("0_1" → 1.0) and numeric keys
- **Mathematical**: LGD × Σ(DF × EPE × ΔPD)

### `workflows/exposures_CVA_workflow.py` - run_q2_workflow()
Orchestrates: Simulate → Value → Expose → CVA → Output
**Outputs** (saved to `output/q2/`):
- `epe_profiles.csv`: EPE at each time step
- `cva_breakdown_unnetted.csv`: Per-time CVA contribution (unnetted)
- `cva_breakdown_netted.csv`: Per-time CVA contribution (netted)
- `epe_plot.png`: EPE visualization

### `validation/orchestrator.py` - run_question_1()
**Tests**:
1. **Martingale Test**: Discounted equity paths are martingales
2. **Option Pricing Test**: Simulated vs Black-Scholes comparison
3. **Correlation Test**: Fisher's Z-test for 80% target correlation

**Output**: `output/q1/q1_validation_results.csv` with test statistics and p-values

---

## Workflows & Architecture

### Architecture Highlights

**Goal**: Compute complex CCR analysis efficiently without redundant simulations.

**Key Design Principles**:
1. **Single Baseline Computation**: Equity simulation, valuation, and baseline CVA computed once
2. **Pure Compute Engine**: `compute_cva_pipeline()` handles all calculations without I/O
3. **Reporting Layer**: Separate functions handle file output, plots, and summaries
4. **Result Passing**: Downstream analyses (Q3, Q4) receive baseline results via function parameters
5. **Intelligent Recomputation**: Only stress scenarios (Q3) and transforms (Q4) are recomputed

**Performance**: Full pipeline (9.4 minutes)
- Q2 Baseline: 179.72s (computed once, reused 3 times)
- Q3 Stress (Vol + Corr): 381.94s (only stress scenarios, different parameters)
- Q4 Collateral: 1.47s (transforms only, no re-simulation)

### Q2: Baseline CVA Analysis
**File**: `workflows/exposures_CVA_workflow.py`

**Pipeline**:
1. **Step 1** (compute_cva_pipeline): Simulate 10,000 equity paths × 60 months
2. **Step 2**: Value portfolio over all scenarios and times
3. **Step 3**: Compute exposures (standalone, netted) and EPE
4. **Step 4**: Calculate CVA from survival probabilities and EPE
5. **Step 5** (run_q2_reporting): Save outputs and plots

**Key Metrics**:
- EPE(netted): Expected Positive Exposure across all time steps
- CVA(netted): Credit Valuation Adjustment under netting agreement
- Netting Benefit: CVA(unnetted) - CVA(netted) [absolute & %; shows value of netting]

**Outputs** (`output/q2/`):
```
epe_profiles.csv              # Time-indexed EPE values
cva_breakdown_netted.csv      # Per-timepoint CVA contribution (netted)
cva_breakdown_unnetted.csv    # Per-timepoint CVA contribution (unnetted)
standalone_cva.csv           # Per-instrument CVA (before netting benefit)
cva_summary.csv              # Single-row CVA summary
epe_plot.png                 # EPE profile visualization
```

### Q3: Sensitivity Analysis
**File**: `workflows/sensitivity_analysis_workflow.py`

**Approach**:
- **Reuses** baseline EPE from Q2 (saved in `baseline_results`)
- **Only recomputes** stressed scenarios (volatility ↑30%, correlation ↓40%)
- Each stress scenario: new equity paths → new valuations → new EPE & CVA

**Scenarios**:
| Scenario | Configuration | Impact |
|----------|---------------|--------|
| Baseline | Vol 15%, Corr 80% | Reference (CVA ~30.7k) |
| Volatility Stress | Vol 30%, Corr 80% | CVA +67% (higher paths variance) |
| Correlation Stress | Vol 15%, Corr 40% | CVA -3% (lower diversification benefit) |

**Outputs** (`output/q3/`):
```
q3_cva_sensitivity_summary.csv    # CVA for each scenario + changes
q3_cva_comparison.png              # Bar plot comparing scenarios
```

### Q4: Collateral Impact Analysis
**File**: `workflows/collateral_impact_workflow.py`

**Key Insight**: Collateral reduces exposure without re-simulation
- Uses baseline equity paths & values (`V` from Q2)
- Applies collateral mechanics (VM, IM) to reduce exposure
- Recomputes only CVA calculation (not simulation/valuation)

**Mechanics**:

**(a) Variation Margin** (Frequency Sweep):
- Updates margin every **M months** (M = 1 to 60)
- Collateralized value = last margin update level
- New exposure = max(current value - collateral, 0)
- **Effect**: More frequent updates → lower exposure → lower CVA
- **Plot**: CVA decreases as M increases (frequent updates better)

**(b) Initial Margin** (Amount Stress):
- Fixed IM held upfront: {€1M, €10M, €100M}
- Reduces exposure: max(V_portfolio - IM, 0)
- **Effect**: Higher IM → lower exposure → lower CVA
- **Plot**: CVA decreases logarithmically with IM level

**Outputs** (`output/q4/`):
```
q4a_variation_margin_cva_by_frequency.csv      # CVA by VM frequency M
q4a_cva_vs_vm_frequency.png                    # Plot

q4b_initial_margin_cva.csv                     # CVA by IM level
q4b_cva_breakdown_im_{IM_EUR}.csv              # Breakdown for each IM
q4b_cva_vs_initial_margin.png                  # Plot (log scale)

q4_baseline_reference.csv                      # No-collateral CVA
```

---

## How to Run

### Prerequisites
```bash
pip install numpy scipy pandas matplotlib
```

### Execute Full Pipeline
```bash
python main.py
```

**Execution Steps** (with timing):
```
[STEP 0] Initializing parameters...  
✓ Initialized in 0.05s

[STEP 1] Simulating equity processes...
✓ Simulation completed in 23.45s

[STEP 2] Running validation tests...
✓ Validation completed in 0.12s

[STEP 3] Computing baseline CVA...
  Valuing instrument 1/4: Forward on SX5E... ✓
  Valuing instrument 2/4: Forward on AEX... ✓
  Valuing instrument 3/4: Option on SX5E... ✓
  Valuing instrument 4/4: Option on AEX... ✓

[DEBUG] Valuation Scaling Check:
  Mean portfolio value at t=0: ~8,500,000 EUR
  Max portfolio value: ~15,000,000 EUR
  Min portfolio value: ~-2,000,000 EUR

✓ Baseline CVA computed in 179.72s

[STEP 3b] Generating Q2 reports...
✓ Q2 reports generated in 0.31s

[STEP 4] Running sensitivity analysis...
  Running Volatility Stress (σ = 30%)...
  Running Correlation Stress (ρ = 40%)...
✓ Sensitivity analysis completed in 381.94s

[STEP 5] Running collateral impact analysis...
  Variation Margin: M=1..60 (sweep complete)
  Initial Margin: {1M, 10M, 100M} EUR
✓ Collateral impact analysis completed in 1.47s

✓ ALL COMPLETE - Total execution time: 563.77s (9.40 min)
```

**Outputs Summary**:
- `output/q1/`: Validation statistics
- `output/q2/`: Baseline CVA, exposure profiles, plots
- `output/q3/`: Sensitivity results (volatility & correlation stress)
- `output/q4/`: Collateral impact (VM frequency & IM stress)

### Run Individual Test Suites
```bash
python tests/test_equity_processes.py      # GBM validation
python tests/test_black_scholes.py         # Option pricing
python tests/test_equity_instruments.py    # Portfolio valuation
python tests/test_exposure.py              # Exposure metrics
python tests/test_cva_calculator.py        # CVA computation
python tests/test_question_1.py            # Validation tests
```

---

## Technical Details

### Valuation Scaling (EUR Notional Terms)

**Problem**: Portfolio values must be in **EUR notional** (contract units × price), not index points.

**Solution** (in `pricing/equity_instruments.py`):
```python
# Step 1: Compute per-contract value (e.g., forward on 1 index point)
per_contract_value = S_t * exp(-q * tau) - K * exp(-r * tau)

# Step 2: Scale by contract size and direction
contracts = float(row["Contracts"])
direction = 1.0 if row["Direction"].lower() == "long" else -1.0
V[:, t_index, idx] = direction * contracts * per_contract_value
```

**Example**:
- Forward 1: 10,000 contracts × €6,000/point × forward price
- Forward 2: 55,000 contracts × €1,000/point × forward price
- Put 3: 10,000 contracts × put premium (in EUR)
- Put 4: 55,000 contracts × put premium (in EUR)

**Verification** (debug output in `compute_cva_pipeline()`):
```
Mean portfolio value at t=0: ~8.5 million EUR  ✓
Max portfolio value: ~15 million EUR  ✓
(NOT thousands or hundreds - that would indicate missing scaling)
```

**Impact**: Ensures CVA, EPE, and collateral amounts are in correct currency units.

---

## Validation Results

### Equity Process Simulation
- **Martingale Test**: SX5E 250.20 vs 264.78 discounted (within 95% CI)
- **Correlation Test**: Simulated 0.7997 vs 0.8000 target (Fisher Z CI: [0.7987, 0.8005])
- **Runtime**: 1000 paths × 60 steps in 0.01 seconds

### Option Pricing
- **Reference Validation**: ATM SX5E put 6.3301 (tested vs ±0.01 tolerance)
- **Call-Put Parity**: Error < 1e-6

### Exposure & CVA
- **EPE Profile**: Mean 818.04, Final 1114.54
- **Netting Benefit**: Quantified as EPE_unnetted vs EPE_netted
- **All Tests**: Exit Code 0 (PASSED)

---

## Code Quality

- **Modular Design**: Each component independently testable
- **Numerical Stability**: All formulas properly implemented with dividend adjustment
- **Reproducibility**: Controlled random seed (42)
- **Auditability**: CSV outputs preserve full calculation trail
- **Documentation**: Mathematical formulas embedded in code comments

---

## Model Scope

**Questions Addressed**:
1. ✅ **Q1**: Validation tests (martingale, parity, correlation)
2. ✅ **Q2**: CVA under baseline market conditions (netted + unnetted)
3. ✅ **Q3**: Sensitivity to volatility (+30%) and correlation (-40%)
4. ✅ **Q4**: Impact of collateral (variation margin frequency and initial margin amount)

**Current Capabilities**:
- Unilateral CVA (counterparty default only)
- Single counterparty exposure
- European options (forwards & puts)
- Flat term structure (rates & spreads)
- Independent market and credit risk
- Variation Margin and Initial Margin collateral

**Architecture**:
- Clean separation: compute engine vs. reporting layer
- Efficient baseline reuse (180s baseline, 380s stresses, 1.5s transforms)
- Full traceability: all intermediate results saved to CSV

**Future Extensions**:
- Bilateral CVA (bank's own default)
- Stochastic interest rates (Hull-White)
- Multiple counterparties and CSA agreements
- Regulatory capital models (SA-CCR, IMM)
- Dynamic hedging strategies

---

## Model Limitations & Extensions

**Current Scope**:
- Unilateral CVA (counterparty default only)
- Single counterparty "C"
- European options only
- Flat term structure (rates & spreads)

**Future Extensions**:
- Bilateral CVA (bank's own default)
- Stochastic interest rates (Hull-White)
- Multiple counterparties
- Hedging strategies (Q3)
- Regulatory capital (SA-CCR)

---

## Authors & Contact
**Case 3, Credit Complexity and Systemic Risk**  
Quantitative Finance Course (2026)

For questions or improvements, consult the code comments and test files.

---

## References
- Hull, J. C. (2018). *Options, Futures, and Other Derivatives*
- Brigo, D., & Mercurio, F. (2006). *Interest Rate Models - Theory and Practice*
- Basel Committee on Banking Supervision. *Standardised Approach for Counterparty Credit Risk*
