# CVA Equity Derivatives Modeling (Case 3, 2026)

## Overview
This project implements a complete **Counterparty Credit Risk (CCR)** modeling pipeline for an equity derivatives portfolio exposed to two underlyings (SX5E, AEX). The system computes **Credit Valuation Adjustment (CVA)** through Monte Carlo simulation of equity paths, portfolio valuation, exposure measurement, and risk-neutral default probabilities.

**Key Outcome**: CVA quantifies the credit cost of counterparty default, integrated into pricing and risk management frameworks.

---

## Project Structure

```
.
├── main.py                           # 3-step orchestrator (initialize → simulate → CVA workflow)
├── data/
│   ├── config.py                     # Centralized parameters (credit, rates, equity, simulation)
│   ├── portfolio.py                  # 4-instrument portfolio (2 forwards, 2 put options)
│   └── orchestrator.py               # Aggregates config + portfolio
├── models/
│   └── equity_processes.py           # GBM simulation with Cholesky correlation (80%)
├── pricing/
│   ├── black_scholes.py              # European option pricing (call/put)
│   └── equity_instruments.py         # Portfolio valuation across paths × times
├── cva/
│   ├── exposure.py                   # Exposure metrics (standalone, netted, EPE)
│   └── cva_calculator.py             # CVA computation from piecewise hazard rates
├── workflows/
│   └── exposures_CVA_workflow.py     # Question 2: full pipeline (simulate → value → expose → CVA)
├── validation/
│   └── orchestrator.py               # Question 1: martingale, parity, correlation tests
├── utils/
│   └── output_writer.py              # CSV/JSON output infrastructure
├── output/
│   ├── q1/                           # Validation results (CSV)
│   └── q2/                           # CVA outputs (CSV, PNG plots)
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

## How to Run

### Prerequisites
```bash
pip install numpy scipy pandas matplotlib
```

### Execute Full Pipeline
```bash
python main.py
```
**Execution Steps**:
1. **Step 0**: Initialize config + portfolio
2. **Step 1**: Simulate 10,000 equity paths (60 months)
3. **Step 2**: Validate (martingale, parity, correlation)
4. **Step 3**: Compute exposures and CVA, save outputs

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

## Validation Results (from Test Runs)

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
