# Regime Shift

Multi-asset market regime detection with a Gaussian Hidden Markov Model, regime-conditioned convex portfolio optimisation on SPY, TLT, GLD, and synthetic cash, and a walk-forward backtest with transaction-cost sensitivity. Macro features come from FRED; the walk-forward loop refits the HMM and portfolios monthly with no lookahead.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your FRED API key
```

Get a free API key at [FRED](https://fred.stlouisfed.org/docs/api/api_key.html).

Open this folder as the workspace root in Cursor/VS Code (`.vscode/settings.json` sets the Jupyter notebook root to the project directory).

## Notebooks

Run in order:

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | `notebooks/datapipeline.ipynb` | Download ETF prices (yfinance), pull FRED macro series, build synthetic CASH from 1M T-bill yield, export modeling dataset and return series |
| 2 | `notebooks/hmmregimedetection.ipynb` | Fit a 3-state Gaussian HMM on standardized features; label states as **Crisis**, **Recovery**, and **Bullish** via profile matching; export daily regime probabilities |
| 3 | `notebooks/portfoliooptimisation.ipynb` | Ledoit–Wolf covariance; min-variance and max-Sharpe solvers (cvxpy); per-regime μ/Σ and optimal weights; blend portfolios by HMM probabilities (full-sample, in-sample) |
| 4 | `notebooks/walkforwardbacktest.ipynb` | Walk-forward validation: refit HMM and regime portfolios at each month-end using only past data; simulate strategy with drift between rebalances; compare to 60/40 and equal-weight benchmarks |
| 5 | `notebooks/sensitivityanalysis.ipynb` | Stress-test the walk-forward strategy across transaction costs (0–50 bps); turnover per rebalance; calendar-year returns vs 60/40 |

### Data pipeline (notebook 1)

- **Assets**: SPY (S&P 500), TLT (long Treasuries), GLD (gold), ^VIX (fear gauge), plus synthetic **CASH** from the 1M T-bill yield (`DGS1MO`).
- **Macro (FRED)**: yield spread (`T10Y2Y`), CPI (YoY), unemployment, 10Y and 1M yields.
- **Publication lags**: monthly series shifted 21 trading days; daily series shifted 1 day.
- **Sample**: prices from 2005-01-01; merged `dataset.csv` runs **2006-02-02 → 2024-12-31** (~4,760 trading days).

### HMM regimes (notebook 2)

Three states are fit on standardized features: `SPYlogret`, `SPYvol21d` (21-day annualized vol), `vix`, `yieldspread`, `unrate`.

The middle state is **Recovery** (post-crisis rally, elevated unemployment, steep yield curve), not a classic bear market. Labeling scores each raw HMM state mean against three economic profiles (Crisis / Recovery / Bullish). The model uses `random_state=42` for reproducibility.

### Portfolio optimisation (notebook 3)

- **Baseline strategies** on a rolling 252-day window: min variance, max Sharpe, equal weight.
- **Regime-conditioned**: estimate μ and Σ separately per regime (fallback to pooled estimates when a regime has too few days), solve min-var and max-Sharpe per regime, then blend weights each day with `pcrisis`, `precovery`, `pbull` from the HMM.

### Walk-forward backtest (notebook 4)

Notebook 4 avoids lookahead by refitting at each rebalance date `t` on `dataset.loc[:t]` only.

- **Rebalance schedule**: last business day of each month from **Feb 2009 → Dec 2024** (191 dates; ~3 years burn-in for the HMM).
- **At each date**: refit 3-state Gaussian HMM → profile-match Crisis / Recovery / Bull → estimate per-regime μ/Σ on training labels → solve regime portfolios → blend by soft probabilities at `t`.
- **Regime portfolios** (with cash caps on the CASH leg):
  - Crisis: min variance, max 80% cash
  - Recovery: max Sharpe, max 40% cash
  - Bullish: max Sharpe, max 50% cash
- **Simulation**: weights drift daily; reset to target on rebalance days; **10 bps** turnover cost on rebalance.
- **Benchmarks**: 60% SPY / 40% TLT and equal weight (25% each asset), same cost assumption.

Net performance at **10 bps** (2009–2024, from last run):

| | Ann. return | Ann. vol | Sharpe | Sortino | Max DD | Calmar |
|---|------------:|---------:|-------:|--------:|-------:|-------:|
| Strategy | 5.5% | 6.7% | 0.82 | 0.97 | −13.5% | 0.41 |
| 60/40 | 10.7% | 10.2% | 1.05 | 1.38 | −27.7% | 0.39 |
| Equal weight | 6.7% | 6.9% | 0.98 | 1.37 | −17.6% | 0.38 |

The strategy trades lower return for materially lower volatility and drawdown versus static benchmarks in this run. It tends to outperform 60/40 in stress years (e.g. 2018, 2022) but lags in strong equity rallies.

### Sensitivity analysis (notebook 5)

Replays the walk-forward target weights from notebook 4 under varying transaction costs:

| Cost | Ann. return | Sharpe | Max DD |
|------|------------:|-------:|-------:|
| 0 bps | 6.0% | 0.89 | −12.5% |
| 10 bps | 5.5% | 0.82 | −13.5% |
| 50 bps | 3.6% | 0.53 | −17.8% |

Typical rebalance turnover: **~38% mean**, **~19% median** per month-end rebalance (annualized ~457% of notional traded, reflecting frequent regime-driven reallocations).

## Data layout

Generated locally (CSVs are gitignored; rerun notebooks after clone):

```
data/
  raw/
    prices.csv      # ETF/index close prices
    rawmacro.csv    # FRED series (yield spread, CPI, unemployment, yields)
  processed/
    dataset.csv     # merged features for HMM (log returns, VIX, macro)
    logrt.csv
    simplert.csv    # simple returns (SPY, TLT, GLD, CASH) for portfolios
    vix.csv
    cashprice.csv
    regimeprobs.csv # HMM output: pcrisis, precovery, pbull, regime, regimename
    backtest/       # walk-forward + sensitivity outputs (notebooks 4–5)
      targetweights.csv
      actualweights.csv
      regimeprobswf.csv
      strategyreturns.csv
      benchmark6040.csv
      benchmarkeqw.csv
      performancemetrics.csv
      sensitivitycosts.csv
      turnover.csv
      annualperformance.csv
```

`dataset.csv` columns: `SPYlogret`, `TLTlogret`, `GLDlogret`, `CASHlogret`, `VIX`, `YIELDSPREAD`, `UNRATE`, `DGS10`, `CPIYOY`.

## Dependencies

`pandas`, `numpy`, `matplotlib`, `yfinance`, `fredapi`, `python-dotenv`, `jupyter`, `hmmlearn`, `cvxpy`, `scikit-learn` (Ledoit–Wolf shrinkage).

## Project structure

```
regime-shift/
  notebooks/
    datapipeline.ipynb
    hmmregimedetection.ipynb
    portfoliooptimisation.ipynb
    walkforwardbacktest.ipynb
    sensitivityanalysis.ipynb
  data/raw/
  data/processed/
  data/processed/backtest/
  requirements.txt
  .env.example        # FRED_API_KEY template (copy to .env)
```
