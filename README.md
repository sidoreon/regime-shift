# Regime Shift

Multi-asset market regime detection with a Gaussian Hidden Markov Model, then regime-conditioned convex portfolio optimisation on SPY, TLT, GLD, and synthetic cash, using FRED macro features. A walk-forward backtest refits the HMM and portfolios monthly with no lookahead.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your FRED API key
```

Get a free API key at [FRED](https://fred.stlouisfed.org/docs/api/api_key.html).

Open this folder as the workspace root in Cursor/VS Code (`.vscode/settings.json` sets the notebook working directory).

## Notebooks

Run in order:

| Notebook | Purpose |
|----------|---------|
| `notebooks/datapipeline.ipynb` | Download ETF prices (yfinance), pull FRED macro series, build synthetic CASH from 1M T-bill yield, export modeling dataset and return series |
| `notebooks/hmmregimedetection.ipynb` | Fit a 3-state Gaussian HMM on standardized features (`SPYlogret`, `SPYvol21d`, `vix`, `yieldspread`, `unrate`); label states as **Crisis**, **Recovery**, and **Bullish** via profile matching; export daily regime probabilities |
| `notebooks/portfoliooptimisation.ipynb` | Ledoit–Wolf covariance; min-variance and max-Sharpe solvers (cvxpy); per-regime μ/Σ and optimal weights; blend portfolios by HMM probabilities; plot dynamic allocation vs regime probabilities (full-sample, in-sample) |
| `notebooks/walkforwardbacktest.ipynb` | Walk-forward validation: refit HMM and regime portfolios at each month-end using only past data; simulate strategy with drift between rebalances; compare to 60/40 and equal-weight benchmarks with transaction costs |

### HMM regimes

The middle state is **Recovery** (post-crisis rally, elevated unemployment, steep yield curve), not a classic bear market. Labeling uses three economic profiles scored against each raw HMM state mean. The model is fit with `random_state=42` for reproducibility.

### Portfolio optimisation

- **Baseline strategies** on a rolling 252-day window: min variance, max Sharpe, equal weight.
- **Regime-conditioned**: estimate μ and Σ separately per regime (fallback to pooled estimates when a regime has too few days), solve min-var and max-Sharpe per regime, then blend weights each day with `pcrisis`, `precovery`, `pbull` from the HMM.

### Walk-forward backtest

Notebook 4 avoids lookahead by refitting at each rebalance date `t` on `dataset.loc[:t]` only.

- **Rebalance schedule**: last business day of each month from Feb 2009 through end of sample (~191 dates; ~3 years burn-in for the HMM).
- **At each date**: refit 3-state Gaussian HMM → profile-match Crisis / Recovery / Bull → estimate per-regime μ/Σ on training labels → solve regime portfolios → blend by soft probabilities at `t`.
- **Regime portfolios** (with cash caps on the CASH leg):
  - Crisis: min variance, max 80% cash
  - Recovery: max Sharpe, max 40% cash
  - Bullish: max Sharpe, max 50% cash
- **Simulation**: weights drift daily; reset to target on rebalance days; **10 bps** turnover cost on rebalance.
- **Benchmarks**: 60% SPY / 40% TLT and equal weight (25% each asset), same cost assumption.

Example net performance (2009–2024, from last run):

| | Ann. return | Ann. vol | Sharpe | Max DD |
|---|------------:|---------:|-------:|-------:|
| Strategy | 4.5% | 5.8% | 0.77 | −11.4% |
| 60/40 | 10.7% | 10.2% | 1.05 | −27.7% |
| Equal weight | 6.7% | 6.9% | 0.98 | −17.6% |

The strategy trades lower return for materially lower volatility and drawdown versus static benchmarks in this run.

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
    backtest/       # walk-forward outputs (notebook 4)
      targetweights.csv
      actualweights.csv
      regimeprobswf.csv
      strategyreturns.csv
      benchmark6040.csv
      benchmarkeqw.csv
      performancemetrics.csv
```

`dataset.csv` columns: `SPYlogret`, `TLTlogret`, `GLDlogret`, `CASHlogret`, `VIX`, `YIELDSPREAD`, `UNRATE`, `DGS10`, `CPIYOY`.

## Dependencies

`pandas`, `numpy`, `matplotlib`, `yfinance`, `fredapi`, `python-dotenv`, `jupyter`, `hmmlearn`, `cvxpy` (portfolio solvers), `sklearn` (Ledoit–Wolf shrinkage).

## Project structure

```
regime-shift/
  notebooks/
    datapipeline.ipynb
    hmmregimedetection.ipynb
    portfoliooptimisation.ipynb
    walkforwardbacktest.ipynb
  data/raw/
  data/processed/
  data/processed/backtest/
  requirements.txt
  .env.example        # FRED_API_KEY template (copy to .env)
```
