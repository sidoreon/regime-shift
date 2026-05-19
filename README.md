# Regime Shift

Multi-asset market regime detection with a Gaussian Hidden Markov Model, then regime-conditioned convex portfolio optimization on SPY, TLT, GLD, and synthetic cash, using FRED macro features.

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
| `notebooks/portfoliooptimization.ipynb` | Ledoit–Wolf covariance; min-variance and max-Sharpe solvers (cvxpy); per-regime μ/Σ and optimal weights; blend portfolios by HMM probabilities; plot dynamic allocation vs regime probabilities |

### HMM regimes

The middle state is **Recovery** (post-crisis rally, elevated unemployment, steep yield curve), not a classic bear market. Labeling uses three economic profiles scored against each raw HMM state mean. The model is fit with `random_state=42` for reproducibility.

### Portfolio optimization

- **Baseline strategies** on a rolling 252-day window: min variance, max Sharpe, equal weight.
- **Regime-conditioned**: estimate μ and Σ separately per regime (fallback to pooled estimates when a regime has too few days), solve min-var and max-Sharpe per regime, then blend weights each day with `pcrisis`, `precovery`, `pbull` from the HMM.

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
```

`dataset.csv` columns: `SPYlogret`, `TLTlogret`, `GLDlogret`, `CASHlogret`, `VIX`, `YIELDSPREAD`, `UNRATE`, `DGS10`, `CPIYOY`.

## Dependencies

`pandas`, `numpy`, `matplotlib`, `yfinance`, `fredapi`, `jupyter`, `hmmlearn`, `cvxpy` (portfolio solvers), `sklearn` (Ledoit–Wolf shrinkage).

## Project structure

```
regime-shift/
  notebooks/
    datapipeline.ipynb
    hmmregimedetection.ipynb
    portfoliooptimization.ipynb
  data/raw/
  data/processed/
  requirements.txt
  .env.example        # FRED_API_KEY template (copy to .env)
```
