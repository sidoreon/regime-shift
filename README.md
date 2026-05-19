# Regime Shift

Hidden Markov Model regime detection on multi-asset market data (SPY, TLT, GLD, synthetic cash) with FRED macro features.

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
| `notebooks/datapipeline.ipynb` | Download prices (yfinance), pull FRED macro series, build synthetic CASH from 1M T-bill yield, merge modeling dataset |
| `notebooks/hmmregimedetection.ipynb` | Fit Gaussian HMM on processed features for regime detection |

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
    simplert.csv
    vix.csv
    cashprice.csv
```

`dataset.csv` columns: `SPYlogret`, `TLTlogret`, `GLDlogret`, `CASHlogret`, `VIX`, `YIELDSPREAD`, `UNRATE`, `DGS10`, `CPIYOY`.

## Project structure

```
regime-shift/
  notebooks/          # pipeline + HMM notebooks
  data/raw/           # raw downloads
  data/processed/     # model-ready outputs
  requirements.txt
  .env.example        # FRED_API_KEY template (copy to .env)
```
