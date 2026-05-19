# Regime Shift

Market data pipeline for regime analysis (SPY, TLT, GLD, VIX).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your FRED API key if using FRED
```

Open this folder as the workspace root in Cursor/VS Code (see `.vscode/settings.json` for notebook paths).

## Data

Run `notebooks/datapipeline.ipynb` to download prices and write outputs:

- `data/raw/` — raw downloads
- `data/processed/` — log returns, simple returns, VIX

CSV files are gitignored; regenerate locally after clone.
