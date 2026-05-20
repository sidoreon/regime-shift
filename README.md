# Regime Shift: Macro-Aware Tactical Asset Allocation Engine

A dynamic multi-asset allocation system that uses unsupervised machine learning to detect unobservable economic regimes and shifts portfolio weights between equities, fixed income, gold, and cash based on the inferred market state. The full pipeline is built from scratch — data ingestion through walk-forward validation — with no look-ahead bias and explicit transaction cost modelling.

**Backtest period:** February 2009 – December 2024 (16 years, fully out-of-sample)
**Universe:** SPY (equities), TLT (long Treasuries), GLD (gold), synthetic CASH (1-month T-bill)
**Rebalance:** Monthly, with 10 bps per-rebalance transaction cost

---

## Headline Results

| Metric | Strategy | 60/40 | Equal Weight |
| --- | --- | --- | --- |
| Annualized return | 5.49% | 10.70% | 6.74% |
| Annualized volatility | 6.70% | 10.20% | 6.86% |
| Sharpe ratio | 0.82 | 1.05 | 0.98 |
| Sortino ratio | 0.97 | 1.38 | 1.37 |
| Max drawdown | −13.50% | −27.71% | −17.60% |
| Calmar ratio | 0.41 | 0.39 | 0.38 |

**Interpretation:** This strategy does not maximize total return. It targets *return per unit of tail risk*. On the Calmar ratio — annualized return divided by maximum drawdown — the strategy beats 60/40 (0.41 vs 0.39) at roughly **two-thirds the volatility and half the maximum drawdown**. In 2022, when stocks and bonds fell together and 60/40 lost 23.4%, the strategy lost only 10.9%, a 12.5 percentage-point outperformance in the year that broke the traditional balanced portfolio.

---

## Why this exists

Static portfolios assume a stationary world: they hold the same weights through bull markets, crises, and recoveries. The 60/40 portfolio collapsed in 2022 because both of its legs sold off simultaneously, invalidating its core diversification assumption. The 60/40 lost a quarter of its value in a single year, even though it has averaged double-digit returns over the past decade.

A regime-aware allocator addresses this by treating the economic environment as an unobservable state to be inferred from market data. When the inferred regime suggests stress, the optimizer's objective function shifts from maximizing return to minimizing variance. When the regime suggests recovery, the objective tilts aggressively toward risk assets. The capital allocation responds to the macro environment rather than ignoring it.

This project implements that idea end-to-end and tests it rigorously.

---

## Pipeline

The system is built as five Jupyter notebooks, executed in order:

| # | Notebook | Role |
| --- | --- | --- |
| 1 | `notebooks/datapipeline.ipynb` | Data ingestion, macro feature construction, synthetic cash |
| 2 | `notebooks/hmmregimedetection.ipynb` | HMM training, regime identification, label assignment |
| 3 | `notebooks/portfoliooptimisation.ipynb` | Convex optimizers, in-sample regime-conditional portfolios |
| 4 | `notebooks/walkforwardbacktest.ipynb` | Production backtest with monthly refit and transaction costs |
| 5 | `notebooks/sensitivityanalysis.ipynb` | Cost sweep, turnover, annual performance, robustness checks |

Notebook 4 is the authoritative source of performance numbers. Notebook 3 is an in-sample exploration of the optimizer mechanics and should not be confused with the walk-forward backtest. Run notebooks 1–4 in sequence to generate all backtest artifacts; notebook 5 is the sensitivity tear sheet that consumes those artifacts.

### Notebook 1 — Data pipeline

Price data for SPY, TLT, GLD, and the VIX index is pulled from Yahoo Finance using `yfinance` with `auto_adjust=True` to get total-return-adjusted prices. Macro data — the 10-year minus 2-year Treasury yield spread, headline CPI, unemployment rate, 10-year and 1-month Treasury yields — comes from the FRED API.

Two design decisions in this stage materially shape downstream results. First, every macro series is **lagged to respect publication delays**: monthly series (CPI, unemployment) are shifted by 21 trading days to reflect the roughly one-month lag between observation and release, and daily yield series are shifted by one day. This is the single most important defense against look-ahead bias and is preserved throughout the rest of the pipeline.

Second, CASH is **synthesized rather than held as an ETF**. The 1-month T-bill yield (FRED series `DGS1MO`) is converted to a daily return (`yield/100/252`) and compounded into a price index. This sidesteps the inception-date limitation of cash ETFs like BIL (which only began trading in mid-2007) and gives a clean cash return series back to 2005.

The output of this stage is `data/processed/dataset.csv` — 4,760 daily rows from 2006-02-02 to 2024-12-31, with zero missing values across nine columns: log returns for the four assets, the VIX level, and four macro features.

### Notebook 2 — HMM regime detection

A three-state Gaussian Hidden Markov Model is fit on five standardized features: SPY log return, 21-day annualized realized volatility of SPY, the VIX level, the 10y–2y yield spread, and the unemployment rate. The model is `hmmlearn.GaussianHMM` with `n_components=3`, `covariance_type='full'`, and a fixed random seed for reproducibility.

The HMM finds three clusters in feature space, but the cluster labels are arbitrary — there is no inherent meaning to "state 0" vs "state 2." Labels are assigned by **profile matching** against expected economic signatures. For each cluster, we score how well its mean feature vector matches the canonical profile of each regime:

- **Crisis** expects negative SPY returns, high realized vol, high VIX, and high unemployment.
- **Recovery** expects positive returns, falling volatility, a steep yield curve, and still-elevated unemployment.
- **Bullish** expects positive returns, low vol, low VIX, and low unemployment.

The cluster with the highest crisis score is labeled Crisis; the remaining cluster best matching the recovery profile is Recovery; the last is Bullish. This mapping is critical because it must be applied **consistently across every walk-forward refit** in notebook 4 — without it, the regime-to-portfolio mapping would shuffle randomly every month.

A noteworthy finding from the full-sample fit: the middle regime that emerges is not a generic "bear market" but specifically *post-crisis recovery*. The cluster activates during 2009–2014 (post-GFC) and 2020–2021 (post-COVID), characterized by positive equity returns alongside elevated unemployment and a steep yield curve. This is the macro signature of the early-cycle Fed-supported rally, and it is statistically distinct from both crisis and mature-bull regimes. The strategy benefits substantially from treating these periods as their own state.

The notebook outputs `regimeprobs.csv` with soft posterior probabilities (`pcrisis`, `precovery`, `pbull`), hard Viterbi labels, and named regimes. Soft probabilities are used by the optimizer because they avoid the portfolio thrashing that hard labels would cause at regime boundaries.

### Notebook 3 — Convex portfolio optimization

Three optimizers are implemented in CVXPY:

- **Minimum variance**: `minimize wᵀΣw` subject to `sum(w) = 1, w ≥ 0`. The classical Markowitz problem in its convex form.
- **Maximum Sharpe**: the naive ratio `μᵀw / √(wᵀΣw)` is non-convex, but it admits a clean convex reformulation. Introducing `y = kw` and fixing `μᵀy = 1` reduces the problem to `minimize yᵀΣy` subject to that affine constraint, then normalizing `w = y / sum(y)` recovers the max-Sharpe weights.
- **Equal weight**: `w = 1/n`. No optimization; serves as a benchmark.

Expected returns are estimated as the annualized sample mean of regime-conditional returns. Covariances use **Ledoit-Wolf shrinkage** (`sklearn.covariance.LedoitWolf`), which shrinks the noisy sample covariance toward a structured estimator. This trades a small amount of bias for a large reduction in estimation variance and produces materially more stable optimizer outputs.

Each regime requires at least 60 days of observations to estimate its own parameters; below that threshold, the optimizer falls back to pooled full-sample estimates.

This notebook explores the optimizer behavior on full-sample data and produces illustrative regime-conditional portfolios. It is *not* the production strategy — that lives in notebook 4 with walk-forward refits and tighter constraints.

### Notebook 4 — Walk-forward backtest

This is the authoritative production backtest. At each month-end rebalance date `t`:

1. Slice `dataset.loc[:t]` — strictly no future data.
2. Rebuild features from the sliced dataset; standardize using statistics computed only from the training window.
3. Refit the HMM on the training features.
4. Apply the profile-matching logic to identify Crisis, Recovery, and Bull states in the newly fitted model.
5. Extract soft regime probabilities at the current date.
6. Compute hard regime labels over the training period and use them to estimate regime-conditional `(μ, Σ)`.
7. Solve the regime-specific optimizer for each of the three states.
8. Blend the three regime-optimal portfolios using the soft probabilities at date `t`.
9. Store the resulting target weights and move forward to the next rebalance.

A `np.random.seed(42)` and `random_state=42` lock ensures reproducibility. The training data assertion `result["trainingfeatures"].index[-1] <= testdate` is checked at every refit to guarantee no look-ahead.

The regime-conditional policies use slightly tighter constraints than notebook 3 to prevent the optimizer from over-allocating to cash in the Bullish regime — a known failure mode of the unconstrained problem when CASH has a competitive Sharpe ratio over long historical windows:

| Regime | Objective | Constraints |
| --- | --- | --- |
| Crisis | Minimum variance | Max 80% CASH |
| Recovery | Maximum Sharpe | Max 40% CASH, min 30% SPY |
| Bullish | Maximum Sharpe | Max 30% CASH, min 25% SPY |

The minimum SPY constraint in Recovery and Bullish materially improves total return in sustained bull markets without compromising crisis protection. Crisis remains effectively unconstrained on the upside since min-variance naturally concentrates in low-volatility assets.

**Simulation mechanics.** Weights are set to the target on each rebalance day. Between rebalances, weights drift with asset returns and are renormalized daily. Transaction costs are applied on rebalance days as `(bps/10000) × Σ|w_target − w_pre_rebalance|`, deducted from that day's return. The 60/40 and equal-weight benchmarks use identical simulation infrastructure and the same cost model, ensuring an apples-to-apples comparison.

**Calendar.** Rebalances occur on the last business day of each month, aligned to actual trading days. Backtest begins 2009-02-27 after a three-year burn-in period for the HMM to have enough data to fit. The total backtest spans 191 rebalance dates.

### Notebook 5 — Sensitivity analysis

The headline metrics are computed at 10 bps. To assess robustness, the strategy is rerun at 0, 5, 10, 20, and 50 bps using the saved target weights from notebook 4:

| Cost | Ann. return | Sharpe | Max drawdown |
| --- | --- | --- | --- |
| 0 bps | 5.98% | 0.89 | −12.5% |
| 5 bps | 5.73% | 0.86 | −13.0% |
| 10 bps | 5.49% | 0.82 | −13.5% |
| 20 bps | 5.01% | 0.75 | −14.6% |
| 50 bps | 3.58% | 0.53 | −17.8% |

The strategy remains profitable across all realistic ETF cost levels. ETF bid-ask spreads on SPY, TLT, and GLD are typically well under 1 bp; even allowing for slippage and market impact, costs of 10–20 bps are conservative upper bounds. At 50 bps — implausibly high for liquid ETF trading — the strategy still earns 3.58% annualized.

**Turnover.** Turnover per rebalance is the sum of absolute weight changes, `Σ|w_target − w_pre_rebalance|` (one-way). Mean turnover is roughly 38% per rebalance; annualized turnover is mean × 12 ≈ 457%. This is high but concentrated at genuine regime transitions: long stretches of stable Bull regime produce near-zero turnover, while regime flips (e.g., the COVID transition in early 2020) produce large rebalances. The cost sensitivity table demonstrates that this turnover does not destroy the strategy's economics, but it is a real concern documented in the limitations section.

**Annual returns vs 60/40 (at 10 bps).**

| Year | Strategy | 60/40 | Outperformance |
| --- | --- | --- | --- |
| 2009 | +7.45% | +23.89% | −16.44% |
| 2010 | +11.40% | +13.85% | −2.45% |
| 2011 | +13.78% | +15.10% | −1.32% |
| 2012 | +0.10% | +10.79% | −10.68% |
| 2013 | −1.54% | +12.16% | −13.70% |
| 2014 | +10.57% | +18.93% | −8.36% |
| 2015 | −3.99% | +0.56% | −4.56% |
| 2016 | +4.68% | +8.08% | −3.40% |
| 2017 | +11.04% | +16.60% | −5.57% |
| 2018 | −0.52% | −2.87% | **+2.36%** |
| 2019 | +14.19% | +24.90% | −10.71% |
| 2020 | +6.77% | +19.91% | −13.14% |
| 2021 | +4.62% | +14.46% | −9.85% |
| 2022 | −10.89% | −23.38% | **+12.48%** |
| 2023 | +7.59% | +16.39% | −8.80% |
| 2024 | +15.70% | +10.70% | **+5.00%** |

The pattern is consistent: the strategy underperforms in sustained bull years and outperforms substantially when the traditional balanced portfolio struggles. The outperformance is concentrated in exactly the years that matter most for capital preservation.

---

## Setup and reproduction

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then edit and add FRED_API_KEY
```

A free FRED API key is required for macro data ingestion in notebook 1. Request one at <https://fred.stlouisfed.org/docs/api/api_key.html>.

Run notebooks in order: 1 → 2 → 3 → 4 → 5. Each notebook writes its outputs to `data/processed/`, which the next consumes. The walk-forward backtest in notebook 4 takes about 30 seconds across 191 rebalances on a modern laptop. All data files are gitignored, so a fresh clone requires running the full pipeline once to regenerate them.

Open the repository at its root in your editor of choice so notebook-relative paths (`data/processed/...`) resolve correctly.

---

## Repository structure

```
regime-shift/
├── notebooks/
│   ├── datapipeline.ipynb
│   ├── hmmregimedetection.ipynb
│   ├── portfoliooptimisation.ipynb
│   ├── walkforwardbacktest.ipynb
│   └── sensitivityanalysis.ipynb
├── data/
│   ├── raw/             # gitignored — recreated from yfinance/FRED
│   └── processed/       # gitignored — pipeline outputs
│       └── backtest/    # notebooks 4–5 artifacts
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

After a full pipeline run, `data/processed/` will contain `dataset.csv`, `logrt.csv`, `simplert.csv`, `vix.csv`, `cashprice.csv`, `regimeprobs.csv`, and the `backtest/` subdirectory with `targetweights.csv`, `actualweights.csv`, `regimeprobswf.csv`, `strategyreturns.csv`, `benchmark6040.csv`, `benchmarkeqw.csv`, `performancemetrics.csv`, `sensitivitycosts.csv`, `turnover.csv`, and `annualperformance.csv`.

---

## Dependencies

The project runs on Python 3.9+ and uses:

- `pandas`, `numpy`, `matplotlib` for data manipulation and plotting
- `yfinance` for price data; `fredapi` for FRED macro series; `python-dotenv` for API key management
- `hmmlearn` for the Gaussian HMM
- `cvxpy` for convex optimization
- `scikit-learn` for Ledoit-Wolf covariance shrinkage
- `jupyter` for the notebook environment

Full versions in `requirements.txt`.

---

## Limitations and honest caveats

This is a research project and should be read as one. Several deliberate tradeoffs and known limitations:

**Underperforms 60/40 on total return.** The strategy earns 5.49% annualized vs 60/40's 10.70% over the backtest. This is structural, not a bug: the strategy averages 30% SPY allocation against 60/40's 60%, and SPY had its strongest 16-year stretch in modern history during this period. The strategy is designed for drawdown-constrained investors who would not accept 60/40's −27.7% peak loss; for investors who can tolerate that drawdown, 60/40 is the better choice.

**High turnover.** Mean rebalance turnover is ~38% (sum of absolute weight changes); annualized as mean × 12 ≈ 457%. This is high for a tactical allocator and reflects sharp regime transitions producing large rebalances. The cost sensitivity analysis shows the strategy survives at realistic cost levels, but a production implementation would benefit from a turnover penalty in the optimizer objective or smoothing of regime probabilities — both explored briefly and left as future work.

**Recovery regime is post-crisis, not bear.** The middle regime that the HMM identifies is the post-crisis recovery state, not a classic grinding-bear market. The strategy does not have a separate state for slow declines like 2022; that environment is split between Crisis and Bull in the regime probabilities. A four-state HMM may better separate panic crisis from grinding bear; this is a natural extension.

**Synthetic CASH.** The cash leg is constructed from the 1-month T-bill yield rather than holding a money market fund or short-term Treasury ETF. This means the backtest abstracts away the small but real differences between T-bill yield and what an investor actually earns in cash-like instruments. The headline numbers should be read with that abstraction in mind.

**Walk-forward refit cadence.** The HMM is refit monthly. This is computationally cheap but can produce probabilities that snap between months when the model parameters shift slightly. Smoothing the soft probabilities (e.g., with an exponential moving average) would reduce turnover at the cost of some responsiveness; this was experimented with and removed for clarity.

**Data not in git.** All CSVs under `data/` are gitignored. Reproducing the results requires running notebooks 1 and 2 to regenerate `dataset.csv` and `regimeprobs.csv` before the backtest can run. A FRED API key is required.

**Single seed.** The HMM uses `random_state=42` throughout for reproducibility. Multi-seed sensitivity (fitting with several seeds and taking the best log-likelihood) was used during development but the production pipeline uses a single locked seed to avoid label drift. The walk-forward refit at each date uses the same seed, so the model is deterministic given the data.

---

## Methodology alignment

| Project brief requirement | Implementation |
| --- | --- |
| HMM regime classifier, no manual labels | Gaussian HMM with profile-matching on cluster means |
| Dynamic constraint mapping by regime | Min-variance for Crisis, max-Sharpe for Recovery/Bull, regime-specific cash and SPY constraints |
| Walk-forward validation, no look-ahead | `dataset.loc[:t]` slicing at every refit, with assertion check |
| Transaction friction modelling | 10 bps per rebalance default, sensitivity tested 0–50 bps |
| Benchmark against static portfolios | 60/40 and equal-weight, identical cost model |
| Avoid portfolio thrashing | Partial — soft regime probabilities and constraints reduce thrashing; high turnover remains a known limitation |
| Performance tear sheet | Sharpe, Sortino, max drawdown, Calmar, turnover, annual returns vs benchmarks |
| Reproducibility guide | This README; pipeline runs end-to-end from a fresh clone |