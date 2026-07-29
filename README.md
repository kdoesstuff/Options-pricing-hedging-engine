# Option Pricing, Risk Analysis & Hedging Lab

**Multi-model option pricing, Greeks, implied volatility, dynamic hedging, and strategy backtesting, with an interactive Streamlit app.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Built as the course project for **MTL732 Financial Mathematics, IIT Delhi**.

---

## What's inside

**Pricing models** - four independent engines, cross-validated against each other:

| Model | Method | Notes |
|---|---|---|
| Black–Scholes–Merton | Closed form | European calls/puts; the analytical benchmark |
| Binomial tree | CRR + pure binomial, backward induction | European **and American** exercise, early-exercise premium |
| Monte Carlo | Risk-neutral GBM simulation | Standard-error bands, convergence diagnostics |
| Machine learning | Gradient-boosted trees (scikit-learn) | Trained on 400k BSM-priced samples, test RMSE ≈ $0.87 |

**Risk & Greeks** - full sensitivities (Δ, Γ, ν, Θ, ρ), Delta/Gamma surfaces across spot and volatility, and implied-volatility recovery via Newton-Raphson and Brent solvers.

**Hedging simulation** - discrete time delta and delta-gamma hedging over GBM paths with transaction costs; P&L distribution vs. rebalancing frequency.

**Trading strategies** - covered call, long straddle, delta-neutral speculation backtested on real equity data against a buy-and-hold benchmark (Sharpe, drawdown, cumulative P&L), plus **15+ option payoff structures** (spreads, straddles, strangles, butterflies, iron condors, collars) with automated breakeven and max-profit/loss detection.

**Convergence analysis** - CRR → BSM as tree steps grow (N ≤ 1000) and Monte Carlo stabilization, with absolute/relative error plots.

**Market data** - live prices and historical volatility via yfinance for any ticker, with bundled 1-year snapshots (AAPL, MSFT, GOOGL, TSLA, AMZN) as an automatic offline fallback, so the app works even when the data API is unavailable.

## Quick start

```bash
git clone <this-repo>
cd <repo-folder>
pip install -r requirements.txt

# Web app (recommended)
streamlit run streamlit_gui.py

# Terminal interface
python project_main.py --nogui --real
```

The app opens at `http://localhost:8501`: pick a ticker (or set parameters manually), then explore the tabs - model comparison, Greeks surfaces, binomial trees, payoff diagrams, hedging simulation, and strategy backtests.

## Project structure

```
├── streamlit_gui.py             # Interactive web app (Streamlit + Plotly)
├── project_main.py              # Terminal interface
├── models/
│   ├── pricing_models.py        # BSM, CRR/binomial, Monte Carlo, ML pricer
│   ├── greeks.py                # Greeks + implied-vol solvers (Newton-Raphson, Brent)
│   └── strategies.py            # Hedging simulator + strategy backtests
├── strategies/
│   └── options_payoff_strategies.py  # 15+ payoff structures & comparison
├── utils/
│   ├── data_handler.py          # yfinance fetch + offline CSV fallback
│   ├── real_data_handler.py     # Bundled market-data management
│   ├── visualization.py         # Matplotlib/Plotly figures
│   └── tree_printer.py          # ASCII binomial-tree rendering
├── data/
│   ├── real_data/               # 1y OHLCV snapshots (5 tickers)
│   ├── ml_model.pkl             # Trained ML pricing model
│   └── sample_options_data.csv
├── setup_sample_data.py         # Regenerate ML model / sample data
└── update_real_data.py          # Refresh market-data snapshots
```

## Sample: model agreement

For S=100, K=110, T=0.25y, r=4%, σ=30% (European call):

```
BSM (closed form)   $2.7730
CRR (500 steps)     $2.7719
Monte Carlo (100k)  $2.7694 ± 0.02
```

CRR discretization error → 0 as N grows; Monte Carlo error shrinks at the O(M^-1/2) rate - both verified in the convergence tab.

## Theory notes

The models assume frictionless markets, constant r and σ, and GBM dynamics under the risk-neutral measure. CRR parameters are u = e^(σ√Δt), d = 1/u, p = (e^(rΔt) − d)/(u − d); American options are valued by comparing continuation vs. immediate exercise at every node. Implied volatility inverts BSM with Brent's method (Newton–Raphson as fallback), and hedging error is analyzed as a function of rebalancing frequency, illustrating the discrete-hedging results of Black–Scholes theory.

A full write-up (derivations, plots, and interpretation) is in the [project report](docs/MTL732_Project_Report.pdf).

## Contributors

Aditya Narware · Kedhar Karthik Naidu Rejeti · Sanyam Garg
MTL732 Financial Mathematics, IIT Delhi (2025)

## License

MIT — see [LICENSE](LICENSE).
