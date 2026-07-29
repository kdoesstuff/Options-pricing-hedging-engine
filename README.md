# Option Pricing, Risk Analysis & Hedging Engine

Option pricing, Greeks, implied volatility, dynamic hedging and strategy backtesting in Python, with an interactive Streamlit app.

Live demo: [options-pricing-hedging-engine.streamlit.app](https://options-pricing-hedging-engine.streamlit.app/)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This was built as our course project for MTL732 (Financial Mathematics) at IIT Delhi.

## Features

### Pricing models

| Model | Method | Notes |
|---|---|---|
| Black-Scholes-Merton | Closed form | European calls and puts, used as the benchmark |
| Binomial tree | CRR and pure binomial, backward induction | Handles European and American exercise |
| Monte Carlo | Risk-neutral GBM simulation | Reports standard error and convergence |
| ML pricer | Gradient boosted trees (scikit-learn) | Trained on 400k BSM-priced samples, test RMSE around $0.87 |

### Risk and Greeks

Full sensitivities (Delta, Gamma, Vega, Theta, Rho), Delta/Gamma surfaces across spot and volatility, and implied volatility solved with Brent's method (Newton-Raphson as fallback).

### Hedging simulation

Discrete-time delta and delta-gamma hedging of a short option over simulated GBM paths. The simulator rebalances at your chosen frequency and reports the P&L distribution and how much hedging cuts risk vs an unhedged position.

### Trading strategies

Covered call, long straddle and delta-neutral speculation backtested on real stock data against buy-and-hold (Sharpe ratio, drawdown, cumulative P&L). There are also 15+ option payoff structures (spreads, straddles, strangles, butterflies, iron condors, collars) with breakeven and max profit/loss computed automatically.

### Convergence analysis

CRR converging to BSM as tree steps grow (N up to 1000), and Monte Carlo stabilization, with absolute and relative error plots.

### Market data

Live prices and historical volatility from yfinance for any ticker. The repo also bundles 1-year snapshots for AAPL, MSFT, GOOGL, TSLA and AMZN, which the app falls back to automatically if the data API is down or rate-limited.

## Quick start

```bash
git clone https://github.com/kdoesstuff/Options-pricing-hedging-engine.git
cd Options-pricing-hedging-engine
pip install -r requirements.txt

# Web app
streamlit run streamlit_gui.py

# Terminal interface
python project_main.py --nogui --real
```

The app opens at `http://localhost:8501`. Pick a ticker (or set parameters manually in the sidebar), then go through the tabs: model comparison, Greeks surfaces, binomial trees, payoff diagrams, hedging simulation and strategy backtests.

## Project structure

```
├── streamlit_gui.py             # Web app (Streamlit + Plotly)
├── project_main.py              # Terminal interface
├── models/
│   ├── pricing_models.py        # BSM, CRR/binomial, Monte Carlo, ML pricer
│   ├── greeks.py                # Greeks + implied vol solvers
│   └── strategies.py            # Hedging simulator + strategy backtests
├── strategies/
│   └── options_payoff_strategies.py  # Payoff structures and comparison
├── utils/
│   ├── data_handler.py          # yfinance fetch + offline CSV fallback
│   ├── real_data_handler.py     # Bundled market data management
│   ├── visualization.py         # Matplotlib/Plotly figures
│   └── tree_printer.py          # ASCII binomial tree printing
├── data/
│   ├── real_data/               # 1y OHLCV snapshots (5 tickers)
│   ├── ml_model.pkl             # Trained ML pricing model
│   └── sample_options_data.csv
├── setup_sample_data.py         # Regenerate ML model / sample data
└── update_real_data.py          # Refresh market data snapshots
```

## Sample output

For S=100, K=110, T=0.25y, r=4%, sigma=30% (European call):

```
BSM (closed form)   $2.7730
CRR (500 steps)     $2.7719
Monte Carlo (100k)  $2.7694 (se 0.02)
```

CRR discretization error goes to 0 as N grows, and Monte Carlo error shrinks at the usual 1/sqrt(M) rate. Both are plotted in the convergence tab.

## Theory notes

The models assume frictionless markets, constant r and sigma, and GBM dynamics under the risk-neutral measure. CRR parameters are u = e^(sigma*sqrt(dt)), d = 1/u, p = (e^(r*dt) - d)/(u - d). American options are valued by comparing continuation value vs immediate exercise at every node. Hedging error is analyzed as a function of rebalancing frequency, which shows the discrete-hedging behaviour you expect from Black-Scholes theory.

The full write-up with derivations, plots and interpretation is in the [project report](docs/MTL732_Project_Report.pdf).

## Contributors

Aditya Narware, Kedhar Karthik Naidu Rejeti, Sanyam Garg

MTL732 Financial Mathematics, IIT Delhi (2025)

## License

MIT, see [LICENSE](LICENSE).
