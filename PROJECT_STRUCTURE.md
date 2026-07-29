# Project Structure

See the [README](README.md) for the full annotated layout, features, and quick-start commands.

```
├── streamlit_gui.py             # Interactive web app (Streamlit + Plotly)
├── project_main.py              # Terminal interface
├── models/                      # BSM, binomial/CRR, Monte Carlo, ML pricer; Greeks & IV; hedging + backtests
├── strategies/                  # 15+ option payoff structures and comparisons
├── utils/                       # Data handling (live + offline fallback), plotting, tree printing
├── data/                        # Market-data snapshots, trained ML model, sample data, config
├── docs/                        # Course project report (PDF)
├── setup_sample_data.py         # Regenerate ML model / sample data
└── update_real_data.py          # Refresh market-data snapshots
```
