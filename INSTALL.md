# Installation

## Prerequisites

- Python 3.9+ (3.10/3.11 recommended)
- Internet connection (optional, the bundled market data snapshots are used as a fallback)

## Setup

```bash
pip install -r requirements.txt
```

Or with a virtual environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Web app (recommended)
streamlit run streamlit_gui.py
# opens at http://localhost:8501

# Terminal interface
python project_main.py --nogui --real    # bundled/live market data
python project_main.py --nogui --sample  # synthetic sample data
```

## Optional maintenance

```bash
python update_real_data.py    # refresh the 1y market-data snapshots (needs internet)
python setup_sample_data.py   # regenerate the ML model and sample data
```

## Troubleshooting

- **yfinance errors / no internet**: the app automatically falls back to the bundled AAPL/MSFT/GOOGL/TSLA/AMZN snapshots. You can also enter all option parameters manually in the sidebar.
- **GUI won't start**: `pip install --upgrade streamlit plotly`, then `streamlit run streamlit_gui.py`. Manually visit `http://localhost:8501` if the browser doesn't open.
- **Slow Monte Carlo / trees**: reduce the simulation count or tree steps in the sidebar.
