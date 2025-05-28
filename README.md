# Macro Sentiment Trading

## Research Motivation
Extracting macro‐alpha from global news sentiment is a frontier in systematic trading. This project implements an interpretable machine learning framework to link macro news sentiment to asset returns, inspired by [arXiv:2505.16136v1](https://arxiv.org/abs/2505.16136v1).

## Project Overview
- **Goal:** Build an interpretable ML framework to extract macro‐alpha from global news sentiment.
- **Data sources:**
  - GDELT v2 API (no key required) for worldwide macro headlines (EventCodes 100–199)
  - Yahoo Finance (tickers: `EURUSD=X`, `USDJPY=X`, `ZN=F`)
- **NLP model:** FinBERT (`ProsusAI/finbert` via HuggingFace Transformers; requires `HUGGINGFACE_TOKEN` env var if rate‐limited)
- **Features:** Daily mean sentiment, sentiment dispersion, news volume, article impact, Goldstein scores; plus lags, moving averages (5/20 days), rolling std, sentiment acceleration
- **Predictive models:** Logistic Regression (with standard scaling) and XGBoost (tree‐based, non‐linear), tuned via time‐series cross‐validation
- **Backtest:** 5‐fold expanding window (train starts Jan 2015), realistic transaction costs (0.02% FX, 0.05% bonds), output Sharpe, CAGR, max drawdown
- **Interpretability:** SHAP to explain XGBoost feature contributions

## Project Structure
```
macro_sentiment_trading/
├── data/
│   ├── raw/        # GDELT CSVs, raw market data, raw headlines
│   └── processed/  # cleaned sentiment & feature tables
├── notebooks/      # EDA and prototyping
├── src/
│   ├── __init__.py
│   ├── ingestion.py    # download & filter GDELT events and market data
│   ├── headlines.py    # scrape & extract article headlines
│   ├── sentiment.py    # FinBERT sentiment scoring pipeline
│   ├── features.py     # compute daily aggregates, lags, MAs, rolling stats
│   ├── model.py        # train and tune logistic regression & XGBoost
│   ├── backtest.py     # expanding-window out-of-sample backtest, P&L simulation
│   └── utils.py        # configuration loader, logging, date helpers
├── requirements.txt    # Python dependencies
├── .env.example        # template for environment variables
├── Dockerfile          # containerization (optional)
└── README.md           # overview, research context, setup, usage
```

## Setup Instructions
1. **Clone the repository**
2. **Create a virtual environment**
   ```sh
   python -m venv .venv
   ```
3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
4. **Create your `.env` file** (see `.env.example` for required variables)

## Environment Variables (`.env`)
- `GDELT_API_URL` (default v2 events endpoint)
- `DATABASE_URL` (e.g. `postgresql://user:pass@localhost:5432/macro_sentiment`)
- `HUGGINGFACE_TOKEN` (if needed for FinBERT)

## Usage
Fetch and process data, run models, and backtest:
```sh
python src/ingestion.py   # Download & filter GDELT and market data
python src/headlines.py  # Scrape & extract article headlines
python src/sentiment.py  # Run FinBERT sentiment scoring
python src/features.py   # Compute features
python src/model.py      # Train and tune models
python src/backtest.py   # Run backtest and P&L simulation
```

## SHAP Interpretability
- SHAP plots are generated in `model.py` and/or `backtest.py`.
- Use these to interpret XGBoost feature contributions and model decisions.

## Extending the Framework
- Add new assets by updating tickers and data ingestion logic.
- Adapt to intraday feeds by modifying feature engineering and backtest frequency.
- Integrate with a scheduler (e.g., Airflow) for daily ingestion and retraining.

---
**Reference:** [arXiv:2505.16136v1](https://arxiv.org/abs/2505.16136v1)
