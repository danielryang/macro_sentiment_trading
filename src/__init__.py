"""
Macro Sentiment Trading

Goal: Build an interpretable ML framework to extract macro‐alpha from global news sentiment.

- Data: GDELT v2 API (macro headlines, EventCodes 100–199), Yahoo Finance (EURUSD=X, USDJPY=X, ZN=F)
- NLP: FinBERT (ProsusAI/finbert via HuggingFace Transformers)
- Features: Daily mean sentiment, dispersion, news volume, impact, Goldstein scores, lags, MAs, rolling stats
- Models: Logistic Regression (scaled), XGBoost (tree-based), time-series CV
- Backtest: Expanding window, realistic transaction costs, Sharpe/CAGR/drawdown
- Interpretability: SHAP for XGBoost

See README.md for full context and usage.
"""
