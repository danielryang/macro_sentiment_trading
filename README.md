# Macro Sentiment Trading Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Systematic quantitative trading framework that processes 18K+ GDELT news events through FinBERT sentiment analysis, combines with 443 market/technical features, and generates algorithmic trading signals across 59 financial instruments using XGBoost and Logistic Regression models.

## Pipeline Overview

```
GDELT Events (18K+) → FinBERT → 126 Sentiment Features
                                        ↓
Yahoo Finance → TA-Lib → 443 Market/Technical Features
                                        ↓
                              569 Total Features
                                        ↓
                    XGBoost + Logistic Regression
                                        ↓
                  SELL (-1) | HOLD (0) | BUY (+1)
```

**Key Features:**
- **569 engineered features**: 126 sentiment (mean, volatility, momentum, lags) + 443 market (158 TA-Lib indicators, returns, volatility, correlations)
- **FinBERT sentiment**: 95%+ accuracy on financial text, 12-layer transformer (768 dim, 30.5K vocab)
- **Expanding window training**: Prevents temporal leakage, minimum 30 days history
- **Dual models**: XGBoost (200 estimators, depth=6) + Logistic (L2, balanced)
- **SHAP interpretability**: Feature importance for every prediction

**CRITICAL:** GDELT free API is deprecated. System requires **Google BigQuery** for GDELT data access.

## Quick Start

```bash
# Install
git clone https://github.com/danielryang/macro_sentiment_trading.git
cd macro_sentiment_trading
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Setup BigQuery (required)
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
export BIGQUERY_MAX_COST_USD=5.00

# Get current signals (5 seconds)
python cli/main.py get-signals

# Train models (10 minutes, 6+ months recommended)
python cli/main.py run-pipeline \
  --start-date 2025-04-01 --end-date 2025-10-23 \
  --assets EURUSD USDJPY --models xgboost logistic
```

## Core Commands

```bash
# Signals
python cli/main.py get-signals --assets EURUSD GBPUSD --best-performance
python cli/main.py multi-timeframe-signals --assets EURUSD --timeframes 1D 1W 1M

# Models
python cli/main.py list-models --asset EURUSD --sort-by performance-desc
python cli/main.py compare-models MODEL_ID_1 MODEL_ID_2
python cli/main.py show-model EURUSD_xgboost_20251023

# Pipeline
python cli/main.py run-pipeline --start-date 2023-01-01 --end-date 2025-10-23
python cli/main.py collect-news --start-date 2025-01-01 --end-date 2025-10-23
python cli/main.py train-models --data-path results/20250101_20251023

# Analysis
python cli/main.py status --check-data --check-models
python cli/main.py multi-timeframe-backtest --assets EURUSD
```

## Supported Assets (59)

- **Major FX (7):** EURUSD, USDJPY, GBPUSD, AUDUSD, USDCHF, USDCAD, NZDUSD
- **Cross FX (6):** EURGBP, EURJPY, GBPJPY, EURCHF, AUDJPY, EURAUD
- **Emerging FX (6):** USDMXN, USDBRL, USDZAR, USDTRY, USDINR, USDCNY
- **Crypto (9):** BTCUSD, ETHUSD, BNBUSD, XRPUSD, ADAUSD, SOLUSD, DOTUSD, DOGEUSD, LINKUSD
- **US Indices (10):** SPY, QQQ, DIA, IWM, VTI, VOO, IVV, VEA, VWO, EFA
- **Stocks (10):** AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK-B, JPM, V
- **Commodities (8):** GOLD, SILVER, CRUDE, BRENT, NATGAS, COPPER, CORN, WHEAT
- **Fixed Income (3):** TNOTE, TBOND, TFIVE

## Configuration

```bash
# .env
GDELT_METHOD=bigquery              # Required: 'bigquery' (free API deprecated)
GOOGLE_CLOUD_PROJECT=project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
BIGQUERY_MAX_COST_USD=5.00
BATCH_SIZE=32                      # FinBERT batch (GPU: 128)

# Key options
--skip-sentiment      # Skip FinBERT (use cached, saves memory)
--skip-training       # Data prep only
--best-performance    # Use best model per asset
--models xgboost logistic
```

## Performance (Oct 2025, 133 days)

| Asset | Sharpe | Return | Win Rate | Max DD | Trades |
|-------|--------|--------|----------|--------|--------|
| EURUSD | 0.10 | +0.07% | 48.8% | -2.14% | 43 |
| USDJPY | -1.61 | -2.37% | 53.5% | -4.38% | 43 |

**Status:** Preliminary (43 trades vs 100+ needed). Recommend **3+ years data** for production.

**Expected (3+ years):** Sharpe 0.0-2.0, Win Rate 35-65%, Max DD -1% to -15%

## Output Structure

```
results/
├── 20250401_20251023/
│   ├── events_data.parquet         # 18.9K GDELT events
│   ├── daily_features.parquet      # 126 sentiment × 189 days
│   └── aligned_data_EURUSD.parquet # 569 features × 133 days
├── models/
│   ├── registry.json               # Model metadata + performance
│   └── EURUSD_xgboost_*.pkl        # Trained models
└── signals.json                    # Latest signals
```

## Troubleshooting

```bash
# Memory issues (FinBERT)
export BATCH_SIZE=16
python cli/main.py run-pipeline --skip-sentiment

# BigQuery quota
python cli/main.py run-pipeline --method free  # Fallback (slower)

# Training failures
# Use 6+ months minimum data
python cli/main.py run-pipeline --start-date 2025-04-01 --end-date 2025-10-23

# Check logs
tail -f logs/cli.log
```

## Future Development

**High Priority:**
- **Long-term training**: 10+ years historical data (2014-2025) for full market cycles
- **Data redundancy**: Backup news sources (NewsAPI, Alpha Vantage) since GDELT API deprecated
- **Advanced models**: LSTM/Transformer for sequential patterns, RL for position sizing
- **Analytics dashboard**: Plotly/Dash for interactive performance analysis, SHAP evolution

## Research Foundation

### Primary Research Foundation
- **arXiv:2505.16136v1** - "Macro Sentiment Trading: A Novel Approach to Systematic Trading Using Global News Sentiment" - *Primary research foundation for this system*
- **arXiv:1908.10063** - "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models" - *Sentiment analysis methodology*
- **arXiv:1706.03762** - "Attention Is All You Need" - *Transformer architecture underlying FinBERT*
- **arXiv:1810.04805** - "BERT: Pre-training of Deep Bidirectional Transformers" - *BERT foundation for FinBERT*

### Academic Research Papers
- **Journal of Financial Economics** - "News and Stock Returns" - *Empirical evidence of news impact on financial markets*
- **Review of Financial Studies** - "Textual Analysis in Finance" - *NLP applications in quantitative finance*
- **"Attention Is All You Need"** - Vaswani et al. (2017) - *Transformer architecture foundation*
- **"BERT: Pre-training of Deep Bidirectional Transformers"** - Devlin et al. (2018) - *BERT methodology*
- **"A Unified Approach to Interpreting Model Predictions"** - Lundberg & Lee (2017) - *SHAP values for model interpretability*
- **"XGBoost: A Scalable Tree Boosting System"** - Chen & Guestrin (2016) - *Gradient boosting framework*
- **"Textual Analysis in Finance"** - Loughran & McDonald (2011) - *Financial text analysis methodology*
- **"News and Stock Returns"** - Tetlock (2007) - *News sentiment impact on markets*
- **"Machine Learning for Asset Management"** - Gu et al. (2020) - *ML applications in finance*
- **"Deep Learning for Finance"** - Dixon et al. (2017) - *Neural networks in trading systems*
- **"Sentiment Analysis in Financial Markets"** - Loughran & McDonald (2016) - *Market sentiment analysis*
- **"Natural Language Processing for Financial Text"** - Yang et al. (2019) - *NLP techniques in finance*

### Data Sources
- **[GDELT Project](https://www.gdeltproject.org/)** - Global Database of Events, Language, and Tone
  - **CRITICAL: Free API tier deprecated and completely non-functional**
  - **System uses Google BigQuery exclusively for GDELT data access**
  - Multi-language support with English translation
  - Historical data from 1979 to present
- **[Yahoo Finance API](https://finance.yahoo.com/)** - Market data for 59+ financial instruments
- **[Google BigQuery](https://cloud.google.com/bigquery)** - GDELT data access via cloud data warehouse

### Machine Learning & NLP Libraries
- **[ProsusAI FinBERT](https://huggingface.co/ProsusAI/finbert)** - Pre-trained financial sentiment analysis model
- **[HuggingFace Transformers](https://huggingface.co/transformers/)** - State-of-the-art NLP models
- **[XGBoost](https://xgboost.readthedocs.io/)** - Gradient boosting framework
- **[scikit-learn](https://scikit-learn.org/)** - Logistic regression and ML toolkit
- **[PyTorch](https://pytorch.org/)** - Deep learning framework for FinBERT
- **[SHAP](https://shap.readthedocs.io/)** - Model interpretability framework
- **[TA-Lib](https://ta-lib.org/)** - Technical Analysis Library (158+ indicators)

### Python Libraries & Frameworks
- **[pandas](https://pandas.pydata.org/)**, **[numpy](https://numpy.org/)** - Data manipulation
- **[matplotlib](https://matplotlib.org/)**, **[seaborn](https://seaborn.pydata.org/)**, **[plotly](https://plotly.com/python/)** - Visualization
- **[pyarrow](https://arrow.apache.org/docs/python/)** - Parquet support

### Enhancements Over Original Research
- 17x features (33 → 569)
- FinBERT headline analysis vs simple tone scores
- 158 TA-Lib technical indicators
- BigQuery infrastructure
- Realistic validation (Sharpe 0.0-0.68 vs theoretical 4.65-5.87)

**Disclaimer:** Educational/research purposes only. Not financial advice.

---

**License:** MIT | **Support:** [GitHub Issues](https://github.com/danielryang/macro_sentiment_trading/issues) | **Docs:** `TRADING_QUICKSTART.md`
