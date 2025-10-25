# Macro Sentiment Trading Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**Macro Sentiment Trading Pipeline** is a systematic quantitative trading framework that processes global news sentiment and market data through sequential phases to generate algorithmic trading signals across 59 financial instruments. The system operates through the following phase-by-phase methodology:

**Phase 1: Data Ingestion** - The system ingests news headlines from the GDELT (Global Database of Events, Language, and Tone) database via Google BigQuery, capturing global news events. **Note:** The GDELT API free tier is deprecated and non-functional; the system currently relies on cached BigQuery data. Simultaneously, market data is collected from Yahoo Finance for 59 financial instruments including major currency pairs, cryptocurrencies, equity indices, individual equities, commodities, and fixed income instruments.

**Phase 2: Sentiment Analysis** - Each news headline undergoes tokenization and processing through the pre-trained FinBERT transformer model to generate sentiment scores ranging from -1 (negative) to +1 (positive). The process begins with text preprocessing: headlines are cleaned, normalized, and tokenized using the WordPiece tokenizer with a vocabulary size of 30,522 tokens. Each headline is converted to input IDs, attention masks, and token type IDs, then fed through the transformer's 12 attention layers (12 heads, 768 hidden dimensions, 3,072 intermediate dimensions). The model outputs contextualized embeddings that are passed through a classification head to generate sentiment scores. The sentiment analysis pipeline processes headlines in batches of 32, applying the FinBERT model to extract contextual sentiment information from financial news text with 95%+ accuracy on financial sentiment classification tasks.

**Phase 3: Feature Engineering** - Sentiment scores are aggregated with market data to create a comprehensive feature matrix of 569 engineered features designed to capture both macro sentiment dynamics and market microstructure patterns. The 126 sentiment-based features include: mean daily sentiment (captures overall market mood), sentiment volatility (measures uncertainty and market stress), news volume (indicates information flow intensity), sentiment momentum (trends in sentiment changes), and various lagged transformations (1-day, 3-day, 7-day lags to capture delayed market reactions). The 443 market/technical features comprise: 158 TA-Lib technical indicators (RSI, MACD, Bollinger Bands, etc. for trend and momentum analysis), lagged returns (1-day to 20-day returns for momentum capture), volatility measures (realized and implied volatility for risk assessment), cross-asset correlations (inter-market relationships and contagion effects), and interaction terms between sentiment and technical indicators (capturing how sentiment amplifies or dampens technical signals). This multi-dimensional feature space enables the models to identify complex patterns where news sentiment interacts with market microstructure to predict directional price movements.

**Phase 4: Model Training** - The system employs a dual-model approach using Logistic Regression with L2 regularization (C=1.0, max_iter=1000, class_weight='balanced') and XGBoost gradient boosting algorithms (n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8). Features are standardized using StandardScaler before Logistic Regression training, while XGBoost handles feature scaling internally. Models are trained on an expanding window methodology with a minimum of 30 days of historical data, starting from day 30 and expanding the training window by one day for each subsequent prediction. This prevents temporal data leakage by ensuring models only use information available at the time of prediction. The training process includes robust data validation: minimum 10 samples per class, target distribution validation, and comprehensive error handling for assets with insufficient data.

**Phase 5: Signal Generation** - Trained models generate trading signals for each asset, with predictions accompanied by SHAP (SHapley Additive exPlanations) values for model interpretability. The system produces binary directional signals (long/short) with confidence scores.

**Phase 6: Validation & Persistence** - Model performance is validated through comprehensive backtesting with transaction cost modeling. Successful models are persisted in a model registry system with automated performance analytics and monitoring capabilities.

## Future Work & Development Roadmap

### High-Priority TODO Items

**Long-Term Model Training** - Train models on 10+ years of historical data (2014-2025) to capture full market cycles including:
- Multiple economic cycles (expansion, contraction, recovery)
- Various market regimes (bull markets, bear markets, sideways markets)
- Major geopolitical events and their sentiment impact
- Different volatility environments and their effect on sentiment-momentum relationships

**GDELT API Migration** - **CRITICAL:** The GDELT API free tier is deprecated and no longer functional. The system currently relies on cached BigQuery data, but for production use, we need to:
- Implement alternative news data sources (NewsAPI, Alpha Vantage News, Reuters API)
- Set up GDELT BigQuery paid access for real-time data
- Develop fallback mechanisms for when primary data sources are unavailable
- Ensure data continuity and quality across different news providers

**Advanced Analytics & Visualizations** - Develop comprehensive analytical dashboards including:
- Interactive performance analytics with Plotly/Dash
- Real-time sentiment-momentum correlation heatmaps
- SHAP value evolution over time for model interpretability
- Cross-asset sentiment contagion analysis
- Market regime detection and regime-specific model performance
- Feature importance evolution and stability analysis
- Risk-adjusted performance metrics across different market conditions

**Enhanced Model Architecture** - Implement additional advanced ML techniques:
- **Currently Implemented:** Dual-model ensemble (Logistic Regression + XGBoost), SHAP interpretability, expanding window backtesting
- **Future Enhancements:**
  - Deep learning models (LSTM, Transformer) for sequential pattern recognition
  - Reinforcement learning for dynamic position sizing
  - Multi-asset portfolio optimization with sentiment constraints
  - Ensemble methods combining multiple timeframes (daily, weekly, monthly)

**Disclaimer:** Educational/research purposes only. Not financial advice.

---

## Quick Start

### Installation

```bash
git clone https://github.com/danielryang/macro_sentiment_trading.git
cd macro_sentiment_trading
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### API Setup (Choose One)

**Option 1: BigQuery (Recommended - Fast)**
```bash
# 1. Create GCP project: https://console.cloud.google.com
# 2. Enable BigQuery API
# 3. Create service account with BigQuery User role
# 4. Download JSON credentials

# Set environment variables
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
export BIGQUERY_MAX_COST_USD=5.00
```

**Option 2: Free GDELT API (Slow)**
```bash
# No setup needed - automatically used if BigQuery not configured
# Processes raw CSV files from GDELT (slower, no caching)
```

**GPU Setup (Optional - 5x faster FinBERT)**
```bash
# CUDA 11.8+ required
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Get Trading Signals (5 seconds)

```bash
python cli/main.py get-signals
```

Output:
```
CURRENT TRADING SIGNALS
Generated at: 2025-10-24 15:30:00

EUR/USD (EURUSD):
  Xgboost : BUY  (+1) | Confidence:  65.2%
  Logistic: HOLD ( 0) | Confidence:  52.1%

USD/JPY (USDJPY):
  Xgboost : SELL (-1) | Confidence:  58.4%
```

### Train New Models (10 minutes)

```bash
# Minimum 6 months data recommended
python cli/main.py run-pipeline \
  --start-date 2025-04-01 \
  --end-date 2025-10-23 \
  --assets EURUSD USDJPY \
  --models xgboost logistic
```

---

## Core Commands

### Signal Generation
```bash
# Current signals for default assets (EURUSD, USDJPY, TNOTE)
python cli/main.py get-signals

# Specific assets with best models
python cli/main.py get-signals --assets EURUSD GBPUSD BTCUSD --best-performance

# Save to file
python cli/main.py get-signals --output-file signals.json

# Multi-timeframe forecasts (1D/1W/1M/1Q)
python cli/main.py multi-timeframe-signals --assets EURUSD --timeframes 1D 1W 1M
```

### Model Management
```bash
# List all trained models
python cli/main.py list-models

# Filter by asset/type, sort by performance
python cli/main.py list-models --asset EURUSD --sort-by performance-desc --limit 5

# Show detailed model info
python cli/main.py show-model EURUSD_xgboost_20251023_abc123

# Compare models
python cli/main.py compare-models MODEL_ID_1 MODEL_ID_2

# Registry status
python cli/main.py model-registry status
```

### Data Pipeline
```bash
# Complete pipeline (data → training → signals)
python cli/main.py run-pipeline \
  --start-date 2023-01-01 --end-date 2025-10-23 \
  --assets EURUSD USDJPY BTCUSD

# Just collect data (skip training)
python cli/main.py collect-news --start-date 2025-01-01 --end-date 2025-10-23

# Process sentiment only
python cli/main.py process-sentiment --data-path results/events_data_*.parquet

# Align market data
python cli/main.py process-market --start-date 2025-01-01 --end-date 2025-10-23

# Train on existing data
python cli/main.py train-models --data-path results/20250101_20251023 --assets EURUSD
```

### Analysis & Monitoring
```bash
# System health check
python cli/main.py status --check-data --check-models

# Multi-timeframe backtest
python cli/main.py multi-timeframe-backtest --assets EURUSD --models xgboost

# Generate visualizations
python cli/main.py visualize --results-path results --types performance shap

# Check data integrity
python cli/main.py check-parquet --directory results --fix
```

---

## Architecture Overview

### Data Flow (569 Features Total)

```
GDELT Events (18K+ events)
    ↓ FinBERT Sentiment Analysis
Sentiment Features (126)
    ├─ Mean sentiment, volatility, volume
    ├─ GDELT tone & Goldstein scale
    ├─ Article impact, lags (1-3 days)
    └─ Moving averages, acceleration
    ↓
Yahoo Finance Market Data
    ↓ TA-Lib Technical Indicators
Market Features (443)
    ├─ RSI, SMA, Bollinger Bands, MACD
    ├─ Lagged returns (1-10 days)
    ├─ Volatility measures (ATR, stdev)
    └─ 158 TA-Lib indicators + derivatives
    ↓
Feature Alignment & Engineering
    ↓ 569 Total Features
XGBoost + Logistic Regression
    ↓ 3-Class Output
Signals: SELL (-1) | HOLD (0) | BUY (+1)
```

### Model Architecture
- **XGBoost**: 100 estimators, max_depth=6, no scaling
- **Logistic Regression**: L2 penalty, StandardScaler, balanced weights
- **Target**: Volatility-adjusted returns (±0.5σ thresholds)
- **Backtesting**: Expanding window (2-year train, 1-year test)
- **Transaction Costs**: 1bp FX, 2bp futures

---

## Configuration

### Environment Variables (.env)
```bash
# Data Collection
GDELT_METHOD=bigquery              # 'bigquery' or 'free'
GOOGLE_CLOUD_PROJECT=project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
BIGQUERY_MAX_COST_USD=5.00

# Model Training
BATCH_SIZE=32                      # FinBERT batch size (GPU: 128)
LOG_LEVEL=INFO

# Paths
DATA_DIR=data
RESULTS_DIR=results
MODELS_DIR=results/models
```

### Command Options
```bash
# Pipeline control
--skip-news          # Skip GDELT collection
--skip-sentiment     # Skip FinBERT (memory-intensive)
--skip-market        # Skip market data
--skip-training      # Data prep only
--force-refresh      # Ignore cache

# Model selection
--models xgboost logistic           # Model types
--model-ids MODEL_ID_1 MODEL_ID_2   # Specific models
--best-performance                  # Use best per asset
--performance-metric accuracy       # Ranking metric

# Date filtering
--start-date 2023-01-01
--end-date 2025-10-23
--train-date 20240101               # Filter by training date
--training-window 2023010120231231  # Filter by data period
```

---

## Supported Assets (59 Total)

**Major FX (7):** EURUSD, USDJPY, GBPUSD, AUDUSD, USDCHF, USDCAD, NZDUSD
**Cross FX (6):** EURGBP, EURJPY, GBPJPY, EURCHF, AUDJPY, EURAUD
**Emerging Market FX (6):** USDMXN, USDBRL, USDZAR, USDTRY, USDINR, USDCNY
**Cryptocurrencies (9):** BTCUSD, ETHUSD, BNBUSD, XRPUSD, ADAUSD, SOLUSD, DOTUSD, DOGEUSD, LINKUSD
**US Equity Indices (10):** SPY, QQQ, DIA, IWM, VTI, VOO, IVV, VEA, VWO, EFA
**Individual Stocks (10):** AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK-B, JPM, V
**Commodities (8):** GOLD, SILVER, CRUDE, BRENT, NATGAS, COPPER, CORN, WHEAT
**Fixed Income (3):** TNOTE, TBOND, TFIVE

**Note:** MATICUSD was removed due to data unavailability (delisted/unavailable on Yahoo Finance).

Add custom assets in `src/market_processor.py`:
```python
# Add to appropriate category dictionary
crypto = {
    'NEWCRYPTO': 'NEWCRYPTO-USD',  # Add your asset here
    # ... existing assets
}
```

---

## Performance Results

### Oct 2025 Validation (133 days, 569 features)

| Asset | Sharpe | Calmar | Return | Win Rate | Max DD | Trades |
|-------|--------|--------|--------|----------|--------|--------|
| EURUSD | 0.10 | 0.20 | +0.07% | 48.8% | -2.14% | 43 |
| USDJPY | -1.61 | -3.17 | -2.37% | 53.5% | -4.38% | 43 |

**Status:** Preliminary (43 trades vs 100+ needed for statistical significance)
**Recommendation:** 3+ years data (500-750 days) for production validation

### Expected Performance (3+ years)
- **Sharpe Ratio:** 0.0-2.0 (excellent >1.5)
- **Win Rate:** 35-65% (good >50%)
- **Max Drawdown:** -1% to -15% (excellent <-5%)

---

## Output Structure

```
results/
├── 20250401_20251023/              # Time-windowed directory
│   ├── events_data.parquet         # GDELT events (18.9K rows)
│   ├── daily_features.parquet      # Sentiment (126 features × 189 days)
│   ├── aligned_data_EURUSD.parquet # Full features (569 × 133 days)
│   └── aligned_data_USDJPY.parquet
├── models/
│   ├── registry.json               # Model metadata + performance
│   ├── EURUSD_xgboost_*.pkl        # Trained models
│   └── EURUSD_xgboost_*_scaler.pkl # StandardScaler (logistic only)
├── signals.json                    # Latest trading signals
└── logs/
    └── cli.log                     # Detailed execution logs
```

---

## Troubleshooting

**Import/Dependency Errors:**
```bash
pip install -r requirements.txt
# If venv issues: deactivate, delete venv/, recreate
```

**BigQuery Quota Exceeded:**
```bash
python cli/main.py run-pipeline --method free  # Fallback to free API
# Or increase: BIGQUERY_MAX_COST_USD=10.00
```

**Memory Issues (FinBERT OOM):**
```bash
# Reduce batch size
export BATCH_SIZE=16

# Skip sentiment (use cached)
python cli/main.py run-pipeline --skip-sentiment

# Process fewer assets
python cli/main.py run-pipeline --assets EURUSD
```

**No Models Found:**
```bash
# Train models first
python cli/main.py run-pipeline --start-date 2025-04-01 --end-date 2025-10-23

# Check status
python cli/main.py status --check-models
```

**Training Failures (Single Target Class):**
```bash
# Use 6+ months of data minimum
python cli/main.py run-pipeline --start-date 2025-04-01 --end-date 2025-10-23
```

**Detailed Logs:** `logs/cli.log` contains full error traces

---

## Research Foundation

**Primary Papers:**
1. [Macro Sentiment Trading (arXiv:2505.16136v1)](https://arxiv.org/abs/2505.16136v1) - Original sentiment→price framework
2. [FinBERT (arXiv:1908.10063)](https://arxiv.org/abs/1908.10063) - Financial text sentiment (97% accuracy)

**Enhancements:**
- 17x feature increase (33 → 569)
- FinBERT headline analysis vs simple tone scores
- TA-Lib technical integration (158 indicators)
- Production infrastructure (BigQuery, model registry)
- Realistic performance validation (Sharpe 0.0-0.68 vs theoretical 4.65-5.87)

---

## Development

**Structure:**
```
cli/
├── commands/          # All CLI commands (16 total)
└── main.py           # Entry point + argument parsing

src/
├── data_collector.py      # GDELT BigQuery/free API
├── sentiment_analyzer.py  # FinBERT (126 sentiment features)
├── market_processor.py    # Yahoo Finance + TA-Lib (443 features)
├── feature_pipeline.py    # Feature engineering orchestration
├── model_trainer.py       # XGBoost + Logistic training
├── model_persistence.py   # Model registry + save/load
└── config.py             # Environment configuration
```

**Testing:**
```bash
python cli/main.py status  # System check
python cli/main.py get-signals --assets EURUSD  # Quick test
```

---

## Sources and References

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
  - **Note: Free API tier deprecated and non-functional**
  - System currently uses Google BigQuery access for data
  - Multi-language support with English translation
  - Historical data from 1979 to present
- **[Yahoo Finance API](https://finance.yahoo.com/)** - Market data for 59+ financial instruments
  - Real-time and historical OHLCV data
  - Currency pairs, stocks, commodities, bonds, and indices
- **[Google BigQuery](https://cloud.google.com/bigquery)** - GDELT data access
  - Public datasets for GDELT queries
  - Cloud-based data warehouse for large-scale analysis

### Machine Learning & NLP Libraries
- **[ProsusAI FinBERT](https://huggingface.co/ProsusAI/finbert)** - Pre-trained financial sentiment analysis model
- **[HuggingFace Transformers](https://huggingface.co/transformers/)** - State-of-the-art NLP models
- **[XGBoost](https://xgboost.readthedocs.io/)** - Gradient boosting framework for machine learning
- **[scikit-learn](https://scikit-learn.org/)** - Logistic regression and comprehensive ML toolkit
- **[PyTorch](https://pytorch.org/)** - Deep learning framework for FinBERT

### Model Interpretability
- **[SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/)** - Lundberg & Lee (2017)
  - Model-agnostic interpretability framework
  - Feature importance and prediction explanations
- **[LIME](https://github.com/marcotcr/lime)** - Local Interpretable Model-agnostic Explanations
  - Local model interpretability for individual predictions

### Python Libraries & Frameworks
- **[pandas](https://pandas.pydata.org/)** - Data manipulation and analysis
- **[numpy](https://numpy.org/)** - Numerical computing foundation
- **[matplotlib](https://matplotlib.org/)** & **[seaborn](https://seaborn.pydata.org/)** - Data visualization
- **[plotly](https://plotly.com/python/)** - Interactive visualizations and dashboards
- **[pyarrow](https://arrow.apache.org/docs/python/)** - Parquet file format support
- **[requests](https://requests.readthedocs.io/)** - HTTP library for API calls
- **[argparse](https://docs.python.org/3/library/argparse.html)** - Command-line interface
- **[logging](https://docs.python.org/3/library/logging.html)** - Application logging

### Financial Data & APIs
- **[Yahoo Finance](https://finance.yahoo.com/)** - Real-time and historical market data
- **[Google Cloud BigQuery](https://cloud.google.com/bigquery)** - Cloud data warehouse
- **[GDELT BigQuery Datasets](https://console.cloud.google.com/marketplace/product/gdelt-bq)** - Public datasets for global event data

### Technical Analysis
- **[TA-Lib](https://ta-lib.org/)** - Technical Analysis Library
  - 158+ technical indicators
  - C++ implementation with Python bindings
  - Industry-standard technical analysis functions

### Research Methodology
- **Expanding Window Backtesting** - Prevents look-ahead bias in model evaluation
- **Cross-Validation** - Time series cross-validation for robust model selection
- **Feature Engineering** - 569 engineered features (126 sentiment + 443 market/technical)
- **Transaction Cost Modeling** - Realistic trading costs for accurate performance measurement
- **Risk Management** - Position sizing and portfolio optimization techniques

---

## License & Credits

**License:** MIT

**Acknowledgments:**
- GDELT Project (global news events)
- ProsusAI (FinBERT model)
- HuggingFace Transformers
- Yahoo Finance API
- TA-Lib (technical analysis library)

---

## Documentation

- `CLAUDE.md` - Troubleshooting & known issues
- `TRADING_QUICKSTART.md` - Beginner guide
- `results/*/BACKTEST_VALIDATION_REPORT.md` - Performance analysis

**Support:** [GitHub Issues](https://github.com/danielryang/macro_sentiment_trading/issues)
