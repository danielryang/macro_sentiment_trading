# Macro Sentiment Trading Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-ready quantitative trading system combining GDELT news sentiment, FinBERT NLP, and machine learning with **569 engineered features** for systematic signals across 35+ assets (FX, crypto, equities, commodities).

**Latest (Oct 2025):** 569-feature architecture (17x increase), FinBERT headline analysis, validated backtest infrastructure, model registry system.

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

## Supported Assets (35+)

**FX (10):** EURUSD, USDJPY, GBPUSD, AUDUSD, USDCHF, USDCAD, NZDUSD, EURGBP, EURJPY, GBPJPY
**Crypto (6):** BTCUSD, ETHUSD, ADAUSD, DOGEUSD, SOLUSD, AVAXUSD
**Equities (3):** SPY (S&P 500), QQQ (Nasdaq), IWM (Russell 2000)
**Commodities (2):** GOLD, TNOTE (Treasury futures)

Add custom assets in `src/market_processor.py`:
```python
ASSET_MAPPINGS = {
    'CUSTOM': 'TICKER-SYMBOL'  # Yahoo Finance format
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
