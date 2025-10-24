# Macro Sentiment Trading Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready quantitative trading system combining GDELT news data, FinBERT transformer sentiment analysis, and machine learning with **569 engineered features** to generate systematic trading signals for FX, crypto, and commodities.

**⚡ Latest (Oct 2025):** Enhanced architecture with 569 features (17x increase), FinBERT headline analysis, validated backtesting infrastructure.

**⚠️ Disclaimer:** Educational and research purposes only. Not financial advice.

---

## Key Features

- **🧠 FinBERT Analysis**: Transformer-based sentiment on 18K+ headlines (CUDA-accelerated)
- **📊 569 Features**: 126 sentiment + 443 market/technical indicators (auto-generated)
- **🎯 Multi-Asset**: 35+ assets (EURUSD, USDJPY, BTCUSD, GOLD, SP500)
- **🔬 Academic Rigor**: Based on [arXiv:2505.16136v1](https://arxiv.org/abs/2505.16136v1), enhanced with [FinBERT](https://arxiv.org/abs/1908.10063)
- **⚡ Fast**: 18K events processed in ~9 minutes end-to-end
- **✅ Validated**: Backtesting infrastructure tested on 2025 data

---

## Architecture

### Feature Engineering (569 Total)

**Sentiment (126 features):**
- FinBERT scores (5 base + 40 derivatives)
- GDELT tone (2 base + 16 derivatives)
- Article impact (1 base + 8 derivatives)
- Goldstein scale (2 base + 16 derivatives)
- News volume & acceleration (36)
- Cross-lag interactions (47)

**Market/Technical (443 features):**
- RSI, SMA, Bollinger Bands, MACD
- Lagged returns (1-5 days, multiple windows)
- Volatility measures (20+ variants)
- Moving averages & momentum (176)
- Cross-feature interactions (200+)

### Models
- **XGBoost Classifier**: Tree-based ensemble, no scaling needed
- **Logistic Regression**: Linear model with StandardScaler
- **3-Class Output**: SELL (-1), HOLD (0), BUY (+1)

---

## Quick Start

### Installation

```bash
git clone https://github.com/danielryang/macro_sentiment_trading.git
cd macro_sentiment_trading
python -m venv venv
venv\Scripts\activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Get Current Trading Signals

```bash
python cli/main.py get-signals
```

**Output:**
```
EUR/USD: Xgboost: SELL (-1) | Confidence: 50.0%
USD/JPY: Xgboost: BUY  (+1) | Confidence: 50.0%
```

### Run Full Pipeline

```bash
# 7 months of data (Apr-Oct 2025)
python cli/main.py run-pipeline \
  --start-date 2025-04-01 \
  --end-date 2025-10-23 \
  --assets EURUSD USDJPY
```

**Pipeline Stages:**
1. **Data Collection** (GDELT BigQuery): 18,900 events, 18,864 headlines
2. **Sentiment Processing** (FinBERT): 189 days × 69 sentiment features
3. **Market Alignment** (Yahoo Finance): 133 days × 577 total features
4. **Model Training**: 4 models (2 assets × 2 algorithms)
5. **Signal Generation**: Production-ready JSON output

**Runtime:** ~10 minutes for 7 months

---

## Common Commands

```bash
# Check system status
python cli/main.py status

# Get signals for specific assets
python cli/main.py get-signals --assets EURUSD GBPUSD

# Save signals to file
python cli/main.py get-signals --output-file signals.json

# Train models on historical data
python cli/main.py train-models \
  --assets EURUSD \
  --data-path results/aligned_data_EURUSD.parquet

# Multi-timeframe analysis (1D, 1W, 1M)
python cli/main.py multi-timeframe-signals \
  --assets EURUSD GBPUSD \
  --timeframes 1D 1W 1M

# Run backtesting
python cli/main.py multi-timeframe-backtest \
  --assets EURUSD \
  --models xgboost
```

---

## Performance Results

### Validated Backtest (Oct 2025, 133 days, 569 features)

| Asset | Sharpe | Calmar | Return | Win Rate | Max DD | Trades |
|-------|--------|--------|--------|----------|--------|--------|
| EURUSD | 0.10 | 0.20 | +0.07% | 48.8% | -2.14% | 43 |
| USDJPY | -1.61 | -3.17 | -2.37% | 53.5% | -4.38% | 43 |

**Status:** Preliminary (43 trades vs 100+ recommended)

### Statistical Significance Requirements

| Requirement | Current | Target | Coverage |
|-------------|---------|--------|----------|
| **Trading Days** | 133 (6 months) | 500-750 (2-3 years) | 18-27% |
| **Test Trades** | 43 per asset | 100+ | 43% |
| **Confidence** | Preliminary | 95% with 100+ trades | - |

**Recommendation:** Collect 3+ years of data for institutional-grade validation.

### Expected Ranges (with 3+ years)
- Sharpe Ratio: 0.0-2.0 (excellent > 1.5)
- Win Rate: 35-65% (good > 50%)
- Max Drawdown: -1% to -15% (excellent < -5%)

---

## Configuration

### Environment Variables (.env)

```bash
# BigQuery (optional, faster than free API)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
BIGQUERY_MAX_COST_USD=5.00

# Performance
BATCH_SIZE=128  # FinBERT batch size
LOG_LEVEL=INFO
```

### Date Range Guidelines

```bash
# Data exploration (fast)
--start-date 2025-01-01 --end-date 2025-01-31 --skip-training

# Minimum for training (6 months)
--start-date 2025-04-01 --end-date 2025-10-23

# Recommended for production (3+ years)
--start-date 2022-01-01 --end-date 2025-10-23
```

---

## Supported Assets (35+)

**FX:** EURUSD, USDJPY, GBPUSD, AUDUSD, USDCHF, USDCAD, NZDUSD, EURGBP, EURJPY, GBPJPY
**Crypto:** BTCUSD, ETHUSD, ADAUSD, DOGEUSD, SOLUSD, AVAXUSD
**Equities:** SPY, QQQ, IWM
**Commodities:** GOLD, TNOTE (Treasury futures)

---

## Output Files

```
results/
├── lifecycle_2025/
│   ├── events_data_20250401_20251023.parquet       # GDELT data
│   ├── daily_features_20250401_20251023.parquet    # 69 sentiment features
│   ├── 20250401_20251023/
│   │   ├── aligned_data_EURUSD.parquet             # 577 features
│   │   └── aligned_data_USDJPY.parquet
│   ├── models/
│   │   ├── EURUSD_xgboost_20251023_*.pkl           # Trained models
│   │   └── training_metrics.parquet
│   └── production_signals.json                     # Trading signals
```

---

## Research Foundation

### Primary Papers
1. **Macro Sentiment Trading** ([arXiv:2505.16136v1](https://arxiv.org/abs/2505.16136v1))
   - Original framework: Sentiment → price prediction
   - Baseline: 33 features, Sharpe 4.65-5.87 (theoretical)

2. **FinBERT** ([arXiv:1908.10063](https://arxiv.org/abs/1908.10063))
   - Deep learning for financial sentiment
   - 97% accuracy on Financial PhraseBank

### Our Enhancements
- **17x More Features**: 33 → 569 features
- **FinBERT on Headlines**: Transformer analysis vs simple tone scores
- **Technical Integration**: TA-Lib indicators + price action
- **Production Infrastructure**: BigQuery caching, 9-minute pipeline
- **Validated Backtest**: Real 2025 data, realistic performance

---

## Troubleshooting

**Import Errors:**
```bash
pip install -r requirements.txt  # Reinstall dependencies
```

**BigQuery Quota:**
```bash
# Use free API instead
python cli/main.py run-pipeline --method free --start-date 2025-04-01 --end-date 2025-10-23
```

**Memory Issues:**
```bash
# Reduce assets or date range
python cli/main.py run-pipeline --assets EURUSD --start-date 2025-09-01 --end-date 2025-10-23
```

**Logs:** Check `logs/cli.log` for detailed errors

---

## Development

### Code Organization
```
cli/commands/     # CLI command implementations
src/              # Core pipeline modules
├── data_collector.py           # GDELT collection
├── sentiment_analyzer.py       # FinBERT processing
├── market_processor.py         # Feature engineering
├── model_trainer.py            # ML training (569 features)
└── model_persistence.py        # Model saving/loading
```

### Running Tests
```bash
python cli/main.py status  # System check
python cli/main.py get-signals --assets EURUSD  # Test signal generation
```

---

## License & Acknowledgments

**License:** MIT

**Credits:**
- GDELT Project (global news database)
- ProsusAI (FinBERT model)
- HuggingFace (Transformers library)
- Yahoo Finance (market data)

---

## References

**Full Documentation:**
- `CLAUDE.md` - Troubleshooting & known issues
- `TRADING_QUICKSTART.md` - Quick start guide
- `results/lifecycle_2025/BACKTEST_VALIDATION_REPORT.md` - Detailed performance analysis

**Support:** [GitHub Issues](https://github.com/danielryang/macro_sentiment_trading/issues)
