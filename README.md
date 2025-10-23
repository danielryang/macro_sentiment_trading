# Macro Sentiment Trading Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A quantitative finance system implementing research for macro sentiment-based trading. This pipeline combines global news event data from GDELT, financial sentiment analysis using FinBERT, and machine learning models to generate systematic trading signals for major currency pairs and financial assets.

This software is for educational and research purposes only. Not financial advice. Trading involves risk.

## Objective

This system demonstrates how news sentiment from macro-relevant events contains predictive information for future price movements in major financial assets. The pipeline implements:

- **Global News Analysis**: Real-time processing of worldwide events from GDELT database
- **Financial Sentiment Analysis**: Natural Language Processing using FinBERT for market sentiment extraction
- **Machine Learning Models**: Ensemble methods (Logistic Regression, XGBoost) for signal generation
- **Multi-Timeframe Analysis**: Signals across 1D, 1W, 1M, 1Q, 1Y horizons
- **Validation**: Expanding window backtesting to prevent look-ahead bias
- **Model Interpretability**: SHAP analysis for feature importance and model explainability

## Usage

### Quick Start

```bash
# Clone and setup
git clone https://github.com/danielryang/macro_sentiment_trading.git
cd macro_sentiment_trading
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install -e .

# Get current trading signals
python cli/main.py get-signals

# Run full pipeline (1 year analysis)
python cli/main.py run-pipeline --start-date 2023-01-01 --end-date 2023-12-31
```

### Key Commands

```bash
# Multi-timeframe signals
python cli/main.py multi-timeframe-signals --assets EURUSD GBPUSD --timeframes 1D 1W 1M 1Y

# Backtesting
python cli/main.py multi-timeframe-backtest --assets EURUSD --models xgboost

# Data collection only
python cli/main.py collect-news --start-date 2024-01-01 --end-date 2024-01-31

# System status
python cli/main.py status
```

### Example Usage & Notebooks

For detailed examples and interactive tutorials, see the Jupyter notebooks:

- **`notebooks/01_training_simulation.ipynb`** - Complete pipeline walkthrough
- **`notebooks/02_signal_generation.ipynb`** - Signal generation examples  
- **`notebooks/03_alpha_analytics.ipynb`** - Performance analysis and visualization

```bash
# Launch Jupyter to explore notebooks
jupyter notebook notebooks/
```

## API Setup & Configuration

### GDELT BigQuery Setup (Recommended)

1. **Create Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create new project or select existing
   - Enable BigQuery API

2. **Enable Billing** (Required for BigQuery):
   - Go to Billing in Google Cloud Console
   - Link a payment method
   - **Note**: GDELT queries are typically free within 1TB/month limit

3. **Create Service Account**:
   ```bash
   # Download credentials file
   # Save as: macro-sentiment-trading-credentials.json
   ```

4. **Set Environment Variables**:
   ```bash
   # Add to .env file
   GOOGLE_APPLICATION_CREDENTIALS=path/to/macro-sentiment-trading-credentials.json
   GOOGLE_CLOUD_PROJECT=your-project-id
   GDELT_METHOD=bigquery
   ```

### GDELT Free API Setup (Alternative)

1. **No Setup Required**:
   - Uses GDELT's free API endpoints
   - Limited to recent data (last 2 years)
   - Slower but no billing required

2. **Configure for Free Method**:
   ```bash
   # Add to .env file
   GDELT_METHOD=free
   ```

### Yahoo Finance API (No Setup Required)

- **Automatic**: No API keys needed
- **Rate Limits**: Built-in throttling
- **Coverage**: 35+ financial instruments

### Optional: Advanced Configuration

```bash
# .env file example
GDELT_METHOD=bigquery                    # 'bigquery' or 'free'
GOOGLE_APPLICATION_CREDENTIALS=credentials.json
GOOGLE_CLOUD_PROJECT=your-project-id
BIGQUERY_MAX_COST_USD=5.00              # Cost limit for safety
CACHE_SENTIMENT=true                    # Enable sentiment caching
LOG_LEVEL=INFO                          # DEBUG, INFO, WARNING, ERROR
```

### Supported Assets

**35+ Financial Instruments**: EURUSD, USDJPY, GBPUSD, AUDUSD, BTCUSD, ETHUSD, SPY, QQQ, GOLD, TNOTE, and more.

### Performance Expectations

- **Runtime**: 1 month (5 min), 1 year (45 min), 3+ years (3 hours)
- **Signal Quality**: Sharpe 0.0-2.0, Win Rate 35-65%, Max Drawdown -1% to -15%
- **Assets**: 35+ FX pairs, cryptocurrencies, equities, commodities

## Sources & References

### Primary Research
- **arXiv:2505.16136v1** - "Macro Sentiment Trading: A Novel Approach to Systematic Trading Using Global News Sentiment" (Primary research foundation)
- **arXiv:1908.10063** - "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models" (Sentiment analysis foundation)
- **arXiv:1706.03762** - "Attention Is All You Need" (Transformer architecture for FinBERT)
- **arXiv:1810.04805** - "BERT: Pre-training of Deep Bidirectional Transformers" (BERT foundation)
- **Journal of Financial Economics** - "News and Stock Returns" (News impact on markets)
- **Review of Financial Studies** - "Textual Analysis in Finance" (NLP applications in finance)

### Data Sources
- **GDELT Project** - Global Database of Events, Language, and Tone (https://www.gdeltproject.org/)
- **Yahoo Finance API** - Market data for 35+ financial instruments
- **Google BigQuery** - GDELT data access via BigQuery public datasets

### Machine Learning & NLP
- **ProsusAI FinBERT** - "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models" (HuggingFace)
- **HuggingFace Transformers** - Pre-trained language models and tokenizers
- **XGBoost** - Gradient boosting framework for machine learning
- **scikit-learn** - Logistic regression and other ML algorithms

### Model Interpretability
- **SHAP (SHapley Additive exPlanations)** - Lundberg & Lee (2017) - Model interpretability framework
- **LIME** - Local Interpretable Model-agnostic Explanations

### Financial Data & APIs
- **Yahoo Finance** - Real-time and historical market data
- **Google Cloud BigQuery** - Cloud data warehouse for GDELT queries
- **GDELT BigQuery Datasets** - Public datasets for global event data

### Python Libraries & Frameworks
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning toolkit
- **xgboost** - Gradient boosting
- **transformers** - HuggingFace transformers library
- **torch** - PyTorch for deep learning
- **matplotlib/seaborn** - Data visualization
- **plotly** - Interactive visualizations
- **pyarrow** - Parquet file format support
- **requests** - HTTP library for API calls
- **argparse** - Command-line interface
- **logging** - Application logging

### Academic & Research Papers
- **"Attention Is All You Need"** - Vaswani et al. (2017) - Transformer architecture
- **"BERT: Pre-training of Deep Bidirectional Transformers"** - Devlin et al. (2018)
- **"A Unified Approach to Interpreting Model Predictions"** - Lundberg & Lee (2017) - SHAP values
- **"XGBoost: A Scalable Tree Boosting System"** - Chen & Guestrin (2016)
- **"Textual Analysis in Finance"** - Loughran & McDonald (2011) - Financial text analysis
- **"News and Stock Returns"** - Tetlock (2007) - News sentiment impact
- **"Machine Learning for Asset Management"** - Gu et al. (2020) - ML in finance
- **"Deep Learning for Finance"** - Dixon et al. (2017) - Neural networks in trading
- **"Sentiment Analysis in Financial Markets"** - Loughran & McDonald (2016) - Market sentiment
- **"Natural Language Processing for Financial Text"** - Yang et al. (2019) - NLP in finance

### Development & Infrastructure
- **Git** - Version control
- **MIT License** - Open source licensing
- **Python 3.8+** - Programming language
- **Jupyter Notebooks** - Interactive development environment

### Documentation & Standards
- **Markdown** - Documentation format
- **Black** - Python code formatting
- **PEP 8** - Python style guide
- **Semantic Versioning** - Version numbering

---

**Quick Reference**: `python cli/main.py --help` for all commands, `python cli/main.py status` for system check.