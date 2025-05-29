# Macro Sentiment Trading Pipeline

This project implements an end-to-end pipeline for trading macro assets based on news sentiment analysis. The pipeline combines news data from GDELT, sentiment analysis using FinBERT, and machine learning models to generate trading signals.

## Pipeline Overview

1. **News Collection and Filtering**
   - Sources daily event records from GDELT v2 API
   - Filters for macro-relevant events (EventCode 100-199)
   - Retains top 100 events per day by article coverage

2. **Headline Extraction**
   - Extracts headlines from article URLs
   - Cleans and normalizes text
   - Truncates to 512 WordPiece tokens

3. **Sentiment Scoring**
   - Uses FinBERT model for sentiment analysis
   - Computes polarity scores (-1 to +1)
   - Generates daily sentiment features

4. **Market Data Processing**
   - Downloads price data for EUR/USD, USD/JPY, and Treasury futures
   - Computes returns and technical features
   - Aligns market data with sentiment features

5. **Predictive Modeling**
   - Implements both logistic regression and XGBoost models
   - Uses expanding window backtest approach
   - Includes transaction costs in performance calculation

6. **Model Interpretation**
   - Computes SHAP values for feature importance
   - Generates performance metrics and visualizations

## Project Structure

```
.
├── src/
│   ├── news_collector.py      # GDELT data collection
│   ├── headline_processor.py  # Headline extraction and cleaning
│   ├── sentiment_analyzer.py  # FinBERT sentiment analysis
│   ├── market_processor.py    # Market data processing
│   ├── model_trainer.py       # Model training and backtesting
│   └── main.py               # Pipeline orchestration
├── data/                     # Data storage
├── results/                  # Backtest results and metrics
├── notebooks/               # Jupyter notebooks for analysis
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/macro_sentiment_trading.git
cd macro_sentiment_trading
```

2. Create a virtual environment:
   
   a) On Windows:
   ```sh
   python -m venv venv
   venv\Scripts\activate
   ```
   
   b) On macOS/Linux:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Setup (Alternative Installation)

If you prefer using Docker, you can set up the environment using the provided Dockerfile:

1. Build the Docker image:
```bash
docker build -t macro-sentiment-trading .
```

2. Run the container:
```bash
docker run -it macro-sentiment-trading
```

For development with data persistence:
```bash
docker run -it \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    macro-sentiment-trading
```

Note: Replace `$(pwd)` with `%cd%` on Windows PowerShell.

## Usage

Run the complete pipeline:
```bash
python src/main.py
```

The pipeline will:
1. Collect news data from GDELT
2. Process headlines and compute sentiment scores
3. Download and process market data
4. Train models and run backtests
5. Generate performance metrics and SHAP values
6. Save results to the `results/` directory

## Results

The pipeline generates several output files:
- `results/{asset}_{model}_results.csv`: Backtest results for each asset and model
- `results/{asset}_shap_values.csv`: SHAP values for feature importance
- `results/performance_metrics.csv`: Summary performance metrics

## Dependencies

- Python 3.8+
- See `requirements.txt` for full list of dependencies

## Extending the Framework

The pipeline is designed to be modular and extensible. Here are some ways to extend it:

1. **Add New Assets**
   - Add new tickers to the `assets` dictionary in `MarketProcessor`
   - Adjust transaction costs in `main.py` if needed
   - The pipeline will automatically handle the new assets

2. **Customize Feature Engineering**
   - Add new market features in `MarketProcessor.compute_market_features()`
   - Add new sentiment features in `SentimentAnalyzer.compute_daily_features()`
   - The feature alignment process will automatically include new features

3. **Add New Models**
   - Add new model classes to `ModelTrainer.models`
   - Implement required methods (fit, predict_proba)
   - The backtesting framework will automatically include new models

4. **Modify Backtest Parameters**
   - Adjust fold duration in `ModelTrainer.backtest()`
   - Change transaction costs in `main.py`
   - Modify performance metrics in `ModelTrainer.compute_metrics()`

5. **Integrate with Production Systems**
   - Add database integration for storing results
   - Implement real-time data feeds
   - Add API endpoints for model predictions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- GDELT Project for providing the news data API
- ProsusAI for developing and open-sourcing the FinBERT model
- HuggingFace for maintaining the transformers library
- Yahoo Finance for market data access
- The open-source community for the various Python packages used in this project

## Reference

This implementation is based on the research paper: [arXiv:2505.16136v1](https://arxiv.org/abs/2505.16136v1)


