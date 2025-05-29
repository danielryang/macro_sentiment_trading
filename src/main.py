"""
Main Pipeline Orchestration Module

This module orchestrates the complete macro sentiment trading pipeline, which includes:
1. News collection and filtering from GDELT
2. Headline extraction and processing
3. Sentiment analysis using FinBERT
4. Market data processing and feature engineering
5. Model training and backtesting
6. Performance evaluation and SHAP analysis

The pipeline processes data from 2015-01-01 to 2025-04-30 and generates trading signals
for EUR/USD, USD/JPY, and Treasury futures based on news sentiment analysis.

Inputs:
    - GDELT v2 API for news data
    - Yahoo Finance for market data
    - FinBERT model for sentiment analysis

Outputs:
    - Backtest results for each asset and model
    - SHAP values for feature importance
    - Performance metrics and visualizations

Reference: arXiv:2505.16136v1
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
from news_collector import GDELTCollector
from headline_processor import HeadlineProcessor
from sentiment_analyzer import SentimentAnalyzer
from market_processor import MarketProcessor
from model_trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(start_date: str, end_date: str):
    """
    Run the complete trading pipeline.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
    logger.info("Starting pipeline...")
    
    # Initialize components
    news_collector = GDELTCollector()
    headline_processor = HeadlineProcessor()
    sentiment_analyzer = SentimentAnalyzer()
    market_processor = MarketProcessor()
    model_trainer = ModelTrainer()
    
    # Step 1: Collect and process news
    logger.info("Collecting news data...")
    events_df = news_collector.fetch_events(start_date, end_date)
    
    logger.info("Processing headlines...")
    events_df = headline_processor.process_articles(events_df)
    
    # Step 2: Compute sentiment scores
    logger.info("Computing sentiment scores...")
    sentiment_df = sentiment_analyzer.compute_sentiment(events_df['headline'].tolist())
    sentiment_df['date'] = events_df['date']
    
    # Step 3: Compute daily sentiment features
    logger.info("Computing daily sentiment features...")
    daily_features = sentiment_analyzer.compute_daily_features(sentiment_df)
    
    # Step 4: Process market data
    logger.info("Processing market data...")
    market_data = market_processor.fetch_market_data(start_date, end_date)
    
    # Add market features
    for asset_name in market_data:
        market_data[asset_name] = market_processor.compute_market_features(
            market_data[asset_name]
        )
    
    # Step 5: Align features
    logger.info("Aligning features...")
    aligned_data = market_processor.align_features(market_data, daily_features)
    
    # Step 6: Train models and backtest
    logger.info("Training models and running backtest...")
    results = {}
    metrics = {}
    
    for asset_name, data in aligned_data.items():
        logger.info(f"Processing {asset_name}...")
        
        # Set transaction costs
        transaction_cost = 0.0002 if asset_name in ['EURUSD', 'USDJPY'] else 0.0005
        
        # Run backtest
        asset_results = model_trainer.backtest(data, transaction_cost)
        results[asset_name] = asset_results
        
        # Compute metrics
        asset_metrics = {}
        for model_name, model_results in asset_results.items():
            asset_metrics[model_name] = model_trainer.compute_metrics(
                model_results['returns']
            )
        metrics[asset_name] = asset_metrics
        
        # Generate SHAP values for XGBoost model
        if 'xgboost' in asset_results:
            logger.info(f"Computing SHAP values for {asset_name}...")
            shap_values = model_trainer.explain_predictions(
                model_trainer.models['xgboost'],
                data
            )
            shap_values.to_csv(f'results/{asset_name}_shap_values.csv')
    
    # Save results
    logger.info("Saving results...")
    for asset_name, asset_results in results.items():
        for model_name, model_results in asset_results.items():
            model_results.to_csv(f'results/{asset_name}_{model_name}_results.csv')
            
    # Save metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('results/performance_metrics.csv')
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run pipeline for the specified date range
    start_date = "2015-02-18"
    end_date = datetime.now().strftime("%Y-%m-%d")  # Use current date as end date
    run_pipeline(start_date, end_date) 