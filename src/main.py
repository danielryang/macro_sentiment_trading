"""
Main Pipeline Orchestration Module

This module orchestrates the complete macro sentiment trading pipeline, which includes:
1. News collection and filtering from GDELT
2. Headline extraction and processing
3. Sentiment analysis using FinBERT
4. Market data processing and feature engineering
5. Model training and backtesting
6. Performance evaluation and SHAP analysis

The pipeline processes data from 2015-01-01 to the present and generates trading signals
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
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.news_collector import GDELTCollector
    from src.headline_processor import HeadlineProcessor
    from src.sentiment_analyzer import SentimentAnalyzer
    from src.market_processor import MarketProcessor
    from src.model_trainer import ModelTrainer
    from src.config_validator import ConfigValidator
except ImportError:
    from .news_collector import GDELTCollector
    from .headline_processor import HeadlineProcessor
    from .sentiment_analyzer import SentimentAnalyzer
    from .market_processor import MarketProcessor
    from .model_trainer import ModelTrainer
    from .config_validator import ConfigValidator

# Configure logging with both console and file handlers
def setup_logging(log_level: str = "INFO", log_file: str = "pipeline.log"):
    """Setup comprehensive logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

def run_pipeline(start_date: str, end_date: str, should_collect_news: bool, should_process_headlines: bool, validate_config: bool = True):
    """
    Run the complete trading pipeline.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        should_collect_news: Boolean flag to control news collection
        should_process_headlines: Boolean flag to control headline processing
        validate_config: Whether to validate configuration before running
    """
    logger.info("Starting pipeline...")
    
    # Validate configuration if requested
    if validate_config:
        logger.info("Validating configuration...")
        validator = ConfigValidator()
        is_valid = validator.validate_environment()
        
        if not is_valid:
            logger.error("Configuration validation failed. Please fix the errors and try again.")
            validator.print_validation_report()
            return None
        
        # Validate date range
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            today = datetime.now()
            
            if end_dt > today:
                logger.error(f"End date {end_date} is in the future. Using today's date instead.")
                end_date = today.strftime("%Y-%m-%d")
                end_dt = today
                
            if start_dt >= end_dt:
                logger.error(f"Start date {start_date} must be before end date {end_date}")
                return None
                
            if (end_dt - start_dt).days < 30:
                logger.warning(f"Date range is only {(end_dt - start_dt).days} days, which may be too small for reliable results")
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            return None
    
    # Initialize components
    news_collector = GDELTCollector()
    headline_processor = HeadlineProcessor()
    sentiment_analyzer = SentimentAnalyzer()
    market_processor = MarketProcessor()
    model_trainer = ModelTrainer()
    
    # Step 1: Collect and process news
    gdelt_data_path = f'data/news/gdelt_{start_date}_{end_date}.parquet'
    
    if should_collect_news:
        logger.info("Collecting news data...")
        events_df = news_collector.fetch_events(start_date, end_date)
    else:
        logger.info("Skipping news collection - using existing data")
        events_df = pd.read_parquet(gdelt_data_path)
    
    if should_process_headlines:
        logger.info("Processing headlines...")
        events_df = headline_processor.process_articles(events_df)
    else:
        logger.info("Skipping headline processing - headlines already in data")
    
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
    
    # Step 6: Train models and backtest with SHAP analysis
    logger.info("Training models and running backtest...")
    results = {}
    metrics = {}
    all_shap_values = {}
    
    for asset_name, data in aligned_data.items():
        logger.info(f"Processing {asset_name}...")
        
        # Set realistic transaction costs based on typical spreads
        # EURUSD/USDJPY: ~0.5-1.0 pips = 0.00005-0.0001, using 0.0001 (0.01%) for conservative estimate
        # Treasury futures: ~0.5-1.0 ticks = 0.0001-0.0002, using 0.0002 (0.02%) for conservative estimate
        transaction_cost = 0.0001 if asset_name in ['EURUSD', 'USDJPY'] else 0.0002
        
        # Run backtest with automatic SHAP analysis and result saving
        asset_results, asset_metrics, asset_shap = model_trainer.backtest(
            data, 
            transaction_cost, 
            asset_name=asset_name, 
            save_results=True
        )
        
        results[asset_name] = asset_results
        metrics[asset_name] = asset_metrics
        all_shap_values[asset_name] = asset_shap
        
        logger.info(f"COMPLETED {asset_name}: {len(asset_results)} models, {len(asset_shap)} SHAP analyses")
    
    # Save comprehensive performance summary
    logger.info("Saving comprehensive performance summary...")
    try:
        # Create a comprehensive metrics summary
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv('results/comprehensive_performance_metrics.csv')
        logger.info("SAVED comprehensive performance metrics")
        
        # Create SHAP summary if available
        if all_shap_values:
            shap_summary = {}
            for asset_name, asset_shap in all_shap_values.items():
                for model_name, shap_data in asset_shap.items():
                    if 'importance' in shap_data:
                        importance_df = shap_data['importance']
                        # Get top 5 features for summary
                        top_features = importance_df.head(5)
                        shap_summary[f"{asset_name}_{model_name}"] = top_features['feature'].tolist()
            
            # Save SHAP summary
            import json
            with open('results/shap_features_summary.json', 'w') as f:
                json.dump(shap_summary, f, indent=2)
            logger.info("SAVED SHAP features summary")
    except Exception as e:
        logger.error(f"ERROR saving summary files: {e}")
    
    logger.info("Pipeline completed successfully!")

def main():
    """Entry point for command line execution."""
    # Create necessary directories
    import os
    os.makedirs('results', exist_ok=True)
    os.makedirs('data/news', exist_ok=True)
    os.makedirs('data/raw/gdelt', exist_ok=True)
    
    # Control flags for pipeline steps
    start_date = "2015-02-18"
    end_date = datetime.now().strftime("%Y-%m-%d")  # Use current date as end date
    
    # Check for existing data files
    gdelt_data_path = f'data/news/gdelt_{start_date}_{end_date}.parquet'
    should_collect_news = not os.path.exists(gdelt_data_path)  # Skip if data exists
    should_process_headlines = should_collect_news  # If we need to collect news, we also need to process headlines
    
    print(f"News collection: {'enabled' if should_collect_news else 'skipped - using existing data'}")
    print(f"Headline processing: {'enabled' if should_process_headlines else 'skipped - using existing data'}")
    
    run_pipeline(start_date, end_date, should_collect_news, should_process_headlines)

if __name__ == "__main__":
    main() 