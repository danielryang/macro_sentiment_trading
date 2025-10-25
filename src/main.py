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
    from src.config import Config
except ImportError:
    from .news_collector import GDELTCollector
    from .headline_processor import HeadlineProcessor
    from .sentiment_analyzer import SentimentAnalyzer
    from .market_processor import MarketProcessor
    from .model_trainer import ModelTrainer
    from .config import Config

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
        validator = Config()
        validation_result = validator.validate_configuration()
        
        if not validation_result.get('valid', True):
            logger.error("Configuration validation failed. Please fix the errors and try again.")
            logger.error(f"Validation issues: {validation_result}")
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
    
    # Step 1: Collect and process news (using the same method as notebook)
    if should_collect_news:
        logger.info("Collecting news data...")
        from src.data_collector import collect_and_process_news
        
        # Use the same caching-enabled helper function as the notebook
        events_df = collect_and_process_news(
            start_date=start_date,
            end_date=end_date,
            force_refresh=False,  # Use cache if available
            use_method=None,      # Auto-detect (BigQuery preferred)
            top_n_per_day=100
        )
        logger.info(f"Collected {len(events_df)} news events")
    else:
        logger.info("Skipping news collection - using existing data")
        # Try to load from cache first
        cache_file = f"data/cache/events_data_{start_date}_{end_date}.parquet"
        if os.path.exists(cache_file):
            events_df = pd.read_parquet(cache_file)
            logger.info(f"Loaded {len(events_df)} events from cache")
        else:
            logger.error("No cached data found and news collection is disabled")
            return None
    
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
    
    # Step 6: Train models first (same as notebook)
    logger.info("Training models...")
    trained_models = {}
    
    for asset_name, data in aligned_data.items():
        logger.info(f"Training models for {asset_name}...")
        logger.info(f"  Data shape: {data.shape}")
        
        # Check minimum data requirements
        if len(data) < 30:
            logger.error(f"SKIPPING {asset_name}: Insufficient data ({len(data)} samples, need 30+)")
            continue
        
        if 'target' not in data.columns:
            logger.error(f"SKIPPING {asset_name}: No target column found")
            continue
        
        # Check target distribution
        target_counts = data['target'].value_counts()
        min_class_samples = target_counts.min()
        if min_class_samples < 5:
            logger.error(f"SKIPPING {asset_name}: Insufficient samples per class ({min_class_samples}, need 5+)")
            continue
        
        try:
            # Train both models (same as notebook)
            models, scalers, feature_cols = model_trainer.train_models(data)
            
            trained_models[asset_name] = {
                'models': models,
                'scalers': scalers,
                'feature_cols': feature_cols
            }
            
            logger.info(f"SUCCESS: Trained {len(models)} models for {asset_name}")
            
        except Exception as e:
            logger.error(f"ERROR: Failed to train models for {asset_name}: {str(e)}")
            continue
    
    # Step 7: Run backtest (same as notebook)
    logger.info("Running backtests...")
    results = {}
    metrics = {}
    all_shap_values = {}
    
    successful_assets = []
    failed_assets = []
    
    for asset_name, data in aligned_data.items():
        logger.info(f"Processing {asset_name}...")
        
        # CRITICAL: Validate data before processing
        if data is None or len(data) == 0:
            logger.error(f"SKIPPING {asset_name}: No data available")
            failed_assets.append(asset_name)
            continue
        
        if 'target' not in data.columns:
            logger.error(f"SKIPPING {asset_name}: No target column found")
            failed_assets.append(asset_name)
            continue
        
        # Check minimum data requirements
        if len(data) < 30:
            logger.error(f"SKIPPING {asset_name}: Insufficient data ({len(data)} samples, need 30+)")
            failed_assets.append(asset_name)
            continue
        
        # Check target distribution
        target_counts = data['target'].value_counts()
        min_class_samples = target_counts.min()
        if min_class_samples < 5:
            logger.error(f"SKIPPING {asset_name}: Insufficient samples per class ({min_class_samples}, need 5+)")
            failed_assets.append(asset_name)
            continue
        
        logger.info(f"  Data shape: {data.shape}")
        logger.info(f"  Target distribution: {dict(target_counts)}")
        
        try:
            # Set realistic transaction costs based on typical spreads
            # EURUSD/USDJPY: ~0.5-1.0 pips = 0.00005-0.0001, using 0.0001 (0.01%) for conservative estimate
            # Treasury futures: ~0.5-1.0 ticks = 0.0001-0.0002, using 0.0002 (0.02%) for conservative estimate
            transaction_cost = 0.0001 if asset_name in ['EURUSD', 'USDJPY'] else 0.0002
            
            # Use the same approach as the notebook - manual backtesting
            from src.performance_metrics import PerformanceAnalyzer
            perf_analyzer = PerformanceAnalyzer()
            
            # Get trained models for this asset (same as notebook)
            if asset_name not in trained_models:
                logger.error(f"No trained models found for {asset_name}")
                continue
                
            models = trained_models[asset_name]['models']
            scalers = trained_models[asset_name]['scalers']
            feature_cols = trained_models[asset_name]['feature_cols']
            
            asset_results = {}
            asset_metrics = {}
            asset_shap = {}
            
            # Loop through each model (same as notebook)
            for model_name, model in models.items():
                logger.info(f"  Backtesting {model_name} for {asset_name}...")
                
                # Get scaler if available
                scaler = scalers.get(model_name, None)
                
                # Generate signals (same as notebook)
                signals = model_trainer.generate_signals(model, data, scaler=scaler, feature_cols=feature_cols)
                
                # Compute strategy returns (same as notebook)
                strategy_returns = model_trainer.compute_returns(signals, data, transaction_cost)
                
                # Compute metrics (same as notebook)
                metrics = perf_analyzer.compute_comprehensive_metrics(strategy_returns)
                
                # Store results (same structure as notebook)
                asset_results[model_name] = {
                    'signals': signals,
                    'returns': strategy_returns,
                    'metrics': metrics
                }
                asset_metrics[model_name] = metrics
            
            results[asset_name] = asset_results
            metrics[asset_name] = asset_metrics
            all_shap_values[asset_name] = asset_shap
            
            logger.info(f"SUCCESS: Completed {asset_name}: {len(asset_results)} models, {len(asset_shap)} SHAP analyses")
            successful_assets.append(asset_name)
            
        except Exception as e:
            logger.error(f"ERROR: Failed {asset_name}: {str(e)}")
            failed_assets.append(asset_name)
            continue
    
    # Log summary
    logger.info(f"Pipeline completed!")
    logger.info(f"  SUCCESS: {len(successful_assets)} assets")
    logger.info(f"  FAILED: {len(failed_assets)} assets")
    
    if successful_assets:
        logger.info(f"  Successful: {', '.join(successful_assets)}")
    
    if failed_assets:
        logger.info(f"  Failed: {', '.join(failed_assets)}")
        logger.info("  Note: Failed assets may have insufficient data, date misalignment, or other data quality issues")
    
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
    return True

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