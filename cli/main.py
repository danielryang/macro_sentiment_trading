#!/usr/bin/env python3
"""
Main CLI Entry Point for Macro Sentiment Trading Pipeline

This is the primary command-line interface that orchestrates all pipeline operations.
Provides comprehensive logging, error handling, and progress tracking.
"""

import sys
import os
import argparse
import logging
from datetime import datetime, date
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file before any other imports
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed

from cli.config import CLIConfig

# Lazy imports for commands - imported only when needed to avoid slow startup


def setup_logging(log_level: str = "INFO", log_file: str = "logs/cli.log") -> logging.Logger:
    """Setup comprehensive logging with both file and console handlers."""
    # Create logs directory if it doesn't exist
    Path(log_file).parent.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def validate_date_range(start_date: str, end_date: str) -> tuple[date, date]:
    """Validate date range format and logic."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        if start >= end:
            raise argparse.ArgumentTypeError("Start date must be before end date")
        
        if end > date.today():
            raise argparse.ArgumentTypeError("End date cannot be in the future")
            
        return start, end
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid date format (use YYYY-MM-DD): {e}")


def validate_date_range_for_training(start_date: str, end_date: str, skip_training: bool = False) -> tuple[date, date]:
    """Validate date range with additional checks for model training."""
    start, end = validate_date_range(start_date, end_date)
    
    # Check minimum dataset size for training
    if not skip_training:
        days = (end - start).days + 1  # Include both start and end dates
        if days < 180:  # Minimum 6 months for reliable training
            raise argparse.ArgumentTypeError(
                f"Date range too small for reliable model training: {days} days. "
                f"Minimum 180 days (6 months) required for meaningful results. "
                f"Use --skip-training for shorter ranges or data exploration."
            )
        
        if days < 365:
            print(f"WARNING: Small dataset ({days} days). Recommend 365+ days (1+ years) for robust training.")
    
    return start, end


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all commands."""
    
    parser = argparse.ArgumentParser(
        prog="macro-sentiment-trading",
        description="Macro Sentiment Trading Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the complete pipeline
  python -m cli.main run-pipeline --start-date 2020-01-01 --end-date 2023-12-31

  # Collect news data only
  python -m cli.main collect-news --start-date 2020-01-01 --end-date 2020-01-31

  # Train models with custom parameters
  python -m cli.main train-models --data-path data/processed/aligned_data.parquet --models xgboost,logistic

  # Generate visualizations
  python -m cli.main visualize --results-path results/ --output-dir visualizations/

  # Get current trading signals
  python -m cli.main get-signals --output-file signals/current_signals.json

  # Get comprehensive multi-timeframe forecasts
  python -m cli.main forecast-signals --output-file forecasts/comprehensive_forecast.json

  # Run multi-timeframe backtesting
  python -m cli.main multi-timeframe-backtest --assets EURUSD USDJPY --models xgboost

  # Check pipeline status
  python -m cli.main status
        """
    )
    
    # Global options
    parser.add_argument("--config", help="Path to configuration file (default: cli/config.yaml)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Set the logging level (default: INFO)")
    parser.add_argument("--log-file", default="logs/cli.log", help="Path to log file (default: logs/cli.log)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    parser.add_argument("--output", action="store_true", help="Save comprehensive output and logs with all parameters")
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run Pipeline Command
    run_parser = subparsers.add_parser("run-pipeline", help="Run the complete trading pipeline")
    run_parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    run_parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    run_parser.add_argument("--assets", nargs="*", default=["EURUSD", "USDJPY", "TNOTE"],
                           help="Assets to process (default: EURUSD USDJPY TNOTE)")
    run_parser.add_argument("--models", nargs="*", default=["logistic", "xgboost"],
                           help="Models to train (default: logistic xgboost)")
    run_parser.add_argument("--output-dir", default="results", help="Output directory (default: results)")
    run_parser.add_argument("--force-refresh", action="store_true", help="Force refresh of cached data")
    run_parser.add_argument("--method", choices=["free", "bigquery"],
                           help="Force specific GDELT collection method (overrides .env config)")
    run_parser.add_argument("--skip-news", action="store_true", help="Skip news collection")
    run_parser.add_argument("--skip-sentiment", action="store_true", help="Skip sentiment analysis")
    run_parser.add_argument("--skip-market", action="store_true", help="Skip market data processing")
    run_parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    
    # Collect News Command
    collect_parser = subparsers.add_parser("collect-news", help="Collect news data from GDELT with time-windowed filenames")
    collect_parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    collect_parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    collect_parser.add_argument("--output-dir", default="results", help="Output directory (default: results)")
    collect_parser.add_argument("--force-refresh", action="store_true", help="Force refresh of cached data")
    collect_parser.add_argument("--method", choices=["free", "bigquery"], default="bigquery", help="Data collection method (default: bigquery)")
    
    # Process Sentiment Command
    sentiment_parser = subparsers.add_parser("process-sentiment", help="Process sentiment analysis with time-windowed filenames")
    sentiment_parser.add_argument("--data-path", help="Path to news data file")
    sentiment_parser.add_argument("--output-dir", default="results", help="Output directory (default: results)")
    sentiment_parser.add_argument("--model", default="finbert", help="Sentiment model (default: finbert)")
    
    # Process Market Command
    market_parser = subparsers.add_parser("process-market", help="Process market data")
    market_parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    market_parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    market_parser.add_argument("--assets", nargs="*", default=["EURUSD", "USDJPY", "TNOTE"],
                             help="Assets to process (default: EURUSD USDJPY TNOTE)")
    market_parser.add_argument("--output-dir", default="results", help="Output directory (default: results)")
    
    # Train Models Command
    train_parser = subparsers.add_parser("train-models", help="Train trading models with time window support")
    train_parser.add_argument("--data-path", help="Path to time-windowed directory (e.g., results/20240101_20240630) or results directory")
    train_parser.add_argument("--models", nargs="*", default=["logistic", "xgboost"],
                             help="Models to train (default: logistic xgboost)")
    train_parser.add_argument("--output-dir", default="results", help="Output directory (default: results)")
    train_parser.add_argument("--start-date", type=str,
                             help="Start date for training data (YYYY-MM-DD) - filters existing data")
    train_parser.add_argument("--end-date", type=str,
                             help="End date for training data (YYYY-MM-DD) - filters existing data")
    train_parser.add_argument("--assets", nargs="*",
                             help="Specific assets to train (if not specified, trains all assets in directory)")
    
    # Visualize Command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument("--results-path", default="results", help="Path to results directory")
    viz_parser.add_argument("--output-dir", default="results", help="Output directory for plots")
    viz_parser.add_argument("--types", nargs="*", 
                           default=["data", "sentiment", "market", "performance", "features", "shap"],
                           help="Visualization types to generate")
    
    # Get Signals Command  
    signals_parser = subparsers.add_parser("get-signals", help="Get current trading signals from pre-trained models")
    signals_parser.add_argument("--output-file", help="Save signals to JSON file (optional)")
    signals_parser.add_argument("--assets", nargs="*", default=["EURUSD", "USDJPY", "TNOTE"],
                               help="Assets to get signals for (default: EURUSD USDJPY TNOTE)")
    signals_parser.add_argument("--models", nargs="*", default=["logistic", "xgboost"],
                               help="Models to use for signals (default: logistic xgboost)")
    signals_parser.add_argument("--model-ids", nargs="*", 
                               help="Specific model IDs to use (overrides --models)")
    signals_parser.add_argument("--train-date", type=str,
                               help="Filter by training date: YYYYMMDD or YYYYMMDDYYYYMMDD range")
    signals_parser.add_argument("--training-window", type=str,
                               help="Filter by training window: YYYYMMDD or YYYYMMDDYYYYMMDD range")
    signals_parser.add_argument("--best-performance", action="store_true",
                               help="Use only the best performing models for each asset")
    signals_parser.add_argument("--performance-metric", type=str, default="accuracy",
                               help="Performance metric for best model selection (default: accuracy)")
    
    # Forecast Signals Command  
    forecast_parser = subparsers.add_parser("forecast-signals", help="Generate comprehensive multi-timeframe trading forecasts")
    forecast_parser.add_argument("--output-file", help="Save comprehensive forecast to JSON file (optional)")
    forecast_parser.add_argument("--assets", nargs="*", default=["EURUSD", "USDJPY", "TNOTE"],
                                help="Assets to forecast (default: EURUSD USDJPY TNOTE)")
    forecast_parser.add_argument("--models", nargs="*", default=["logistic", "xgboost"],
                                help="Models to use for forecasts (default: logistic xgboost)")
    forecast_parser.add_argument("--timeframes", nargs="*", default=["1D", "1W", "1M", "1Q"],
                                help="Timeframes to analyze (default: 1D 1W 1M 1Q)")
    forecast_parser.add_argument("--include-shap", action="store_true", default=True,
                                help="Include SHAP feature importance analysis (default: True)")
    forecast_parser.add_argument("--risk-analysis", action="store_true", default=True,
                                help="Include comprehensive risk analysis (default: True)")
    
    # Multi-Timeframe Signals Command
    mtf_signals_parser = subparsers.add_parser("multi-timeframe-signals", 
                                              help="Generate multi-timeframe trading signals (1D, 2D, 3D, 1W, 1M, 1Q, 1Y)")
    mtf_signals_parser.add_argument("--output-file", help="Save signals to JSON file (optional)")
    mtf_signals_parser.add_argument("--assets", nargs="*", default=["EURUSD", "USDJPY", "TNOTE"],
                                   help="Assets to get signals for (default: EURUSD USDJPY TNOTE)")
    mtf_signals_parser.add_argument("--models", nargs="*", default=["xgboost", "logistic"],
                                   help="Models to use for signals (default: xgboost logistic)")
    mtf_signals_parser.add_argument("--timeframes", nargs="*", default=["1D", "2D", "3D", "1W", "1M", "1Q", "1Y"],
                                   help="Timeframes to analyze (default: 1D 2D 3D 1W 1M 1Q 1Y)")
    mtf_signals_parser.add_argument("--confidence-threshold", type=float, default=0.6,
                                   help="Minimum confidence threshold for signals (default: 0.6)")
    
    # Multi-Timeframe Backtest Command
    mtf_parser = subparsers.add_parser("multi-timeframe-backtest", help="Run multi-timeframe backtesting analysis")
    mtf_parser.add_argument("--assets", nargs="*", default=["EURUSD", "USDJPY", "TNOTE"],
                           help="Assets to analyze (default: EURUSD USDJPY TNOTE)")
    mtf_parser.add_argument("--models", nargs="*", default=["logistic", "xgboost"],
                           help="Models to test (default: logistic xgboost)")
    mtf_parser.add_argument("--output-dir", default="results", help="Output directory (default: results)")
    mtf_parser.add_argument("--model-ids", nargs="*", 
                           help="Specific model IDs to use (overrides --models)")
    mtf_parser.add_argument("--train-date", type=str,
                           help="Filter by training date: YYYYMMDD or YYYYMMDDYYYYMMDD range")
    mtf_parser.add_argument("--training-window", type=str,
                           help="Filter by training window: YYYYMMDD or YYYYMMDDYYYYMMDD range")
    mtf_parser.add_argument("--best-performance", action="store_true",
                           help="Use only the best performing models for each asset")
    mtf_parser.add_argument("--performance-metric", type=str, default="accuracy",
                           help="Performance metric for best model selection (default: accuracy)")
    
    # Status Command
    status_parser = subparsers.add_parser("status", help="Check pipeline status")
    status_parser.add_argument("--data-dir", default="data", help="Data directory to check (default: data)")
    status_parser.add_argument("--results-dir", default="results", help="Results directory to check (default: results)")
    status_parser.add_argument("--models-dir", default="results/models", help="Models directory to check (default: results/models)")
    status_parser.add_argument("--check-data", action="store_true", help="Check data availability")
    status_parser.add_argument("--check-models", action="store_true", help="Check model status")
    status_parser.add_argument("--check-results", action="store_true", help="Check results status")
    
    # Model Management Commands
    # List Models
    list_models_parser = subparsers.add_parser("list-models", help="List available trained models")
    list_models_parser.add_argument("--asset", help="Filter by asset name")
    list_models_parser.add_argument("--model-type", help="Filter by model type")
    list_models_parser.add_argument("--details", action="store_true", help="Show detailed information")
    list_models_parser.add_argument("--output-file", help="Save model list to file")
    list_models_parser.add_argument("--sort-by", choices=[
        "date", "date-desc", "performance", "performance-desc", 
        "training-window", "training-window-desc", "asset", "model-type"
    ], default="date-desc", help="Sort models by criteria (default: date-desc)")
    list_models_parser.add_argument("--performance-metric", type=str, default="accuracy",
                                   help="Performance metric for sorting (default: accuracy)")
    list_models_parser.add_argument("--limit", type=int, help="Limit number of results shown")
    list_models_parser.add_argument("--training-date-from", type=str,
                                   help="Only show models trained after this date (YYYY-MM-DD)")
    list_models_parser.add_argument("--training-date-to", type=str,
                                   help="Only show models trained before this date (YYYY-MM-DD)")
    
    # Show Model
    show_model_parser = subparsers.add_parser("show-model", help="Show detailed information about a model")
    show_model_parser.add_argument("model_id", help="Model ID to show")
    show_model_parser.add_argument("--output-file", help="Save model details to file")
    
    # Delete Model
    delete_model_parser = subparsers.add_parser("delete-model", help="Delete a trained model")
    delete_model_parser.add_argument("model_id", help="Model ID to delete")
    delete_model_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    
    # Compare Models
    compare_models_parser = subparsers.add_parser("compare-models", help="Compare multiple models")
    compare_models_parser.add_argument("model_ids", nargs="+", help="Model IDs to compare")
    compare_models_parser.add_argument("--output-file", help="Save comparison to file")
    
    # Model Registry
    registry_parser = subparsers.add_parser("model-registry", help="Manage model registry")
    registry_parser.add_argument("action", choices=["status", "cleanup"], help="Registry action to perform")
    registry_parser.add_argument("--remove-orphaned", action="store_true", help="Remove orphaned models during cleanup")
    
    # Add parquet health check command
    parquet_parser = subparsers.add_parser("check-parquet", help="Check parquet file compatibility and health")
    parquet_parser.add_argument("--file-path", type=str, help="Path to parquet file to check")
    parquet_parser.add_argument("--directory", type=str, help="Directory to check all parquet files")
    parquet_parser.add_argument("--fix", action="store_true", help="Fix incompatible parquet files")
    
    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # If no command provided, show help
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    if args.output:
        # Create timestamped comprehensive log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"logs/comprehensive_output_{timestamp}.log"
    
    logger = setup_logging(args.log_level, args.log_file)
    
    # Log startup information
    logger.info("=" * 80)
    logger.info("MACRO SENTIMENT TRADING PIPELINE - STARTED")
    logger.info("=" * 80)
    logger.info(f"Command: {args.command}")
    logger.info(f"Log Level: DEBUG")  # Always log as DEBUG when --output is used
    logger.info(f"Log File: {args.log_file}")
    
    # Log command line parameters
    logger.info("COMMAND LINE PARAMETERS:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    # Log system information
    logger.info("SYSTEM INFORMATION:")
    logger.info(f"  Python Version: {sys.version}")
    logger.info(f"  Platform: {sys.platform}")
    logger.info(f"  Working Directory: {os.getcwd()}")
    logger.info("=" * 80)
    
    try:
        # Load configuration
        config = CLIConfig()
        
        # Command routing with lazy imports (imports only when command is used)
        if args.command == "run-pipeline":
            from cli.commands.run_pipeline import RunPipelineCommand
            # Validate date range for training
            start_date, end_date = validate_date_range_for_training(args.start_date, args.end_date, args.skip_training)
            command = RunPipelineCommand(config, args)

        elif args.command == "collect-news":
            from cli.commands.collect_news import CollectNewsCommand
            validate_date_range(args.start_date, args.end_date)
            command = CollectNewsCommand(config, args)

        elif args.command == "process-sentiment":
            from cli.commands.process_sentiment import ProcessSentimentCommand
            command = ProcessSentimentCommand(config, args)

        elif args.command == "process-market":
            from cli.commands.process_market import ProcessMarketCommand
            validate_date_range(args.start_date, args.end_date)
            command = ProcessMarketCommand(config, args)

        elif args.command == "train-models":
            from cli.commands.train_models import TrainModelsCommand
            command = TrainModelsCommand(config, args)

        elif args.command == "visualize":
            from cli.commands.visualize import VisualizeCommand
            command = VisualizeCommand(config, args)

        elif args.command == "get-signals":
            from cli.commands.get_signals import GetSignalsCommand
            command = GetSignalsCommand(config, args)

        elif args.command == "forecast-signals":
            from cli.commands.forecast_signals import ForecastSignalsCommand
            command = ForecastSignalsCommand(config, args)

        elif args.command == "multi-timeframe-signals":
            from cli.commands.multi_timeframe_signals import MultiTimeframeSignalsCommand
            command = MultiTimeframeSignalsCommand(config, args)

        elif args.command == "multi-timeframe-backtest":
            from cli.commands.multi_timeframe_backtest import MultiTimeframeBacktestCommand
            command = MultiTimeframeBacktestCommand(config, args)

        elif args.command == "status":
            from cli.commands.status import StatusCommand
            command = StatusCommand(config, args)

        elif args.command == "list-models":
            from cli.commands.model_management import ListModelsCommand
            command = ListModelsCommand(config, args)

        elif args.command == "show-model":
            from cli.commands.model_management import ShowModelCommand
            command = ShowModelCommand(config, args)

        elif args.command == "delete-model":
            from cli.commands.model_management import DeleteModelCommand
            command = DeleteModelCommand(config, args)

        elif args.command == "compare-models":
            from cli.commands.model_management import CompareModelsCommand
            command = CompareModelsCommand(config, args)

        elif args.command == "model-registry":
            from cli.commands.model_management import ModelRegistryCommand
            command = ModelRegistryCommand(config, args)

        elif args.command == "check-parquet":
            from cli.commands.check_parquet import CheckParquetCommand
            command = CheckParquetCommand(args)

        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
        
        # Execute the command
        return command.execute()
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    main()