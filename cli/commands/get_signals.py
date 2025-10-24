#!/usr/bin/env python3

"""
Get Trading Signals Command

Provides real-time trading signals from pre-trained models without requiring 
full pipeline training. Designed for non-technical users to get actionable signals.
"""

import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import json

# Add project root to path  
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cli.commands.base import BaseCommand

# Lazy imports - these will be imported only when needed to avoid slow startup
# from src.market_processor import MarketProcessor
# from src.sentiment_analyzer import SentimentAnalyzer
# from src.news_collector import GDELTCollector
# from src.data_collector import UnifiedGDELTCollector
# from src.config import config


class GetSignalsCommand(BaseCommand):
    """Command to generate current trading signals from pre-trained models."""
    
    def __init__(self, config, args):
        super().__init__(config, args)
        self.logger = logging.getLogger(__name__)
        # Use CLI assets or default to original 3
        self.assets = getattr(args, 'assets', ["EURUSD", "USDJPY", "TNOTE"])
        self.models = getattr(args, 'models', ["logistic", "xgboost"])
        
        # Advanced model selection parameters
        self.model_ids = getattr(args, 'model_ids', None)
        self.train_date = getattr(args, 'train_date', None)
        self.training_window = getattr(args, 'training_window', None)
        self.best_performance = getattr(args, 'best_performance', True)  # Default to latest models
        self.performance_metric = getattr(args, 'performance_metric', 'accuracy')
        
        # Asset display mapping
        self.asset_display = {
            "EURUSD": "EUR/USD",
            "USDJPY": "USD/JPY", 
            "TNOTE": "Treasury Notes",
            "GBPUSD": "GBP/USD",
            "AUDUSD": "AUD/USD",
            "USDCHF": "USD/CHF",
            "USDCAD": "USD/CAD",
            "GOLD": "Gold",
            "CRUDE": "Crude Oil",
            "SP500": "S&P 500",
            "VIX": "VIX"
        }
        
        # Signal interpretation
        self.signal_meaning = {
            -1: "SELL",
            0: "HOLD", 
            1: "BUY"
        }
        
        # Signal confidence colors (for future CLI coloring)
        self.signal_colors = {
            -1: "RED",
            0: "YELLOW",
            1: "GREEN"
        }
    
    def execute(self) -> int:
        """Execute the get-signals command."""
        try:
            self.logger.info("=" * 60)
            self.logger.info("GENERATING CURRENT TRADING SIGNALS")
            self.logger.info("=" * 60)
            
            # Step 1: Load pre-trained models
            models_info = self._load_pretrained_models()
            if not models_info:
                self.logger.error("No pre-trained models found. Run 'train-models' first.")
                return 1
            
            # Step 2: Collect latest data  
            latest_data = self._collect_latest_data()
            if latest_data is None:
                self.logger.error("Failed to collect latest market and sentiment data")
                return 1
            
            # Step 3: Generate signals for each asset
            all_signals = {}
            for asset in self.assets:
                asset_signals = self._generate_asset_signals(asset, latest_data, models_info)
                if asset_signals:
                    all_signals[asset] = asset_signals
            
            if not all_signals:
                self.logger.error("Failed to generate signals for any assets")
                return 1
            
            # Step 4: Display signals in user-friendly format
            self._display_signals(all_signals)
            
            # Step 5: Save signals to file if requested
            if hasattr(self.args, 'output_file') and self.args.output_file:
                self._save_signals_to_file(all_signals, self.args.output_file)
            
            self.logger.info("Signal generation completed successfully")
            return 0
            
        except Exception as e:
            self.logger.error(f"Error in get-signals command: {e}", exc_info=True)
            return 1
    
    def _load_pretrained_models(self) -> Dict:
        """Load pre-trained models using advanced selection criteria."""
        from src.model_persistence import ModelPersistence
        
        models_info = {}
        persistence = ModelPersistence()
        
        self.logger.info("Loading pre-trained models with advanced selection...")
        
        # If specific model IDs are provided, load only those
        if self.model_ids:
            self.logger.info(f"Loading specific model IDs: {self.model_ids}")
            for model_id in self.model_ids:
                try:
                    model, scaler, feature_columns, metadata = persistence.load_model(model_id)
                    asset = metadata['asset']
                    model_type = metadata['model_type']
                    
                    if asset not in models_info:
                        models_info[asset] = {}
                    
                    models_info[asset][model_type] = {
                        'model': model,
                        'scaler': scaler,
                        'feature_columns': feature_columns,
                        'metadata': metadata,
                        'model_id': model_id
                    }
                    
                    self.logger.info(f"Loaded specific model {model_id} for {asset} ({model_type})")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load model {model_id}: {e}")
        
        # Otherwise, use advanced filtering
        else:
            for asset in self.assets:
                models_info[asset] = {}
                
                # Get available models for this asset
                available_models = persistence.list_models(asset=asset)
                
                if not available_models:
                    self.logger.warning(f"No models found for {asset}")
                    continue
                
                # Apply date filtering
                if self.train_date or self.training_window:
                    available_models = self._filter_models_by_date(available_models)
                
                # Apply model type filtering
                if self.models:
                    available_models = [m for m in available_models if m['model_type'] in self.models]
                
                # Apply performance filtering
                if self.best_performance:
                    available_models = self._filter_models_by_performance(available_models)
                
                # Load the selected models
                for model_info in available_models:
                    try:
                        model_id = model_info['model_id']
                        model, scaler, feature_columns, metadata = persistence.load_model(model_id)
                        
                        models_info[asset][model_info['model_type']] = {
                            'model': model,
                            'scaler': scaler,
                            'feature_columns': feature_columns,
                            'metadata': metadata,
                            'model_id': model_id
                        }
                        
                        self.logger.info(f"Loaded {asset} {model_info['model_type']} model (ID: {model_id})")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load {asset} {model_info['model_type']} model: {e}")
        
        # Check if we have any models
        total_models = sum(len(asset_models) for asset_models in models_info.values())
        self.logger.info(f"Successfully loaded {total_models} pre-trained models")
        
        return models_info
    
    def _filter_models_by_date(self, models: List[Dict]) -> List[Dict]:
        """Filter models by date range using new YYYYMMDD format."""
        from datetime import datetime
        
        filtered_models = []
        
        for model in models:
            if self.train_date:
                # Filter by training date (when model was trained)
                training_date = model.get('training_date', '')
                if not training_date:
                    continue
                    
                try:
                    # Parse training date from model
                    model_date = datetime.fromisoformat(training_date.replace('Z', '+00:00'))
                    model_date_str = model_date.strftime('%Y%m%d')
                    
                    # Parse filter date(s)
                    if len(self.train_date) == 8:  # Single date: YYYYMMDD
                        if model_date_str != self.train_date:
                            continue
                    elif len(self.train_date) == 16:  # Date range: YYYYMMDDYYYYMMDD
                        start_date = self.train_date[:8]
                        end_date = self.train_date[8:]
                        if not (start_date <= model_date_str <= end_date):
                            continue
                    else:
                        self.logger.warning(f"Invalid train-date format: {self.train_date}")
                        continue
                    
                    filtered_models.append(model)
                    
                except ValueError as e:
                    self.logger.warning(f"Invalid date format in model {model.get('model_id', 'unknown')}: {e}")
                    continue
            
            elif self.training_window:
                # Filter by training window (data period model was trained on)
                training_params = model.get('training_params', {})
                start_date = training_params.get('start_date', '')
                end_date = training_params.get('end_date', '')
                
                if not start_date or not end_date:
                    continue
                    
                try:
                    # Parse training window dates
                    model_start = datetime.fromisoformat(start_date)
                    model_end = datetime.fromisoformat(end_date)
                    model_start_str = model_start.strftime('%Y%m%d')
                    model_end_str = model_end.strftime('%Y%m%d')
                    
                    # Parse filter date(s)
                    if len(self.training_window) == 8:  # Single date: YYYYMMDD
                        filter_date = self.training_window
                        # For single date, check if model was trained exactly on that day
                        if model_start_str != filter_date or model_end_str != filter_date:
                            continue
                    elif len(self.training_window) == 16:  # Date range: YYYYMMDDYYYYMMDD
                        filter_start = self.training_window[:8]
                        filter_end = self.training_window[8:]
                        # Check if training windows overlap
                        if not (model_start_str <= filter_end and model_end_str >= filter_start):
                            continue
                    else:
                        self.logger.warning(f"Invalid training-window format: {self.training_window}")
                        continue
                    
                    filtered_models.append(model)
                    
                except ValueError as e:
                    self.logger.warning(f"Invalid date format in model {model.get('model_id', 'unknown')}: {e}")
                    continue
        
        filter_type = "training window" if self.training_window else "training date"
        self.logger.info(f"Date filtering ({filter_type}): {len(models)} -> {len(filtered_models)} models")
        return filtered_models
    
    def _filter_models_by_performance(self, models: List[Dict]) -> List[Dict]:
        """Filter models to keep only the best performing ones."""
        if not models:
            return models
        
        # Group by model type and select best performer for each type
        model_types = {}
        for model in models:
            model_type = model.get('model_type', 'unknown')
            if model_type not in model_types:
                model_types[model_type] = []
            model_types[model_type].append(model)
        
        best_models = []
        for model_type, type_models in model_types.items():
            # Find the best model by performance metric
            best_model = None
            best_score = -1
            
            for model in type_models:
                metrics = model.get('metrics', {})
                score = metrics.get(self.performance_metric, 0)
                
                if score > best_score:
                    best_score = score
                    best_model = model
            
            if best_model:
                best_models.append(best_model)
                self.logger.info(f"Best {model_type} model: {best_model.get('model_id', 'unknown')} "
                               f"({self.performance_metric}={best_score:.3f})")
        
        self.logger.info(f"Performance filtering: {len(models)} -> {len(best_models)} models")
        return best_models
    
    def _collect_latest_data(self) -> Optional[Dict]:
        """Collect the latest market and sentiment data with cached fallback."""
        self.logger.info("Collecting latest market and sentiment data...")

        try:
            # PRODUCTION OPTIMIZATION: Use cached aligned data instead of fresh collection
            # This makes signals generation instant instead of 2+ hour GDELT downloads

            # First try to use existing aligned data files (fastest)
            aligned_data = self._load_cached_aligned_data()
            if aligned_data:
                self.logger.info("Using cached aligned data for signal generation")
                return aligned_data

            # Fallback: Try to use latest available cached sentiment data
            sentiment_data = self._load_cached_sentiment_data()
            if sentiment_data is not None:
                self.logger.info("Using cached sentiment data for signal generation")

                # Still collect fresh market data (much faster than GDELT)
                end_date = date.today()
                start_date = end_date - timedelta(days=30)
                market_data = self._collect_market_data(start_date, end_date)

                if market_data:
                    # Align cached sentiment with fresh market data
                    aligned_data = self._align_data(sentiment_data, market_data)
                    return aligned_data

            # Last resort: Full fresh collection (slow but comprehensive)
            self.logger.warning("No cached data available, performing full data collection (may be slow)")
            return self._collect_fresh_data()

        except Exception as e:
            self.logger.error(f"Error collecting latest data: {e}")
            return None

    def _load_cached_aligned_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Load cached aligned data files for instant signal generation."""
        try:
            results_dir = Path("results")
            aligned_data = {}

            for asset in self.assets:
                aligned_file = results_dir / f"aligned_data_{asset}.parquet"
                if aligned_file.exists():
                    df = pd.read_parquet(aligned_file)
                    if not df.empty:
                        # Use the most recent data for signal generation
                        aligned_data[asset] = df.tail(30)  # Last 30 rows for features
                        self.logger.debug(f"Loaded {len(aligned_data[asset])} cached rows for {asset}")

            return aligned_data if aligned_data else None

        except Exception as e:
            self.logger.warning(f"Failed to load cached aligned data: {e}")
            return None

    def _load_cached_sentiment_data(self) -> Optional[pd.DataFrame]:
        """Load cached sentiment data if available."""
        try:
            results_dir = Path("results")
            sentiment_files = [
                "daily_features.parquet",
                "sentiment_data.parquet"
            ]

            for filename in sentiment_files:
                file_path = results_dir / filename
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    if not df.empty:
                        self.logger.debug(f"Loaded cached sentiment data from {filename}")
                        return df.tail(60)  # Last 60 days for pattern recognition

            return None

        except Exception as e:
            self.logger.warning(f"Failed to load cached sentiment data: {e}")
            return None

    def _collect_fresh_data(self) -> Optional[Dict]:
        """Original full data collection method (slow but comprehensive)."""
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        self.logger.info(f"Data collection period: {start_date} to {end_date}")

        # Collect sentiment data
        sentiment_data = self._collect_sentiment_data(start_date, end_date)
        if sentiment_data is None:
            return None

        # Collect market data
        market_data = self._collect_market_data(start_date, end_date)
        if not market_data:
            return None

        # Align and merge data
        aligned_data = self._align_data(sentiment_data, market_data)
        return aligned_data

    def _collect_sentiment_data(self, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
        """Collect recent sentiment data from GDELT with timeout protection."""
        try:
            # PRODUCTION OPTIMIZATION: Reduce date range for faster collection
            # Use shorter range for real-time signals (trade-off: speed vs data richness)

            # Limit to last 7 days for production speed
            if (end_date - start_date).days > 7:
                start_date = end_date - timedelta(days=7)
                self.logger.info(f"Limiting data collection to last 7 days for speed: {start_date} to {end_date}")

            # Initialize unified data collector (lazy import)
            from src.data_collector import UnifiedGDELTCollector
            collector = UnifiedGDELTCollector()

            # Collect news events with timeout protection
            self.logger.info("Collecting recent news events (limited for production speed)...")

            # Use threading timeout for cross-platform compatibility
            import threading
            import queue

            def collect_with_timeout():
                """Collect GDELT data in a separate thread with timeout."""
                result_queue = queue.Queue()

                def worker():
                    try:
                        data = collector.fetch_events(
                            start_date=start_date.strftime("%Y-%m-%d"),
                            end_date=end_date.strftime("%Y-%m-%d"),
                            top_n_per_day=50  # Reduced from 100 for speed
                        )
                        result_queue.put(('success', data))
                    except Exception as e:
                        result_queue.put(('error', e))

                # Start worker thread
                thread = threading.Thread(target=worker)
                thread.daemon = True
                thread.start()

                # Wait for result with timeout
                thread.join(timeout=60)  # 60 seconds timeout

                if thread.is_alive():
                    self.logger.warning("GDELT collection timed out after 60 seconds")
                    return pd.DataFrame()

                try:
                    result_type, result = result_queue.get_nowait()
                    if result_type == 'success':
                        return result
                    else:
                        raise result
                except queue.Empty:
                    self.logger.warning("GDELT collection completed but no result returned")
                    return pd.DataFrame()

            events_data = collect_with_timeout()

            if events_data.empty:
                self.logger.warning("No recent news events found")
                return pd.DataFrame()

            # Process sentiment analysis
            self.logger.info("Processing sentiment analysis...")

            # Transform BigQuery data format to match SentimentAnalyzer expectations
            if 'tone' in events_data.columns:
                events_data['polarity'] = events_data['tone'] / 100.0
            if 'goldstein_scale' in events_data.columns:
                events_data['goldstein'] = events_data['goldstein_scale']

            # Lazy import to avoid slow startup
            from src.sentiment_analyzer import SentimentAnalyzer
            sentiment_analyzer = SentimentAnalyzer()
            sentiment_features = sentiment_analyzer.compute_daily_features(events_data)

            return sentiment_features

        except Exception as e:
            self.logger.error(f"Error collecting sentiment data: {e}")
            return None
    
    def _collect_market_data(self, start_date: date, end_date: date) -> Dict[str, pd.DataFrame]:
        """Collect recent market data for all assets."""
        try:
            self.logger.info("Collecting recent market data...")
            # Lazy import to avoid slow startup
            from src.market_processor import MarketProcessor
            market_processor = MarketProcessor()
            
            # Use the standard fetch_market_data method which returns data for all assets
            all_market_data = market_processor.fetch_market_data(
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            
            # Filter for only the assets we need
            market_data = {}
            for asset in self.assets:
                if asset in all_market_data and not all_market_data[asset].empty:
                    market_data[asset] = all_market_data[asset]
                    self.logger.info(f"Collected {len(all_market_data[asset])} days of {asset} data")
                else:
                    self.logger.warning(f"No market data found for {asset}")
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error collecting market data: {e}")
            return {}
    
    def _align_data(self, sentiment_data: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align sentiment and market data for each asset."""
        aligned_data = {}
        # Lazy import to avoid slow startup
        from src.market_processor import MarketProcessor
        market_processor = MarketProcessor()
        
        for asset in self.assets:
            if asset not in market_data:
                continue
                
            try:
                self.logger.info(f"Aligning data for {asset}...")
                
                # Align sentiment and market data
                aligned = market_processor.align_features(
                    {asset: market_data[asset]}, 
                    sentiment_data
                )
                
                if asset in aligned and not aligned[asset].empty:
                    aligned_data[asset] = aligned[asset]
                    self.logger.info(f"Aligned {len(aligned[asset])} rows for {asset}")
                
            except Exception as e:
                self.logger.warning(f"Failed to align data for {asset}: {e}")
        
        return aligned_data
    
    def _generate_asset_signals(self, asset: str, latest_data: Dict, models_info: Dict) -> Dict:
        """Generate trading signals for a specific asset."""
        if asset not in latest_data or asset not in models_info:
            return {}
        
        data = latest_data[asset]
        asset_models = models_info[asset]
        
        if data.empty or not asset_models:
            return {}
        
        signals = {}
        
        # Get the most recent data point for prediction
        latest_row = data.tail(1)
        
        for model_name, model_info in asset_models.items():
            try:
                model = model_info['model']
                scaler = model_info.get('scaler')
                feature_columns = model_info.get('feature_columns', [])
                
                if model is None:
                    continue
                
                # Generate signal using the model_trainer approach
                from src.model_trainer import ModelTrainer
                trainer = ModelTrainer()
                
                signals_series = trainer.generate_signals(
                    model, latest_row, scaler=scaler, feature_columns=feature_columns
                )
                
                if not signals_series.empty:
                    signal_value = signals_series.iloc[0]
                    signals[model_name] = {
                        'signal': int(signal_value),
                        'meaning': self.signal_meaning[signal_value],
                        'confidence': self._calculate_confidence(model, latest_row, scaler, feature_columns),
                        'timestamp': datetime.now().isoformat()
                    }
                
            except Exception as e:
                self.logger.warning(f"Failed to generate {asset} {model_name} signal: {e}")
        
        return signals
    
    def _calculate_confidence(self, model, data: pd.DataFrame, scaler, feature_columns: List[str]) -> float:
        """Calculate prediction confidence based on model probability."""
        try:
            # Prepare features using feature pipeline
            from src.feature_pipeline import FeatureEngineeringPipeline
            
            pipeline = FeatureEngineeringPipeline()
            X, _ = pipeline.process_features(data, feature_columns=feature_columns)
            
            if scaler is not None:
                X = scaler.transform(X)
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                if len(probabilities) > 0 and len(probabilities[0]) > 0:
                    # Return max probability as confidence
                    max_prob = float(np.max(probabilities[0]))
                    self.logger.debug(f"Model probabilities: {probabilities[0]}, max: {max_prob}")
                    return max_prob
                else:
                    self.logger.warning("Model predict_proba returned empty probabilities")
            else:
                self.logger.warning(f"Model {type(model).__name__} does not have predict_proba method")
            
            # Fallback confidence
            return 0.6
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.5  # Default confidence
    
    def _display_signals(self, all_signals: Dict):
        """Display trading signals in a user-friendly format."""
        
        print("\n" + "=" * 80)
        print("                    CURRENT TRADING SIGNALS")
        print("=" * 80)
        print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        for asset, signals in all_signals.items():
            if not signals:
                continue
                
            asset_name = self.asset_display.get(asset, asset)
            print(f"\n{asset_name} ({asset}):")
            print("-" * 40)
            
            for model_name, signal_info in signals.items():
                signal = signal_info['signal']
                meaning = signal_info['meaning']
                confidence = signal_info['confidence']
                
                # Format confidence as percentage
                confidence_pct = confidence * 100
                
                # Create signal display with confidence
                signal_display = f"{meaning:4s} ({signal:+2d})"
                confidence_display = f"{confidence_pct:5.1f}%"
                
                print(f"  {model_name.capitalize():8s}: {signal_display} | Confidence: {confidence_display}")
        
        # Add legend
        print("\n" + "-" * 80)
        print("SIGNAL LEGEND:")
        print("  SELL (-1): Consider selling the asset")
        print("  HOLD ( 0): No clear direction, maintain current position")
        print("  BUY  (+1): Consider buying the asset")
        print("\nCONFIDENCE: Higher percentage indicates stronger model conviction")
        print("\nDISCLAIMER: Signals are for educational purposes only.")
        print("Always conduct your own analysis before making trading decisions.")
        print("=" * 80)
    
    def _save_signals_to_file(self, all_signals: Dict, output_file: str):
        """Save signals to a JSON file for programmatic access."""
        try:
            # Prepare data for JSON serialization
            signals_data = {
                'timestamp': datetime.now().isoformat(),
                'signals': all_signals,
                'metadata': {
                    'assets': list(all_signals.keys()),
                    'models': self.models,
                    'signal_legend': self.signal_meaning
                }
            }
            
            # Save to file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(signals_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Signals saved to {output_path}")
            print(f"\nSignals saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save signals to file: {e}")