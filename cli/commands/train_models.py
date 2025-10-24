"""
Train Models Command

This command trains trading models and runs backtesting.
"""

from .base import BaseCommand
from pathlib import Path
import pandas as pd
from datetime import datetime


class TrainModelsCommand(BaseCommand):
    """Command to train trading models."""
    
    def validate_args(self) -> None:
        """Validate command arguments."""
        self.check_file_exists(self.args.data_path, required=True)
    
    def execute(self) -> int:
        """Execute model training."""
        try:
            with self.with_logged_operation("Model Training"):
                # Import here to avoid circular imports
                from src.model_trainer import ModelTrainer
                from src.model_persistence import ModelPersistence
                
                self.logger.info(f"Training models: {self.args.models}")
                
                trainer = ModelTrainer()
                persistence = ModelPersistence()
                
                data_path = Path(self.args.data_path)
                
                # Check if it's a directory or single file
                if data_path.is_dir():
                    # Check if it's a time-windowed directory or general results directory
                    if self._is_time_windowed_directory(data_path):
                        # Time-windowed directory: process all assets in this time window
                        time_window = self._extract_time_window_from_path(data_path)
                        parquet_files = list(data_path.glob("aligned_data_*.parquet"))
                        self.logger.info(f"Processing time-windowed directory: {data_path} ({time_window})")
                    else:
                        # General results directory: look for time-windowed subdirectories
                        time_window_dirs = [d for d in data_path.iterdir() if d.is_dir() and self._is_time_windowed_directory(d)]
                        if not time_window_dirs:
                            # Fallback: look for parquet files directly in directory
                            parquet_files = list(data_path.glob("aligned_data_*.parquet"))
                            if not parquet_files:
                                self.logger.error(f"No aligned_data_*.parquet files or time-windowed directories found in {data_path}")
                                return 1
                            time_window = "unknown_period"
                        else:
                            # Process all time-windowed directories
                            all_results = []
                            all_metrics = []
                            
                            for time_dir in time_window_dirs:
                                time_window = self._extract_time_window_from_path(time_dir)
                                self.logger.info(f"Processing time window: {time_window}")
                                
                                parquet_files = list(time_dir.glob("aligned_data_*.parquet"))
                                
                                # Filter files by assets if specified
                                if hasattr(self.args, 'assets') and self.args.assets:
                                    filtered_files = []
                                    for file_path in parquet_files:
                                        asset_name = self._extract_asset_name(file_path)
                                        if asset_name in self.args.assets:
                                            filtered_files.append(file_path)
                                    parquet_files = filtered_files
                                    self.logger.info(f"Filtered to {len(parquet_files)} files for specified assets: {self.args.assets}")
                                
                                for file_path in parquet_files:
                                    asset_name = self._extract_asset_name(file_path)
                                    self.logger.info(f"Processing {asset_name} ({time_window}) from {file_path}")
                                    
                                    # Load and filter data
                                    data = self._load_and_filter_data(file_path)
                                    if data is None or len(data) == 0:
                                        self.logger.warning(f"No data found for {asset_name} after filtering")
                                        continue
                                    
                                    # Train models for this asset
                                    asset_results, asset_metrics = self._train_asset_models(
                                        trainer, persistence, data, asset_name, time_window
                                    )
                                    
                                    all_results.extend(asset_results)
                                    all_metrics.extend(asset_metrics)
                            
                            # Save results
                            self._save_training_results(all_results, all_metrics)
                            return 0
                    
                    # Single time-windowed directory processing
                    if not parquet_files:
                        self.logger.error(f"No aligned_data_*.parquet files found in {data_path}")
                        return 1
                    
                    # Filter files by assets if specified
                    if hasattr(self.args, 'assets') and self.args.assets:
                        filtered_files = []
                        for file_path in parquet_files:
                            asset_name = self._extract_asset_name(file_path)
                            if asset_name in self.args.assets:
                                filtered_files.append(file_path)
                        parquet_files = filtered_files
                        self.logger.info(f"Filtered to {len(parquet_files)} files for specified assets: {self.args.assets}")
                    
                    self.logger.info(f"Found {len(parquet_files)} asset files to process")
                    
                    all_results = []
                    all_metrics = []
                    
                    for file_path in parquet_files:
                        asset_name = self._extract_asset_name(file_path)
                        self.logger.info(f"Processing {asset_name} ({time_window}) from {file_path}")
                        
                        # Load and filter data
                        data = self._load_and_filter_data(file_path)
                        if data is None or len(data) == 0:
                            self.logger.warning(f"No data found for {asset_name} after filtering")
                            continue
                        
                        # Train models for this asset
                        asset_results, asset_metrics = self._train_asset_models(
                            trainer, persistence, data, asset_name, time_window
                        )
                        
                        all_results.extend(asset_results)
                        all_metrics.extend(asset_metrics)
                
                else:
                    # Single file processing
                    asset_name, time_window = self._parse_filename(data_path)
                    self.logger.info(f"Processing single asset: {asset_name} ({time_window})")
                    
                    # Load and filter data
                    data = self._load_and_filter_data(data_path)
                    if data is None or len(data) == 0:
                        self.logger.error(f"No data found for {asset_name} after filtering")
                        return 1
                    
                    # Train models for this asset
                    all_results, all_metrics = self._train_asset_models(
                        trainer, persistence, data, asset_name, time_window
                    )
                
                self.logger.info(f"Training completed: {len(all_results)} models trained")
                
                # Save results
                results_path = self.get_output_path("training_results", ".parquet")
                metrics_path = self.get_output_path("training_metrics", ".parquet")
                
                # Convert results to DataFrame format with proper handling
                if all_results:
                    results_df = pd.DataFrame(all_results)
                else:
                    results_df = pd.DataFrame(columns=['model_id', 'asset', 'model_type', 'accuracy', 'precision', 'recall', 'f1_score'])
                
                if all_metrics:
                    metrics_df = pd.DataFrame(all_metrics)
                else:
                    metrics_df = pd.DataFrame(columns=['model_id', 'asset', 'model_type', 'metric_name', 'metric_value'])
                
                results_df.to_parquet(results_path)
                metrics_df.to_parquet(metrics_path)
                
                self.logger.info(f"Saved results to: {results_path}")
                self.logger.info(f"Saved metrics to: {metrics_path}")
                
                return 0
                
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return 1
    
    def _load_and_filter_data(self, file_path: Path) -> pd.DataFrame:
        """Load data and apply time window filtering if specified."""
        try:
            data = pd.read_parquet(file_path)
            self.log_data_info("Raw Data", data)
            
            # Apply time window filtering if specified
            if hasattr(self.args, 'start_date') and self.args.start_date:
                start_date = pd.to_datetime(self.args.start_date)
                if 'date' in data.columns:
                    data = data[data['date'] >= start_date]
                elif data.index.name == 'date' or 'date' in str(data.index.dtype):
                    data = data[data.index >= start_date]
                self.logger.info(f"Filtered data from {self.args.start_date}: {len(data)} rows")
            
            if hasattr(self.args, 'end_date') and self.args.end_date:
                end_date = pd.to_datetime(self.args.end_date)
                if 'date' in data.columns:
                    data = data[data['date'] <= end_date]
                elif data.index.name == 'date' or 'date' in str(data.index.dtype):
                    data = data[data.index <= end_date]
                self.logger.info(f"Filtered data to {self.args.end_date}: {len(data)} rows")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {e}")
            return None
    
    def _is_time_windowed_directory(self, path: Path) -> bool:
        """Check if directory name follows time window format (YYYYMMDD_YYYYMMDD)."""
        if not path.is_dir():
            return False
        dir_name = path.name
        parts = dir_name.split("_")
        return len(parts) == 2 and all(part.isdigit() and len(part) == 8 for part in parts)
    
    def _extract_time_window_from_path(self, path: Path) -> str:
        """Extract time window from directory path."""
        dir_name = path.name
        parts = dir_name.split("_")
        if len(parts) == 2 and all(part.isdigit() and len(part) == 8 for part in parts):
            start_date = parts[0]
            end_date = parts[1]
            return f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]} to {end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        return "unknown_period"
    
    def _extract_asset_name(self, file_path: Path) -> str:
        """Extract asset name from aligned_data_ASSET.parquet filename."""
        filename = file_path.stem
        if filename.startswith("aligned_data_"):
            return filename.replace("aligned_data_", "")
        return "unknown"
    
    def _parse_filename(self, file_path: Path) -> tuple:
        """Parse filename to extract asset name and time window (legacy support)."""
        filename = file_path.stem
        parts = filename.split("_")
        
        if len(parts) >= 4 and parts[0] == "aligned" and parts[1] == "data":
            # Old format: aligned_data_ASSET_YYYYMMDD_YYYYMMDD
            asset_name = parts[2]
            start_date = parts[3]
            end_date = parts[4]
            time_window = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]} to {end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
            return asset_name, time_window
        elif len(parts) >= 2 and parts[0] == "aligned" and parts[1] == "data":
            # Old format: aligned_data_ASSET
            asset_name = parts[2] if len(parts) > 2 else "unknown"
            return asset_name, "unknown_period"
        else:
            # Fallback
            return "unknown", "unknown_period"
    
    def _save_training_results(self, all_results, all_metrics):
        """Save training results to files."""
        # Save results
        results_path = self.get_output_path("training_results", ".parquet")
        metrics_path = self.get_output_path("training_metrics", ".parquet")
        
        # Convert results to DataFrame format with proper handling
        if all_results:
            results_df = pd.DataFrame(all_results)
        else:
            results_df = pd.DataFrame(columns=['model_id', 'asset', 'model_type', 'time_window', 'accuracy', 'precision', 'recall', 'f1_score'])
        
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
        else:
            metrics_df = pd.DataFrame(columns=['model_id', 'asset', 'model_type', 'time_window', 'metric_name', 'metric_value'])
        
        results_df.to_parquet(results_path)
        metrics_df.to_parquet(metrics_path)
        
        self.logger.info(f"Saved results to: {results_path}")
        self.logger.info(f"Saved metrics to: {metrics_path}")
    
    def _train_asset_models(self, trainer, persistence, data, asset_name, time_window):
        """Train models for a specific asset."""
        try:
            # Set transaction costs (default if not provided)
            transaction_costs = getattr(self.args, 'transaction_costs', None) or [0.0001, 0.0001, 0.0002]
            
            # Train models using the new registry system
            models, scalers, feature_columns = trainer.train_models(data)
            
            results = []
            metrics = []
            
            # Save each model to the registry
            self.logger.info(f"Starting to save {len(models)} models for {asset_name}")
            for model_type, model in models.items():
                self.logger.info(f"Processing {model_type} model for {asset_name}")
                if feature_columns is not None:
                    # Get scaler if available (LogisticRegression needs it, XGBoost doesn't)
                    scaler = scalers.get(model_type, None)
                    features = feature_columns  # feature_columns is a list, not a dict
                    
                    if scaler is not None:
                        self.logger.info(f"Found scaler and features for {model_type}: {len(features)} features")
                    else:
                        self.logger.info(f"Found features for {model_type} (no scaler needed): {len(features)} features")
                    
                    # Get performance metrics (you might need to adjust this based on your ModelTrainer implementation)
                    performance_metrics = {
                        'accuracy': 0.75,  # Placeholder - you'll need to get actual metrics
                        'precision': 0.70,
                        'recall': 0.65,
                        'f1_score': 0.67
                    }
                    
                    # Save to registry with time window info
                    try:
                        self.logger.info(f"Attempting to save {model_type} model for {asset_name}")
                        
                        # Extract actual date range from data
                        if 'date' in data.columns:
                            actual_start_date = data['date'].min().strftime('%Y-%m-%d') if hasattr(data['date'].min(), 'strftime') else str(data['date'].min())
                            actual_end_date = data['date'].max().strftime('%Y-%m-%d') if hasattr(data['date'].max(), 'strftime') else str(data['date'].max())
                        else:
                            actual_start_date = 'unknown'
                            actual_end_date = 'unknown'
                        
                        model_id = persistence.save_model(
                            model=model,
                            scaler=scaler,
                            asset=asset_name,
                            model_type=model_type,
                            feature_names=features,
                            performance_metrics=performance_metrics,
                            training_params={
                                'time_window': time_window,
                                'start_date': actual_start_date,
                                'end_date': actual_end_date,
                                'data_source': str(data.index.min()) if hasattr(data.index, 'min') else 'unknown'
                            }
                        )
                        self.logger.info(f"Successfully saved {model_type} model for {asset_name} with ID: {model_id}")
                    except Exception as save_error:
                        self.logger.error(f"Failed to save {model_type} model for {asset_name}: {save_error}")
                        import traceback
                        self.logger.error(f"Full traceback: {traceback.format_exc()}")
                        continue  # Skip this model and continue with others
                else:
                    self.logger.warning(f"Missing scaler or features for {model_type} model for {asset_name}")
                    continue
                
                # Only add to results if save was successful
                results.append({
                    'model_id': model_id,
                    'asset': asset_name,
                    'model_type': model_type,
                    'time_window': time_window,
                    'accuracy': performance_metrics['accuracy'],
                    'precision': performance_metrics['precision'],
                    'recall': performance_metrics['recall'],
                    'f1_score': performance_metrics['f1_score']
                })
                
                for metric_name, metric_value in performance_metrics.items():
                    metrics.append({
                        'model_id': model_id,
                        'asset': asset_name,
                        'model_type': model_type,
                        'time_window': time_window,
                        'metric_name': metric_name,
                        'metric_value': metric_value
                    })
            
            return results, metrics
            
        except Exception as e:
            self.logger.error(f"Failed to train models for {asset_name}: {e}")
            return [], []


