"""
Process Market Command

This command processes market data and aligns it with sentiment features.
"""

from pathlib import Path
import pandas as pd
from .base import BaseCommand


class ProcessMarketCommand(BaseCommand):
    """Command to process market data."""
    
    def validate_args(self) -> None:
        """Validate command arguments."""
        # Validate date range
        from datetime import datetime
        try:
            start_date = datetime.strptime(self.args.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(self.args.end_date, "%Y-%m-%d")

            if start_date > end_date:
                raise ValueError("Start date must be before or equal to end date")

            if end_date > datetime.now():
                raise ValueError("End date cannot be in the future")
                
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}")
    
    def execute(self) -> int:
        """Execute market data processing."""
        try:
            with self.with_logged_operation("Market Data Processing"):
                # Import here to avoid circular imports
                from src.market_processor import MarketProcessor
                
                self.logger.info(f"Processing market data for assets: {self.args.assets}")
                
                processor = MarketProcessor()
                
                # Fetch market data
                market_data = processor.fetch_market_data(
                    start_date=self.args.start_date,
                    end_date=self.args.end_date
                )
                
                # Add market features (only for requested assets)
                processed_market_data = {}
                for asset_name in self.args.assets:
                    if asset_name in market_data:
                        processed_market_data[asset_name] = processor.compute_market_features(
                            market_data[asset_name]
                        )
                        self.log_data_info(f"Market Data - {asset_name}", processed_market_data[asset_name])

                # Load daily features if available
                daily_features = None
                start_date_str = self.args.start_date.replace("-", "")
                end_date_str = self.args.end_date.replace("-", "")

                # Try time-windowed filename first
                daily_features_path = Path(self.args.output_dir) / f"daily_features_{start_date_str}_{end_date_str}.parquet"
                if not daily_features_path.exists():
                    # Fallback to generic filename
                    daily_features_path = Path(self.get_output_path("daily_features", ".parquet"))

                if daily_features_path.exists():
                    daily_features = pd.read_parquet(daily_features_path)
                    self.logger.info(f"Loaded daily features from: {daily_features_path}")
                else:
                    self.logger.warning("Daily features not found, creating dummy features")
                    # Create dummy daily features for alignment
                    daily_features = self._create_dummy_daily_features(
                        self.args.start_date, self.args.end_date
                    )
                
                # Align features (only for processed assets)
                aligned_data = processor.align_features(processed_market_data, daily_features)

                # Create time-windowed directory (date strings already created above)
                time_window_dir = Path(self.args.output_dir) / f"{start_date_str}_{end_date_str}"
                time_window_dir.mkdir(parents=True, exist_ok=True)
                
                # Save aligned data in time-windowed directory
                for asset_name, data in aligned_data.items():
                    # Simple filename in time-windowed directory: aligned_data_ASSET.parquet
                    asset_path = time_window_dir / f"aligned_data_{asset_name}.parquet"
                    data.to_parquet(asset_path)
                    self.logger.info(f"Saved aligned data for {asset_name} ({self.args.start_date} to {self.args.end_date}) to: {asset_path}")
                
                self.logger.info(f"Created time-windowed directory: {time_window_dir}")
                
                return 0
                
        except Exception as e:
            self.logger.error(f"Market data processing failed: {e}", exc_info=True)
            return 1
    
    def _create_dummy_daily_features(self, start_date: str, end_date: str):
        """Create dummy daily features for testing."""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
        
        dummy_features = pd.DataFrame({
            'date': dates,
            'mean_sentiment': np.random.normal(0, 0.1, len(dates)),
            'sentiment_std': np.random.uniform(0.05, 0.2, len(dates)),
            'news_volume': np.random.poisson(100, len(dates)),
            'goldstein_mean': np.random.normal(0, 0.5, len(dates)),
            'goldstein_std': np.random.uniform(0.1, 0.3, len(dates)),
            'article_impact': np.random.poisson(50, len(dates))
        })
        
        return dummy_features


