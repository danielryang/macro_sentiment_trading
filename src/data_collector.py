"""
Unified Data Collector with Method Selection

Provides a single interface for GDELT data collection that can switch
between free (raw files) and BigQuery methods based on configuration.
"""

import os
import logging
from typing import Optional, Union
import pandas as pd

from .config import config, get_collector_class, print_configuration_status
from .headline_processor import HeadlineProcessor

logger = logging.getLogger(__name__)


class UnifiedGDELTCollector:
    """
    Unified GDELT data collector that automatically selects the best available method.
    """
    
    def __init__(self, force_method: Optional[str] = None):
        """
        Initialize the unified collector.
        
        Args:
            force_method: Force a specific method ('free' or 'bigquery'), 
                         overriding environment configuration
        """
        self.force_method = force_method
        self.config = config
        self.headline_processor = HeadlineProcessor()
        self.data_dir = "data"  # Default data directory
        
        # Determine which method to use
        if force_method:
            if force_method not in ['free', 'bigquery']:
                raise ValueError("force_method must be 'free' or 'bigquery'")
            self.method = force_method
        else:
            self.method = self.config.method
            
        # Validate and potentially fallback
        self._validate_and_initialize()
    
    def _validate_and_initialize(self):
        """Validate configuration and initialize the appropriate collector."""
        status = self.config.validate_configuration()
        
        if self.method == 'bigquery':
            if not status['bigquery_available']:
                logger.warning("BigQuery method requested but not available, falling back to free method")
                self.method = 'free'
        
        # Initialize the appropriate collector
        if self.method == 'bigquery':
            from .gdelt_bigquery_collector import GDELTBigQueryCollector
            self.collector = GDELTBigQueryCollector(
                project_id=self.config.bigquery_project_id,
                credentials_path=self.config.bigquery_credentials_path
            )
            logger.info("Initialized BigQuery GDELT collector")
        else:
            from .news_collector import GDELTCollector
            self.collector = GDELTCollector()
            logger.info("Initialized free GDELT API collector")
    
    def fetch_events(
        self,
        start_date: str,
        end_date: str,
        top_n_per_day: int = 100,
        include_headlines: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch GDELT events using the configured method.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format  
            top_n_per_day: Number of top events per day to fetch
            include_headlines: Whether to process headlines
            **kwargs: Additional arguments passed to the specific collector
            
        Returns:
            DataFrame with events and optionally headlines
        """
        logger.info(f"Fetching GDELT events using {self.method} method")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        if self.method == 'bigquery':
            return self._fetch_bigquery_events(
                start_date, end_date, top_n_per_day, include_headlines, **kwargs
            )
        else:
            return self._fetch_free_events(
                start_date, end_date, top_n_per_day, include_headlines, **kwargs
            )
    
    def _fetch_bigquery_events(
        self, 
        start_date: str, 
        end_date: str, 
        top_n_per_day: int,
        include_headlines: bool,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch events using BigQuery method."""
        # Estimate cost first
        cost_estimate = self.collector.estimate_query_cost(start_date, end_date)
        
        if cost_estimate['estimated_cost_usd'] > self.config.max_cost_usd:
            raise ValueError(
                f"Estimated query cost ${cost_estimate['estimated_cost_usd']:.3f} "
                f"exceeds limit ${self.config.max_cost_usd:.2f}. "
                f"Adjust date range or increase BIGQUERY_MAX_COST_USD"
            )
        
        logger.info(f"Estimated BigQuery cost: ${cost_estimate['estimated_cost_usd']:.3f}")
        
        # Calculate daily limit based on total desired events
        days = pd.date_range(start_date, end_date).shape[0]
        total_limit = top_n_per_day * days
        
        # Filter out parameters that BigQuery collector doesn't expect
        bigquery_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['force_refresh', 'top_n_per_day', 'include_headlines']}
        
        # Fetch hybrid data: GKG headlines + events metadata (HYBRID APPROACH)
        events_df = self.collector.fetch_hybrid_data(
            start_date=start_date,
            end_date=end_date,
            event_codes=[100, 110, 120, 130, 140, 150, 160, 170, 180, 190],  # Macro events
            limit=total_limit
        )
        
        logger.info(f"Retrieved {len(events_df)} events from BigQuery")
        
        # GKG data includes actual headlines for FinBERT sentiment analysis
        if include_headlines and not events_df.empty:
            logger.info("GKG data includes actual headlines for FinBERT sentiment analysis")
            logger.info("Headlines are ready for sentiment analysis")
            # Headlines are already populated from GKG table
            
        return events_df
    
    def _fetch_free_events(
        self,
        start_date: str,
        end_date: str,
        top_n_per_day: int,
        include_headlines: bool,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch events using free GDELT API method."""
        # Filter out parameters that free collector doesn't expect
        free_kwargs = {k: v for k, v in kwargs.items() if k == 'force_refresh'}

        # Use existing free method (GDELTCollector.fetch_events only accepts start_date, end_date, force_refresh)
        events_df = self.collector.fetch_events(
            start_date=start_date,
            end_date=end_date,
            **free_kwargs
        )

        logger.info(f"Retrieved {len(events_df)} events from free GDELT API")

        # Process headlines if requested and events exist
        if include_headlines and not events_df.empty:
            logger.info("Processing headlines...")
            events_df = self.headline_processor.process_articles(events_df)

        return events_df
    
    def get_method_info(self) -> dict:
        """Get information about the current method."""
        info = {
            "method": self.method,
            "cost_estimate": None,
            "features": {
                "free": True,
                "headlines_included": self.method == 'bigquery',
                "fast_retrieval": self.method == 'bigquery',
                "unlimited_historical": True
            }
        }
        
        if self.method == 'bigquery':
            # Cost estimate info not yet implemented
            info["cost_estimate"] = "BigQuery method active"

        return info
    
    def print_status(self):
        """Print current collector status."""
        print_configuration_status()

        info = self.get_method_info()
        print(f"\nACTIVE Method: {info['method']}")

        if info['cost_estimate'] and isinstance(info['cost_estimate'], dict):
            print(f"COST estimates: {info['cost_estimate'].get('estimated_costs', 'N/A')}")
        elif info['cost_estimate']:
            print(f"Cost estimate: {info['cost_estimate']}")


def collect_and_process_news(
    start_date: str,
    end_date: str,
    force_refresh: bool = False,
    use_method: Optional[str] = None,
    top_n_per_day: int = 100
) -> pd.DataFrame:
    """
    Convenience function for notebook use - collects and processes news data with intelligent caching.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        force_refresh: Force refresh of cached data
        use_method: Force specific method ('free' or 'bigquery')
        top_n_per_day: Number of events per day to collect

    Returns:
        DataFrame with events and headlines
    """
    # Validate date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    current_date = pd.Timestamp.now()

    if start_dt >= end_dt:
        raise ValueError(f"Start date {start_date} must be before end date {end_date}")

    if end_dt > current_date:
        logger.warning(f"WARNING: End date {end_date} is in the future. Clamping to current date {current_date.strftime('%Y-%m-%d')}")
        end_dt = current_date
        end_date = end_dt.strftime('%Y-%m-%d')

    if start_dt > current_date:
        raise ValueError(f"Start date {start_date} cannot be in the future")

    # ENHANCED CACHE DETECTION - Check for existing cache files intelligently
    cache_dir = os.path.join('data', 'news')
    method = use_method or config.method

    # Primary cache path (method-specific)
    primary_cache_path = os.path.join(cache_dir, f'gdelt_{method}_{start_date}_{end_date}.parquet')

    # Alternative cache paths (check both methods)
    alternative_cache_paths = []
    for alt_method in ['bigquery', 'free']:
        if alt_method != method:
            alt_path = os.path.join(cache_dir, f'gdelt_{alt_method}_{start_date}_{end_date}.parquet')
            if os.path.exists(alt_path):
                alternative_cache_paths.append((alt_method, alt_path))

    # Check for superset cache files (longer date ranges that contain our range)
    superset_cache_paths = []
    if os.path.exists(cache_dir):
        for filename in os.listdir(cache_dir):
            if filename.startswith('gdelt_') and filename.endswith('.parquet'):
                try:
                    # Parse filename: gdelt_{method}_{start}_{end}.parquet
                    parts = filename.replace('.parquet', '').split('_')
                    if len(parts) >= 4:
                        file_method = parts[1]
                        file_start = '_'.join(parts[2:-1])  # Handle dates with hyphens
                        file_end = parts[-1]

                        # Check if this cache file contains our date range
                        file_start_dt = pd.to_datetime(file_start)
                        file_end_dt = pd.to_datetime(file_end)

                        if file_start_dt <= start_dt and file_end_dt >= end_dt:
                            file_path = os.path.join(cache_dir, filename)
                            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                            superset_cache_paths.append((file_method, file_path, file_size_mb, file_start, file_end))
                except:
                    continue

    # Try to load from cache (priority order: primary -> alternative -> superset)
    cache_to_use = None

    if not force_refresh:
        # 1. Primary cache path
        if os.path.exists(primary_cache_path):
            cache_to_use = (method, primary_cache_path, "primary")

        # 2. Alternative method cache (exact date match)
        elif alternative_cache_paths:
            alt_method, alt_path = alternative_cache_paths[0]
            cache_to_use = (alt_method, alt_path, "alternative")
            logger.info(f"[CACHE] Primary cache not found, using {alt_method} cache instead")

        # 3. Superset cache (contains our date range)
        elif superset_cache_paths:
            # Sort by size (prefer smaller, more specific files)
            superset_cache_paths.sort(key=lambda x: x[2])
            file_method, file_path, file_size_mb, file_start, file_end = superset_cache_paths[0]
            cache_to_use = (file_method, file_path, "superset")
            logger.info(f"[CACHE] Exact cache not found, using superset cache {file_start} to {file_end} ({file_size_mb:.1f} MB)")

        # Attempt to load the selected cache
        if cache_to_use:
            cache_method, cache_path, cache_type = cache_to_use
            try:
                logger.info(f"[CACHE] Loading {cache_type} cached data from: {cache_path}")

                # Enhanced loading with multiple fallback strategies
                events_df = None

                # Strategy 1: Try pyarrow engine (best for BigQuery data types)
                try:
                    events_df = pd.read_parquet(cache_path, engine='pyarrow')
                except Exception as e1:
                    logger.warning(f"PyArrow loading failed: {e1}. Trying fallback methods...")

                    # Strategy 2: Try fastparquet engine
                    try:
                        events_df = pd.read_parquet(cache_path, engine='fastparquet')
                        logger.info("Successfully loaded with fastparquet engine")
                    except Exception as e2:
                        logger.warning(f"FastParquet loading failed: {e2}. Trying data type conversion...")

                        # Strategy 3: Try to use pre-fixed version if available
                        try:
                            fixed_cache_path = cache_path.replace('.parquet', '.fixed.parquet')
                            if os.path.exists(fixed_cache_path):
                                logger.info(f"Found pre-fixed cache version: {fixed_cache_path}")
                                events_df = pd.read_parquet(fixed_cache_path, engine='pyarrow')
                                logger.info("Successfully loaded pre-fixed cache version")
                            else:
                                # Strategy 4: Manual raw PyArrow handling for BigQuery files
                                import pyarrow.parquet as pq
                                import pyarrow as pa

                                logger.info("Attempting low-level PyArrow fix for BigQuery data types...")

                                # Read the raw arrow table
                                parquet_file = pq.ParquetFile(cache_path)
                                table = parquet_file.read()

                                # Convert problematic columns at the Arrow level
                                new_schema = []
                                column_arrays = []

                                for i, field in enumerate(table.schema):
                                    col_name = field.name
                                    col_array = table.column(i)

                                    # Handle dbdate columns
                                    if str(field.type) == 'dbdate' or 'date' in col_name.lower():
                                        # Convert to timestamp
                                        try:
                                            # Try to convert dbdate to timestamp
                                            timestamp_array = pa.compute.cast(col_array, pa.timestamp('us'))
                                            new_schema.append(pa.field(col_name, pa.timestamp('us')))
                                            column_arrays.append(timestamp_array)
                                            logger.info(f"Converted {col_name} from {field.type} to timestamp")
                                        except:
                                            # If conversion fails, convert to string
                                            string_array = pa.compute.cast(col_array, pa.string())
                                            new_schema.append(pa.field(col_name, pa.string()))
                                            column_arrays.append(string_array)
                                            logger.info(f"Converted {col_name} from {field.type} to string")
                                    else:
                                        # Keep other columns as-is
                                        new_schema.append(field)
                                        column_arrays.append(col_array)

                                # Create new table with fixed schema
                                fixed_schema = pa.schema(new_schema)
                                fixed_table = pa.table(column_arrays, schema=fixed_schema)

                                # Convert to pandas
                                events_df = fixed_table.to_pandas()
                                logger.info("Successfully loaded with low-level PyArrow conversion")

                        except Exception as e3:
                            logger.warning(f"All loading strategies failed: {e3}")
                            # Re-raise the original error since all strategies failed
                            raise e1

                if events_df is None:
                    raise ValueError("All loading strategies failed")

                # Filter to requested date range if using superset cache
                if cache_type == "superset":
                    original_len = len(events_df)
                    events_df = events_df[
                        (events_df['date'] >= start_date) &
                        (events_df['date'] <= end_date)
                    ].copy()
                    logger.info(f"[FILTER] Filtered from {original_len} to {len(events_df)} events for date range")

                cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
                logger.info(f"[OK] Loaded {len(events_df)} cached events ({cache_size_mb:.1f} MB)")
                logger.info(f"[DATE] Date range: {events_df['date'].min()} to {events_df['date'].max()}")

                # Update method if we used alternative cache
                if cache_type == "alternative":
                    method = cache_method
                    logger.info(f"[METHOD] Switched to {method} method based on available cache")

                return events_df

            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}. Removing corrupted cache and collecting fresh data...")
                # Remove corrupted cache file to prevent future issues
                try:
                    os.remove(cache_path)
                    logger.info(f"Removed corrupted cache file: {cache_path}")
                except Exception as remove_error:
                    logger.warning(f"Could not remove corrupted cache: {remove_error}")
    
    # Collect fresh data
    logger.info(f"[FRESH] Collecting fresh data for {start_date} to {end_date}")
    collector = UnifiedGDELTCollector(force_method=use_method)
    
    # Print status for user awareness
    collector.print_status()
    
    # Fetch events
    events_df = collector.fetch_events(
        start_date=start_date,
        end_date=end_date,
        top_n_per_day=top_n_per_day,
        include_headlines=True
    )
    
    # Save results to cache
    if not events_df.empty:
        # Define cache path for fresh data
        cache_path = os.path.join(cache_dir, f'gdelt_{method}_{start_date}_{end_date}.parquet')
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # Fix data types for parquet compatibility
        events_df_clean = events_df.copy()
        for col in events_df_clean.columns:
            # Convert any problematic dtypes to standard pandas dtypes
            if hasattr(events_df_clean[col].dtype, 'name'):
                dtype_name = events_df_clean[col].dtype.name
                if 'dbdate' in dtype_name or 'date' in dtype_name.lower():
                    # Convert date-like columns to standard datetime
                    events_df_clean[col] = pd.to_datetime(events_df_clean[col], errors='coerce')
                elif 'object' in dtype_name and events_df_clean[col].dtype == 'object':
                    # Keep object columns as-is but ensure they're clean
                    continue
        
        # Use robust parquet saving to prevent dbdate issues
        from .parquet_utils import save_parquet_robust
        save_parquet_robust(events_df_clean, cache_path)
        cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    logger.info(f"[SAVE] Cached {len(events_df)} events to: {cache_path} ({cache_size_mb:.1f} MB)")
    logger.info(f"[INFO] Next time you run this date range, data will load instantly!")
    
    return events_df

