"""
Configuration module for the macro sentiment trading system.

This module provides configuration management, environment variable handling,
and collector class selection based on available credentials.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Type
from dataclasses import dataclass

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system environment variables only

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for the trading system."""

    # Data collection settings
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    results_dir: str = "results"
    models_dir: str = "results/models"

    # GDELT settings
    gdelt_api_key: Optional[str] = None
    gdelt_base_url: str = "https://api.gdeltproject.org/api/v2"

    # BigQuery settings
    bigquery_project_id: Optional[str] = None
    bigquery_credentials_path: Optional[str] = None
    bigquery_dataset: str = "gdelt"
    max_cost_usd: float = 5.0  # Default max cost for BigQuery queries
    
    # Yahoo Finance settings
    yahoo_finance_base_url: str = "https://query1.finance.yahoo.com/v8/finance/chart"
    
    # Model settings
    default_models: list = None
    default_assets: list = None
    default_timeframes: list = None
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "logs/trading_system.log"
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.default_models is None:
            self.default_models = ["logistic", "xgboost"]
        if self.default_assets is None:
            self.default_assets = ["EURUSD", "USDJPY", "TNOTE"]
        if self.default_timeframes is None:
            self.default_timeframes = ["1D", "1W", "1M", "1Q"]
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the current configuration and return status.
        
        Returns:
            Dict containing validation results
        """
        status = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'method': 'bigquery'  # Default method
        }
        
        # Check BigQuery configuration
        if self.bigquery_project_id and self.bigquery_credentials_path:
            if Path(self.bigquery_credentials_path).exists():
                status['method'] = 'bigquery'
                status['bigquery_available'] = True
            else:
                status['errors'].append(f"BigQuery credentials file not found: {self.bigquery_credentials_path}")
                status['valid'] = False
        else:
            status['warnings'].append("BigQuery not configured, falling back to free method")
            status['method'] = 'free'
            status['bigquery_available'] = False
        
        # Check GDELT API key
        if self.gdelt_api_key:
            status['gdelt_api_available'] = True
        else:
            status['warnings'].append("GDELT API key not configured")
            status['gdelt_api_available'] = False
        
        return status
    
    @property
    def method(self) -> str:
        """Get the current data collection method."""
        status = self.validate_configuration()
        return status['method']
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls(
            data_dir=os.getenv('DATA_DIR', 'data'),
            cache_dir=os.getenv('CACHE_DIR', 'data/cache'),
            results_dir=os.getenv('RESULTS_DIR', 'results'),
            models_dir=os.getenv('MODELS_DIR', 'results/models'),
            gdelt_api_key=os.getenv('GDELT_API_KEY'),
            gdelt_base_url=os.getenv('GDELT_BASE_URL', 'https://api.gdeltproject.org/api/v2'),
            # Support both naming conventions for BigQuery
            bigquery_project_id=os.getenv('BIGQUERY_PROJECT_ID') or os.getenv('GOOGLE_CLOUD_PROJECT'),
            bigquery_credentials_path=os.getenv('BIGQUERY_CREDENTIALS_PATH') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
            bigquery_dataset=os.getenv('BIGQUERY_DATASET', 'gdelt'),
            max_cost_usd=float(os.getenv('BIGQUERY_MAX_COST_USD', '5.0')),
            yahoo_finance_base_url=os.getenv('YAHOO_FINANCE_BASE_URL', 'https://query1.finance.yahoo.com/v8/finance/chart'),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_file=os.getenv('LOG_FILE', 'logs/trading_system.log')
        )

# Global configuration instance
config = Config.from_env()

def get_collector_class() -> Type:
    """
    Determine the appropriate collector class based on available credentials.
    
    Returns:
        Type: The collector class to use (GDELTBigQueryCollector or GDELTCollector)
    """
    # Check for BigQuery credentials
    if config.bigquery_project_id and config.bigquery_credentials_path:
        if Path(config.bigquery_credentials_path).exists():
            try:
                from .gdelt_bigquery_collector import GDELTBigQueryCollector
                logger.info("Using BigQuery collector (credentials found)")
                return GDELTBigQueryCollector
            except ImportError as e:
                logger.warning(f"BigQuery collector not available: {e}")
    
    # Fallback to free GDELT API
    try:
        from .news_collector import GDELTCollector
        logger.info("Using free GDELT API collector")
        return GDELTCollector
    except ImportError as e:
        logger.error(f"Free GDELT collector not available: {e}")
        raise ImportError("No GDELT collectors available")

def print_configuration_status():
    """Print the current configuration status."""
    logger.info("=" * 60)
    logger.info("CONFIGURATION STATUS")
    logger.info("=" * 60)
    
    # Data directories
    logger.info(f"Data Directory: {config.data_dir}")
    logger.info(f"Cache Directory: {config.cache_dir}")
    logger.info(f"Results Directory: {config.results_dir}")
    logger.info(f"Models Directory: {config.models_dir}")
    
    # GDELT settings
    logger.info(f"GDELT API Key: {'Set' if config.gdelt_api_key else 'Not set'}")
    logger.info(f"GDELT Base URL: {config.gdelt_base_url}")
    
    # BigQuery settings
    logger.info(f"BigQuery Project ID: {'Set' if config.bigquery_project_id else 'Not set'}")
    logger.info(f"BigQuery Credentials: {'Set' if config.bigquery_credentials_path else 'Not set'}")
    if config.bigquery_credentials_path:
        logger.info(f"Credentials Path: {config.bigquery_credentials_path}")
        logger.info(f"Credentials Exist: {Path(config.bigquery_credentials_path).exists()}")
    
    # Model settings
    logger.info(f"Default Models: {config.default_models}")
    logger.info(f"Default Assets: {config.default_assets}")
    logger.info(f"Default Timeframes: {config.default_timeframes}")
    
    # Logging settings
    logger.info(f"Log Level: {config.log_level}")
    logger.info(f"Log File: {config.log_file}")
    
    logger.info("=" * 60)

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        config.data_dir,
        config.cache_dir,
        config.results_dir,
        config.models_dir,
        Path(config.log_file).parent
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

def get_asset_config(asset: str) -> Dict[str, Any]:
    """
    Get configuration for a specific asset.
    
    Args:
        asset: Asset symbol (e.g., 'EURUSD', 'USDJPY')
        
    Returns:
        Dict containing asset-specific configuration
    """
    asset_configs = {
        'EURUSD': {
            'symbol': 'EURUSD=X',
            'name': 'EUR/USD',
            'category': 'forex',
            'base_currency': 'EUR',
            'quote_currency': 'USD'
        },
        'USDJPY': {
            'symbol': 'USDJPY=X',
            'name': 'USD/JPY',
            'category': 'forex',
            'base_currency': 'USD',
            'quote_currency': 'JPY'
        },
        'TNOTE': {
            'symbol': '^TNX',
            'name': '10-Year Treasury Note',
            'category': 'bonds',
            'base_currency': 'USD',
            'quote_currency': 'USD'
        },
        'GBPUSD': {
            'symbol': 'GBPUSD=X',
            'name': 'GBP/USD',
            'category': 'forex',
            'base_currency': 'GBP',
            'quote_currency': 'USD'
        },
        'AUDUSD': {
            'symbol': 'AUDUSD=X',
            'name': 'AUD/USD',
            'category': 'forex',
            'base_currency': 'AUD',
            'quote_currency': 'USD'
        },
        'USDCHF': {
            'symbol': 'USDCHF=X',
            'name': 'USD/CHF',
            'category': 'forex',
            'base_currency': 'USD',
            'quote_currency': 'CHF'
        },
        'USDCAD': {
            'symbol': 'USDCAD=X',
            'name': 'USD/CAD',
            'category': 'forex',
            'base_currency': 'USD',
            'quote_currency': 'CAD'
        },
        'GOLD': {
            'symbol': 'GC=F',
            'name': 'Gold Futures',
            'category': 'commodities',
            'base_currency': 'USD',
            'quote_currency': 'USD'
        },
        'CRUDE': {
            'symbol': 'CL=F',
            'name': 'Crude Oil Futures',
            'category': 'commodities',
            'base_currency': 'USD',
            'quote_currency': 'USD'
        },
        'SP500': {
            'symbol': '^GSPC',
            'name': 'S&P 500',
            'category': 'indices',
            'base_currency': 'USD',
            'quote_currency': 'USD'
        },
        'VIX': {
            'symbol': '^VIX',
            'name': 'VIX',
            'category': 'indices',
            'base_currency': 'USD',
            'quote_currency': 'USD'
        }
    }
    
    return asset_configs.get(asset, {
        'symbol': asset,
        'name': asset,
        'category': 'unknown',
        'base_currency': 'USD',
        'quote_currency': 'USD'
    })

def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model type.
    
    Args:
        model_type: Model type (e.g., 'logistic', 'xgboost')
        
    Returns:
        Dict containing model-specific configuration
    """
    model_configs = {
        'logistic': {
            'class_name': 'LogisticRegression',
            'module': 'sklearn.linear_model',
            'default_params': {
                'random_state': 42,
                'max_iter': 1000
            }
        },
        'xgboost': {
            'class_name': 'XGBClassifier',
            'module': 'xgboost',
            'default_params': {
                'random_state': 42,
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1
            }
        }
    }
    
    return model_configs.get(model_type, {
        'class_name': model_type,
        'module': 'unknown',
        'default_params': {}
    })

