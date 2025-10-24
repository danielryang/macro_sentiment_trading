"""
CLI Configuration Module

This module provides configuration management for the CLI interface.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class CLIConfig:
    """Configuration manager for CLI operations."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize CLI configuration."""
        self.config_file = config_file
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment."""
        config = {
            'log_level': 'INFO',
            'log_file': 'logs/cli.log',
            'output_dir': 'results',
            'data_dir': 'data',
            'models_dir': 'results/models',
            'cache_dir': 'data/cache',
            'force_refresh': False,
            'verbose': False,
            'dry_run': False
        }
        
        # Override with environment variables if present
        if os.getenv('LOG_LEVEL'):
            config['log_level'] = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FILE'):
            config['log_file'] = os.getenv('LOG_FILE')
        if os.getenv('OUTPUT_DIR'):
            config['output_dir'] = os.getenv('OUTPUT_DIR')
        if os.getenv('DATA_DIR'):
            config['data_dir'] = os.getenv('DATA_DIR')
        if os.getenv('MODELS_DIR'):
            config['models_dir'] = os.getenv('MODELS_DIR')
        if os.getenv('CACHE_DIR'):
            config['cache_dir'] = os.getenv('CACHE_DIR')
        if os.getenv('FORCE_REFRESH', '').lower() in ('true', '1', 'yes'):
            config['force_refresh'] = True
        if os.getenv('VERBOSE', '').lower() in ('true', '1', 'yes'):
            config['verbose'] = True
        if os.getenv('DRY_RUN', '').lower() in ('true', '1', 'yes'):
            config['dry_run'] = True
            
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
    
    def get_log_level(self) -> str:
        """Get logging level."""
        return self.config.get('log_level', 'INFO')
    
    def get_log_file(self) -> str:
        """Get log file path."""
        return self.config.get('log_file', 'logs/cli.log')
    
    def get_output_dir(self) -> str:
        """Get output directory."""
        return self.config.get('output_dir', 'results')
    
    def get_data_dir(self) -> str:
        """Get data directory."""
        return self.config.get('data_dir', 'data')
    
    def get_models_dir(self) -> str:
        """Get models directory."""
        return self.config.get('models_dir', 'results/models')
    
    def get_cache_dir(self) -> str:
        """Get cache directory."""
        return self.config.get('cache_dir', 'data/cache')
    
    def is_force_refresh(self) -> bool:
        """Check if force refresh is enabled."""
        return self.config.get('force_refresh', False)
    
    def is_verbose(self) -> bool:
        """Check if verbose mode is enabled."""
        return self.config.get('verbose', False)
    
    def is_dry_run(self) -> bool:
        """Check if dry run mode is enabled."""
        return self.config.get('dry_run', False)
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = getattr(logging, self.get_log_level().upper(), logging.INFO)
        log_file = self.get_log_file()
        
        # Ensure log directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

