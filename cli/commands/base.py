"""
Base Command Class

This module provides the base command class that all CLI commands inherit from.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class BaseCommand:
    """Base class for all CLI commands."""
    
    def __init__(self, config, args):
        """Initialize the base command."""
        self.config = config
        self.args = args
        self.logger = logging.getLogger(self.__class__.__name__)
        self.command_name = getattr(self, 'command_name', self.__class__.__name__.lower())
    
    def add_arguments(self, parser):
        """Add command-specific arguments. Override in subclasses."""
        pass
    
    def validate_args(self) -> None:
        """Validate command arguments. Override in subclasses."""
        pass
    
    def execute(self) -> int:
        """Execute the command. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def check_file_exists(self, file_path: str, required: bool = True) -> None:
        """Check if a file exists and raise error if required but missing."""
        path = Path(file_path)
        if required and not path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
        elif not required and not path.exists():
            self.logger.warning(f"Optional file not found: {file_path}")
    
    def get_output_path(self, filename: str, extension: str = ".parquet") -> str:
        """Get output path for a file."""
        output_dir = getattr(self.args, 'output_dir', 'results')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return str(Path(output_dir) / f"{filename}{extension}")
    
    def log_data_info(self, name: str, data) -> None:
        """Log information about a DataFrame."""
        if hasattr(data, 'shape'):
            self.logger.info(f"[DATA] {name} - Shape: {data.shape}")
        elif hasattr(data, '__len__'):
            self.logger.info(f"[DATA] {name} - Length: {len(data)}")
        else:
            self.logger.info(f"[DATA] {name} - Type: {type(data)}")
    
    def with_logged_operation(self, operation_name: str):
        """Context manager for logging operations."""
        return LoggedOperation(self.logger, operation_name)
    
    def save_json(self, data: Dict[str, Any], file_path: str) -> None:
        """Save data to JSON file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        self.logger.info(f"Saved JSON data to: {file_path}")
    
    def load_json(self, file_path: str) -> Dict[str, Any]:
        """Load data from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def get_timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")


class LoggedOperation:
    """Context manager for logging operations with timing."""
    
    def __init__(self, logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"[START] {self.operation_name} - START")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            if exc_type is None:
                self.logger.info(f"[TIME] {self.operation_name} completed in {duration:.2f}s")
                self.logger.info(f"[DONE] {self.operation_name} - COMPLETE")
            else:
                self.logger.error(f"[FAIL] {self.operation_name} failed after {duration:.2f}s")
                self.logger.error(f"[ERROR] {self.operation_name} - FAILED")
















