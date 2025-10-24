#!/usr/bin/env python3
"""
Check parquet file compatibility and health
"""

import os
import sys
from pathlib import Path
from cli.commands.base import BaseCommand
from src.parquet_utils import test_parquet_compatibility, read_parquet_robust

class CheckParquetCommand(BaseCommand):
    """Check parquet file compatibility."""
    
    def __init__(self, config, args):
        super().__init__(config, args)
        self.command_name = "check-parquet"
    
    def add_arguments(self, parser):
        """Add command-specific arguments."""
        parser.add_argument(
            '--file-path', 
            type=str, 
            help='Path to parquet file to check'
        )
        parser.add_argument(
            '--directory', 
            type=str, 
            help='Directory to check all parquet files'
        )
        parser.add_argument(
            '--fix', 
            action='store_true', 
            help='Fix incompatible parquet files'
        )
    
    def execute(self) -> int:
        """Execute parquet health check."""
        try:
            with self.with_logged_operation("Parquet Health Check"):
                if self.args.file_path:
                    return self._check_single_file(self.args.file_path)
                elif self.args.directory:
                    return self._check_directory(self.args.directory)
                else:
                    self.logger.error("Please specify --file-path or --directory")
                    return 1
                    
        except Exception as e:
            self.logger.error(f"Parquet check failed: {e}")
            return 1
    
    def _check_single_file(self, file_path: str) -> int:
        """Check a single parquet file."""
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return 1
        
        self.logger.info(f"Checking parquet file: {file_path}")
        
        # Test compatibility
        if test_parquet_compatibility(file_path):
            self.logger.info("✅ File is compatible")
            
            # Show basic info
            try:
                df = read_parquet_robust(file_path)
                self.logger.info(f"📊 Shape: {df.shape}")
                self.logger.info(f"📋 Columns: {list(df.columns)}")
                return 0
            except Exception as e:
                self.logger.error(f"❌ Error reading file: {e}")
                return 1
        else:
            self.logger.error("❌ File is incompatible")
            return 1
    
    def _check_directory(self, directory: str) -> int:
        """Check all parquet files in a directory."""
        if not os.path.exists(directory):
            self.logger.error(f"Directory not found: {directory}")
            return 1
        
        parquet_files = list(Path(directory).glob("**/*.parquet"))
        
        if not parquet_files:
            self.logger.warning(f"No parquet files found in: {directory}")
            return 0
        
        self.logger.info(f"Found {len(parquet_files)} parquet files")
        
        compatible_count = 0
        incompatible_files = []
        
        for file_path in parquet_files:
            if test_parquet_compatibility(str(file_path)):
                compatible_count += 1
                self.logger.info(f"✅ {file_path.name}")
            else:
                incompatible_files.append(str(file_path))
                self.logger.error(f"❌ {file_path.name}")
        
        self.logger.info(f"📊 Summary: {compatible_count}/{len(parquet_files)} files compatible")
        
        if incompatible_files:
            self.logger.warning(f"❌ Incompatible files: {len(incompatible_files)}")
            if self.args.fix:
                return self._fix_incompatible_files(incompatible_files)
            else:
                self.logger.info("💡 Use --fix to attempt repairs")
        
        return 0 if incompatible_files else 0
    
    def _fix_incompatible_files(self, incompatible_files: list) -> int:
        """Attempt to fix incompatible parquet files."""
        self.logger.info(f"🔧 Attempting to fix {len(incompatible_files)} incompatible files")
        
        fixed_count = 0
        for file_path in incompatible_files:
            try:
                # Try to read and re-save the file
                df = read_parquet_robust(file_path)
                
                # Create backup
                backup_path = file_path.replace('.parquet', '.backup.parquet')
                os.rename(file_path, backup_path)
                
                # Save with robust method
                from src.parquet_utils import save_parquet_robust
                save_parquet_robust(df, file_path)
                
                # Test the fixed file
                if test_parquet_compatibility(file_path):
                    self.logger.info(f"✅ Fixed: {os.path.basename(file_path)}")
                    fixed_count += 1
                    # Remove backup
                    os.unlink(backup_path)
                else:
                    # Restore backup
                    os.rename(backup_path, file_path)
                    self.logger.error(f"❌ Could not fix: {os.path.basename(file_path)}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error fixing {os.path.basename(file_path)}: {e}")
        
        self.logger.info(f"🔧 Fixed {fixed_count}/{len(incompatible_files)} files")
        return 0
