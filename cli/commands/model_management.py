"""
Model Management Commands

This module provides CLI commands for managing trained models including:
1. Listing available models
2. Showing model details
3. Deleting models
4. Comparing models
5. Model registry management
"""

from .base import BaseCommand
from src.model_persistence import ModelPersistence
import pandas as pd
from typing import List, Dict, Any
import json
from pathlib import Path


class ListModelsCommand(BaseCommand):
    """Command to list available trained models."""
    
    def execute(self) -> int:
        """Execute model listing with advanced sorting and filtering."""
        try:
            with self.with_logged_operation("List Models"):
                persistence = ModelPersistence()
                
                # Get filters and sorting options from args
                asset_filter = getattr(self.args, 'asset', None)
                model_type_filter = getattr(self.args, 'model_type', None)
                show_details = getattr(self.args, 'details', False)
                sort_by = getattr(self.args, 'sort_by', 'date-desc')
                performance_metric = getattr(self.args, 'performance_metric', 'accuracy')
                limit = getattr(self.args, 'limit', None)
                training_date_from = getattr(self.args, 'training_date_from', None)
                training_date_to = getattr(self.args, 'training_date_to', None)
                
                # List models with basic filters
                models = persistence.list_models(asset=asset_filter, model_type=model_type_filter)
                
                if not models:
                    self.logger.info("No models found.")
                    if asset_filter:
                        self.logger.info(f"Filter: asset={asset_filter}")
                    if model_type_filter:
                        self.logger.info(f"Filter: model_type={model_type_filter}")
                    return 0
                
                # Apply date filtering
                if training_date_from or training_date_to:
                    models = self._filter_models_by_date(models, training_date_from, training_date_to)
                
                # Apply sorting
                models = self._sort_models(models, sort_by, performance_metric)
                
                # Apply limit
                if limit:
                    models = models[:limit]
                
                self.logger.info(f"Found {len(models)} models (sorted by {sort_by}):")
                print("\n" + "="*80)
                print("AVAILABLE MODELS")
                print("="*80)
                
                for i, model in enumerate(models, 1):
                    model_id = model['model_id']
                    asset = model['asset']
                    model_type = model['model_type']
                    training_date = model['training_date']
                    feature_count = model['feature_count']
                    version = model.get('version', '1.0.0')
                    
                    print(f"\n{i}. {model_id}")
                    print(f"   Asset: {asset}")
                    print(f"   Type: {model_type}")
                    print(f"   Training Date: {training_date}")
                    print(f"   Features: {feature_count}")
                    print(f"   Version: {version}")
                    
                    # Show performance metrics if available
                    if 'metrics' in model and model['metrics']:
                        metrics = model['metrics']
                        if performance_metric in metrics:
                            print(f"   {performance_metric.title()}: {metrics[performance_metric]:.4f}")
                        if show_details:
                            print(f"   All Metrics:")
                            for metric, value in metrics.items():
                                if isinstance(value, float):
                                    print(f"     {metric}: {value:.4f}")
                                else:
                                    print(f"     {metric}: {value}")
                    
                    # Show training window if available
                    if 'training_params' in model and model['training_params']:
                        params = model['training_params']
                        if 'start_date' in params and 'end_date' in params:
                            start_date = params['start_date']
                            end_date = params['end_date']
                            print(f"   Training Window: {start_date} to {end_date}")
                
                print(f"\nTotal: {len(models)} models")
                if limit:
                    print(f"Showing top {limit} models")
                
                # Save to file if requested
                if hasattr(self.args, 'output_file') and self.args.output_file:
                    output_data = {
                        'models': models,
                        'filters': {
                            'asset': asset_filter,
                            'model_type': model_type_filter,
                            'training_date_from': training_date_from,
                            'training_date_to': training_date_to
                        },
                        'sorting': {
                            'sort_by': sort_by,
                            'performance_metric': performance_metric
                        },
                        'total_count': len(models)
                    }
                    
                    with open(self.args.output_file, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    
                    self.logger.info(f"Model list saved to: {self.args.output_file}")
                
                return 0
                
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return 1
    
    def _filter_models_by_date(self, models: List[Dict], date_from: str = None, date_to: str = None) -> List[Dict]:
        """Filter models by training date range."""
        from datetime import datetime
        
        filtered_models = []
        
        for model in models:
            training_date = model.get('training_date', '')
            if not training_date:
                continue
                
            try:
                model_date = datetime.fromisoformat(training_date.replace('Z', '+00:00'))
                
                # Check date from filter
                if date_from:
                    from_date = datetime.fromisoformat(date_from)
                    if model_date < from_date:
                        continue
                
                # Check date to filter
                if date_to:
                    to_date = datetime.fromisoformat(date_to)
                    if model_date > to_date:
                        continue
                
                filtered_models.append(model)
                
            except ValueError as e:
                self.logger.warning(f"Invalid date format in model {model.get('model_id', 'unknown')}: {e}")
                continue
        
        self.logger.info(f"Date filtering: {len(models)} -> {len(filtered_models)} models")
        return filtered_models
    
    def _sort_models(self, models: List[Dict], sort_by: str, performance_metric: str) -> List[Dict]:
        """Sort models by the specified criteria."""
        from datetime import datetime
        
        if sort_by == "date":
            return sorted(models, key=lambda x: x.get('training_date', ''))
        elif sort_by == "date-desc":
            return sorted(models, key=lambda x: x.get('training_date', ''), reverse=True)
        elif sort_by == "performance":
            return sorted(models, key=lambda x: x.get('metrics', {}).get(performance_metric, 0))
        elif sort_by == "performance-desc":
            return sorted(models, key=lambda x: x.get('metrics', {}).get(performance_metric, 0), reverse=True)
        elif sort_by == "training-window":
            return sorted(models, key=lambda x: self._get_training_window_days(x))
        elif sort_by == "training-window-desc":
            return sorted(models, key=lambda x: self._get_training_window_days(x), reverse=True)
        elif sort_by == "asset":
            return sorted(models, key=lambda x: x.get('asset', ''))
        elif sort_by == "model-type":
            return sorted(models, key=lambda x: x.get('model_type', ''))
        else:
            return models
    
    def _get_training_window_days(self, model: Dict) -> int:
        """Calculate training window in days from model metadata."""
        try:
            params = model.get('training_params', {})
            if 'start_date' in params and 'end_date' in params:
                start_date = datetime.fromisoformat(params['start_date'])
                end_date = datetime.fromisoformat(params['end_date'])
                return (end_date - start_date).days
        except (ValueError, KeyError):
            pass
        return 0


class ShowModelCommand(BaseCommand):
    """Command to show detailed information about a specific model."""
    
    def validate_args(self) -> None:
        """Validate command arguments."""
        if not hasattr(self.args, 'model_id') or not self.args.model_id:
            raise ValueError("Model ID is required")
    
    def execute(self) -> int:
        """Execute model details display."""
        try:
            with self.with_logged_operation("Show Model"):
                persistence = ModelPersistence()
                model_id = self.args.model_id
                
                # Get model metadata
                metadata = persistence.registry.get_model_metadata(model_id)
                
                if not metadata:
                    self.logger.error(f"Model {model_id} not found in registry")
                    return 1
                
                # Display model information
                print("\n" + "="*80)
                print(f"MODEL DETAILS: {model_id}")
                print("="*80)
                
                print(f"\nBasic Information:")
                print(f"  Asset: {metadata['asset']}")
                print(f"  Model Type: {metadata['model_type']}")
                print(f"  Training Date: {metadata['training_date']}")
                print(f"  Registered: {metadata['registered_at']}")
                print(f"  Version: {metadata.get('version', '1.0.0')}")
                print(f"  Feature Count: {metadata['feature_count']}")
                
                # Show file paths
                print(f"\nFile Locations:")
                print(f"  Model File: {metadata['model_path']}")
                if metadata.get('scaler_path'):
                    print(f"  Scaler File: {metadata['scaler_path']}")
                print(f"  Features File: {metadata['feature_path']}")
                
                # Show performance metrics
                if 'metrics' in metadata and metadata['metrics']:
                    print(f"\nPerformance Metrics:")
                    for metric, value in metadata['metrics'].items():
                        if isinstance(value, float):
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: {value}")
                
                # Show training parameters
                if 'training_params' in metadata and metadata['training_params']:
                    print(f"\nTraining Parameters:")
                    for param, value in metadata['training_params'].items():
                        print(f"  {param}: {value}")
                
                # Show feature names if available
                try:
                    feature_path = metadata.get('feature_path')
                    if feature_path:
                        with open(feature_path, 'r') as f:
                            features = json.load(f)
                        
                        print(f"\nFeatures Used ({len(features)}):")
                        for i, feature in enumerate(features, 1):
                            print(f"  {i:2d}. {feature}")
                except Exception as e:
                    self.logger.warning(f"Could not load feature names: {e}")
                
                # Save to file if requested
                if hasattr(self.args, 'output_file') and self.args.output_file:
                    with open(self.args.output_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    self.logger.info(f"Model details saved to: {self.args.output_file}")
                
                return 0
                
        except Exception as e:
            self.logger.error(f"Failed to show model: {e}")
            return 1


class DeleteModelCommand(BaseCommand):
    """Command to delete a trained model."""
    
    def validate_args(self) -> None:
        """Validate command arguments."""
        if not hasattr(self.args, 'model_id') or not self.args.model_id:
            raise ValueError("Model ID is required")
    
    def execute(self) -> int:
        """Execute model deletion."""
        try:
            with self.with_logged_operation("Delete Model"):
                persistence = ModelPersistence()
                model_id = self.args.model_id
                
                # Check if model exists
                metadata = persistence.registry.get_model_metadata(model_id)
                if not metadata:
                    self.logger.error(f"Model {model_id} not found in registry")
                    return 1
                
                # Show model info before deletion
                print(f"\nModel to be deleted:")
                print(f"  ID: {model_id}")
                print(f"  Asset: {metadata['asset']}")
                print(f"  Type: {metadata['model_type']}")
                print(f"  Training Date: {metadata['training_date']}")
                
                # Confirm deletion unless --force is used
                if not getattr(self.args, 'force', False):
                    confirm = input("\nAre you sure you want to delete this model? (yes/no): ")
                    if confirm.lower() not in ['yes', 'y']:
                        self.logger.info("Model deletion cancelled")
                        return 0
                
                # Delete the model
                success = persistence.delete_model(model_id)
                
                if success:
                    self.logger.info(f"Model {model_id} deleted successfully")
                    return 0
                else:
                    self.logger.error(f"Failed to delete model {model_id}")
                    return 1
                
        except Exception as e:
            self.logger.error(f"Failed to delete model: {e}")
            return 1


class CompareModelsCommand(BaseCommand):
    """Command to compare multiple models."""
    
    def validate_args(self) -> None:
        """Validate command arguments."""
        if not hasattr(self.args, 'model_ids') or len(self.args.model_ids) < 2:
            raise ValueError("At least 2 model IDs are required for comparison")
    
    def execute(self) -> int:
        """Execute model comparison."""
        try:
            with self.with_logged_operation("Compare Models"):
                persistence = ModelPersistence()
                model_ids = self.args.model_ids
                
                # Get metadata for all models
                models_data = []
                for model_id in model_ids:
                    metadata = persistence.registry.get_model_metadata(model_id)
                    if not metadata:
                        self.logger.error(f"Model {model_id} not found")
                        return 1
                    models_data.append(metadata)
                
                # Display comparison
                print("\n" + "="*100)
                print("MODEL COMPARISON")
                print("="*100)
                
                # Basic information comparison
                print(f"\nBasic Information:")
                print(f"{'Metric':<20} {'Model 1':<30} {'Model 2':<30}")
                print("-" * 80)
                
                for i, model in enumerate(models_data):
                    model_num = f"Model {i+1}"
                    print(f"{'Model ID':<20} {model['model_id']:<30}")
                    print(f"{'Asset':<20} {model['asset']:<30}")
                    print(f"{'Type':<20} {model['model_type']:<30}")
                    print(f"{'Training Date':<20} {model['training_date']:<30}")
                    print(f"{'Features':<20} {model['feature_count']:<30}")
                    print(f"{'Version':<20} {model.get('version', '1.0.0'):<30}")
                    if i < len(models_data) - 1:
                        print("-" * 80)
                
                # Performance metrics comparison
                if all('metrics' in model and model['metrics'] for model in models_data):
                    print(f"\nPerformance Metrics Comparison:")
                    print(f"{'Metric':<20} {'Model 1':<15} {'Model 2':<15} {'Difference':<15}")
                    print("-" * 65)
                    
                    # Get all unique metrics
                    all_metrics = set()
                    for model in models_data:
                        all_metrics.update(model['metrics'].keys())
                    
                    for metric in sorted(all_metrics):
                        values = []
                        for model in models_data:
                            value = model['metrics'].get(metric, 'N/A')
                            if value != 'N/A' and isinstance(value, (int, float)):
                                values.append(value)
                            else:
                                values.append(None)
                        
                        if len(values) >= 2 and values[0] is not None and values[1] is not None:
                            diff = values[1] - values[0]
                            print(f"{metric:<20} {values[0]:<15.4f} {values[1]:<15.4f} {diff:<15.4f}")
                        else:
                            print(f"{metric:<20} {values[0]:<15} {values[1]:<15} {'N/A':<15}")
                
                # Feature comparison
                try:
                    features_sets = []
                    for model in models_data:
                        feature_path = model.get('feature_path')
                        if feature_path:
                            with open(feature_path, 'r') as f:
                                features = json.load(f)
                            features_sets.append(set(features))
                        else:
                            features_sets.append(set())
                    
                    if len(features_sets) >= 2:
                        common_features = features_sets[0].intersection(features_sets[1])
                        unique_1 = features_sets[0] - features_sets[1]
                        unique_2 = features_sets[1] - features_sets[0]
                        
                        print(f"\nFeature Comparison:")
                        print(f"  Common features: {len(common_features)}")
                        print(f"  Unique to Model 1: {len(unique_1)}")
                        print(f"  Unique to Model 2: {len(unique_2)}")
                        
                        if unique_1:
                            print(f"  Model 1 unique: {list(unique_1)}")
                        if unique_2:
                            print(f"  Model 2 unique: {list(unique_2)}")
                
                except Exception as e:
                    self.logger.warning(f"Could not compare features: {e}")
                
                # Save comparison to file if requested
                if hasattr(self.args, 'output_file') and self.args.output_file:
                    comparison_data = {
                        'model_ids': model_ids,
                        'models': models_data,
                        'comparison_timestamp': pd.Timestamp.now().isoformat()
                    }
                    
                    with open(self.args.output_file, 'w') as f:
                        json.dump(comparison_data, f, indent=2)
                    
                    self.logger.info(f"Model comparison saved to: {self.args.output_file}")
                
                return 0
                
        except Exception as e:
            self.logger.error(f"Failed to compare models: {e}")
            return 1


class ModelRegistryCommand(BaseCommand):
    """Command to manage the model registry."""
    
    def execute(self) -> int:
        """Execute registry management."""
        try:
            with self.with_logged_operation("Model Registry"):
                persistence = ModelPersistence()
                
                action = getattr(self.args, 'action', 'status')
                
                if action == 'status':
                    # Show registry status
                    models = persistence.list_models()
                    
                    print("\n" + "="*60)
                    print("MODEL REGISTRY STATUS")
                    print("="*60)
                    
                    print(f"Total Models: {len(models)}")
                    
                    if models:
                        # Group by asset
                        assets = {}
                        model_types = {}
                        
                        for model in models:
                            asset = model['asset']
                            model_type = model['model_type']
                            
                            if asset not in assets:
                                assets[asset] = 0
                            assets[asset] += 1
                            
                            if model_type not in model_types:
                                model_types[model_type] = 0
                            model_types[model_type] += 1
                        
                        print(f"\nModels by Asset:")
                        for asset, count in sorted(assets.items()):
                            print(f"  {asset}: {count}")
                        
                        print(f"\nModels by Type:")
                        for model_type, count in sorted(model_types.items()):
                            print(f"  {model_type}: {count}")
                        
                        # Show latest models
                        latest_models = sorted(models, key=lambda x: x['training_date'], reverse=True)[:5]
                        print(f"\nLatest 5 Models:")
                        for model in latest_models:
                            print(f"  {model['model_id']}: {model['asset']} ({model['model_type']}) - {model['training_date']}")
                
                elif action == 'cleanup':
                    # Clean up orphaned models (models in registry but files missing)
                    models = persistence.list_models()
                    orphaned = []
                    
                    for model in models:
                        model_path = model.get('model_path')
                        if model_path and not Path(model_path).exists():
                            orphaned.append(model)
                    
                    if orphaned:
                        print(f"\nFound {len(orphaned)} orphaned models:")
                        for model in orphaned:
                            print(f"  {model['model_id']}")
                        
                        if getattr(self.args, 'remove_orphaned', False):
                            for model in orphaned:
                                persistence.registry.delete_model(model['model_id'])
                            print(f"Removed {len(orphaned)} orphaned models from registry")
                    else:
                        print("\nNo orphaned models found")
                
                return 0
                
        except Exception as e:
            self.logger.error(f"Registry management failed: {e}")
            return 1
