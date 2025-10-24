"""
Model Persistence Module

This module provides utilities for saving and loading trained models, including:
1. Model versioning and metadata tracking
2. Serialization and deserialization of models and scalers
3. Model registry management
4. Model artifact organization

The module ensures that models can be properly saved during training and loaded
for inference in production environments.
"""

import os
import pickle
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import hashlib

import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default locations
DEFAULT_MODELS_DIR = "results/models"
DEFAULT_REGISTRY_FILE = "results/models/registry.json"

class ModelRegistry:
    """Manages the model registry for tracking trained models."""
    
    def __init__(self, registry_file: str = DEFAULT_REGISTRY_FILE):
        """
        Initialize the model registry.
        
        Args:
            registry_file: Path to the registry JSON file
        """
        self.registry_file = registry_file
        self.registry_path = Path(registry_file)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """
        Load the model registry from disk.
        
        Returns:
            Dict[str, Any]: The model registry data
        """
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model registry: {e}. Creating a new one.")
        
        # Create a new registry if it doesn't exist or can't be loaded
        return {
            "models": {},
            "last_updated": datetime.now().isoformat(),
            "version": "1.0"
        }
    
    def _save_registry(self):
        """Save the model registry to disk."""
        self.registry["last_updated"] = datetime.now().isoformat()
        
        try:
            # Ensure the directory exists
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a backup of the existing registry
            if self.registry_path.exists():
                backup_path = self.registry_path.with_suffix('.backup')
                shutil.copy2(self.registry_path, backup_path)
            
            # Save the registry
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
            
            logger.info(f"Successfully saved model registry to {self.registry_path}")
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
            raise  # Re-raise the exception so the caller knows it failed
    
    def register_model(self, model_id: str, metadata: Dict[str, Any]) -> str:
        """
        Register a model in the registry.
        
        Args:
            model_id: Unique identifier for the model
            metadata: Model metadata
            
        Returns:
            str: The registered model ID
        """
        # Ensure metadata has required fields
        required_fields = ["asset", "model_type", "training_date", "feature_count"]
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing required metadata field: {field}")
        
        # Add additional metadata
        metadata["registered_at"] = datetime.now().isoformat()
        metadata["version"] = metadata.get("version", "1.0.0")
        
        # Add to registry
        self.registry["models"][model_id] = metadata
        
        try:
            self._save_registry()
            logger.info(f"Registered model {model_id} in registry")
            return model_id
        except Exception as e:
            # Remove from registry if save failed
            if model_id in self.registry["models"]:
                del self.registry["models"][model_id]
            logger.error(f"Failed to save registry after registering model {model_id}: {e}")
            raise
    
    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a registered model.
        
        Args:
            model_id: The model ID to look up
            
        Returns:
            Optional[Dict[str, Any]]: The model metadata or None if not found
        """
        return self.registry["models"].get(model_id)
    
    def list_models(self, asset: Optional[str] = None, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List registered models, optionally filtered by asset and model type.
        
        Args:
            asset: Filter by asset name
            model_type: Filter by model type
            
        Returns:
            List[Dict[str, Any]]: List of model entries with metadata
        """
        models = []
        
        for model_id, metadata in self.registry["models"].items():
            if asset and metadata.get("asset") != asset:
                continue
                
            if model_type and metadata.get("model_type") != model_type:
                continue
                
            models.append({
                "model_id": model_id,
                **metadata
            })
        
        # Sort by training date (newest first)
        models.sort(key=lambda x: x.get("training_date", ""), reverse=True)
        
        return models
    
    def get_latest_model(self, asset: str, model_type: str) -> Optional[str]:
        """
        Get the latest model ID for a specific asset and model type.
        
        Args:
            asset: Asset name
            model_type: Model type
            
        Returns:
            Optional[str]: The latest model ID or None if not found
        """
        models = self.list_models(asset=asset, model_type=model_type)
        
        if not models:
            return None
            
        return models[0]["model_id"]
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id: The model ID to delete
            
        Returns:
            bool: True if the model was deleted, False otherwise
        """
        if model_id in self.registry["models"]:
            del self.registry["models"][model_id]
            self._save_registry()
            logger.info(f"Deleted model {model_id} from registry")
            return True
            
        return False

class ModelPersistence:
    """Handles saving and loading of trained models."""
    
    def __init__(self, models_dir: str = DEFAULT_MODELS_DIR):
        """
        Initialize the model persistence manager.
        
        Args:
            models_dir: Directory to store model files
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry()
    
    def _generate_model_id(self, asset: str, model_type: str, features: List[str]) -> str:
        """
        Generate a unique model ID based on asset, model type, and features.
        
        Args:
            asset: Asset name
            model_type: Model type
            features: List of feature names
            
        Returns:
            str: A unique model ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feature_hash = hashlib.md5(",".join(sorted(features)).encode()).hexdigest()[:8]
        return f"{asset}_{model_type}_{timestamp}_{feature_hash}"
    
    def save_model(self, 
                  model: Any, 
                  scaler: Any, 
                  asset: str, 
                  model_type: str,
                  feature_names: List[str],
                  performance_metrics: Dict[str, float],
                  training_params: Dict[str, Any] = None,
                  version: str = "1.0.0") -> str:
        """
        Save a trained model with its associated artifacts.
        
        Args:
            model: The trained model object
            scaler: The feature scaler object
            asset: Asset name (e.g., "EURUSD")
            model_type: Model type (e.g., "logistic", "xgboost")
            feature_names: List of feature names used for training
            performance_metrics: Dictionary of performance metrics
            training_params: Dictionary of training parameters
            version: Model version string
            
        Returns:
            str: The unique model ID
        """
        # Generate a unique model ID
        model_id = self._generate_model_id(asset, model_type, feature_names)
        
        # Create model directory
        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler if provided
        if scaler is not None:
            scaler_path = model_dir / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save feature names
        feature_path = model_dir / "features.json"
        with open(feature_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        # Save performance metrics
        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(performance_metrics, f, indent=2)
        
        # Prepare metadata
        metadata = {
            "asset": asset,
            "model_type": model_type,
            "training_date": datetime.now().isoformat(),
            "feature_count": len(feature_names),
            "metrics": performance_metrics,
            "version": version,
            "training_params": training_params or {},
            "model_path": str(model_path),
            "scaler_path": str(scaler_path) if scaler is not None else None,
            "feature_path": str(feature_path)
        }
        
        # Register model
        self.registry.register_model(model_id, metadata)
        
        logger.info(f"Saved model {model_id} for {asset} ({model_type})")
        return model_id
    
    def load_model(self, model_id: str) -> Tuple[Any, Any, List[str], Dict[str, Any]]:
        """
        Load a model and its associated artifacts by ID.
        
        Args:
            model_id: The model ID to load
            
        Returns:
            Tuple[Any, Any, List[str], Dict[str, Any]]: 
                Tuple of (model, scaler, feature_names, metadata)
        """
        # Get model metadata
        metadata = self.registry.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found in registry")
        
        # Load model
        model_path = Path(metadata["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler if available
        scaler = None
        if metadata.get("scaler_path"):
            scaler_path = Path(metadata["scaler_path"])
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
        
        # Load feature names
        feature_path = Path(metadata["feature_path"])
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
            
        with open(feature_path, 'r') as f:
            feature_names = json.load(f)
        
        logger.info(f"Loaded model {model_id} ({metadata['model_type']})")
        return model, scaler, feature_names, metadata
    
    def load_latest_model(self, asset: str, model_type: str) -> Optional[Tuple[Any, Any, List[str], Dict[str, Any]]]:
        """
        Load the latest model for a specific asset and model type.
        
        Args:
            asset: Asset name
            model_type: Model type
            
        Returns:
            Optional[Tuple[Any, Any, List[str], Dict[str, Any]]]: 
                Tuple of (model, scaler, feature_names, metadata) or None if not found
        """
        model_id = self.registry.get_latest_model(asset, model_type)
        
        if not model_id:
            logger.warning(f"No {model_type} model found for {asset}")
            return None
            
        return self.load_model(model_id)
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model and its associated artifacts.
        
        Args:
            model_id: The model ID to delete
            
        Returns:
            bool: True if the model was deleted, False otherwise
        """
        # Get model metadata
        metadata = self.registry.get_model_metadata(model_id)
        if not metadata:
            logger.warning(f"Model {model_id} not found in registry")
            return False
        
        # Delete model directory
        model_dir = Path(metadata["model_path"]).parent
        if model_dir.exists():
            try:
                shutil.rmtree(model_dir)
                logger.info(f"Deleted model directory: {model_dir}")
            except Exception as e:
                logger.error(f"Failed to delete model directory: {e}")
                return False
        
        # Remove from registry
        return self.registry.delete_model(model_id)
    
    def list_models(self, asset: Optional[str] = None, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available models, optionally filtered by asset and model type.
        
        Args:
            asset: Filter by asset name
            model_type: Filter by model type
            
        Returns:
            List[Dict[str, Any]]: List of model entries with metadata
        """
        return self.registry.list_models(asset=asset, model_type=model_type)
    
    def get_model_performance(self, model_id: str) -> Dict[str, float]:
        """
        Get performance metrics for a model.
        
        Args:
            model_id: The model ID
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        metadata = self.registry.get_model_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not found in registry")
            
        return metadata.get("metrics", {})

def save_model(model: Any, 
               scaler: Any, 
               asset: str, 
               model_type: str,
               feature_names: List[str],
               performance_metrics: Dict[str, float],
               training_params: Dict[str, Any] = None,
               version: str = "1.0.0") -> str:
    """
    Convenience function to save a model.
    
    Args:
        model: The trained model object
        scaler: The feature scaler object
        asset: Asset name (e.g., "EURUSD")
        model_type: Model type (e.g., "logistic", "xgboost")
        feature_names: List of feature names used for training
        performance_metrics: Dictionary of performance metrics
        training_params: Dictionary of training parameters
        version: Model version string
        
    Returns:
        str: The unique model ID
    """
    persistence = ModelPersistence()
    return persistence.save_model(
        model=model,
        scaler=scaler,
        asset=asset,
        model_type=model_type,
        feature_names=feature_names,
        performance_metrics=performance_metrics,
        training_params=training_params,
        version=version
    )

def load_model(model_id: str) -> Tuple[Any, Any, List[str], Dict[str, Any]]:
    """
    Convenience function to load a model by ID.
    
    Args:
        model_id: The model ID to load
        
    Returns:
        Tuple[Any, Any, List[str], Dict[str, Any]]: 
            Tuple of (model, scaler, feature_names, metadata)
    """
    persistence = ModelPersistence()
    return persistence.load_model(model_id)

def load_latest_model(asset: str, model_type: str) -> Optional[Tuple[Any, Any, List[str], Dict[str, Any]]]:
    """
    Convenience function to load the latest model for an asset and model type.
    
    Args:
        asset: Asset name
        model_type: Model type
        
    Returns:
        Optional[Tuple[Any, Any, List[str], Dict[str, Any]]]: 
            Tuple of (model, scaler, feature_names, metadata) or None if not found
    """
    persistence = ModelPersistence()
    return persistence.load_latest_model(asset, model_type)

def list_models(asset: Optional[str] = None, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to list available models.
    
    Args:
        asset: Filter by asset name
        model_type: Filter by model type
        
    Returns:
        List[Dict[str, Any]]: List of model entries with metadata
    """
    persistence = ModelPersistence()
    return persistence.list_models(asset=asset, model_type=model_type)

def main():
    """Test the model persistence functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Persistence Utility")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List models command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--asset", help="Filter by asset")
    list_parser.add_argument("--model-type", help="Filter by model type")
    
    # Show model command
    show_parser = subparsers.add_parser("show", help="Show model details")
    show_parser.add_argument("model_id", help="Model ID to show")
    
    # Delete model command
    delete_parser = subparsers.add_parser("delete", help="Delete a model")
    delete_parser.add_argument("model_id", help="Model ID to delete")
    
    args = parser.parse_args()
    
    persistence = ModelPersistence()
    
    if args.command == "list":
        models = persistence.list_models(asset=args.asset, model_type=args.model_type)
        
        if not models:
            print("No models found.")
            return
            
        print(f"Found {len(models)} models:")
        for model in models:
            print(f"  - {model['model_id']}: {model['asset']} ({model['model_type']}) - {model['training_date']}")
    
    elif args.command == "show":
        try:
            metadata = persistence.registry.get_model_metadata(args.model_id)
            
            if not metadata:
                print(f"Model {args.model_id} not found.")
                return
                
            print(f"Model: {args.model_id}")
            print(f"  Asset: {metadata['asset']}")
            print(f"  Type: {metadata['model_type']}")
            print(f"  Training Date: {metadata['training_date']}")
            print(f"  Feature Count: {metadata['feature_count']}")
            print(f"  Version: {metadata['version']}")
            print("  Metrics:")
            
            for metric, value in metadata.get("metrics", {}).items():
                print(f"    {metric}: {value}")
        except Exception as e:
            print(f"Error showing model: {e}")
    
    elif args.command == "delete":
        try:
            success = persistence.delete_model(args.model_id)
            
            if success:
                print(f"Model {args.model_id} deleted successfully.")
            else:
                print(f"Failed to delete model {args.model_id}.")
        except Exception as e:
            print(f"Error deleting model: {e}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

