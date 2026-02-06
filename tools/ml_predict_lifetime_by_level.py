#!/usr/bin/env python3
"""
ML Model for predicting file lifetime by level
Supports different models for different levels
This module will be called via pybind11 from C++
"""

import numpy as np
import pickle
import os
import json
from pathlib import Path
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None

class FileLifetimePredictorByLevel:
    """Predictor class that loads models by level and provides fast prediction"""
    
    def __init__(self, models_base_path=None):
        """
        Initialize predictor and load models for all levels
        
        Args:
            models_base_path: Base path to model directory
                           If None, uses default path from environment variable
        """
        if models_base_path is None:
            models_base_path = os.getenv('ROCKSDB_ML_MODELS_PATH', 
                                        '/home/usr/test_environment/workplace/models')
        
        self.models_base_path = Path(models_base_path)
        self.models = {}  # level -> model dict
        self.scalers = {}  # level -> scaler
        self.configs = {}  # level -> config
        
        # Load models for all levels
        self._load_all_models()
    
    def _load_all_models(self):
        """Load models for all levels"""
        # Try simple format first (model_level_{level}_lgb.txt + scaler_level_{level}.pkl)
        # This is the format used by train_models_by_level.py
        for level in range(1, 7):
            model_loaded = False
            
            # Try simple format: model_level_{level}_lgb.txt
            model_file = self.models_base_path / f"model_level_{level}_lgb.txt"
            scaler_file = self.models_base_path / f"scaler_level_{level}.pkl"
            metrics_file = self.models_base_path / "metrics_lgb.json"
            
            if model_file.exists() and scaler_file.exists():
                try:
                    # Load LGB model
                    if lgb is not None:
                        with open(model_file, 'r') as f:
                            model = lgb.Booster(model_str=f.read())
                    else:
                        print(f"Warning: LightGBM not available, skipping Level {level}")
                        continue
                    
                    # Load scaler (simple format: just StandardScaler)
                    with open(scaler_file, 'rb') as f:
                        scaler = pickle.load(f)
                    
                    # Load metrics if available
                    test_r2 = 0.0
                    if metrics_file.exists():
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics_data = json.load(f)
                            level_metrics = next((m for m in metrics_data if m.get('level') == level), None)
                            if level_metrics:
                                test_r2 = level_metrics.get('test', {}).get('r2', 0.0)
                        except Exception:
                            pass
                    
                    # Create simple config
                    config = {
                        'model_type': 'lgb',
                        'test_r2': test_r2,
                        'preprocessing': 'standard',
                        'use_log_transform': False,
                        'selected_features': None
                    }
                    
                    self.models[level] = model
                    self.scalers[level] = scaler
                    self.configs[level] = config
                    
                    print(f"Loaded model for Level {level}: LGB "
                          f"(test R²={test_r2:.4f})")
                    model_loaded = True
                except Exception as e:
                    print(f"Warning: Failed to load simple model for Level {level}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Fallback to old format (level_{level}_best_model.pkl)
            if not model_loaded:
                model_file = self.models_base_path / f"level_{level}_best_model.pkl"
                scaler_file = self.models_base_path / f"level_{level}_best_scaler.pkl"
                features_file = self.models_base_path / f"level_{level}_best_features.pkl"
                config_file = self.models_base_path / f"level_{level}_best_config.json"
                
                if model_file.exists() and config_file.exists():
                    try:
                        # Load model
                        with open(model_file, 'rb') as f:
                            model_data = pickle.load(f)
                        
                        # Extract model (scaler is in separate file)
                        if isinstance(model_data, dict):
                            model = model_data.get('model')
                        else:
                            model = model_data
                        
                        # Load scaler from separate file
                        scaler = None
                        if scaler_file.exists():
                            with open(scaler_file, 'rb') as f:
                                scaler = pickle.load(f)
                        
                        # Load selected features from separate file (if exists)
                        selected_features = None
                        if features_file.exists():
                            with open(features_file, 'rb') as f:
                                selected_features = pickle.load(f)
                        
                        # Load config
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        
                        # Update config with selected_features if not already present
                        if selected_features is not None and 'selected_features' not in config:
                            config['selected_features'] = selected_features
                        
                        self.models[level] = model
                        self.scalers[level] = scaler
                        self.configs[level] = config
                        
                        print(f"Loaded model for Level {level}: {config.get('model_type', 'unknown')} "
                              f"(test R²={config.get('test_r2', 0):.4f})")
                    except Exception as e:
                        print(f"Warning: Failed to load model for Level {level}: {e}")
                        import traceback
                        traceback.print_exc()
                        self.models[level] = None
                        self.scalers[level] = None
                        self.configs[level] = None
    
    def _apply_preprocessing(self, features, preprocessing, scaler):
        """Apply preprocessing to features"""
        # If scaler is None, skip preprocessing (raw features)
        if scaler is None:
            return features
        
        if preprocessing == 'standard' or preprocessing == 'none':
            if scaler:
                return scaler.transform(features.reshape(1, -1))[0]
            return features
        elif preprocessing == 'log_all':
            features_log = np.log1p(features - features.min() + 1)
            if scaler:
                return scaler.transform(features_log.reshape(1, -1))[0]
            return features_log
        elif preprocessing == 'selective_log':
            # Selective log transformation (same as training)
            # Training uses: global_min = min(X_train[:, feat_idx].min(), X_test[:, feat_idx].min())
            # Then: shift = -global_min + 1, and log1p(X + shift)
            # For prediction, we use shift = 1 (assuming features are already non-negative)
            # This matches training when global_min >= 0
            large_range_features = [49, 44, 39, 8, 34, 29, 24, 51, 50, 55, 6, 10, 54, 57, 12, 11, 58, 59, 5, 56, 13]
            features_processed = features.copy()
            for feat_idx in large_range_features:
                if feat_idx < len(features):
                    # Training: shift = -global_min + 1, where global_min = min(train_min, test_min)
                    # For prediction, we use shift = 1 (equivalent to global_min = 0)
                    # This is correct because scaler was trained on log-transformed data
                    shift = 1
                    features_processed[feat_idx] = np.log1p(features[feat_idx] + shift)
            if scaler:
                return scaler.transform(features_processed.reshape(1, -1))[0]
            return features_processed
        else:
            return features
    
    def _apply_feature_selection(self, features, selected_features):
        """Apply feature selection"""
        if selected_features is None:
            return features
        
        # If selected_features is a selector object (sklearn SelectKBest), use transform
        if hasattr(selected_features, 'transform'):
            return selected_features.transform(features.reshape(1, -1))[0]
        
        # If selected_features is a list/array of indices, use indexing
        if isinstance(selected_features, (list, np.ndarray)) and len(selected_features) > 0:
            return features[selected_features]
        
        return features
    
    def _add_feature_interactions(self, features):
        """Add feature interactions (for Level 2) - matches training code"""
        n_features = len(features)
        # Select top 10 features for interaction (same as training)
        top_features = min(10, n_features)
        interactions = []
        
        for i in range(top_features):
            for j in range(i+1, top_features):
                interactions.append(features[i] * features[j])
        
        if interactions:
            # Stack interactions and limit to 20 (same as training)
            interactions_array = np.column_stack(interactions) if len(interactions) > 1 else np.array(interactions).reshape(-1, 1)
            interactions_array = interactions_array[:, :min(20, interactions_array.shape[1])] if interactions_array.ndim > 1 else interactions_array[:20]
            if interactions_array.ndim == 1:
                interactions_array = interactions_array.reshape(-1, 1)
            return np.hstack([features.reshape(1, -1), interactions_array])[0] if features.ndim == 1 else np.hstack([features, interactions_array])
        return features
    
    def predict(self, features, level):
        """
        Predict file lifetime from features for a specific level
        
        Args:
            features: numpy array of 69 features
            level: target level (1-6)
            
        Returns:
            predicted lifetime in seconds (float)
        """
        if level not in self.models or self.models[level] is None:
            # Fallback: return default prediction
            return self._fallback_predict(features, level)
        
        try:
            config = self.configs[level]
            model = self.models[level]
            scaler = self.scalers[level]
            
            # Convert to numpy array if needed
            features = np.array(features, dtype=np.float64)
            
            # Apply feature selection BEFORE preprocessing (matching training order)
            # In training: feature selection -> scaling -> model
            selected_features = config.get('selected_features')
            if selected_features:
                features = self._apply_feature_selection(features, selected_features)
            
            # Apply preprocessing (scaling)
            # If scaler is None, skip preprocessing (use raw features)
            preprocessing_method = config.get('preprocessing', 'standard')
            if scaler is None:
                preprocessing_method = 'none'  # Skip preprocessing
            features_proc = self._apply_preprocessing(features, preprocessing_method, scaler)
            
            # Add feature interactions if needed
            if config.get('use_feature_interactions'):
                features_proc = self._add_feature_interactions(features_proc)
            
            # Reshape for model prediction
            if features_proc.ndim == 1:
                features_proc = features_proc.reshape(1, -1)
            
            # Predict (XGBoost needs DMatrix, LightGBM can use numpy array)
            model_type = config.get('model_type', 'lgb')
            if model_type == 'xgb' and xgb is not None:
                dmatrix = xgb.DMatrix(features_proc)
                prediction = model.predict(dmatrix)[0]
            else:
                prediction = model.predict(features_proc)[0]
            
            # If use_log_transform is True, convert back from log space
            if config.get('use_log_transform', False):
                prediction = np.expm1(prediction)  # exp(x) - 1, inverse of log1p
            
            # Ensure non-negative and convert to seconds
            lifetime_seconds = max(0.0, float(prediction))
            
            return lifetime_seconds
        except Exception as e:
            print(f"Warning: Prediction failed for Level {level}: {e}")
            return self._fallback_predict(features, level)
    
    def _fallback_predict(self, features, level):
        """
        Fallback prediction when model is not available
        Uses simple heuristics based on level
        """
        # Simple heuristic: higher level -> longer lifetime
        base_lifetime = (level + 1) * 50.0
        return base_lifetime


# Global predictor instance (loaded once)
_predictor_instance = None

def get_predictor():
    """Get or create global predictor instance"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FileLifetimePredictorByLevel()
    return _predictor_instance

def initialize():
    """
    Initialize the predictor (called from C++)
    Returns True if initialization successful, False otherwise
    """
    try:
        predictor = get_predictor()
        # Check if at least one model is loaded
        if predictor and len(predictor.models) > 0:
            return True
        return False
    except Exception as e:
        print(f"Warning: Predictor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def predict_file_lifetime_by_level(features, level):
    """
    Predict file lifetime (called from C++)
    
    Args:
        features: numpy array of 69 features
        level: target level (1-6)
    
    Returns:
        predicted lifetime in seconds
    """
    predictor = get_predictor()
    return predictor.predict(features, level)


# For testing/standalone use
if __name__ == '__main__':
    # Test the predictor
    predictor = FileLifetimePredictorByLevel()
    
    # Create dummy features (69 features)
    dummy_features = np.random.rand(69)
    
    # Test prediction for each level
    for level in range(1, 7):
        lifetime = predictor.predict(dummy_features, level)
        print(f"Level {level}: Predicted lifetime = {lifetime:.2f} seconds")

