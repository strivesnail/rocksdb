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
    
    def __init__(self, models_base_path=None, lazy_load=True):
        """
        Initialize predictor and load models for all levels
        
        Args:
            models_base_path: Base path to model directory
                           If None, uses default path from environment variable
            lazy_load: If True, only load models when needed (for process pool workers)
        """
        if models_base_path is None:
            models_base_path = os.getenv('ROCKSDB_ML_MODELS_PATH', 
                                        '/home/usr/test_environment/workplace/models')
        
        self.models_base_path = Path(models_base_path)
        self.models = {}  # level -> model dict
        self.scalers = {}  # level -> scaler
        self.configs = {}  # level -> config
        self.lazy_load = lazy_load
        
        # Load models for all levels (unless lazy_load is True)
        if not lazy_load:
            self._load_all_models()
    
    def _load_all_models(self):
        """Load models for all levels"""
        # Try simple format first (model_level_{level}_lgb.txt + scaler_level_{level}.pkl)
        # This is the format used by train_models_by_level.py
        for level in range(1, 7):
            model_loaded = False
            
            # Try simple format: model_level_{level}_lgb.txt or model_level_{level}_xgb.json
            lgb_model_file = self.models_base_path / f"model_level_{level}_lgb.txt"
            xgb_model_file = self.models_base_path / f"model_level_{level}_xgb.json"
            scaler_file = self.models_base_path / f"scaler_level_{level}.pkl"
            metrics_file = self.models_base_path / "metrics_lgb.json"
            
            # Check for LightGBM format first
            if lgb_model_file.exists() and scaler_file.exists():
                model_file = lgb_model_file
                model_type = 'lgb'
            # Check for XGBoost format
            elif xgb_model_file.exists() and scaler_file.exists():
                model_file = xgb_model_file
                model_type = 'xgb'
            else:
                model_file = None
                model_type = None
            
            if model_file and model_file.exists() and scaler_file.exists():
                try:
                    # Load model (LightGBM or XGBoost)
                    if model_type == 'lgb':
                        if lgb is not None:
                            with open(model_file, 'r') as f:
                                model = lgb.Booster(model_str=f.read())
                        else:
                            raise RuntimeError(f"LightGBM not available, cannot load model for Level {level}")
                    elif model_type == 'xgb':
                        if xgb is not None:
                            model = xgb.Booster()
                            model.load_model(str(model_file))
                        else:
                            raise RuntimeError(f"XGBoost not available, cannot load model for Level {level}")
                    else:
                        raise RuntimeError(f"Unknown model type for Level {level}")
                    
                    # Load scaler (simple format: just StandardScaler)
                    with open(scaler_file, 'rb') as f:
                        scaler_data = pickle.load(f)
                        # Handle case where pickle contains a tuple (scaler, ...) or just scaler
                        if isinstance(scaler_data, tuple):
                            # If it's a tuple, try to find a valid scaler in all elements
                            scaler = None
                            for idx, item in enumerate(scaler_data):
                                if hasattr(item, 'transform') and hasattr(item, 'fit'):
                                    scaler = item
                                    print(f"Found scaler at tuple index {idx} for Level {level} (simple format)")
                                    break
                            
                            if scaler is None:
                                # Log all tuple elements for debugging
                                print(f"ERROR: Scaler file for Level {level} contains tuple with {len(scaler_data)} elements:")
                                for idx, item in enumerate(scaler_data):
                                    print(f"  Element {idx}: type={type(item)}, has_transform={hasattr(item, 'transform') if hasattr(item, '__class__') else False}, has_fit={hasattr(item, 'fit') if hasattr(item, '__class__') else False}")
                                raise ValueError(f"Scaler file for Level {level} contains tuple but no valid scaler found in any element")
                        elif hasattr(scaler_data, 'transform') and hasattr(scaler_data, 'fit'):
                            scaler = scaler_data
                        else:
                            raise ValueError(f"Scaler file for Level {level} does not contain a valid scaler object (type: {type(scaler_data)})")
                    
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
                        'model_type': model_type,
                        'test_r2': test_r2,
                        'preprocessing': 'standard',
                        'use_log_transform': False,
                        'selected_features': None
                    }
                    
                    self.models[level] = model
                    self.scalers[level] = scaler
                    self.configs[level] = config
                    
                    print(f"Loaded model for Level {level}: {model_type.upper()} "
                          f"(test R²={test_r2:.4f})")
                    model_loaded = True
                except Exception as e:
                    # Expose the problem directly - don't hide it
                    error_msg = f"Failed to load simple model for Level {level}: {e}"
                    print(f"ERROR: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(error_msg) from e
            
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
                                scaler_data = pickle.load(f)
                                # Handle case where pickle contains a tuple (model, scaler) or just scaler
                                if isinstance(scaler_data, tuple):
                                    # If it's a tuple, try to find a valid scaler in all elements
                                    scaler = None
                                    for idx, item in enumerate(scaler_data):
                                        if hasattr(item, 'transform') and hasattr(item, 'fit'):
                                            scaler = item
                                            print(f"Found scaler at tuple index {idx} for Level {level}")
                                            break
                                    
                                    if scaler is None:
                                        # Log all tuple elements for debugging
                                        print(f"ERROR: Scaler file for Level {level} contains tuple with {len(scaler_data)} elements:")
                                        for idx, item in enumerate(scaler_data):
                                            print(f"  Element {idx}: type={type(item)}, has_transform={hasattr(item, 'transform') if hasattr(item, '__class__') else False}, has_fit={hasattr(item, 'fit') if hasattr(item, '__class__') else False}")
                                        raise ValueError(f"Scaler file for Level {level} contains tuple but no valid scaler found in any element")
                                elif hasattr(scaler_data, 'transform') and hasattr(scaler_data, 'fit'):
                                    scaler = scaler_data
                                else:
                                    raise ValueError(f"Scaler file for Level {level} does not contain a valid scaler object (type: {type(scaler_data)})")
                        
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
                        # Expose the problem directly - don't hide it
                        error_msg = f"Failed to load model for Level {level}: {e}"
                        print(f"ERROR: {error_msg}")
                        import traceback
                        traceback.print_exc()
                        raise RuntimeError(error_msg) from e
    
    def _apply_preprocessing(self, features, preprocessing, scaler):
        """Apply preprocessing to features"""
        # If scaler is None, skip preprocessing (raw features)
        if scaler is None:
            return features
        
        # Scaler must have transform method - validate before use
        if not hasattr(scaler, 'transform'):
            raise ValueError(f"Scaler does not have transform method (type: {type(scaler)}). This should not happen if loading was correct.")
        
        # Additional safety: check if transform is callable
        if not callable(getattr(scaler, 'transform', None)):
            raise ValueError(f"Scaler.transform is not callable (type: {type(scaler)}). This should not happen if loading was correct.")
        
        if preprocessing == 'standard' or preprocessing == 'none':
            return scaler.transform(features.reshape(1, -1))[0]
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
    
    def _load_model_for_level(self, level):
        """Load model for a specific level (lazy loading)"""
        if level in self.models and self.models[level] is not None:
            return  # Already loaded
        
        # Load single level model
        model_loaded = False
        # Try simple format first
        lgb_model_file = self.models_base_path / f"model_level_{level}_lgb.txt"
        xgb_model_file = self.models_base_path / f"model_level_{level}_xgb.json"
        scaler_file = self.models_base_path / f"scaler_level_{level}.pkl"
        
        model_type = None
        model = None
        
        if lgb_model_file.exists():
            model_type = 'lgb'
            if lgb is not None:
                with open(lgb_model_file, 'r') as f:
                    model = lgb.Booster(model_str=f.read())
            else:
                raise RuntimeError(f"LightGBM not available, cannot load model for Level {level}")
        elif xgb_model_file.exists():
            model_type = 'xgb'
            if xgb is not None:
                model = xgb.Booster()
                model.load_model(str(xgb_model_file))
            else:
                raise RuntimeError(f"XGBoost not available, cannot load model for Level {level}")
        
        if model_type:
            # Load scaler
            scaler = None
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    scaler_data = pickle.load(f)
                    if isinstance(scaler_data, tuple):
                        for item in scaler_data:
                            if hasattr(item, 'transform') and hasattr(item, 'fit'):
                                scaler = item
                                break
                        if scaler is None:
                            raise ValueError(f"Scaler file for Level {level} contains tuple but no valid scaler found")
                    elif hasattr(scaler_data, 'transform') and hasattr(scaler_data, 'fit'):
                        scaler = scaler_data
                    else:
                        raise ValueError(f"Scaler file for Level {level} does not contain a valid scaler object")
            
            # Load config (use default if not found)
            config = {
                'model_type': model_type,
                'preprocessing': 'standard' if scaler else 'none',
                'use_log_transform': False,
                'use_feature_interactions': False,
                'selected_features': None
            }
            
            self.models[level] = model
            self.scalers[level] = scaler
            self.configs[level] = config
            # 延迟加载日志已移除
            model_loaded = True
        
        if not model_loaded:
            raise RuntimeError(f"Failed to load model for Level {level}")
    
    def predict(self, features, level):
        """
        Predict file lifetime from features for a specific level
        
        Args:
            features: numpy array of 69 features
            level: target level (1-6)
            
        Returns:
            predicted lifetime in seconds (float)
            
        Raises:
            ValueError: if model is not loaded or prediction fails
        """
        # Lazy load model if needed
        if self.lazy_load and (level not in self.models or self.models[level] is None):
            self._load_model_for_level(level)
        
        if level not in self.models or self.models[level] is None:
            raise ValueError(f"Model for Level {level} is not loaded. Available levels: {list(self.models.keys())}")
        
        config = self.configs[level]
        model = self.models[level]
        scaler = self.scalers.get(level)  # Use .get() to handle missing scaler
        
        # Convert to numpy array if needed
        features = np.array(features, dtype=np.float64)
        
        # Apply feature selection BEFORE preprocessing (matching training order)
        # In training: feature selection -> scaling -> model
        selected_features = config.get('selected_features')
        if selected_features:
            features = self._apply_feature_selection(features, selected_features)
        
        # Apply preprocessing (scaling)
        # If scaler is None or not in dict, skip preprocessing (use raw features)
        preprocessing_method = config.get('preprocessing', 'standard')
        if scaler is None:
            preprocessing_method = 'none'  # Skip preprocessing
            print(f"WARNING: Level {level} has no scaler, skipping preprocessing")
        # Scaler validation should have been done during loading, so we don't check here
        # But add extra safety check
        if scaler is not None and not hasattr(scaler, 'transform'):
            raise ValueError(f"Level {level} scaler is not None but has no transform method (type: {type(scaler)})")
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
    
    # Removed _fallback_predict - no fallback, expose problems directly


# Global predictor instance (loaded once)
_predictor_instance = None
_process_pool = None
_use_process_pool = False

def get_predictor():
    """Get or create global predictor instance"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = FileLifetimePredictorByLevel()
    return _predictor_instance

def _init_worker():
    """Initialize worker process for process pool"""
    # Each worker process needs its own predictor instance
    import os
    import sys
    import multiprocessing
    worker_id = multiprocessing.current_process().pid
    try:
        # Ensure Python path is set
        tools_path = os.getenv('ROCKSDB_TOOLS_PATH', '/home/usr/test_environment/rocksdb/tools')
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)
        
        global _predictor_instance
        # Use lazy_load=True to avoid loading all models during initialization
        _predictor_instance = FileLifetimePredictorByLevel(lazy_load=True)
    except Exception as e:
        # 进程池日志已移除
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise

def _predict_worker(args):
    """Worker function for process pool"""
    import multiprocessing
    worker_id = multiprocessing.current_process().pid
    features, level = args
    predictor = get_predictor()
    result = predictor.predict(features, level)
    # 进程池日志已移除
    return result

def initialize():
    """
    Initialize the predictor (called from C++)
    Returns True if initialization successful, False otherwise
    
    Raises exception if initialization fails - expose problems directly
    """
    global _process_pool, _use_process_pool
    
    # Check if process pool should be used
    use_pool = os.getenv('ROCKSDB_ML_USE_PROCESS_POOL', '0')
    pool_size = int(os.getenv('ROCKSDB_ML_PROCESS_POOL_SIZE', '4'))
    
    if use_pool == '1' and pool_size > 0:
        _use_process_pool = True
        try:
            from multiprocessing import Pool
            # Initialize main process predictor first
            # For process pool, use lazy_load=False for main process (load all models)
            # Workers will use lazy_load=True (load on demand)
            predictor = get_predictor()
            # If using lazy_load, models won't be loaded yet, so check if predictor exists
            if predictor and (len(predictor.models) > 0 or predictor.lazy_load):
                # Create process pool with worker initialization
                _process_pool = Pool(processes=pool_size, initializer=_init_worker)
                return True
            else:
                raise RuntimeError(f"Predictor initialization failed: No models loaded. Available levels: {list(predictor.models.keys())}")
        except Exception as e:
            # 进程池日志已移除
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Process pool initialization failed: {e}") from e
    else:
        # No process pool, use direct call
        _use_process_pool = False
        predictor = get_predictor()
        if predictor and len(predictor.models) > 0:
            return True
        # Expose the problem - no models loaded
        raise RuntimeError(f"Predictor initialization failed: No models loaded. Available levels: {list(predictor.models.keys())}")

def predict_file_lifetime_by_level(features, level):
    """
    Predict file lifetime (called from C++)
    
    Args:
        features: numpy array of 69 features
        level: target level (1-6)
    
    Returns:
        predicted lifetime in seconds (float)
    """
    global _process_pool, _use_process_pool
    
    if _use_process_pool and _process_pool is not None:
        # Use process pool for parallel execution
        try:
            import time
            start_time = time.time()
            result = _process_pool.apply(_predict_worker, args=((features, level),))
            return result
        except Exception as e:
            # 进程池日志已移除
            import traceback
            traceback.print_exc()
            raise
    else:
        # Direct call (with GIL protection from C++)
        import time
        start_time = time.time()
        predictor = get_predictor()
        result = predictor.predict(features, level)
        elapsed = (time.time() - start_time) * 1000  # ms
        # 进程池日志已移除
        return result


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

