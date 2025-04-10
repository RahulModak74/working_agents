#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any, List, Optional, Union
import logging
from pathlib import Path

# Import scikit-learn components
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sklearn_adapter")

# Initialize the tool registry
TOOL_REGISTRY = {}

# Model registry with available models
CLASSIFICATION_MODELS = {
    "logistic": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "svc": SVC,
    "knn": KNeighborsClassifier,
    "naive_bayes": GaussianNB
}

REGRESSION_MODELS = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elastic_net": ElasticNet,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "svr": SVR
}

# Directory to store trained models
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")
os.makedirs(MODELS_DIR, exist_ok=True)

class SklearnModelManager:
    """Manager for training, evaluating, and using sklearn models"""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.trained_models = {}
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load existing models from disk"""
        model_files = list(Path(self.models_dir).glob("*.pkl"))
        for model_file in model_files:
            model_id = model_file.stem
            try:
                self.trained_models[model_id] = self._load_model(model_id)
                logger.info(f"Loaded existing model: {model_id}")
            except Exception as e:
                logger.warning(f"Failed to load model {model_id}: {e}")
    
    def _load_model(self, model_id: str):
        """Load a model from disk"""
        model_path = os.path.join(self.models_dir, f"{model_id}.pkl")
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def _save_model(self, model_id: str, model_data: Dict[str, Any]):
        """Save a model to disk"""
        model_path = os.path.join(self.models_dir, f"{model_id}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def _prepare_data(self, data: Union[str, pd.DataFrame], target_column: str,
                      features: List[str] = None, categorical_features: List[str] = None):
        """Prepare data for training or prediction"""
        # Load data if path is provided
        if isinstance(data, str):
            if data.endswith('.csv'):
                df = pd.read_csv(data)
            elif data.endswith('.xlsx') or data.endswith('.xls'):
                df = pd.read_excel(data)
            else:
                raise ValueError(f"Unsupported file format: {data}")
        else:
            df = data.copy()
        
        # Select features if specified
        if features:
            X = df[features]
        else:
            X = df.drop(columns=[target_column])
            features = X.columns.tolist()
        
        # Extract target if available
        y = df[target_column] if target_column in df.columns else None
        
        # Identify categorical features if not specified
        if categorical_features is None:
            categorical_features = []
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].nunique() < 10:
                    categorical_features.append(col)
        
        # Identify numerical features
        numerical_features = [col for col in features if col not in categorical_features]
        
        return X, y, numerical_features, categorical_features
    
    def _create_preprocessing_pipeline(self, numerical_features: List[str], 
                                       categorical_features: List[str],
                                       scaling: str = 'standard'):
        """Create a preprocessing pipeline for the data"""
        # Numerical preprocessing
        if scaling == 'standard':
            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
        elif scaling == 'minmax':
            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', MinMaxScaler())
            ])
        else:  # No scaling
            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean'))
            ])
        
        # Categorical preprocessing
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Create the preprocessor
        transformers = []
        if numerical_features:
            transformers.append(('num', num_transformer, numerical_features))
        if categorical_features:
            transformers.append(('cat', cat_transformer, categorical_features))
        
        preprocessor = ColumnTransformer(transformers=transformers)
        return preprocessor
    
    def train_model(self, data: Union[str, pd.DataFrame], 
                   model_type: str = 'classification',
                   algorithm: str = 'random_forest',
                   target_column: str = 'target',
                   features: List[str] = None,
                   categorical_features: List[str] = None,
                   scaling: str = 'standard',
                   test_size: float = 0.2,
                   model_params: Dict[str, Any] = None,
                   do_grid_search: bool = False,
                   grid_params: Dict[str, List[Any]] = None,
                   cv_folds: int = 5,
                   model_id: str = None) -> Dict[str, Any]:
        """Train a model and return metrics"""
        
        # Generate model_id if not provided
        if model_id is None:
            model_id = f"{model_type}_{algorithm}_{int(time.time())}"
        
        try:
            # Prepare data
            X, y, numerical_features, categorical_features = self._prepare_data(
                data, target_column, features, categorical_features
            )
            
            # Check if we have a target column
            if y is None:
                return {"error": f"Target column '{target_column}' not found in the data"}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Create preprocessing pipeline
            preprocessor = self._create_preprocessing_pipeline(
                numerical_features, categorical_features, scaling
            )
            
            # Select model class
            if model_type == 'classification':
                if algorithm not in CLASSIFICATION_MODELS:
                    return {"error": f"Unknown classification algorithm: {algorithm}"}
                model_class = CLASSIFICATION_MODELS[algorithm]
            else:  # regression
                if algorithm not in REGRESSION_MODELS:
                    return {"error": f"Unknown regression algorithm: {algorithm}"}
                model_class = REGRESSION_MODELS[algorithm]
            
            # Create model instance with params
            model_params = model_params or {}
            base_model = model_class(**model_params)
            
            # Create full pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', base_model)
            ])
            
            # Grid search if requested
            if do_grid_search and grid_params:
                # Prefix parameter names with 'model__'
                prefix_grid = {'model__' + k: v for k, v in grid_params.items()}
                
                # Create grid search
                grid_search = GridSearchCV(
                    pipeline, prefix_grid, cv=cv_folds, scoring='accuracy' if model_type == 'classification' else 'r2',
                    n_jobs=-1
                )
                
                # Fit grid search
                grid_search.fit(X_train, y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                # Make predictions
                y_pred = best_model.predict(X_test)
                
                # Store the pipeline
                train_model = best_model
                grid_results = {
                    'best_params': {k.replace('model__', ''): v for k, v in best_params.items()},
                    'cv_results': {
                        'mean_test_score': float(grid_search.cv_results_['mean_test_score'][grid_search.best_index_]),
                        'std_test_score': float(grid_search.cv_results_['std_test_score'][grid_search.best_index_])
                    }
                }
            else:
                # Just fit the pipeline
                pipeline.fit(X_train, y_train)
                
                # Make predictions
                y_pred = pipeline.predict(X_test)
                
                # Store the pipeline
                train_model = pipeline
                grid_results = None
            
            # Calculate metrics
            if model_type == 'classification':
                metrics = {
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'precision': float(precision_score(y_test, y_pred, average='weighted')),
                    'recall': float(recall_score(y_test, y_pred, average='weighted')),
                    'f1': float(f1_score(y_test, y_pred, average='weighted')),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                }
            else:  # regression
                metrics = {
                    'mse': float(mean_squared_error(y_test, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    'mae': float(mean_absolute_error(y_test, y_pred)),
                    'r2': float(r2_score(y_test, y_pred))
                }
            
            # Create model data object
            model_data = {
                'model_id': model_id,
                'model_type': model_type,
                'algorithm': algorithm,
                'pipeline': train_model,
                'features': features,
                'target_column': target_column,
                'numerical_features': numerical_features,
                'categorical_features': categorical_features,
                'metrics': metrics,
                'grid_search': grid_results,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            # Save the model
            self.trained_models[model_id] = model_data
            self._save_model(model_id, model_data)
            
            # Prepare the result
            result = {
                'model_id': model_id,
                'model_type': model_type,
                'algorithm': algorithm,
                'metrics': metrics,
                'features': {
                    'total': len(features),
                    'numerical': len(numerical_features),
                    'categorical': len(categorical_features),
                    'names': features
                },
                'samples': {
                    'training': len(X_train),
                    'test': len(X_test)
                }
            }
            
            # Add grid search results if available
            if grid_results:
                result['grid_search'] = grid_results
            
            return result
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {"error": f"Error training model: {str(e)}"}
    
    def predict(self, model_id: str, data: Union[str, pd.DataFrame],
               target_column: str = None, features: List[str] = None) -> Dict[str, Any]:
        """Make predictions with a trained model"""
        
        try:
            # Check if model exists
            if model_id not in self.trained_models:
                # Try to load from disk
                try:
                    self.trained_models[model_id] = self._load_model(model_id)
                except:
                    return {"error": f"Model not found: {model_id}"}
            
            # Get model data
            model_data = self.trained_models[model_id]
            pipeline = model_data['pipeline']
            model_type = model_data['model_type']
            
            # Prepare data for prediction
            if target_column is None:
                target_column = model_data['target_column']
            
            # Handle the case where the target column might not be in the prediction data
            has_target = True
            if isinstance(data, pd.DataFrame) and target_column not in data.columns:
                # Create temporary target with dummy values for preprocessing
                data[target_column] = 0
                has_target = False
            elif isinstance(data, str):
                # Check if file exists
                if not os.path.exists(data):
                    return {"error": f"Data file not found: {data}"}
                
                # Load data
                if data.endswith('.csv'):
                    df = pd.read_csv(data)
                elif data.endswith('.xlsx') or data.endswith('.xls'):
                    df = pd.read_excel(data)
                else:
                    return {"error": f"Unsupported file format: {data}"}
                
                # Check if target column exists
                has_target = target_column in df.columns
                if not has_target:
                    df[target_column] = 0
                
                data = df
            
            # Prepare data
            X, y, _, _ = self._prepare_data(
                data, target_column, features or model_data['features']
            )
            
            # Make predictions
            predictions = pipeline.predict(X)
            
            # For classification, also get probabilities if model supports it
            probabilities = None
            if model_type == 'classification':
                if hasattr(pipeline, 'predict_proba'):
                    try:
                        proba = pipeline.predict_proba(X)
                        probabilities = proba.tolist()
                    except:
                        pass
            
            # Calculate metrics if target is available
            metrics = None
            if has_target and y is not None:
                if model_type == 'classification':
                    metrics = {
                        'accuracy': float(accuracy_score(y, predictions)),
                        'precision': float(precision_score(y, predictions, average='weighted')),
                        'recall': float(recall_score(y, predictions, average='weighted')),
                        'f1': float(f1_score(y, predictions, average='weighted'))
                    }
                else:  # regression
                    metrics = {
                        'mse': float(mean_squared_error(y, predictions)),
                        'rmse': float(np.sqrt(mean_squared_error(y, predictions))),
                        'mae': float(mean_absolute_error(y, predictions)),
                        'r2': float(r2_score(y, predictions))
                    }
            
            # Prepare the result
            result = {
                'model_id': model_id,
                'model_type': model_type,
                'algorithm': model_data['algorithm'],
                'predictions': predictions.tolist(),
            }
            
            # Add probabilities if available
            if probabilities:
                result['probabilities'] = probabilities
            
            # Add metrics if available
            if metrics:
                result['metrics'] = metrics
            
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {"error": f"Error making predictions: {str(e)}"}
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a trained model"""
        
        try:
            # Check if model exists
            if model_id not in self.trained_models:
                # Try to load from disk
                try:
                    self.trained_models[model_id] = self._load_model(model_id)
                except:
                    return {"error": f"Model not found: {model_id}"}
            
            # Get model data
            model_data = self.trained_models[model_id]
            
            # Remove pipeline from result to make it serializable
            result = {k: v for k, v in model_data.items() if k != 'pipeline'}
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"error": f"Error getting model info: {str(e)}"}
    
    def list_models(self) -> Dict[str, Any]:
        """List all trained models"""
        
        try:
            result = []
            
            for model_id, model_data in self.trained_models.items():
                # Create a summary of the model
                summary = {
                    'model_id': model_id,
                    'model_type': model_data['model_type'],
                    'algorithm': model_data['algorithm'],
                    'metrics': model_data['metrics'],
                    'features': {
                        'total': len(model_data['features']),
                        'names': model_data['features']
                    }
                }
                
                result.append(summary)
            
            return {"models": result}
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return {"error": f"Error listing models: {str(e)}"}
    
    def evaluate_model(self, model_id: str, data: Union[str, pd.DataFrame],
                      target_column: str = None, features: List[str] = None,
                      cv_folds: int = 5) -> Dict[str, Any]:
        """Evaluate a model with cross-validation"""
        
        try:
            # Check if model exists
            if model_id not in self.trained_models:
                # Try to load from disk
                try:
                    self.trained_models[model_id] = self._load_model(model_id)
                except:
                    return {"error": f"Model not found: {model_id}"}
            
            # Get model data
            model_data = self.trained_models[model_id]
            pipeline = model_data['pipeline']
            model_type = model_data['model_type']
            
            # Prepare data for evaluation
            if target_column is None:
                target_column = model_data['target_column']
            
            # Prepare data
            X, y, _, _ = self._prepare_data(
                data, target_column, features or model_data['features']
            )
            
            # Check if we have a target column
            if y is None:
                return {"error": f"Target column '{target_column}' not found in the data"}
            
            # Choose scoring metric based on model type
            if model_type == 'classification':
                scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            else:  # regression
                scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
            
            # Perform cross-validation
            cv_results = {}
            for score in scoring:
                cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring=score)
                cv_results[score] = {
                    'mean': float(cv_scores.mean()),
                    'std': float(cv_scores.std()),
                    'values': cv_scores.tolist()
                }
            
            # Prepare the result
            result = {
                'model_id': model_id,
                'model_type': model_type,
                'algorithm': model_data['algorithm'],
                'cross_validation': {
                    'folds': cv_folds,
                    'results': cv_results
                },
                'samples': len(X)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {"error": f"Error evaluating model: {str(e)}"}
    
    def delete_model(self, model_id: str) -> Dict[str, Any]:
        """Delete a trained model"""
        
        try:
            # Check if model exists
            if model_id not in self.trained_models:
                # Check if it exists on disk
                model_path = os.path.join(self.models_dir, f"{model_id}.pkl")
                if not os.path.exists(model_path):
                    return {"error": f"Model not found: {model_id}"}
            
            # Remove from memory
            if model_id in self.trained_models:
                del self.trained_models[model_id]
            
            # Remove from disk
            model_path = os.path.join(self.models_dir, f"{model_id}.pkl")
            if os.path.exists(model_path):
                os.remove(model_path)
            
            return {"success": True, "message": f"Model {model_id} deleted successfully"}
            
        except Exception as e:
            logger.error(f"Error deleting model: {str(e)}")
            return {"error": f"Error deleting model: {str(e)}"}

# Create global instance
MODEL_MANAGER = SklearnModelManager()

#---------------------------
# Tool Registration Functions
#---------------------------

def ml_train_model(**kwargs) -> Dict[str, Any]:
    """
    Train a machine learning model.
    
    Args:
        data: Path to data file or DataFrame
        model_type: 'classification' or 'regression'
        algorithm: Algorithm name (e.g., 'random_forest', 'logistic')
        target_column: Name of the target column
        features: List of feature columns (optional)
        categorical_features: List of categorical features (optional)
        scaling: Scaling method ('standard', 'minmax', or None)
        test_size: Fraction of data to use for testing
        model_params: Parameters for the model
        do_grid_search: Whether to perform grid search
        grid_params: Parameters for grid search
        cv_folds: Number of cross-validation folds
        model_id: Custom ID for the model (optional)
    
    Returns:
        Dict with training results and metrics
    """
    return MODEL_MANAGER.train_model(**kwargs)

def ml_predict(**kwargs) -> Dict[str, Any]:
    """
    Make predictions with a trained model.
    
    Args:
        model_id: ID of the model to use
        data: Path to data file or DataFrame
        target_column: Name of the target column (optional)
        features: List of feature columns (optional)
    
    Returns:
        Dict with predictions and metrics if target is available
    """
    return MODEL_MANAGER.predict(**kwargs)

def ml_get_model_info(**kwargs) -> Dict[str, Any]:
    """
    Get information about a trained model.
    
    Args:
        model_id: ID of the model
    
    Returns:
        Dict with model information
    """
    return MODEL_MANAGER.get_model_info(**kwargs.get('model_id', ''))

def ml_list_models(**kwargs) -> Dict[str, Any]:
    """
    List all trained models.
    
    Returns:
        Dict with list of model summaries
    """
    return MODEL_MANAGER.list_models()

def ml_evaluate_model(**kwargs) -> Dict[str, Any]:
    """
    Evaluate a model with cross-validation.
    
    Args:
        model_id: ID of the model to evaluate
        data: Path to data file or DataFrame
        target_column: Name of the target column (optional)
        features: List of feature columns (optional)
        cv_folds: Number of cross-validation folds
    
    Returns:
        Dict with evaluation results
    """
    return MODEL_MANAGER.evaluate_model(**kwargs)

def ml_delete_model(**kwargs) -> Dict[str, Any]:
    """
    Delete a trained model.
    
    Args:
        model_id: ID of the model to delete
    
    Returns:
        Dict with deletion status
    """
    return MODEL_MANAGER.delete_model(**kwargs.get('model_id', ''))

def ml_get_available_algorithms(**kwargs) -> Dict[str, Any]:
    """
    Get available ML algorithms for classification and regression.
    
    Returns:
        Dict with available algorithms
    """
    return {
        "classification": list(CLASSIFICATION_MODELS.keys()),
        "regression": list(REGRESSION_MODELS.keys())
    }

# Register tools
TOOL_REGISTRY["ml:train_model"] = ml_train_model
TOOL_REGISTRY["ml:predict"] = ml_predict
TOOL_REGISTRY["ml:get_model_info"] = ml_get_model_info
TOOL_REGISTRY["ml:list_models"] = ml_list_models
TOOL_REGISTRY["ml:evaluate_model"] = ml_evaluate_model
TOOL_REGISTRY["ml:delete_model"] = ml_delete_model
TOOL_REGISTRY["ml:get_available_algorithms"] = ml_get_available_algorithms

# Print initialization message
print("✅ Scikit-learn ML tools registered successfully")
