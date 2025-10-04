"""
Policy Tree training module for Voyager Coupon System

Trains shallow decision trees on user data to create explainable cohorts
with feasible actions that respect segment constraints.
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import logging

from .features import build_features, prepare_training_data, validate_features
from .feasible import choose_actions_batch, validate_action_selection
from .tags import extract_tags_batch
from .constraints import MAX_DEPTH, MIN_LEAF_FRAC, MIN_SUPPORT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fit_policy_tree(
    df: pd.DataFrame,
    target_col: str = 'chosen_action',
    weight_col: str = 'utility_score',
    max_depth: int = MAX_DEPTH,
    min_samples_leaf_frac: float = MIN_LEAF_FRAC,
    class_weight: str = 'balanced',
    random_state: int = 42
) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
    """
    Fit a policy tree on user data
    
    Args:
        df: DataFrame with user features and chosen actions
        target_col: Target variable column name
        weight_col: Sample weight column name
        max_depth: Maximum tree depth
        min_samples_leaf_frac: Minimum fraction of samples per leaf
        class_weight: Class weighting strategy
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (fitted_estimator, metadata)
    """
    logger.info(f"Training policy tree on {len(df)} samples")
    
    # Prepare training data
    X, y, sample_weights, feature_names = prepare_training_data(
        df, target_col, weight_col
    )
    
    if y is None:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    # Encode target variable to handle mixed types (int and string)
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    # Convert all target values to strings first to ensure uniform type
    y_str = y.astype(str)
    y_encoded = label_encoder.fit_transform(y_str)
    
    # Store label mapping for later use
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    
    # Validate features
    validation_report = validate_features(X, pd.Series(y_encoded))
    if not validation_report['is_valid']:
        logger.warning(f"Feature validation issues: {validation_report['issues']}")
    
    # Calculate min_samples_leaf
    min_samples_leaf = max(1, int(min_samples_leaf_frac * len(df)))
    
    logger.info(f"Tree parameters: max_depth={max_depth}, min_samples_leaf={min_samples_leaf}")
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target classes: {y.value_counts().to_dict()}")
    logger.info(f"Encoded target classes: {dict(zip(*np.unique(y_encoded, return_counts=True)))}")
    
    # Initialize classifier
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        criterion='gini'
    )
    
    # Fit the model
    if sample_weights is not None:
        clf.fit(X, y_encoded, sample_weight=sample_weights)
        logger.info("Trained with sample weights (utility scores)")
    else:
        clf.fit(X, y_encoded)
        logger.info("Trained without sample weights")
    
    # Calculate training metrics
    y_pred = clf.predict(X)
    train_accuracy = accuracy_score(y_encoded, y_pred)
    
    # Get feature importance
    feature_importance = dict(zip(feature_names, clf.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    logger.info(f"Training accuracy: {train_accuracy:.3f}")
    logger.info(f"Tree depth: {clf.get_depth()}")
    logger.info(f"Number of leaves: {clf.get_n_leaves()}")
    logger.info(f"Top features: {[f[0] for f in top_features[:5]]}")
    
    # Create metadata
    metadata = {
        'training_timestamp': datetime.now().isoformat(),
        'n_samples': len(df),
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'target_classes': y.value_counts().to_dict(),
        'tree_depth': clf.get_depth(),
        'n_leaves': clf.get_n_leaves(),
        'training_accuracy': train_accuracy,
        'feature_importance': feature_importance,
        'top_features': dict(top_features),
        'label_encoder': label_encoder,
        'label_mapping': label_mapping,
        'hyperparameters': {
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'class_weight': class_weight,
            'random_state': random_state
        },
        'validation_report': validation_report
    }
    
    return clf, metadata

def save_model(
    estimator: DecisionTreeClassifier,
    metadata: Dict[str, Any],
    filepath: str,
    version: str = None
) -> str:
    """
    Save trained model and metadata to file
    
    Args:
        estimator: Trained decision tree
        metadata: Model metadata
        filepath: Output file path
        version: Model version (defaults to timestamp)
        
    Returns:
        Actual filepath used
    """
    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure filepath has version
    path = Path(filepath)
    if version not in path.stem:
        versioned_path = path.parent / f"{path.stem}_{version}{path.suffix}"
    else:
        versioned_path = path
    
    # Create directory if needed
    versioned_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Package model artifacts
    artifacts = {
        'estimator': estimator,
        'metadata': metadata,
        'version': version,
        'voyager_policy_tree': True  # Marker for validation
    }
    
    # Save to file
    joblib.dump(artifacts, versioned_path)
    logger.info(f"Model saved to: {versioned_path}")
    
    return str(versioned_path)

def load_model(filepath: str) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
    """
    Load trained model and metadata from file
    
    Args:
        filepath: Model file path
        
    Returns:
        Tuple of (estimator, metadata)
    """
    try:
        artifacts = joblib.load(filepath)
        
        # Validate artifacts
        if not isinstance(artifacts, dict) or 'voyager_policy_tree' not in artifacts:
            raise ValueError("Invalid model file format")
        
        estimator = artifacts['estimator']
        metadata = artifacts['metadata']
        
        logger.info(f"Model loaded from: {filepath}")
        logger.info(f"Model version: {artifacts.get('version', 'unknown')}")
        logger.info(f"Training samples: {metadata.get('n_samples', 'unknown')}")
        
        return estimator, metadata
        
    except Exception as e:
        logger.error(f"Failed to load model from {filepath}: {e}")
        raise

def train_pipeline(
    input_data: pd.DataFrame,
    output_path: str = "artifacts/policy_tree.joblib",
    test_size: float = 0.2,
    **kwargs
) -> Tuple[str, Dict[str, Any]]:
    """
    Complete training pipeline from raw data to saved model
    
    Args:
        input_data: Raw user data DataFrame
        output_path: Output model path
        test_size: Fraction of data for testing
        **kwargs: Additional parameters for fit_policy_tree
        
    Returns:
        Tuple of (saved_model_path, evaluation_metrics)
    """
    logger.info("Starting policy tree training pipeline")
    
    # Step 1: Extract tags from decision factors
    if 'decision_factors' in input_data.columns:
        logger.info("Extracting tags from decision factors")
        input_data = extract_tags_batch(input_data)
    
    # Step 2: Choose feasible actions
    logger.info("Selecting feasible actions")
    input_data = choose_actions_batch(input_data)
    
    # Step 3: Validate action selection
    validation_report = validate_action_selection(input_data)
    if not validation_report['is_valid']:
        logger.error(f"Action validation failed: {validation_report}")
        raise ValueError("Invalid action selection - check segment constraints")
    
    logger.info(f"Action selection valid - {validation_report['violation_rate']:.1%} violations")
    
    # Step 4: Split data for evaluation
    if test_size > 0:
        train_df, test_df = train_test_split(
            input_data, 
            test_size=test_size, 
            stratify=input_data.get('segment'),
            random_state=42
        )
        logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test")
    else:
        train_df = input_data
        test_df = None
        logger.info("Using all data for training (no test split)")
    
    # Step 5: Train model
    estimator, metadata = fit_policy_tree(train_df, **kwargs)
    
    # Step 6: Evaluate on test set if available
    evaluation_metrics = {'validation_report': validation_report}
    
    if test_df is not None:
        X_test, y_test, _, _ = prepare_training_data(test_df)
        
        # Encode test targets using the same encoder
        label_encoder = metadata.get('label_encoder')
        if label_encoder:
            y_test_str = y_test.astype(str)  # Convert to string first
            y_test_encoded = label_encoder.transform(y_test_str)
        else:
            y_test_encoded = y_test
        
        y_pred = estimator.predict(X_test)
        
        test_accuracy = accuracy_score(y_test_encoded, y_pred)
        evaluation_metrics.update({
            'test_accuracy': test_accuracy,
            'test_samples': len(test_df),
            'classification_report': classification_report(y_test_encoded, y_pred, output_dict=True)
        })
        
        logger.info(f"Test accuracy: {test_accuracy:.3f}")
    
    # Step 7: Save model
    saved_path = save_model(estimator, metadata, output_path)
    
    logger.info("Training pipeline completed successfully")
    return saved_path, evaluation_metrics

def main():
    """CLI entry point for training"""
    parser = argparse.ArgumentParser(description='Train Policy Tree for Voyager Coupon System')
    parser.add_argument('--input', required=True, help='Input data file (CSV or Parquet)')
    parser.add_argument('--output', default='artifacts/policy_tree.joblib', help='Output model path')
    parser.add_argument('--max-depth', type=int, default=MAX_DEPTH, help='Maximum tree depth')
    parser.add_argument('--min-leaf-frac', type=float, default=MIN_LEAF_FRAC, help='Min leaf fraction')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load input data
    input_path = Path(args.input)
    if input_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    logger.info(f"Loaded {len(df)} samples from {input_path}")
    
    # Run training pipeline
    try:
        saved_path, metrics = train_pipeline(
            df,
            output_path=args.output,
            max_depth=args.max_depth,
            min_samples_leaf_frac=args.min_leaf_frac,
            test_size=args.test_size,
            random_state=args.random_seed
        )
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {saved_path}")
        print(f"🎯 Test accuracy: {metrics.get('test_accuracy', 'N/A')}")
        print(f"🔍 Validation violations: {metrics['validation_report']['violation_rate']:.1%}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
