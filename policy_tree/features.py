"""
Feature engineering and binning for Policy Tree Cohorting

Converts user data into features suitable for decision tree training,
including one-hot encoding for categorical variables and binning for numerical scores.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .constraints import DEFAULT_BINS
from core.domain.enums import DomainType, SegmentType

def bin_scores(
    df: pd.DataFrame,
    score_cols: List[str] = None,
    bins_config: Dict[str, List[float]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Bin numerical scores into ordinal categories
    
    Args:
        df: DataFrame with score columns
        score_cols: List of score column names to bin
        bins_config: Custom binning configuration
        
    Returns:
        Tuple of (binned_df, metadata)
    """
    if score_cols is None:
        score_cols = ['churn_score', 'activity_score', 'cart_abandon_score', 
                     'price_sensitivity', 'family_score']
    
    if bins_config is None:
        bins_config = DEFAULT_BINS
    
    result_df = df.copy()
    bin_metadata = {}
    
    for col in score_cols:
        if col not in df.columns:
            continue
            
        if col not in bins_config:
            # Default to 3 equal-width bins
            bins = [df[col].quantile(0.33), df[col].quantile(0.67)]
        else:
            bins = bins_config[col]
        
        # Create bin labels
        bin_labels = [f"{col}_low", f"{col}_mid"]
        if len(bins) > 1:
            bin_labels.append(f"{col}_high")
        if len(bins) > 2:
            bin_labels.extend([f"{col}_very_high"] * (len(bins) - 2))
        
        # Apply binning
        binned_col = f"{col}_binned"
        result_df[binned_col] = pd.cut(
            df[col], 
            bins=[-np.inf] + bins + [np.inf], 
            labels=range(len(bins)+1),  # Use numeric labels instead of strings
            include_lowest=True
        ).astype(int)  # Ensure integer type
        
        # Store metadata
        bin_metadata[col] = {
            'bins': bins,
            'labels': bin_labels[:len(bins)+1],
            'binned_column': binned_col
        }
    
    return result_df, bin_metadata

def create_one_hot_features(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    prefix_sep: str = '_'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create one-hot encoded features for categorical variables
    
    Args:
        df: DataFrame with categorical columns
        categorical_cols: List of categorical column names
        prefix_sep: Separator for one-hot column names
        
    Returns:
        Tuple of (one_hot_df, feature_names)
    """
    if categorical_cols is None:
        categorical_cols = ['segment', 'domain']
    
    result_df = df.copy()
    feature_names = []
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
        
        # Handle enum types and convert to string
        if col == 'segment' and len(df) > 0 and hasattr(df[col].iloc[0], 'value'):
            values = df[col].apply(lambda x: x.value if hasattr(x, 'value') else str(x))
        elif col == 'domain' and len(df) > 0 and hasattr(df[col].iloc[0], 'value'):
            values = df[col].apply(lambda x: x.value if hasattr(x, 'value') else str(x))
        else:
            values = df[col].astype(str)
        
        # Create one-hot encoding
        one_hot = pd.get_dummies(values, prefix=col, prefix_sep=prefix_sep, dtype=int)
        
        # Add to result dataframe
        for one_hot_col in one_hot.columns:
            result_df[one_hot_col] = one_hot[one_hot_col].astype(int)  # Ensure integer type
            feature_names.append(one_hot_col)
    
    return result_df, feature_names

def create_tag_features(
    df: pd.DataFrame,
    tags_col: str = 'tags',
    prefix: str = 'tag_'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create binary features from tags (multi-hot encoding)
    
    Args:
        df: DataFrame with tags column
        tags_col: Column name containing tags
        prefix: Prefix for tag feature columns
        
    Returns:
        Tuple of (tagged_df, tag_feature_names)
    """
    result_df = df.copy()
    
    # Get all unique tags
    all_tags = set()
    for tag_list in df[tags_col]:
        if isinstance(tag_list, (list, set)):
            all_tags.update(tag_list)
        elif pd.notna(tag_list):
            # Handle string representation of lists
            if isinstance(tag_list, str) and tag_list.startswith('['):
                try:
                    import ast
                    parsed_tags = ast.literal_eval(tag_list)
                    all_tags.update(parsed_tags)
                except:
                    pass
    
    # Create binary columns for each tag
    tag_feature_names = []
    for tag in sorted(all_tags):
        col_name = f"{prefix}{tag}"
        result_df[col_name] = df[tags_col].apply(
            lambda tags: 1 if (isinstance(tags, (list, set)) and tag in tags) else 0
        ).astype(int)  # Ensure integer type
        tag_feature_names.append(col_name)
    
    return result_df, tag_feature_names

def build_features(
    df: pd.DataFrame,
    score_cols: List[str] = None,
    categorical_cols: List[str] = None,
    tags_col: str = 'tags',
    bins_config: Dict[str, List[float]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build complete feature set for policy tree training
    
    Args:
        df: Input DataFrame with user data
        score_cols: Score columns to bin
        categorical_cols: Categorical columns to one-hot encode
        tags_col: Tags column name
        bins_config: Custom binning configuration
        
    Returns:
        Tuple of (feature_df, metadata)
    """
    if score_cols is None:
        score_cols = ['churn_score', 'activity_score', 'cart_abandon_score', 
                     'price_sensitivity', 'family_score']
    
    if categorical_cols is None:
        categorical_cols = ['segment', 'domain']
    
    result_df = df.copy()
    metadata = {
        'score_bins': {},
        'categorical_features': [],
        'tag_features': [],
        'feature_columns': []
    }
    
    # 1. Bin numerical scores
    if any(col in df.columns for col in score_cols):
        result_df, bin_metadata = bin_scores(result_df, score_cols, bins_config)
        metadata['score_bins'] = bin_metadata
        
        # Add binned columns to features
        for col, bin_info in bin_metadata.items():
            metadata['feature_columns'].append(bin_info['binned_column'])
    
    # 2. One-hot encode categorical variables
    if any(col in df.columns for col in categorical_cols):
        result_df, categorical_features = create_one_hot_features(
            result_df, categorical_cols
        )
        metadata['categorical_features'] = categorical_features
        metadata['feature_columns'].extend(categorical_features)
    
    # 3. Create tag features
    if tags_col in df.columns:
        result_df, tag_features = create_tag_features(result_df, tags_col)
        metadata['tag_features'] = tag_features
        metadata['feature_columns'].extend(tag_features)
    
    return result_df, metadata

def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = 'chosen_action',
    weight_col: str = 'utility_score'
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series], List[str]]:
    """
    Prepare data for policy tree training
    
    Args:
        df: Feature DataFrame
        target_col: Target variable column name
        weight_col: Sample weight column name
        
    Returns:
        Tuple of (X_features, y_target, sample_weights, feature_names)
    """
    # Build features
    feature_df, metadata = build_features(df)
    feature_columns = metadata['feature_columns']
    
    # Extract features matrix
    X = feature_df[feature_columns]
    
    # Extract target variable
    y = feature_df[target_col] if target_col in feature_df.columns else None
    
    # Extract sample weights
    weights = feature_df[weight_col] if weight_col in feature_df.columns else None
    
    return X, y, weights, feature_columns

def get_feature_importance_names(
    feature_names: List[str],
    metadata: Dict[str, Any]
) -> Dict[str, str]:
    """
    Create human-readable names for features
    
    Args:
        feature_names: List of feature column names
        metadata: Feature metadata
        
    Returns:
        Dictionary mapping feature names to readable descriptions
    """
    readable_names = {}
    
    for feature in feature_names:
        # Score bins
        if '_binned' in feature:
            base_score = feature.replace('_binned', '')
            readable_names[feature] = f"{base_score.replace('_', ' ').title()}"
        
        # One-hot categorical
        elif feature.startswith('segment_'):
            segment = feature.replace('segment_', '')
            readable_names[feature] = f"Segment: {segment.replace('_', ' ').title()}"
        elif feature.startswith('domain_'):
            domain = feature.replace('domain_', '')
            readable_names[feature] = f"Domain: {domain.replace('_', ' ').title()}"
        
        # Tags
        elif feature.startswith('tag_'):
            tag = feature.replace('tag_', '')
            readable_names[feature] = f"Tag: {tag.replace('_', ' ').title()}"
        
        # Default
        else:
            readable_names[feature] = feature.replace('_', ' ').title()
    
    return readable_names

def validate_features(X: pd.DataFrame, y: pd.Series = None) -> Dict[str, Any]:
    """
    Validate feature matrix for training
    
    Args:
        X: Feature matrix
        y: Target variable (optional)
        
    Returns:
        Validation report
    """
    report = {
        'n_samples': len(X),
        'n_features': len(X.columns),
        'feature_names': list(X.columns),
        'missing_values': X.isnull().sum().to_dict(),
        'feature_types': X.dtypes.to_dict(),
        'constant_features': [],
        'is_valid': True,
        'issues': []
    }
    
    # Check for constant features
    for col in X.columns:
        if X[col].nunique() <= 1:
            report['constant_features'].append(col)
            report['issues'].append(f"Constant feature: {col}")
    
    # Check for missing values
    missing_cols = [col for col, count in report['missing_values'].items() if count > 0]
    if missing_cols:
        report['issues'].append(f"Missing values in: {missing_cols}")
    
    # Check target variable if provided
    if y is not None:
        report['target_classes'] = y.value_counts().to_dict()
        report['n_classes'] = y.nunique()
        
        if report['n_classes'] < 2:
            report['issues'].append("Target has fewer than 2 classes")
            report['is_valid'] = False
    
    # Set overall validity
    if report['issues']:
        report['is_valid'] = len(report['constant_features']) == 0 and len(missing_cols) == 0
    
    return report
