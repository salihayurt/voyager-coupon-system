"""
Inference and cohort summarization for Policy Tree Cohorting

Loads trained models and generates explainable cohorts with business rules,
feasible actions, and performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from collections import Counter

from .train import load_model
from .features import build_features, get_feature_importance_names
from .constraints import MIN_SUPPORT

logger = logging.getLogger(__name__)

def extract_rule_from_path(
    estimator,
    leaf_id: int,
    feature_names: List[str],
    metadata: Dict[str, Any]
) -> str:
    """
    Extract human-readable rule from decision tree path to leaf
    
    Args:
        estimator: Trained decision tree
        leaf_id: Leaf node ID
        feature_names: List of feature names
        metadata: Feature metadata
        
    Returns:
        Human-readable rule string
    """
    # Get path from root to leaf
    tree = estimator.tree_
    
    # Find path to leaf
    path_conditions = []
    
    def traverse_path(node_id, conditions):
        if tree.children_left[node_id] == tree.children_right[node_id]:
            # Leaf node
            if node_id == leaf_id:
                return conditions
            else:
                return None
        
        # Internal node - try both branches
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feature_name = feature_names[feature_idx]
        
        # Try left branch (<=)
        left_conditions = conditions + [(feature_name, '<=', threshold)]
        result = traverse_path(tree.children_left[node_id], left_conditions)
        if result is not None:
            return result
        
        # Try right branch (>)
        right_conditions = conditions + [(feature_name, '>', threshold)]
        result = traverse_path(tree.children_right[node_id], right_conditions)
        if result is not None:
            return result
        
        return None
    
    # Get all leaf IDs and find the one matching our target
    leaf_ids = estimator.apply(np.array([[0] * len(feature_names)]))
    
    # Traverse tree to build rule
    for node_id in range(tree.node_count):
        if tree.children_left[node_id] == tree.children_right[node_id]:  # Leaf
            if node_id == leaf_id:
                # Build path to this leaf
                path = []
                current = node_id
                
                # Work backwards from leaf to root (simplified approach)
                # For a more robust implementation, we'd need to track the full path
                break
    
    # Simplified rule extraction - convert conditions to readable format
    rule_parts = []
    
    for feature_name, operator, threshold in path_conditions:
        readable_condition = _format_condition(feature_name, operator, threshold, metadata)
        if readable_condition:
            rule_parts.append(readable_condition)
    
    return " & ".join(rule_parts) if rule_parts else "Default rule"

def _format_condition(
    feature_name: str, 
    operator: str, 
    threshold: float, 
    metadata: Dict[str, Any]
) -> str:
    """
    Format a single condition into human-readable text
    
    Args:
        feature_name: Feature name
        operator: Comparison operator
        threshold: Threshold value
        metadata: Feature metadata
        
    Returns:
        Formatted condition string
    """
    # Handle one-hot encoded features
    if feature_name.startswith('segment_'):
        segment = feature_name.replace('segment_', '')
        if operator == '>' and threshold == 0.5:
            return f"segment={segment}"
        elif operator == '<=' and threshold == 0.5:
            return f"segment≠{segment}"
    
    elif feature_name.startswith('domain_'):
        domain = feature_name.replace('domain_', '')
        if operator == '>' and threshold == 0.5:
            return f"domain={domain}"
        elif operator == '<=' and threshold == 0.5:
            return f"domain≠{domain}"
    
    elif feature_name.startswith('tag_'):
        tag = feature_name.replace('tag_', '')
        if operator == '>' and threshold == 0.5:
            return f"has_{tag}"
        elif operator == '<=' and threshold == 0.5:
            return f"no_{tag}"
    
    # Handle binned scores
    elif '_binned' in feature_name:
        base_score = feature_name.replace('_binned', '')
        # This would need more sophisticated handling based on bin metadata
        return f"{base_score}_{operator}_{threshold:.2f}"
    
    # Default formatting
    return f"{feature_name}{operator}{threshold:.2f}"

def generate_cohorts(
    estimator,
    metadata: Dict[str, Any],
    df: pd.DataFrame,
    min_support: int = MIN_SUPPORT
) -> List[Dict[str, Any]]:
    """
    Generate cohort summaries from trained policy tree
    
    Args:
        estimator: Trained decision tree
        metadata: Model metadata
        df: User data DataFrame
        min_support: Minimum users per cohort
        
    Returns:
        List of cohort dictionaries
    """
    logger.info(f"Generating cohorts from {len(df)} users")
    
    # Build features for prediction
    feature_df, feature_metadata = build_features(df)
    feature_names = metadata.get('feature_names', [])
    
    if not feature_names:
        logger.error("No feature names found in metadata")
        return []
    
    # Get features matrix
    X = feature_df[feature_names]
    
    # Predict leaf assignments
    leaf_ids = estimator.apply(X)
    feature_df['leaf_id'] = leaf_ids
    
    # Group by leaf to create cohorts
    cohorts = []
    
    for leaf_id, group in feature_df.groupby('leaf_id'):
        if len(group) < min_support:
            continue
        
        # Get most common action for this leaf
        if 'chosen_action' in group.columns:
            action_counts = group['chosen_action'].value_counts()
            most_common_action = action_counts.index[0]
        else:
            # Predict action for this leaf
            leaf_predictions = estimator.predict(group[feature_names])
            most_common_action = Counter(leaf_predictions).most_common(1)[0][0]
        
        # Calculate metrics
        avg_profit = group.get('expected_profit', pd.Series([0])).mean()
        avg_conversion = group.get('expected_conversion', pd.Series([0])).mean()
        mean_confidence = group.get('confidence_score', pd.Series([0.5])).mean()
        
        # Skip cohorts with non-positive profit
        if avg_profit <= 0:
            continue
        
        # Extract top tags
        top_tags = []
        if 'tags' in group.columns:
            all_tags = []
            for tag_list in group['tags']:
                if isinstance(tag_list, (list, set)):
                    all_tags.extend(tag_list)
            
            tag_counts = Counter(all_tags)
            top_tags = [tag for tag, count in tag_counts.most_common(2)]
        
        # Generate rule text (simplified)
        rule_text = _generate_simple_rule(group, feature_metadata)
        
        # Create cohort name
        cohort_name = f"Cohort_{leaf_id}_{most_common_action}"
        if top_tags:
            cohort_name += f"_{top_tags[0]}"
        
        cohort = {
            'name': cohort_name,
            'rule': rule_text,
            'action': str(most_common_action),
            'size': len(group),
            'avg_expected_profit': round(avg_profit, 2),
            'avg_expected_conversion': round(avg_conversion, 3),
            'mean_confidence': round(mean_confidence, 2),
            'why_tags': top_tags,
            'leaf_id': int(leaf_id),
            'user_ids': group.get('user_id', group.index).tolist()[:50]  # Preview
        }
        
        cohorts.append(cohort)
    
    # Sort by profit impact (size * profit)
    cohorts.sort(key=lambda x: x['size'] * x['avg_expected_profit'], reverse=True)
    
    logger.info(f"Generated {len(cohorts)} cohorts with min_support={min_support}")
    
    return cohorts

def _generate_simple_rule(group: pd.DataFrame, metadata: Dict[str, Any]) -> str:
    """
    Generate simplified rule text based on group characteristics
    
    Args:
        group: User group DataFrame
        metadata: Feature metadata
        
    Returns:
        Rule text string
    """
    rule_parts = []
    
    # Most common segment
    if 'segment' in group.columns:
        segment_counts = group['segment'].value_counts()
        if len(segment_counts) == 1 or segment_counts.iloc[0] > len(group) * 0.8:
            most_common_segment = segment_counts.index[0]
            segment_name = most_common_segment.replace('_', ' ').title()
            rule_parts.append(f"segment={segment_name}")
    
    # Most common domain
    if 'domain' in group.columns:
        domain_counts = group['domain'].value_counts()
        if len(domain_counts) == 1 or domain_counts.iloc[0] > len(group) * 0.8:
            most_common_domain = domain_counts.index[0]
            domain_name = most_common_domain.replace('_', ' ').title()
            rule_parts.append(f"domain={domain_name}")
    
    # Score-based rules (simplified)
    score_cols = ['churn_score', 'price_sensitivity', 'activity_score']
    for col in score_cols:
        if col in group.columns:
            mean_score = group[col].mean()
            if mean_score >= 0.7:
                rule_parts.append(f"high_{col.replace('_score', '')}")
            elif mean_score <= 0.3:
                rule_parts.append(f"low_{col.replace('_score', '')}")
    
    # Tag-based rules
    if 'tags' in group.columns:
        all_tags = []
        for tag_list in group['tags']:
            if isinstance(tag_list, (list, set)):
                all_tags.extend(tag_list)
        
        tag_counts = Counter(all_tags)
        for tag, count in tag_counts.most_common(2):
            if count > len(group) * 0.5:  # Tag appears in >50% of users
                rule_parts.append(f"has_{tag}")
    
    return " & ".join(rule_parts) if rule_parts else "Mixed characteristics"

def load_cohorts_from_artifact(
    model_path: str,
    data_df: pd.DataFrame = None,
    min_support: int = MIN_SUPPORT
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load cohorts from saved model artifact
    
    Args:
        model_path: Path to saved model
        data_df: Optional user data for cohort generation
        min_support: Minimum users per cohort
        
    Returns:
        Tuple of (cohorts_list, model_metadata)
    """
    # Load model
    estimator, metadata = load_model(model_path)
    
    if data_df is not None:
        # Generate fresh cohorts
        cohorts = generate_cohorts(estimator, metadata, data_df, min_support)
    else:
        # Return empty cohorts list if no data provided
        cohorts = []
        logger.warning("No data provided - returning empty cohorts list")
    
    return cohorts, metadata

def filter_cohorts(
    cohorts: List[Dict[str, Any]],
    segment_filter: Optional[str] = None,
    domain_filter: Optional[str] = None,
    min_size: Optional[int] = None,
    min_profit: Optional[float] = None,
    tags_filter: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Filter cohorts based on criteria
    
    Args:
        cohorts: List of cohort dictionaries
        segment_filter: Filter by segment
        domain_filter: Filter by domain
        min_size: Minimum cohort size
        min_profit: Minimum average profit
        tags_filter: Filter by presence of tags
        
    Returns:
        Filtered cohorts list
    """
    filtered = cohorts.copy()
    
    if segment_filter:
        filtered = [c for c in filtered if segment_filter.lower() in c['rule'].lower()]
    
    if domain_filter:
        filtered = [c for c in filtered if domain_filter.lower() in c['rule'].lower()]
    
    if min_size:
        filtered = [c for c in filtered if c['size'] >= min_size]
    
    if min_profit:
        filtered = [c for c in filtered if c['avg_expected_profit'] >= min_profit]
    
    if tags_filter:
        filtered = [
            c for c in filtered 
            if any(tag in c['why_tags'] for tag in tags_filter)
        ]
    
    return filtered

def get_cohort_preview(
    cohort: Dict[str, Any],
    df: pd.DataFrame,
    max_users: int = 50
) -> List[int]:
    """
    Get preview of user IDs for a specific cohort
    
    Args:
        cohort: Cohort dictionary
        df: User data DataFrame
        max_users: Maximum users to return
        
    Returns:
        List of user IDs
    """
    if 'user_ids' in cohort:
        return cohort['user_ids'][:max_users]
    
    # Fallback: return random sample if user_ids not available
    if 'user_id' in df.columns:
        sample_size = min(max_users, len(df))
        return df['user_id'].sample(sample_size).tolist()
    
    return []

def validate_cohorts(cohorts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate generated cohorts
    
    Args:
        cohorts: List of cohort dictionaries
        
    Returns:
        Validation report
    """
    if not cohorts:
        return {
            'is_valid': False,
            'total_cohorts': 0,
            'issues': ['No cohorts generated']
        }
    
    issues = []
    valid_cohorts = 0
    total_users = 0
    
    for i, cohort in enumerate(cohorts):
        # Check required fields
        required_fields = ['name', 'rule', 'action', 'size', 'avg_expected_profit']
        missing_fields = [f for f in required_fields if f not in cohort]
        
        if missing_fields:
            issues.append(f"Cohort {i}: Missing fields {missing_fields}")
            continue
        
        # Check positive profit
        if cohort['avg_expected_profit'] <= 0:
            issues.append(f"Cohort {i}: Non-positive profit {cohort['avg_expected_profit']}")
            continue
        
        # Check minimum size
        if cohort['size'] < MIN_SUPPORT:
            issues.append(f"Cohort {i}: Size {cohort['size']} below minimum {MIN_SUPPORT}")
            continue
        
        valid_cohorts += 1
        total_users += cohort['size']
    
    return {
        'is_valid': len(issues) == 0,
        'total_cohorts': len(cohorts),
        'valid_cohorts': valid_cohorts,
        'total_users_covered': total_users,
        'avg_cohort_size': total_users / valid_cohorts if valid_cohorts > 0 else 0,
        'issues': issues[:10]  # First 10 issues
    }
