"""
XAI decision factors to tags conversion for Policy Tree Cohorting

Converts textual decision factors and user scores into structured tags
for feature engineering and rule generation.
"""

import re
import pandas as pd
from typing import Set, List, Dict, Any, Union

# Tag extraction rules and thresholds
TAG_RULES = {
    'high_price_sens': {
        'score_threshold': ('price_sensitivity', 0.6),
        'keywords': ['price sensitive', 'price-sensitive', 'pricing', 'cost conscious']
    },
    'high_churn': {
        'score_threshold': ('churn_score', 0.7),
        'keywords': ['churn', 'risk', 'leaving', 'at risk', 'retention']
    },
    'cart_abandon': {
        'score_threshold': ('cart_abandon_score', 0.7),
        'keywords': ['abandon', 'cart abandon', 'incomplete', 'drop off']
    },
    'loyal_high_value': {
        'segments': ['HIGH_VALUE_CUSTOMERS', 'PREMIUM_CUSTOMERS'],
        'keywords': ['loyal', 'high value', 'premium', 'vip']
    },
    'family_pattern': {
        'score_threshold': ('family_score', 0.4, 'le'),  # family_score <= 0.4
        'keywords': ['family', 'group', 'multiple', 'children']
    },
    'time_urgent': {
        'keywords': ['last minute', 'deadline', 'urgent', 'immediate', 'soon', 'expires']
    },
    'low_activity': {
        'score_threshold': ('activity_score', 0.3, 'le'),
        'keywords': ['inactive', 'low activity', 'dormant', 'engagement']
    },
    'cross_sell': {
        'keywords': ['cross sell', 'cross-sell', 'additional', 'hotel', 'flight', 'upsell']
    }
}

def make_tags(
    decision_factors: str, 
    scores: Dict[str, float],
    segment: str = None
) -> Set[str]:
    """
    Extract tags from decision factors text and user scores
    
    Args:
        decision_factors: XAI decision factors text
        scores: Dictionary of user scores
        segment: User segment (optional)
        
    Returns:
        Set of extracted tags
    """
    tags = set()
    
    # Normalize text for keyword matching
    text_lower = decision_factors.lower() if decision_factors else ""
    
    for tag_name, rules in TAG_RULES.items():
        tag_applies = False
        
        # Check score threshold
        if 'score_threshold' in rules:
            score_rule = rules['score_threshold']
            score_name = score_rule[0]
            threshold = score_rule[1]
            operator = score_rule[2] if len(score_rule) > 2 else 'ge'
            
            if score_name in scores:
                score_value = scores[score_name]
                if operator == 'ge' and score_value >= threshold:
                    tag_applies = True
                elif operator == 'le' and score_value <= threshold:
                    tag_applies = True
        
        # Check segment membership
        if 'segments' in rules and segment:
            if segment.upper() in rules['segments']:
                tag_applies = True
        
        # Check keywords in decision factors
        if 'keywords' in rules:
            for keyword in rules['keywords']:
                if keyword.lower() in text_lower:
                    tag_applies = True
                    break
        
        if tag_applies:
            tags.add(tag_name)
    
    return tags

def extract_tags_batch(
    df: pd.DataFrame,
    decision_factors_col: str = 'decision_factors',
    score_cols: List[str] = None,
    segment_col: str = 'segment'
) -> pd.DataFrame:
    """
    Extract tags for a batch of users
    
    Args:
        df: DataFrame with user data
        decision_factors_col: Column name for decision factors text
        score_cols: List of score column names
        segment_col: Column name for segment
        
    Returns:
        DataFrame with additional 'tags' column containing list of tags
    """
    if score_cols is None:
        score_cols = ['churn_score', 'activity_score', 'cart_abandon_score', 
                     'price_sensitivity', 'family_score']
    
    result_df = df.copy()
    
    def extract_row_tags(row):
        # Build scores dictionary
        scores = {}
        for col in score_cols:
            if col in row:
                scores[col] = row[col]
        
        # Get decision factors and segment
        decision_factors = row.get(decision_factors_col, "")
        segment = row.get(segment_col, "")
        
        # Extract tags
        tags = make_tags(decision_factors, scores, segment)
        return sorted(list(tags))  # Return sorted list for consistency
    
    result_df['tags'] = df.apply(extract_row_tags, axis=1)
    
    return result_df

def get_tag_distribution(df: pd.DataFrame, tags_col: str = 'tags') -> Dict[str, Any]:
    """
    Get distribution statistics for extracted tags
    
    Args:
        df: DataFrame with tags column
        tags_col: Column name containing tags
        
    Returns:
        Tag distribution statistics
    """
    if tags_col not in df.columns:
        return {'error': f'Column {tags_col} not found'}
    
    # Flatten all tags
    all_tags = []
    for tag_list in df[tags_col]:
        if isinstance(tag_list, list):
            all_tags.extend(tag_list)
        elif isinstance(tag_list, set):
            all_tags.extend(list(tag_list))
    
    # Count tag frequencies
    from collections import Counter
    tag_counts = Counter(all_tags)
    
    # Calculate tag co-occurrence
    tag_pairs = Counter()
    for tag_list in df[tags_col]:
        if isinstance(tag_list, (list, set)):
            tags = list(tag_list)
            for i, tag1 in enumerate(tags):
                for tag2 in tags[i+1:]:
                    pair = tuple(sorted([tag1, tag2]))
                    tag_pairs[pair] += 1
    
    return {
        'total_users': len(df),
        'users_with_tags': sum(1 for tags in df[tags_col] if tags),
        'tag_frequencies': dict(tag_counts.most_common()),
        'avg_tags_per_user': sum(len(tags) for tags in df[tags_col]) / len(df),
        'most_common_pairs': dict(tag_pairs.most_common(10)),
        'unique_tags': len(tag_counts)
    }

def create_tag_features(
    df: pd.DataFrame, 
    tags_col: str = 'tags',
    prefix: str = 'tag_'
) -> pd.DataFrame:
    """
    Create binary features from tags for machine learning
    
    Args:
        df: DataFrame with tags column
        tags_col: Column name containing tags
        prefix: Prefix for tag feature columns
        
    Returns:
        DataFrame with binary tag features
    """
    result_df = df.copy()
    
    # Get all unique tags
    all_tags = set()
    for tag_list in df[tags_col]:
        if isinstance(tag_list, (list, set)):
            all_tags.update(tag_list)
    
    # Create binary columns for each tag
    for tag in sorted(all_tags):
        col_name = f"{prefix}{tag}"
        result_df[col_name] = df[tags_col].apply(
            lambda tags: 1 if tag in tags else 0
        )
    
    return result_df

def validate_tags(tags: Union[List[str], Set[str]]) -> Dict[str, Any]:
    """
    Validate extracted tags against known tag rules
    
    Args:
        tags: List or set of extracted tags
        
    Returns:
        Validation report
    """
    known_tags = set(TAG_RULES.keys())
    tag_set = set(tags) if isinstance(tags, list) else tags
    
    unknown_tags = tag_set - known_tags
    
    return {
        'valid_tags': list(tag_set & known_tags),
        'unknown_tags': list(unknown_tags),
        'total_tags': len(tag_set),
        'is_valid': len(unknown_tags) == 0,
        'coverage': len(tag_set & known_tags) / len(known_tags) if known_tags else 0
    }
