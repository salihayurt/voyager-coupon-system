"""
Feasible action selection for Policy Tree Cohorting

Selects the best feasible action for each user based on segment constraints,
user characteristics, and expected utility.
"""

import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional
from .constraints import (
    get_allowed_actions, 
    is_premium_reward_eligible,
    snap_to_allowed_discount,
    PREMIUM_REWARD_UPLIFT
)
from core.domain.enums import SegmentType

def choose_action(
    row: pd.Series,
    allowed_map: Optional[Dict] = None
) -> Tuple[Union[int, str], float]:
    """
    Choose the best feasible action for a single user
    
    Args:
        row: User data row with required fields:
            - segment: User segment
            - recommended_discount_pct: Recommended discount
            - expected_profit: Expected profit
            - confidence_score: Confidence score
            - price_sensitivity: Price sensitivity score
            - options: Optional list of discount options
        allowed_map: Optional override for allowed actions (for testing)
        
    Returns:
        Tuple of (chosen_action, utility_score)
    """
    segment = row.get('segment')
    if isinstance(segment, str):
        segment = SegmentType(segment.lower())
    
    recommended_discount = row.get('recommended_discount_pct', 10)
    expected_profit = row.get('expected_profit', 0.0)
    confidence_score = row.get('confidence_score', 0.5)
    price_sensitivity = row.get('price_sensitivity', 0.5)
    options = row.get('options', [])
    
    # Get allowed actions for segment
    if allowed_map:
        allowed = allowed_map.get(segment.value if isinstance(segment, SegmentType) else segment, set())
    else:
        allowed = get_allowed_actions(segment)
    
    if not allowed:
        return 10, 0.0  # Fallback
    
    # Extract numeric discount options
    allowed_discounts = [x for x in allowed if isinstance(x, int)]
    
    # Build candidate actions
    candidates = []
    
    # 1. Try to use provided options if they intersect with allowed
    if options:
        for option in options:
            if isinstance(option, dict):
                discount = option.get('discount', 0)
            else:
                discount = int(option)
            
            if discount in allowed_discounts:
                utility = expected_profit * confidence_score
                candidates.append((discount, utility))
    
    # 2. If no valid options, snap recommended discount to nearest allowed
    if not candidates and recommended_discount:
        snapped_discount = snap_to_allowed_discount(recommended_discount, segment)
        utility = expected_profit * confidence_score
        candidates.append((snapped_discount, utility))
    
    # 3. Add premium reward if eligible
    if is_premium_reward_eligible(segment, price_sensitivity) and "premium_reward" in allowed:
        base_utility = expected_profit * confidence_score
        premium_utility = base_utility + PREMIUM_REWARD_UPLIFT
        candidates.append(("premium_reward", premium_utility))
    
    # 4. Fallback to minimum allowed discount
    if not candidates and allowed_discounts:
        min_discount = min(allowed_discounts)
        utility = expected_profit * confidence_score * 0.8  # Penalty for fallback
        candidates.append((min_discount, utility))
    
    # Choose action with maximum utility
    if candidates:
        best_action, best_utility = max(candidates, key=lambda x: x[1])
        return best_action, best_utility
    
    # Ultimate fallback
    return 10, 0.0

def choose_actions_batch(
    df: pd.DataFrame,
    allowed_map: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Vectorized action selection for a batch of users
    
    Args:
        df: DataFrame with user data
        allowed_map: Optional override for allowed actions
        
    Returns:
        DataFrame with additional columns: chosen_action, utility_score
    """
    result_df = df.copy()
    
    # Apply action selection to each row
    actions_and_utilities = df.apply(
        lambda row: choose_action(row, allowed_map), 
        axis=1
    )
    
    # Split results into separate columns
    result_df['chosen_action'] = [x[0] for x in actions_and_utilities]
    result_df['utility_score'] = [x[1] for x in actions_and_utilities]
    
    return result_df

def validate_action_selection(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that all chosen actions are legal for their segments
    
    Args:
        df: DataFrame with chosen_action and segment columns
        
    Returns:
        Validation report dictionary
    """
    violations = []
    
    for idx, row in df.iterrows():
        segment = row.get('segment')
        action = row.get('chosen_action')
        
        if isinstance(segment, str):
            segment = SegmentType(segment.lower())
        
        allowed = get_allowed_actions(segment)
        
        if action not in allowed:
            violations.append({
                'index': idx,
                'user_id': row.get('user_id'),
                'segment': segment.value if isinstance(segment, SegmentType) else segment,
                'chosen_action': action,
                'allowed_actions': list(allowed)
            })
    
    return {
        'total_users': len(df),
        'violations': len(violations),
        'violation_rate': len(violations) / len(df) if len(df) > 0 else 0,
        'violation_details': violations[:10],  # First 10 violations
        'is_valid': len(violations) == 0
    }

def get_action_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get distribution of chosen actions across segments
    
    Args:
        df: DataFrame with chosen_action and segment columns
        
    Returns:
        Action distribution statistics
    """
    if 'chosen_action' not in df.columns or 'segment' not in df.columns:
        return {'error': 'Missing required columns: chosen_action, segment'}
    
    # Overall distribution
    action_counts = df['chosen_action'].value_counts().to_dict()
    
    # Distribution by segment
    segment_distributions = {}
    for segment in df['segment'].unique():
        segment_df = df[df['segment'] == segment]
        segment_dist = segment_df['chosen_action'].value_counts().to_dict()
        segment_distributions[segment] = segment_dist
    
    # Utility statistics
    utility_stats = df['utility_score'].describe().to_dict() if 'utility_score' in df.columns else {}
    
    return {
        'total_users': len(df),
        'action_distribution': action_counts,
        'segment_distributions': segment_distributions,
        'utility_statistics': utility_stats,
        'premium_reward_usage': action_counts.get('premium_reward', 0)
    }
