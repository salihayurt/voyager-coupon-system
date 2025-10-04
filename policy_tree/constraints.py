"""
Segment constraints and configuration for Policy Tree Cohorting
"""

from typing import Dict, Set, Union, List
from core.domain.enums import SegmentType

# Allowed actions per segment (discount percentages + premium_reward)
ALLOWED_ACTIONS: Dict[str, Set[Union[int, str]]] = {
    "AT_RISK_CUSTOMERS": {10, 11, 12, 13},
    "HIGH_VALUE_CUSTOMERS": {5, 6, 7, 8, "premium_reward"},
    "STANDARD_CUSTOMERS": {5, 6, 7, 8, 9, 10},
    "PRICE_SENSITIVE_CUSTOMERS": {12, 13, 14, 15},
    "PREMIUM_CUSTOMERS": {5, 6, 7, 8, "premium_reward"},
}

# Segment constraints mapping for SegmentType enum compatibility
SEGMENT_CONSTRAINTS: Dict[SegmentType, Set[Union[int, str]]] = {
    SegmentType.AT_RISK_CUSTOMERS: ALLOWED_ACTIONS["AT_RISK_CUSTOMERS"],
    SegmentType.HIGH_VALUE_CUSTOMERS: ALLOWED_ACTIONS["HIGH_VALUE_CUSTOMERS"],
    SegmentType.STANDARD_CUSTOMERS: ALLOWED_ACTIONS["STANDARD_CUSTOMERS"],
    SegmentType.PRICE_SENSITIVE_CUSTOMERS: ALLOWED_ACTIONS["PRICE_SENSITIVE_CUSTOMERS"],
    SegmentType.PREMIUM_CUSTOMERS: ALLOWED_ACTIONS["PREMIUM_CUSTOMERS"],
}

# Score binning thresholds for feature engineering
DEFAULT_BINS: Dict[str, List[float]] = {
    "churn_score": [0.3, 0.6, 0.8],
    "activity_score": [0.3, 0.7],
    "cart_abandon_score": [0.6, 0.8],
    "price_sensitivity": [0.4, 0.6],
    "family_score": [0.4, 0.6],
}

# Premium reward configuration
PREMIUM_REWARD_UPLIFT: float = 0.02  # Utility uplift for premium reward eligibility
PREMIUM_REWARD_THRESHOLD: float = 0.4  # Price sensitivity threshold for premium reward

# Policy tree hyperparameters
MIN_SUPPORT: int = 50  # Minimum users per cohort
MAX_DEPTH: int = 4  # Maximum tree depth
MIN_LEAF_FRAC: float = 0.02  # Minimum fraction of samples per leaf (2%)

# Premium reward eligible segments
PREMIUM_ELIGIBLE_SEGMENTS: Set[SegmentType] = {
    SegmentType.HIGH_VALUE_CUSTOMERS,
    SegmentType.PREMIUM_CUSTOMERS
}

def get_allowed_actions(segment: Union[SegmentType, str]) -> Set[Union[int, str]]:
    """
    Get allowed actions for a segment
    
    Args:
        segment: SegmentType enum or string
        
    Returns:
        Set of allowed discount percentages and/or premium_reward
    """
    if isinstance(segment, SegmentType):
        return SEGMENT_CONSTRAINTS.get(segment, set())
    elif isinstance(segment, str):
        return ALLOWED_ACTIONS.get(segment, set())
    else:
        return set()

def is_premium_reward_eligible(segment: Union[SegmentType, str], price_sensitivity: float) -> bool:
    """
    Check if user is eligible for premium reward
    
    Args:
        segment: User segment
        price_sensitivity: User price sensitivity score
        
    Returns:
        True if eligible for premium reward
    """
    if isinstance(segment, str):
        segment = SegmentType(segment.lower())
    
    return (segment in PREMIUM_ELIGIBLE_SEGMENTS and 
            price_sensitivity <= PREMIUM_REWARD_THRESHOLD)

def snap_to_allowed_discount(discount: int, segment: Union[SegmentType, str]) -> int:
    """
    Snap a discount to the nearest allowed value for the segment
    
    Args:
        discount: Proposed discount percentage
        segment: User segment
        
    Returns:
        Nearest allowed discount percentage
    """
    allowed = get_allowed_actions(segment)
    allowed_discounts = [x for x in allowed if isinstance(x, int)]
    
    if not allowed_discounts:
        return 10  # Default fallback
    
    # Find nearest allowed discount
    return min(allowed_discounts, key=lambda x: abs(x - discount))

def validate_action(action: Union[int, str], segment: Union[SegmentType, str]) -> bool:
    """
    Validate if an action is allowed for a segment
    
    Args:
        action: Proposed action (discount % or premium_reward)
        segment: User segment
        
    Returns:
        True if action is allowed for segment
    """
    allowed = get_allowed_actions(segment)
    return action in allowed
