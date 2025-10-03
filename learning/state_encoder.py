import numpy as np
from typing import Tuple
from core.domain.user import User
from core.domain.enums import DomainType, SegmentType

class StateEncoder:
    """User nesnesini Q-Learning state'ine çevirir"""
    
    # Discrete buckets
    SCORE_BUCKETS = 3  # Low(0), Medium(1), High(2)
    
    def __init__(self):
        # Enum mappings
        self.segment_to_idx = {seg: idx for idx, seg in enumerate(SegmentType)}
        self.domain_to_idx = {dom: idx for idx, dom in enumerate(DomainType)}
        
        # State space boyutu hesapla
        self.state_space_size = self._calculate_state_space()
    
    def _calculate_state_space(self) -> int:
        """Total state kombinasyonlarını hesapla"""
        return (
            len(SegmentType) *      # 6 segment
            self.SCORE_BUCKETS *    # 3 churn bucket
            self.SCORE_BUCKETS *    # 3 activity bucket
            self.SCORE_BUCKETS *    # 3 cart abandon bucket
            self.SCORE_BUCKETS *    # 3 price sensitivity bucket
            len(DomainType) *       # 4 domain
            2 *                     # is_oneway (0/1)
            2                       # user_basket (0/1)
        )
    
    def encode(self, user: User) -> Tuple[int, ...]:
        """User → State tuple"""
        return (
            self.segment_to_idx[user.segment],
            self._discretize_score(user.scores.churn_score),
            self._discretize_score(user.scores.activity_score),
            self._discretize_score(user.scores.cart_abandon_score),
            self._discretize_score(user.scores.price_sensitivity),
            self.domain_to_idx[user.domain],
            int(user.is_oneway),
            int(user.user_basket)
        )
    
    def _discretize_score(self, score: float) -> int:
        """
        Continuous score (0-1) → Discrete bucket (0, 1, 2)
        0.0-0.33: 0 (Low)
        0.33-0.66: 1 (Medium)
        0.66-1.0: 2 (High)
        """
        if score < 0.33:
            return 0
        elif score < 0.66:
            return 1
        else:
            return 2
    
    def decode_readable(self, state: Tuple[int, ...]) -> dict:
        """State tuple → Human-readable dict (debug için)"""
        segment_names = list(SegmentType)
        domain_names = list(DomainType)
        score_labels = ["Low", "Medium", "High"]
        
        return {
            "segment": segment_names[state[0]].value,
            "churn": score_labels[state[1]],
            "activity": score_labels[state[2]],
            "cart_abandon": score_labels[state[3]],
            "price_sensitivity": score_labels[state[4]],
            "domain": domain_names[state[5]].value,
            "is_oneway": bool(state[6]),
            "user_basket": bool(state[7])
        }
