from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from core.domain.user import User
from core.domain.enums import DomainType


class TriggerType(str, Enum):
    CART_ABANDONMENT = "cart_abandonment"
    UNREGISTERED_USER = "unregistered_user"
    CROSS_SELL_OPPORTUNITY = "cross_sell_opportunity"
    PAST_TRIPS_REMINDER = "past_trips_reminder"
    BUSINESS_RULE_COUPON = "business_rule_coupon"
    SEGMENT_DISCOUNT_CAMPAIGN = "segment_discount_campaign"


class TriggerEvent(BaseModel):
    trigger_id: str
    trigger_type: TriggerType
    user_id: int
    timestamp: datetime
    metadata: Dict[str, Any]


class EventListener:
    """Monitor user actions and fire triggers"""

    def check_cart_abandonment(self, user_id: int, cart_items: List[str], hours_since_add: int) -> Optional[TriggerEvent]:
        """Trigger 1: User added items but didn't purchase"""
        if hours_since_add >= 24 and len(cart_items) > 0:
            return TriggerEvent(
                trigger_id=f"CART_{user_id}_{datetime.now().timestamp()}",
                trigger_type=TriggerType.CART_ABANDONMENT,
                user_id=user_id,
                timestamp=datetime.now(),
                metadata={
                    'cart_items': cart_items,
                    'hours_abandoned': hours_since_add
                }
            )
        return None

    def check_unregistered_user(self, user: User) -> Optional[TriggerEvent]:
        """Trigger 2: user_basket=0 means not registered"""
        if not user.user_basket:
            return TriggerEvent(
                trigger_id=f"UNREG_{user.user_id}_{datetime.now().timestamp()}",
                trigger_type=TriggerType.UNREGISTERED_USER,
                user_id=user.user_id,
                timestamp=datetime.now(),
                metadata={'domain': user.domain.value}
            )
        return None

    def check_cross_sell(self, user_id: int, purchased_domain: DomainType) -> Optional[TriggerEvent]:
        """Trigger 3: After purchase, offer related services"""
        cross_sell_map = {
            DomainType.ENUYGUN_FLIGHT: [DomainType.ENUYGUN_HOTEL, DomainType.ENUYGUN_CAR_RENTAL],
            DomainType.ENUYGUN_HOTEL: [DomainType.ENUYGUN_CAR_RENTAL],
        }

        opportunities = cross_sell_map.get(purchased_domain, [])
        if opportunities:
            return TriggerEvent(
                trigger_id=f"XSELL_{user_id}_{datetime.now().timestamp()}",
                trigger_type=TriggerType.CROSS_SELL_OPPORTUNITY,
                user_id=user_id,
                timestamp=datetime.now(),
                metadata={
                    'purchased': purchased_domain.value,
                    'opportunities': [d.value for d in opportunities]
                }
            )
        return None

    def check_past_trips(self, user_id: int, trip_history: List[Dict[str, Any]]) -> Optional[TriggerEvent]:
        """Trigger 4: Favorite destinations or seasonal patterns"""
        from collections import Counter
        destinations = Counter([trip.get('destination') for trip in trip_history if trip.get('destination')])
        favorite_dest = destinations.most_common(1)[0][0] if destinations else None

        # Check seasonal patterns
        months = []
        for trip in trip_history:
            date_str = trip.get('date')
            try:
                if date_str:
                    months.append(datetime.fromisoformat(date_str).month)
            except Exception:
                continue
        common_month = Counter(months).most_common(1)[0][0] if months else None

        current_month = datetime.now().month
        if common_month and abs(current_month - common_month) <= 1:
            return TriggerEvent(
                trigger_id=f"TRIPS_{user_id}_{datetime.now().timestamp()}",
                trigger_type=TriggerType.PAST_TRIPS_REMINDER,
                user_id=user_id,
                timestamp=datetime.now(),
                metadata={
                    'favorite_destination': favorite_dest or 'your favorite destination',
                    'seasonal_month': common_month,
                    'trip_count': len(trip_history)
                }
            )
        return None

    def check_business_rules(self, user: User, applied_rules: List) -> Optional[TriggerEvent]:
        """Trigger 5: Business rules from workflow (oneway, basket, domain)"""
        if applied_rules:
            return TriggerEvent(
                trigger_id=f"RULE_{user.user_id}_{datetime.now().timestamp()}",
                trigger_type=TriggerType.BUSINESS_RULE_COUPON,
                user_id=user.user_id,
                timestamp=datetime.now(),
                metadata={'rules': [getattr(r, 'message', '') for r in applied_rules]}
            )
        return None


