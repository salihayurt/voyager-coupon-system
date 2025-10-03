from enum import IntEnum

class DiscountAction(IntEnum):
    """Q-Learning action space - indirim oranları"""
    DISCOUNT_5 = 0   # %5
    DISCOUNT_7 = 1   # %7
    DISCOUNT_10 = 2  # %10
    DISCOUNT_12 = 3  # %12
    DISCOUNT_15 = 4  # %15
    DISCOUNT_20 = 5  # %20

class ActionSpace:
    """Action space yönetimi"""
    
    DISCOUNT_MAP = {
        DiscountAction.DISCOUNT_5: 5,
        DiscountAction.DISCOUNT_7: 7,
        DiscountAction.DISCOUNT_10: 10,
        DiscountAction.DISCOUNT_12: 12,
        DiscountAction.DISCOUNT_15: 15,
        DiscountAction.DISCOUNT_20: 20,
    }
    
    @staticmethod
    def get_discount_percentage(action: DiscountAction) -> int:
        """Action → Discount %"""
        return ActionSpace.DISCOUNT_MAP[action]
    
    @staticmethod
    def get_action_from_discount(discount: int) -> DiscountAction:
        """Discount % → Action"""
        for action, disc in ActionSpace.DISCOUNT_MAP.items():
            if disc == discount:
                return action
        raise ValueError(f"Invalid discount: {discount}")
    
    @staticmethod
    def total_actions() -> int:
        return len(DiscountAction)
