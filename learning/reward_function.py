from core.domain.user import User

class RewardCalculator:
    """Q-Learning reward hesaplama"""
    
    def __init__(self, 
                 profit_weight: float = 0.6,
                 conversion_weight: float = 0.4):
        """
        Args:
            profit_weight: Kar'ın ağırlığı
            conversion_weight: Conversion'ın ağırlığı
        """
        self.profit_weight = profit_weight
        self.conversion_weight = conversion_weight
    
    def calculate(self, 
                  user: User,
                  discount_percentage: int,
                  converted: bool,
                  actual_profit: float) -> float:
        """
        Reward = profit_weight * (actual_profit / max_possible_profit) + 
                 conversion_weight * (1 if converted else 0)
        
        Args:
            user: Kullanıcı
            discount_percentage: Verilen indirim
            converted: Kullanıcı satın aldı mı?
            actual_profit: Gerçekleşen kar (TL)
        
        Returns:
            Reward skoru (0-1 arası normalize edilmiş)
        """
        # Conversion component
        conversion_reward = 1.0 if converted else 0.0
        
        # Profit component
        # Max profit: İndirim verilmeseydi elde edilecek kar
        # Basitleştirilmiş formül (gerçekte average_order_value kullanılmalı)
        estimated_order_value = 2000  # TL (örnek)
        max_possible_profit = estimated_order_value * 0.2  # %20 kar marjı
        
        if max_possible_profit > 0:
            profit_reward = actual_profit / max_possible_profit
        else:
            profit_reward = 0.0
        
        # Weighted sum
        total_reward = (
            self.profit_weight * profit_reward +
            self.conversion_weight * conversion_reward
        )
        
        # Penalty: Çok yüksek indirim verdiyse cezalandır
        if discount_percentage > 15:
            total_reward *= 0.9  # %10 penalty
        
        return max(0.0, min(1.0, total_reward))  # Clamp to [0, 1]
    
    def estimate_reward(self,
                       user: User,
                       discount_percentage: int,
                       predicted_conversion_rate: float,
                       estimated_profit: float) -> float:
        """
        Training sırasında gerçek veri yokken reward tahmini
        
        Args:
            predicted_conversion_rate: Agent'ların tahmin ettiği conversion
            estimated_profit: Agent'ların tahmin ettiği kar
        """
        # Simulated conversion (stochastic)
        import random
        converted = random.random() < predicted_conversion_rate
        
        # Use estimated profit
        return self.calculate(
            user=user,
            discount_percentage=discount_percentage,
            converted=converted,
            actual_profit=estimated_profit if converted else 0
        )
