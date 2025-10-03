from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from core.domain.user import User
from core.domain.enums import DomainType, ActionType

@dataclass
class RuleResult:
    """Kural sonucu"""
    applies: bool
    action_type: Optional[ActionType] = None
    target_domain: Optional[DomainType] = None
    discount_boost: float = 0.0
    message: str = ""

class BusinessRule(ABC):
    """Her kural bu base'den türer"""
    
    @abstractmethod
    def evaluate(self, user: User) -> RuleResult:
        """Kuralı değerlendir"""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """Kural önceliği (düşük = önce çalışır)"""
        pass

class LoginNotificationRule(BusinessRule):
    """user_basket=0 ise login bildirimi yolla"""
    
    @property
    def priority(self) -> int:
        return 1  # En yüksek öncelik
    
    def evaluate(self, user: User) -> RuleResult:
        if not user.user_basket:
            return RuleResult(
                applies=True,
                action_type=ActionType.SEND_LOGIN_NOTIFICATION,
                target_domain=None,
                message="Kullanıcı register olmamış - login bildirimi gönder"
            )
        return RuleResult(applies=False)

class OnewayReturnTicketRule(BusinessRule):
    """is_oneway=1 ise dönüş bileti teklifi yap"""
    
    @property
    def priority(self) -> int:
        return 2
    
    def evaluate(self, user: User) -> RuleResult:
        if user.is_oneway and user.user_basket:
            return RuleResult(
                applies=True,
                action_type=ActionType.RETURN_TICKET_OFFER,
                target_domain=user.domain,  # Aynı domain'de dönüş bileti
                discount_boost=0.02,  # %2 ekstra indirim
                message="Tek yön bilet almış - dönüş için indirim sun"
            )
        return RuleResult(applies=False)

class CrossSellHotelRule(BusinessRule):
    """Flight almış ama hotel almamış ise hotel öner"""
    
    @property
    def priority(self) -> int:
        return 3
    
    def evaluate(self, user: User) -> RuleResult:
        # Not: Gerçek implementasyonda user.previous_domains'e bakılmalı
        # Şimdilik basitleştirilmiş versiyon
        if user.domain == DomainType.ENUYGUN_FLIGHT and user.user_basket:
            # Eğer previous_domains varsa ve hotel yoksa
            if user.previous_domains and DomainType.ENUYGUN_HOTEL not in user.previous_domains:
                return RuleResult(
                    applies=True,
                    action_type=ActionType.CROSS_SELL,
                    target_domain=DomainType.ENUYGUN_HOTEL,
                    discount_boost=0.03,  # %3 ekstra
                    message="Uçuş aldı ama otel yok - cross-sell fırsatı"
                )
        return RuleResult(applies=False)

class RulesEngine:
    """Tüm kuralları yönetir"""
    
    def __init__(self):
        self.rules: list[BusinessRule] = [
            LoginNotificationRule(),
            OnewayReturnTicketRule(),
            CrossSellHotelRule()
        ]
        # Önceliğe göre sırala
        self.rules.sort(key=lambda r: r.priority)
    
    def apply_rules(self, user: User) -> list[RuleResult]:
        """Tüm kuralları uygula ve sonuçları döndür"""
        results = []
        for rule in self.rules:
            result = rule.evaluate(user)
            if result.applies:
                results.append(result)
        return results
    
    def get_primary_rule(self, user: User) -> Optional[RuleResult]:
        """En yüksek öncelikli uygulanabilir kuralı döndür"""
        for rule in self.rules:
            result = rule.evaluate(user)
            if result.applies:
                return result
        return None
