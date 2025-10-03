from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from .enums import DomainType, ActionType

class CouponDecision(BaseModel):
    """Agent'ların verdiği kupon kararı"""
    user_id: str
    action_type: ActionType
    discount_percentage: Optional[int] = Field(None, ge=5, le=20, description="İndirim oranı")
    target_domain: Optional[DomainType] = Field(None, description="Hangi domain için kupon")
    
    # Finansal tahminler
    expected_conversion_rate: float = Field(ge=0, le=1, description="Beklenen dönüşüm oranı")
    expected_profit: float = Field(description="Beklenen kar (TL)")
    
    # XAI açıklamaları
    reasoning: list[str] = Field(description="Kararın nedenleri")
    profitability_score: float = Field(ge=0, le=1, description="Profitability agent skoru")
    conversion_score: float = Field(ge=0, le=1, description="Conversion agent skoru")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    is_approved: Optional[bool] = None  # Manager onayı
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "USR_12345",
                "action_type": "offer_discount",
                "discount_percentage": 12,
                "target_domain": "ENUYGUN_HOTEL",
                "expected_conversion_rate": 0.73,
                "expected_profit": 450.0,
                "reasoning": [
                    "Kullanıcı churn riski yüksek (+0.35 etki)",
                    "Uçuş aldı ama otel almadı (cross-sell fırsatı)",
                    "Fiyat duyarlılığı yüksek, 12% optimal nokta"
                ],
                "profitability_score": 0.68,
                "conversion_score": 0.82
            }
        }
