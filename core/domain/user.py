from pydantic import BaseModel, Field, validator
from typing import Optional
from .enums import DomainType, SegmentType

class UserScores(BaseModel):
    """Kullanıcı skorları (ML modelden gelir)"""
    churn_score: float = Field(ge=0, le=1, description="Churn riski skoru")
    activity_score: float = Field(ge=0, le=1, description="Aktivite skoru")
    cart_abandon_score: float = Field(ge=0, le=1, description="Sepet terk etme skoru")
    price_sensitivity: float = Field(ge=0, le=1, description="Fiyat duyarlılığı skoru")

class User(BaseModel):
    """Kullanıcı modeli - agent'lara input olarak verilir"""
    user_id: int
    domain: DomainType
    is_oneway: int = Field(ge=0, le=1, description="Tek yön bilet almış mı (0=hayır, 1=evet)")
    user_basket: int = Field(ge=0, le=1, description="Kullanıcının sepeti var mı (0=hayır, 1=evet)")
    segment: SegmentType
    scores: UserScores
    previous_domains: Optional[list[DomainType]] = None
    
    @validator('is_oneway', 'user_basket')
    def validate_binary_fields(cls, v):
        """Ensure binary fields are 0 or 1"""
        if v not in [0, 1]:
            raise ValueError('Must be 0 or 1')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 12345,
                "domain": "ENUYGUN_FLIGHT",
                "is_oneway": 1,
                "user_basket": 1,
                "segment": "price_sensitive",
                "scores": {
                    "churn_score": 0.35,
                    "activity_score": 0.68,
                    "cart_abandon_score": 0.22,
                    "price_sensitivity": 0.72
                },
                "previous_domains": ["ENUYGUN_FLIGHT"]
            }
        }