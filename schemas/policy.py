"""
Pydantic schemas for Policy Tree Cohorting API

Defines request/response models for the policy cohort endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class ActionType(str, Enum):
    """Available action types"""
    DISCOUNT_5 = "5"
    DISCOUNT_6 = "6"
    DISCOUNT_7 = "7"
    DISCOUNT_8 = "8"
    DISCOUNT_9 = "9"
    DISCOUNT_10 = "10"
    DISCOUNT_11 = "11"
    DISCOUNT_12 = "12"
    DISCOUNT_13 = "13"
    DISCOUNT_14 = "14"
    DISCOUNT_15 = "15"
    PREMIUM_REWARD = "premium_reward"

class SegmentFilter(str, Enum):
    """Available segment filters"""
    AT_RISK_CUSTOMERS = "at_risk_customers"
    HIGH_VALUE_CUSTOMERS = "high_value_customers"
    STANDARD_CUSTOMERS = "standard_customers"
    PRICE_SENSITIVE_CUSTOMERS = "price_sensitive_customers"
    PREMIUM_CUSTOMERS = "premium_customers"

class DomainFilter(str, Enum):
    """Available domain filters"""
    ENUYGUN_HOTEL = "ENUYGUN_HOTEL"
    ENUYGUN_FLIGHT = "ENUYGUN_FLIGHT"
    ENUYGUN_CAR_RENTAL = "ENUYGUN_CAR_RENTAL"
    ENUYGUN_BUS = "ENUYGUN_BUS"
    WINGIE_FLIGHT = "WINGIE_FLIGHT"

class CohortRule(BaseModel):
    """Individual cohort rule with metrics"""
    name: str = Field(..., description="Cohort name/identifier")
    rule: str = Field(..., description="Human-readable rule conditions")
    action: str = Field(..., description="Recommended action (discount % or premium_reward)")
    size: int = Field(..., ge=1, description="Number of users in cohort")
    avg_expected_profit: float = Field(..., description="Average expected profit per user")
    avg_expected_conversion: float = Field(..., ge=0, le=1, description="Average expected conversion rate")
    mean_confidence: float = Field(..., ge=0, le=1, description="Mean confidence score")
    why_tags: List[str] = Field(default_factory=list, description="Key explanatory tags")
    leaf_id: Optional[int] = Field(None, description="Decision tree leaf ID (internal)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "AtRisk_Flight_PriceHigh_12",
                "rule": "segment=AT_RISK_CUSTOMERS & domain=ENUYGUN_FLIGHT & high_price_sensitivity & high_churn",
                "action": "12",
                "size": 14382,
                "avg_expected_profit": 5.12,
                "avg_expected_conversion": 0.091,
                "mean_confidence": 0.73,
                "why_tags": ["high_price_sens", "high_churn"],
                "leaf_id": 7
            }
        }

class CohortFilters(BaseModel):
    """Filters for cohort queries"""
    segment: Optional[SegmentFilter] = Field(None, description="Filter by segment")
    domain: Optional[DomainFilter] = Field(None, description="Filter by domain")
    min_size: Optional[int] = Field(None, ge=1, description="Minimum cohort size")
    min_profit: Optional[float] = Field(None, description="Minimum average profit")
    max_cohorts: Optional[int] = Field(100, ge=1, le=1000, description="Maximum cohorts to return")
    tags: Optional[List[str]] = Field(None, description="Filter by presence of tags")
    
    # Score range filters
    min_churn_score: Optional[float] = Field(None, ge=0, le=1)
    max_churn_score: Optional[float] = Field(None, ge=0, le=1)
    min_price_sensitivity: Optional[float] = Field(None, ge=0, le=1)
    max_price_sensitivity: Optional[float] = Field(None, ge=0, le=1)

class CohortListRequest(BaseModel):
    """Request for listing cohorts"""
    filters: Optional[CohortFilters] = Field(None, description="Optional filters")
    sort_by: Optional[str] = Field("profit_impact", description="Sort criteria: profit_impact, size, conversion")
    sort_desc: bool = Field(True, description="Sort in descending order")

class CohortListResponse(BaseModel):
    """Response with list of cohorts"""
    cohorts: List[CohortRule] = Field(..., description="List of cohort rules")
    total_cohorts: int = Field(..., description="Total number of cohorts")
    total_users_covered: int = Field(..., description="Total users covered by all cohorts")
    model_version: Optional[str] = Field(None, description="Model version used")
    generated_at: Optional[str] = Field(None, description="Generation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "cohorts": [
                    {
                        "name": "AtRisk_Flight_PriceHigh_12",
                        "rule": "segment=AT_RISK_CUSTOMERS & domain=ENUYGUN_FLIGHT & high_price_sensitivity",
                        "action": "12",
                        "size": 14382,
                        "avg_expected_profit": 5.12,
                        "avg_expected_conversion": 0.091,
                        "mean_confidence": 0.73,
                        "why_tags": ["high_price_sens", "high_churn"]
                    }
                ],
                "total_cohorts": 1,
                "total_users_covered": 14382,
                "model_version": "20241003_143022",
                "generated_at": "2024-10-03T14:30:22"
            }
        }

class CohortPreviewRequest(BaseModel):
    """Request for cohort user preview"""
    cohort_name: Optional[str] = Field(None, description="Specific cohort name")
    rule_conditions: Optional[str] = Field(None, description="Rule conditions to match")
    filters: Optional[CohortFilters] = Field(None, description="Additional filters")
    max_users: int = Field(50, ge=1, le=500, description="Maximum users to return")

class UserPreview(BaseModel):
    """User preview information"""
    user_id: int = Field(..., description="User ID")
    segment: str = Field(..., description="User segment")
    domain: str = Field(..., description="User domain")
    recommended_action: str = Field(..., description="Recommended action")
    expected_profit: float = Field(..., description="Expected profit")
    expected_conversion: float = Field(..., description="Expected conversion rate")
    confidence: float = Field(..., description="Confidence score")
    tags: List[str] = Field(default_factory=list, description="User tags")

class CohortPreviewResponse(BaseModel):
    """Response with cohort user preview"""
    cohort: CohortRule = Field(..., description="Cohort information")
    users: List[UserPreview] = Field(..., description="Sample users in cohort")
    total_users_in_cohort: int = Field(..., description="Total users matching this cohort")
    
    class Config:
        json_schema_extra = {
            "example": {
                "cohort": {
                    "name": "AtRisk_Flight_PriceHigh_12",
                    "rule": "segment=AT_RISK_CUSTOMERS & domain=ENUYGUN_FLIGHT",
                    "action": "12",
                    "size": 14382,
                    "avg_expected_profit": 5.12,
                    "avg_expected_conversion": 0.091,
                    "mean_confidence": 0.73,
                    "why_tags": ["high_price_sens", "high_churn"]
                },
                "users": [
                    {
                        "user_id": 12345,
                        "segment": "at_risk_customers",
                        "domain": "ENUYGUN_FLIGHT",
                        "recommended_action": "12",
                        "expected_profit": 5.20,
                        "expected_conversion": 0.089,
                        "confidence": 0.74,
                        "tags": ["high_price_sens", "high_churn"]
                    }
                ],
                "total_users_in_cohort": 14382
            }
        }

class ModelInfo(BaseModel):
    """Model information and statistics"""
    version: str = Field(..., description="Model version")
    training_date: str = Field(..., description="Training timestamp")
    n_samples: int = Field(..., description="Number of training samples")
    n_features: int = Field(..., description="Number of features")
    tree_depth: int = Field(..., description="Decision tree depth")
    n_leaves: int = Field(..., description="Number of leaf nodes")
    training_accuracy: float = Field(..., description="Training accuracy")
    
    class Config:
        protected_namespaces = ()  # Disable protected namespace warnings

class PolicyStatsResponse(BaseModel):
    """Policy system statistics"""
    model_info: ModelInfo = Field(..., description="Model information")
    cohort_stats: Dict[str, Any] = Field(..., description="Cohort statistics")
    action_distribution: Dict[str, int] = Field(..., description="Action distribution")
    segment_coverage: Dict[str, int] = Field(..., description="Users per segment")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_info": {
                    "version": "20241003_143022",
                    "training_date": "2024-10-03T14:30:22",
                    "n_samples": 50000,
                    "n_features": 25,
                    "tree_depth": 4,
                    "n_leaves": 12,
                    "training_accuracy": 0.847
                },
                "cohort_stats": {
                    "total_cohorts": 8,
                    "avg_cohort_size": 6250,
                    "total_users_covered": 50000
                },
                "action_distribution": {
                    "10": 15000,
                    "12": 20000,
                    "15": 10000,
                    "premium_reward": 5000
                },
                "segment_coverage": {
                    "at_risk_customers": 12000,
                    "standard_customers": 25000,
                    "high_value_customers": 8000,
                    "premium_customers": 3000,
                    "price_sensitive_customers": 2000
                }
            }
        }

class TrainingRequest(BaseModel):
    """Request to trigger model retraining"""
    data_source: Optional[str] = Field(None, description="Data source path")
    max_depth: int = Field(4, ge=2, le=10, description="Maximum tree depth")
    min_samples_leaf_frac: float = Field(0.02, ge=0.01, le=0.1, description="Minimum leaf fraction")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Test set fraction")
    force_retrain: bool = Field(False, description="Force retraining even if recent model exists")

class TrainingResponse(BaseModel):
    """Response from training request"""
    status: str = Field(..., description="Training status")
    model_path: Optional[str] = Field(None, description="Path to saved model")
    training_accuracy: Optional[float] = Field(None, description="Training accuracy")
    test_accuracy: Optional[float] = Field(None, description="Test accuracy") 
    n_cohorts: Optional[int] = Field(None, description="Number of cohorts generated")
    training_time: Optional[float] = Field(None, description="Training time in seconds")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "model_path": "artifacts/policy_tree_20241003_143022.joblib",
                "training_accuracy": 0.847,
                "test_accuracy": 0.832,
                "n_cohorts": 8,
                "training_time": 45.2,
                "message": "Model trained successfully with 8 cohorts"
            }
        }

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid segment filter provided",
                "details": {"allowed_segments": ["at_risk_customers", "high_value_customers"]}
            }
        }
