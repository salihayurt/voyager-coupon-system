from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from core.domain.user import User
from core.rules.business_rules import RuleResult

class SharedContext(BaseModel):
    """Shared context that all agents can access and modify"""
    user: User = Field(description="User being analyzed")
    business_rules: List[RuleResult] = Field(description="Applied business rules")
    q_learning_suggestion: Optional[int] = Field(None, description="Q-Learning's recommended discount percentage")
    q_value: Optional[float] = Field(None, description="Q-Learning's confidence score")
    profitability_proposal: Optional[Dict] = Field(None, description="Profitability agent's proposal")
    conversion_proposal: Optional[Dict] = Field(None, description="Conversion agent's proposal")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    
    def add_agent_proposal(self, agent_name: str, proposal: Dict) -> None:
        """Add proposal from specific agent"""
        agent_name_lower = agent_name.lower()
        
        if "profitability" in agent_name_lower:
            self.profitability_proposal = proposal
        elif "conversion" in agent_name_lower:
            self.conversion_proposal = proposal
        else:
            # Store in metadata for other agent types
            if "agent_proposals" not in self.metadata:
                self.metadata["agent_proposals"] = {}
            self.metadata["agent_proposals"][agent_name] = proposal
    
    class Config:
        json_schema_extra = {
            "example": {
                "user": {
                    "user_id": "USR_12345",
                    "domain": "ENUYGUN_FLIGHT",
                    "is_oneway": True,
                    "user_basket": True,
                    "segment": "price_sensitive",
                    "scores": {
                        "churn_score": 0.35,
                        "activity_score": 0.68,
                        "cart_abandon_score": 0.22,
                        "price_sensitivity": 0.72
                    }
                },
                "business_rules": [
                    {
                        "applies": True,
                        "action_type": "return_ticket_offer",
                        "target_domain": "ENUYGUN_FLIGHT",
                        "discount_boost": 0.02,
                        "message": "Tek yön bilet almış - dönüş için indirim sun"
                    }
                ],
                "q_learning_suggestion": 12,
                "q_value": 0.84,
                "profitability_proposal": {
                    "discount_percentage": 10,
                    "expected_profit": 320.0,
                    "profitability_score": 0.78
                },
                "conversion_proposal": {
                    "discount_percentage": 15,
                    "expected_conversion_rate": 0.82,
                    "conversion_score": 0.85
                },
                "metadata": {
                    "session_id": "sess_789",
                    "processing_time": 0.045
                }
            }
        }
