from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any

class AgentMessage(BaseModel):
    """Inter-agent communication message"""
    sender: str = Field(description="Agent name sending the message")
    recipient: Optional[str] = Field(None, description="Target agent name, None for broadcast")
    message_type: str = Field(description="Type of message: proposal, feedback, decision, query")
    content: Dict[str, Any] = Field(description="Message payload")
    timestamp: datetime = Field(default_factory=datetime.now, description="When message was created")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sender": "ProfitabilityAgent",
                "recipient": "CoordinatorAgent",
                "message_type": "proposal",
                "content": {
                    "discount_percentage": 12,
                    "expected_profit": 280.0,
                    "profitability_score": 0.75,
                    "reasoning": [
                        "User has high price sensitivity (0.72)",
                        "Estimated order value: 2000 TL",
                        "12% discount maximizes profit vs conversion tradeoff"
                    ]
                },
                "timestamp": "2025-10-03T10:30:00"
            }
        }
