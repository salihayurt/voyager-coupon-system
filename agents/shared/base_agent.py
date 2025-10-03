from abc import ABC, abstractmethod
from typing import Dict, Any
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.reasoning import ReasoningTools
from .context import SharedContext
from .message import AgentMessage

class BaseVoyagerAgent(ABC):
    """Abstract base class for all Voyager coupon system agents"""
    
    def __init__(self, 
                 name: str, 
                 model: str = "claude-sonnet-4-5-20250929", 
                 temperature: float = 0.7):
        """
        Initialize base agent with Agno integration
        
        Args:
            name: Agent name for identification
            model: LLM model to use
            temperature: Temperature for LLM responses
        """
        self.name = name
        self.model = model
        self.temperature = temperature
        
        # Create Agno Agent instance with proper model and tools
        self.agent = Agent(
            name=name,
            model=Claude(model=model, temperature=temperature),
            tools=[ReasoningTools()],
            instructions=self._setup_instructions()
        )
    
    @abstractmethod
    def _setup_instructions(self) -> str:
        """Setup agent-specific instructions - implemented by subclasses"""
        pass
    
    @abstractmethod
    def make_proposal(self, context: SharedContext) -> Dict[str, Any]:
        """
        Make proposal based on shared context
        
        Args:
            context: Shared context with user data and other agent proposals
            
        Returns:
            Proposal dictionary with:
            - discount: int (recommended discount percentage)
            - reasoning: list[str] (explanation of decision)
            - confidence: float (0-1, confidence in recommendation)
            - expected_conversion: float (predicted conversion rate)
            - expected_profit: float (predicted profit in TL)
        """
        pass
    
    def send_message(self, 
                    recipient: str, 
                    message_type: str, 
                    content: Dict[str, Any]) -> AgentMessage:
        """
        Create and return a message to another agent
        
        Args:
            recipient: Target agent name (None for broadcast)
            message_type: Type of message (proposal, feedback, decision, query)
            content: Message payload
            
        Returns:
            AgentMessage instance
        """
        return AgentMessage(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            content=content
        )
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.model}')"
