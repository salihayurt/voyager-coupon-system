from abc import ABC, abstractmethod
from typing import Dict, Any
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.reasoning import ReasoningTools
import os
from .context import SharedContext
from .message import AgentMessage

class BaseVoyagerAgent(ABC):
    """Abstract base class for all Voyager coupon system agents"""
    
    def __init__(self, 
                 name: str, 
                 model: str = "claude-3-5-sonnet-20241022", 
                 temperature: float = 0.7):
        """
        Initialize base agent with Agno integration
        
        Args:
            name: Agent name for identification
            model: LLM model ID to use (e.g. 'claude-3-5-sonnet-20241022')
            temperature: Temperature for LLM responses
        """
        self.name = name
        self.model = model
        self.temperature = temperature
        
        # Lazily create Agno Agent only if explicitly enabled
        self.agent = None
        if os.getenv("ENABLE_LLM_AGENTS", "false").lower() in ("1", "true", "yes"): 
            self.agent = Agent(
                name=name,
                model=Claude(id=model, temperature=temperature),
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
        """
        pass
    
    def send_message(self, recipient: str, message_type: str, content: Dict[str, Any]) -> AgentMessage:
        """Create and return a message to another agent"""
        return AgentMessage(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            content=content
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.model}')"
