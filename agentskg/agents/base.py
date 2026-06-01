from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Base class for Agent configuration."""

    name: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1000
    enable_thinking: bool = False


class BaseAgent(ABC):
    """Base class for Agents."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._initialize()

    def _initialize(self) -> None:
        """Initialize additional settings for the Agent."""
        pass

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the core logic of the Agent.

        Args:
            context: The execution context, containing input data and other necessary information.

        Returns:
            The execution result.
        """
        pass

    @abstractmethod
    async def validate(self, result: Dict[str, Any]) -> bool:
        """Validate the execution result.

        Args:
            result: The return value from the execute method.

        Returns:
            True if the validation passes, False otherwise.
        """
        pass
