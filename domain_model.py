"""
Data models and enumerations for the testing framework
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
import time


class ProtocolType(Enum):
    """Enumeration of supported agent communication protocols"""
    MCP = "MCP"  # Model Context Protocol
    A2A = "A2A"  # Agent-to-Agent Protocol
    ACP = "ACP"  # Agent Communication Protocol
    STANDARD = "standard"  # Standard API calls


@dataclass
class ReasoningStep:
    """Represents a single step in the agent's reasoning process"""
    step_number: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class TestResult:
    """
    Comprehensive test result containing all relevant information
    
    Attributes:
        model_name: HuggingFace model identifier
        protocol: Communication protocol used
        framework: Agent framework used for execution
        task: The task/query given to the agent
        response: Final response from the agent
        reasoning_steps: List of intermediate reasoning steps
        latency: Total execution time in seconds
        success: Whether the test completed successfully
        error: Error message if test failed
        metadata: Additional framework-specific information
    """
    model_name: str
    protocol: ProtocolType
    framework: str
    task: str
    response: str
    reasoning_steps: List[ReasoningStep]
    latency: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)