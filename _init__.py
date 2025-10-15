"""
HuggingFace Agentic Stack Testing Framework

A comprehensive framework for testing and evaluating different agentic
technology stacks using HuggingFace models.

Package structure:
- models: Data models and enums
- adapters: Model adapters for different providers
- context_protocols: Communication protocol implementations
- agent_protocols: Agent-specific protocol utilities
- frameworks: Framework-specific agent implementations
- tester: Main testing orchestrator
- main: Entry point for running tests
"""

__version__ = "1.0.0"
__author__ = "Agentic Testing Framework Team"

from .models import ProtocolType, ReasoningStep, TestResult
from .adapters import HuggingFaceAdapter
from .tester import AgenticStackTester

__all__ = [
    'ProtocolType',
    'ReasoningStep',
    'TestResult',
    'HuggingFaceAdapter',
    'AgenticStackTester',
]