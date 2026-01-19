"""
Agentic Stack Testing Framework

A framework for testing and evaluating different agentic
technology stacks using different inference providers.

Package structure:
- domain_model: Domain model (dataclasses) and enums
- adapters: Model adapters for different providers
- protocols: Protocol utilities
- frameworks: Framework-specific agent implementations
- tester: Main testing orchestrator
- main: Entry point for running tests
"""

__version__ = "1.0.0"
__author__ = "Vagner Figueredo de Santana"

from .domain_model import ProtocolType, ReasoningStep, TestResult
from .adapters import HuggingFaceAdapter
from .tester import Mastif

__all__ = [
    'ProtocolType',
    'ReasoningStep',
    'TestResult',
    'HuggingFaceAdapter',
    'OpenAIAdapter',
    'Mastif',
]