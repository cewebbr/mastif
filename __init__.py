"""
HuggingFace Agentic Stack Testing Framework

A framework for testing and evaluating different agentic
technology stacks using HuggingFace models.

Package structure:
- domain_model: Domain model (dataclasses) and enums
- adapters: Model adapters for different providers
- protocols: Protocol utilities
- frameworks: Framework-specific agent implementations
- tester: Main testing orchestrator
- main: Entry point for running tests
"""

__version__ = "1.0.0"
__author__ = "Agentic Testing Framework Team"

from .domain_model import ProtocolType, ReasoningStep, TestResult
from .adapters import HuggingFaceAdapter
from .tester import AgenticStackTester

__all__ = [
    'ProtocolType',
    'ReasoningStep',
    'TestResult',
    'HuggingFaceAdapter',
    'AgenticStackTester',
]