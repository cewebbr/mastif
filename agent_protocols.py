"""
Agent protocol utilities and helpers
"""

from typing import Dict, Any


class AgentProtocolManager:
    """
    Manages agent protocol interactions and conversions
    
    Provides utilities for working with different agent protocols,
    including message validation, protocol detection, and conversion.
    """
    
    @staticmethod
    def detect_protocol(message: Dict[str, Any]) -> str:
        """
        Detect which protocol a message uses
        
        Args:
            message: Message dictionary to analyze
            
        Returns:
            Protocol name as string
        """
        if "protocol" in message and message["protocol"].startswith("MCP"):
            return "MCP"
        elif "protocol" in message and message["protocol"].startswith("A2A"):
            return "A2A"
        elif "protocol_version" in message and message["protocol_version"].startswith("ACP"):
            return "ACP"
        else:
            return "UNKNOWN"
    
    @staticmethod
    def validate_message(message: Dict[str, Any], protocol_type: str) -> bool:
        """
        Validate if a message conforms to a specific protocol
        
        Args:
            message: Message to validate
            protocol_type: Expected protocol type
            
        Returns:
            True if valid, False otherwise
        """
        if protocol_type == "MCP":
            return "protocol" in message and "context" in message
        elif protocol_type == "A2A":
            return "protocol" in message and "payload" in message
        elif protocol_type == "ACP":
            return "protocol_version" in message and "agent_context" in message
        return False