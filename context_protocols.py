"""
Context protocol implementations (MCP, A2A, ACP)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import time


class BaseProtocol(ABC):
    """
    Abstract base class for agent communication protocols
    
    Protocols define how messages are structured and exchanged between agents.
    """
    
    @abstractmethod
    def send_message(self, message: str, context: Dict = None) -> Dict:
        """Format an outgoing message according to protocol specifications"""
        pass
    
    @abstractmethod
    def receive_message(self, response: Dict) -> str:
        """Parse an incoming message and extract content"""
        pass


class MCPProtocol(BaseProtocol):
    """
    Model Context Protocol (MCP) Implementation
    
    Provides structured context management for agent interactions,
    including conversation history, available tools, and environment state.
    """
    
    def send_message(self, message: str, context: Dict = None) -> Dict:
        """Format message with MCP structure including full context"""
        context = context or {}
        
        return {
            "protocol": "MCP/1.0",
            "message": message,
            "context": {
                "conversation_id": context.get("conv_id", "test-001"),
                "turn": context.get("turn", 1),
                "environment": context.get("environment", {}),
                "tools_available": context.get("tools", []),
                "memory": context.get("memory", [])
            },
            "metadata": {
                "timestamp": time.time(),
                "priority": "normal"
            }
        }
    
    def receive_message(self, response: Dict) -> str:
        """Extract content from MCP-formatted response"""
        if isinstance(response, dict):
            return response.get("content", str(response))
        return str(response)


class A2AProtocol(BaseProtocol):
    """
    Agent-to-Agent (A2A) Protocol Implementation
    
    Facilitates direct communication between multiple agents in a system,
    enabling coordination, task delegation, and collaborative problem-solving.
    """
    
    def send_message(self, message: str, context: Dict = None) -> Dict:
        """Format message for agent-to-agent communication with routing info"""
        context = context or {}
        
        return {
            "protocol": "A2A/1.0",
            "sender_agent": context.get("sender", "orchestrator"),
            "receiver_agent": context.get("receiver", "worker"),
            "message_type": context.get("msg_type", "task"),
            "payload": {
                "content": message,
                "task_id": context.get("task_id", "task-001"),
                "priority": context.get("priority", "medium")
            },
            "routing": {
                "requires_response": True,
                "timeout": 30
            }
        }
    
    def receive_message(self, response: Dict) -> str:
        """Extract content from A2A-formatted response"""
        if isinstance(response, dict):
            payload = response.get("payload", {})
            return payload.get("content", str(response))
        return str(response)


class ACPProtocol(BaseProtocol):
    """
    Agent Communication Protocol (ACP) Implementation
    
    Provides a standardized messaging format for heterogeneous agent systems,
    including intent recognition, capability negotiation, and collaboration modes.
    """
    
    def send_message(self, message: str, context: Dict = None) -> Dict:
        """Format message with ACP structure including agent capabilities"""
        context = context or {}
        
        return {
            "protocol_version": "ACP/2.0",
            "message": {
                "id": context.get("msg_id", "msg-001"),
                "type": "query",
                "content": message,
                "intent": context.get("intent", "information_seeking")
            },
            "agent_context": {
                "capabilities": context.get("capabilities", ["reasoning", "tool_use"]),
                "state": context.get("state", "active"),
                "collaboration_mode": context.get("collab_mode", "cooperative")
            }
        }
    
    def receive_message(self, response: Dict) -> str:
        """Extract content from ACP-formatted response"""
        if isinstance(response, dict):
            msg = response.get("message", {})
            return msg.get("content", str(response))
        return str(response)