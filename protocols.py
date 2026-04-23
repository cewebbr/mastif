"""
Protocol implementations with encapsulated validation, injection, and telemetry.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
import time
import json


class BaseProtocol(ABC):
    """
    Abstract base class for agent communication protocols with internal telemetry.
    """
    
    def __init__(self):
        # Internal state to track performance across all agent implementations
        self.stats = {
            "total_calls": 0,
            "valid_responses": 0,
            "total_overhead_tokens": 0
        }
    
    @abstractmethod
    def _get_protocol_instructions(self) -> str:
        """Returns the specific string to inject into the prompt for this protocol."""
        pass

    @abstractmethod
    def send_message(self, message: str, context: Dict = None) -> Dict:
        """Format an outgoing message and inject prompt specifics."""
        pass
    
    @abstractmethod
    def receive_message(self, response: Any) -> str:
        """Parse, validate compliance, and log overhead."""
        pass

    def _update_stats(self, payload: Any, is_valid: bool):
        """Internal helper to log token overhead and compliance."""
        self.stats["total_calls"] += 1
        if is_valid:
            self.stats["valid_responses"] += 1
        
        # Approximation: 4 chars per token for the JSON wrapper overhead
        # TODO: Perform a more accurate token count based on the actual payload size and protocol structure
        try:
            overhead_str = json.dumps(payload)
            self.stats["total_overhead_tokens"] += len(overhead_str) // 4
        except Exception:
            pass

    def get_metrics(self) -> Dict:
        """Retrieve telemetry for experiment analysis across the agent's run."""
        compliance = (self.stats["valid_responses"] / self.stats["total_calls"] * 100) if self.stats["total_calls"] > 0 else 0
        return {
            "compliance_rate": f"{compliance:.2f}%",
            "avg_overhead_tokens": self.stats["total_overhead_tokens"] / max(1, self.stats["total_calls"]),
            "total_calls": self.stats["total_calls"]
        }

    def measure_overhead(self, message: str, context: Dict = None) -> Dict:
        """Measure the protocol's serialization and parsing overhead (ms)."""
        t0 = time.perf_counter()
        formatted = self.send_message(message, context or {})
        send_ms = (time.perf_counter() - t0) * 1000

        message_size = len(json.dumps(formatted).encode("utf-8"))

        t1 = time.perf_counter()
        # Test validation logic with a valid structure
        self.receive_message(formatted)
        receive_ms = (time.perf_counter() - t1) * 1000

        return {
            "send_overhead_ms":    round(send_ms, 4),
            "receive_overhead_ms": round(receive_ms, 4),
            "total_overhead_ms":   round(send_ms + receive_ms, 4),
            "message_size_bytes":  message_size,
        }

    def generate_context(self, task: str, turn: int = 1, tools: list = None) -> Dict:
        """Generate a realistic context dict for use with send_message."""
        tools = tools or []
        return {
            "conv_id":      f"conv-{abs(hash(task)) % 100000:05d}",
            "turn":         turn,
            "task_id":      f"task-{abs(hash(task + str(turn))) % 10000:04d}",
            "msg_id":       f"msg-{abs(hash(task)) % 10000:04d}-{turn}",
            "sender":       "orchestrator",
            "receiver":     "worker",
            "msg_type":     "task",
            "tools":        tools,
            "capabilities": ["reasoning", "tool_use"] + (["web_access"] if tools else []),
            "memory": [
                {"role": "system", "content": "You are a web automation agent."},
                {"role": "user",   "content": task[:200]},
            ],
        }


class MCPProtocol(BaseProtocol):
    """Model Context Protocol (MCP) Implementation"""
    
    def _get_protocol_instructions(self) -> str:
        return "\nFORMAT: Respond ONLY in valid MCP/1.0 JSON: {\"protocol\": \"MCP/1.0\", \"content\": \"your_response_here\"}"

    def send_message(self, message: str, context: Dict = None) -> Dict:
        context = context or {}
        injected_msg = f"{message}\n{self._get_protocol_instructions()}"
        
        return {
            "protocol": "MCP/1.0",
            "message": injected_msg,
            "context": {
                "conversation_id": context.get("conv_id", "test-001"),
                "turn": context.get("turn", 1),
                "tools_available": context.get("tools", []),
                "memory": context.get("memory", [])
            }
        }
    
    def receive_message(self, response: Any) -> str:
        is_valid = False
        content = str(response)
        parsed_data = {}
        
        try:
            if isinstance(response, str):
                parsed_data = json.loads(response)
            elif isinstance(response, dict):
                parsed_data = response
                
            if parsed_data.get("protocol") == "MCP/1.0":
                content = parsed_data.get("content", content)
                is_valid = True
        except Exception:
            pass
        
        self._update_stats(parsed_data, is_valid)
        return content


class A2AProtocol(BaseProtocol):
    """Agent-to-Agent (A2A) Protocol Implementation"""
    
    def _get_protocol_instructions(self) -> str:
        return "\nFORMAT: Respond ONLY in valid A2A JSON: {\"payload\": {\"content\": \"your_response_here\"}}"

    def send_message(self, message: str, context: Dict = None) -> Dict:
        context = context or {}
        injected_msg = f"{message}\n{self._get_protocol_instructions()}"
        
        return {
            "protocol": "A2A/1.0",
            "payload": {
                "content": injected_msg,
                "task_id": context.get("task_id", "task-001")
            }
        }
    
    def receive_message(self, response: Any) -> str:
        is_valid = False
        content = str(response)
        parsed_data = {}
        
        try:
            if isinstance(response, str):
                parsed_data = json.loads(response)
            elif isinstance(response, dict):
                parsed_data = response
                
            if "payload" in parsed_data:
                content = parsed_data["payload"].get("content", content)
                is_valid = True
        except Exception:
            pass
        
        self._update_stats(parsed_data, is_valid)
        return content


class ACPProtocol(BaseProtocol):
    """Agent Communication Protocol (ACP) Implementation"""
    
    def _get_protocol_instructions(self) -> str:
        return "\nFORMAT: Respond ONLY in valid ACP/2.0 JSON: {\"message\": {\"content\": \"your_response_here\"}}"

    def send_message(self, message: str, context: Dict = None) -> Dict:
        context = context or {}
        injected_msg = f"{message}\n{self._get_protocol_instructions()}"
        
        return {
            "protocol_version": "ACP/2.0",
            "message": {
                "id": context.get("msg_id", "msg-001"),
                "content": injected_msg
            },
            "agent_context": {
                "capabilities": context.get("capabilities", ["reasoning", "tool_use"])
            }
        }
    
    def receive_message(self, response: Any) -> str:
        is_valid = False
        content = str(response)
        parsed_data = {}
        
        try:
            if isinstance(response, str):
                parsed_data = json.loads(response)
            elif isinstance(response, dict):
                parsed_data = response
                
            if "message" in parsed_data:
                content = parsed_data["message"].get("content", content)
                is_valid = True
        except Exception:
            pass
        
        self._update_stats(parsed_data, is_valid)
        return content