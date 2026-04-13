"""
LangGraph Stateful Workflow Integration

LangGraph provides graph-based agent orchestration with explicit state management.
"""

import json
from typing import List, Dict, Callable, Optional, TypedDict, Annotated
import operator
import sys
sys.path.append('..')
from domain_model import ReasoningStep
from config import ConfigExpert
from tool_pool import ToolPool
from workflow import WorkflowController


class LangGraphAgent:

    def __init__(self, adapter, protocol=None):
        self.adapter = adapter
        self.protocol = protocol
        self.tools: Dict[str, any] = {}
        self.reasoning_steps: List[ReasoningStep] = []
        self._workflow_controller = WorkflowController(
            framework_name="LangGraph",
            generate_fn=self.adapter.generate,
            get_tool_payload_fn=self._get_tool_payload,
        )

    def _get_tool_payload(self) -> list:
        tool_payload = []
        for tool in self.tools.values():
            name = getattr(tool, "name", str(tool))
            try:
                tool_payload.append(ToolPool.get_tool_schema(name))
            except KeyError:
                tool_payload.append({
                    "name": name,
                    "description": getattr(tool, "description", "Custom tool"),
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string", "description": "Tool input text."}},
                        "required": ["input"],
                    },
                })
        return tool_payload

    def add_tool(self, name: str, func: Optional[Callable] = None, description: Optional[str] = None):
        if not hasattr(self, "tools"):
            self.tools = {}
        if name in ToolPool.available_tools:
            # Note: We use the ToolPool to get the framework-specific tool instance
            from langchain_core.tools import Tool
            tool = ToolPool.get_tool(name, framework="langchain")
        else:
            if func is None:
                def _dummy(x):
                    result = f"Tool '{name}' executed with input: {x}"
                    self.reasoning_steps.append(ReasoningStep(
                        step_number=len(self.reasoning_steps) + 1,
                        thought=f"Executed tool: {name}",
                        action=name,
                        action_input=str(x),
                        observation=result
                    ))
                    return result
                func = _dummy
            from langchain_core.tools import Tool
            tool = Tool(name=name, func=func, description=description or f"Custom tool: {name}")
        self.tools[tool.name] = tool

    def run(self, task: str) -> str:
        if self.protocol:
            formatted_msg = self.protocol.send_message(task, {})
            task = f"""Protocol: {self.protocol.__class__.__name__}

{json.dumps(formatted_msg, indent=2)}

Execute according to protocol."""

        self.reasoning_steps = []

        try:
            self.reasoning_steps.append(ReasoningStep(
                step_number=1,
                thought="Initializing LangGraph workflow",
                observation="Workflow controller ready"
            ))

            state = self._workflow_controller.run(
                task=task,
                tools=self.tools,
                role="an AI agent",
                reasoning_steps=self.reasoning_steps,
            )
            
            # Retrieve the dynamic output key from the exit node
            exit_node_key = self._workflow_controller._nodes[self._workflow_controller._exit_node].output_key
            response = state.get(exit_node_key, "No report generated")

            # Format list-based results (like interaction history) as a string
            if isinstance(response, list):
                response = "\n\n".join(str(r) for r in response)
            
            return response

        except Exception as e:
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Error during workflow execution",
                observation=str(e)
            ))
            return f"LangGraph execution error: {str(e)}"