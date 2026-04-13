"""
LlamaIndex Agent Integration

LlamaIndex specializes in data-augmented LLM applications,
particularly for RAG use cases.
"""

import json
from typing import List, Dict, Callable, Optional
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
import sys
sys.path.append('..')
from domain_model import ReasoningStep
from tool_pool import ToolPool
from config import ConfigExpert
from workflow import WorkflowController

# FIXME: There is an ongoing issue with HF API calls in LlamaIndex
# that causes a 'description' response. There is a need to verify
# any conflict with added tools and protocols.

class LlamaIndexAgent:

    def __init__(self, adapter, protocol=None):
        self.adapter = adapter
        self.llm = adapter.get_llamaindex_llm()
        self.protocol = protocol
        self.tools: Dict[str, FunctionTool] = {}
        self.reasoning_steps: List[ReasoningStep] = []

        # Configure LlamaIndex settings
        Settings.llm = self.llm
        Settings.chunk_size = 512

        self._workflow_controller = WorkflowController(
            framework_name="LlamaIndex",
            generate_fn=lambda prompt, **kwargs: self.llm.complete(prompt).text,
            get_tool_payload_fn=self._get_tool_payload,
        )

    def _get_tool_payload(self) -> list:
        tool_payload = []
        for tool in self.tools.values():
            name = tool.metadata.name
            description = tool.metadata.description
            try:
                tool_payload.append(ToolPool.get_tool_schema(name))
            except KeyError:
                tool_payload.append({
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string", "description": "Tool input text."}},
                        "required": ["input"],
                    },
                })
        return tool_payload

    def add_tool(self, name: str, func: Optional[Callable] = None, description: Optional[str] = None):
        if name in ToolPool.available_tools:
            tool = ToolPool.get_tool(name, framework="llamaindex")
        else:
            if func is None:
                def _dummy(query: str) -> str:
                    result = f"Tool '{name}' executed with input: {query}"
                    self.reasoning_steps.append(ReasoningStep(
                        step_number=len(self.reasoning_steps) + 1,
                        thought=f"Executed tool: {name}",
                        action=name,
                        action_input=query,
                        observation=result
                    ))
                    return result
                func = _dummy
            tool = FunctionTool.from_defaults(
                fn=func,
                name=name,
                description=description or f"Custom tool: {name}"
            )
        self.tools[tool.metadata.name] = tool

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
                thought="Initializing LlamaIndex workflow",
                observation=f"Tools available: {list(self.tools.keys())}"
            ))

            state = self._workflow_controller.run(
                task=task,
                tools=self.tools,
                role="an AI agent",
                reasoning_steps=self.reasoning_steps,
            )
            exit_node_key = self._workflow_controller.final_output_key
            return state.get(exit_node_key, "No report generated")

        except Exception as e:
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Error during execution",
                observation=str(e)
            ))
            return f"LlamaIndex execution error: {str(e)}"
