"""
Smolagents Framework Integration

Smolagents is a lightweight agent framework focusing on simplicity
and efficiency with minimal overhead.
"""

import json
from typing import List, Dict, Callable, Optional
from smolagents import ToolCallingAgent, Tool
import sys
sys.path.append('..')
from domain_model import ReasoningStep
from tool_pool import ToolPool
from config import ConfigExpert
from workflow import WorkflowController


class SmolAgentWrapper:

    def __init__(self, adapter, protocol=None):
        self.adapter = adapter
        self.protocol = protocol
        self.tools: Dict[str, Tool] = {}
        self.reasoning_steps: List[ReasoningStep] = []
        self._workflow_controller = WorkflowController(
            framework_name="Smolagents",
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
        if name in ToolPool.available_tools:
            tool = ToolPool.get_tool(name, framework="smolagents")
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

            class _CustomTool(Tool):
                inputs = {"query": {"type": "string", "description": "Input query or argument."}}
                output_type = "string"

                def forward(self_tool, query: str) -> str:
                    return func(query)

            _CustomTool.name = name
            _CustomTool.description = description or f"Custom tool: {name}"
            tool = _CustomTool()

        self.tools[tool.name] = tool

    def _build_smolagents_primitives(self):
        """Instantiate ToolCallingAgent primitives for structural purposes."""
        tool_list = list(self.tools.values())
        config = ConfigExpert.get_instance()
        workflow_cfg = config.get("workflow", {})
        nodes = workflow_cfg.get("nodes", [])

        primitives = {}
        descriptions = {
            "plan":     "Specialized in breaking down complex tasks into actionable research plans.",
            "research": "Specialized in executing research steps and gathering detailed information.",
            "report":   "Specialized in synthesizing findings into coherent and complete reports.",
        }
        for node in nodes:
            node_name = node["name"]
            primitives[node_name] = ToolCallingAgent(
                model=self.adapter,
                tools=tool_list,
                name=node_name.capitalize(),
                description=descriptions.get(node_name, f"Handles the {node_name} phase.")
            )
        return primitives

    def run(self, task: str) -> str:
        if self.protocol:
            formatted_msg = self.protocol.send_message(task, {})
            task = f"""Protocol: {self.protocol.__class__.__name__}

{json.dumps(formatted_msg, indent=2)}

Execute according to protocol."""

        self.reasoning_steps = []

        try:
            # Instantiate Smolagents primitives for structural purposes
            smolagents_agents = self._build_smolagents_primitives()

            self.reasoning_steps.append(ReasoningStep(
                step_number=1,
                thought="Initializing Smolagents workflow",
                observation=f"ToolCallingAgents ready: {list(smolagents_agents.keys())}"
            ))

            state = self._workflow_controller.run(
                task=task,
                tools=self.tools,
                role="an AI agent",
                reasoning_steps=self.reasoning_steps,
            )
            return state.get("final_report", "No report generated")

        except Exception as e:
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Error during workflow execution",
                observation=str(e)
            ))
            return f"Smolagents execution error: {str(e)}"
