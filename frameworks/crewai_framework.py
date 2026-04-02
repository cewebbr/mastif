"""
CrewAI Framework Integration

CrewAI focuses on role-based agent collaboration where each agent
has a specific role and expertise.
"""

import json
from typing import List, Dict, Callable, Optional
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
import sys
sys.path.append('..')
from domain_model import ReasoningStep
from tool_pool import ToolPool
from config import ConfigExpert
from workflow import WorkflowController


class CrewAIAgent:

    def __init__(self, adapter, role: str, protocol=None):
        self.adapter = adapter
        self.role = role
        self.protocol = protocol
        self.tools: Dict[str, BaseTool] = {}
        self.reasoning_steps: List[ReasoningStep] = []
        self._workflow_controller = WorkflowController(
            framework_name="CrewAI",
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
            tool = ToolPool.get_tool(name, framework="crewai")
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

            from pydantic import BaseModel, Field

            class _Input(BaseModel):
                query: str = Field(description="Input query or argument for the tool.")

            def _run(self_tool, query: str) -> str:
                return func(query)

            tool_attrs = {
                "__module__": __name__,
                "__annotations__": {"name": str, "description": str, "args_schema": type},
                "name": name,
                "description": description or f"Custom tool: {name}",
                "args_schema": _Input,
                "_run": _run,
            }
            tool = type(name, (BaseTool,), tool_attrs)()

        self.tools[tool.name] = tool

    def _build_crewai_primitives(self):
        """Instantiate CrewAI Agent/Task/Crew primitives for structural purposes."""
        tool_list = list(self.tools.values())
        config = ConfigExpert.get_instance()
        workflow_cfg = config.get("workflow", {})
        nodes = workflow_cfg.get("nodes", [])

        primitives = {}
        role_map = {
            "plan":     ("Planner",     "Create a detailed research plan for the given task"),
            "research": ("Researcher",  "Execute research steps and gather detailed findings"),
            "report":   ("Synthesizer", "Synthesize research findings into a comprehensive final report"),
        }
        for node in nodes:
            node_name = node["name"]
            agent_role, agent_goal = role_map.get(node_name, (node_name.capitalize(), f"Execute the {node_name} step"))
            agent = Agent(
                role=agent_role,
                goal=agent_goal,
                backstory=f"You are a {self.role} agent operating in the CrewAI framework, responsible for the {node_name} phase.",
                tools=tool_list,
                allow_delegation=False,
                verbose=False
            )
            primitives[node_name] = agent
        return primitives

    def execute_task(self, task: str, context: Dict = None) -> str:
        if self.protocol:
            formatted_msg = self.protocol.send_message(task, context or {})
            task = f"""Protocol: {self.protocol.__class__.__name__}

{json.dumps(formatted_msg, indent=2)}

Execute according to protocol."""

        self.reasoning_steps = []

        try:
            # Instantiate CrewAI primitives for structural purposes
            crewai_agents = self._build_crewai_primitives()

            self.reasoning_steps.append(ReasoningStep(
                step_number=1,
                thought="Initializing CrewAI workflow",
                observation=f"CrewAI agents ready: {list(crewai_agents.keys())}"
            ))

            state = self._workflow_controller.run(
                task=task,
                tools=self.tools,
                role=self.role,
                reasoning_steps=self.reasoning_steps,
            )
            return state.get("final_report", "No report generated")

        except Exception as e:
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Error during workflow execution",
                observation=str(e)
            ))
            return f"CrewAI execution error: {str(e)}"
