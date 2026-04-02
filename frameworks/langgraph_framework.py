"""
LangGraph Stateful Workflow Integration

LangGraph provides graph-based agent orchestration with explicit state management.
"""

import json
from typing import List, Dict, Callable, Optional, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_core.tools import Tool
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
        self.tools: Dict[str, Tool] = {}
        self.reasoning_steps: List[ReasoningStep] = []
        self._workflow_controller = WorkflowController(
            framework_name="LangGraph",
            generate_fn=self.adapter.generate,
            get_tool_payload_fn=self._get_tool_payload,
        )
        # LangGraph StateGraph compiled once alongside the controller
        self._graph = self._build_langgraph()

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
            tool = Tool(name=name, func=func, description=description or f"Custom tool: {name}")
        self.tools[tool.name] = tool

    def _build_langgraph(self):
        """
        Compile a LangGraph StateGraph as a structural primitive.
        The graph mirrors the configured workflow nodes but delegates
        actual generation to WorkflowController.
        """
        config = ConfigExpert.get_instance()
        workflow_cfg = config.get("workflow", {})
        nodes = workflow_cfg.get("nodes", [])

        class AgentState(TypedDict):
            task: str
            plan: str
            research_results: Annotated[list, operator.add]
            final_report: str
            step: int
            max_steps: int

        def make_node(node_name):
            def _node(state: AgentState) -> AgentState:
                # Delegate to WorkflowController for actual execution
                return state
            _node.__name__ = node_name
            return _node

        def should_continue(state: AgentState) -> str:
            if state["step"] > state["max_steps"]:
                return "exit"
            return "loop"

        workflow = StateGraph(AgentState)
        node_names = [n["name"] for n in nodes]

        for name in node_names:
            workflow.add_node(name, make_node(name))

        if node_names:
            workflow.set_entry_point(node_names[0])
            for i in range(len(node_names) - 1):
                current = node_names[i]
                next_node = node_names[i + 1]
                loop_node = nodes[i].get("loop", False)
                if loop_node:
                    workflow.add_conditional_edges(
                        current, should_continue,
                        {"loop": current, "exit": next_node}
                    )
                else:
                    workflow.add_edge(current, next_node)
            workflow.add_edge(node_names[-1], END)

        return workflow.compile()

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
                observation="StateGraph compiled successfully"
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
            return f"LangGraph execution error: {str(e)}"
