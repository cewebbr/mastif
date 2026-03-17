"""
LangChain Agent Integration
s
This module defines a LangChainAgent class that integrates with the LangChain framework
"""

import json
from typing import List, Dict, Callable, Optional
from langchain_community.tools import Tool
import sys
sys.path.append('..')
from domain_model import ReasoningStep
from tool_pool import ToolPool
from config import ConfigExpert


class LangChainAgent:
    """
    LangChain Stateful Workflow Integration

    LangChain provides chain-based agent orchestration with explicit
    state management. Ideal for complex, multi-step workflows
    with conditional branching.
    """

    def __init__(self, adapter, protocol=None):
        """
        Initialize LangChain agent

        Args:
            adapter: HuggingFace model adapter
            protocol: Optional protocol object
        """
        self.adapter = adapter
        self.protocol = protocol
        self.chain = None
        self.tools: Dict[str, Tool] = {}
        self.reasoning_steps: List[ReasoningStep] = []

    def add_tool(self, name: str, func: Optional[Callable] = None, description: Optional[str] = None):
        """
        Add a tool to the agent.

        For builtin tools available in the shared pool ("web_search", "web_browser",
        "wikipedia"), pass only the name — the tool is cloned from ToolPool.

        For custom tools, provide both func and description.

        Args:
            name:        Tool name. Use a pool tool name or any custom name.
            func:        Callable for custom tools. Ignored for pool tools.
            description: Description for custom tools. Ignored for pool tools.
        """
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

            tool = Tool(
                name=name,
                func=func,
                description=description or f"Custom tool: {name}"
            )

        self.tools[tool.name] = tool

    def build_workflow(self):
        """
        Build a multi-step workflow using LangChain

        The workflow consists of:
        1. Planning: Create research plan
        2. Research: Execute research steps (iterative)
        3. Synthesis: Compile final report
        """

        # Node 1: Planning
        def planning_node(state: dict) -> dict:
            """Create a research plan"""
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Creating research plan",
                action="plan",
                action_input=state['task']
            ))

            tools_text = (
                "\n".join(f"- {t.name}: {t.description}" for t in state["tools"].values())
                if state["tools"] else "None"
            )
            prompt = f"""You are an AI agent operating in the LangChain framework.

Task:
{state['task']}

Available Tools:
{tools_text}

Instructions:
• Think step-by-step.
• Create a detailed research plan for this task.
• Use available tools where appropriate.
• Do not skip steps.
• Make intermediate decisions explicit.
• If information is missing, state assumptions clearly.
• If the output format is not provided in the task, favor correctness and completeness over brevity.
"""

            try:
                config = ConfigExpert.get_instance()
                plan = self.adapter.generate(prompt, max_tokens=config.get("max_tokens", 1024))
                state["plan"] = plan
                state["step"] = 1

                self.reasoning_steps.append(ReasoningStep(
                    step_number=len(self.reasoning_steps) + 1,
                    thought="Research plan created",
                    observation=f"Plan generated with {len(plan)} characters"
                ))
            except Exception as e:
                state["plan"] = f"Planning error: {str(e)}"
                self.reasoning_steps.append(ReasoningStep(
                    step_number=len(self.reasoning_steps) + 1,
                    thought="Error in planning phase",
                    observation=str(e)
                ))
            return state

        # Node 2: Research
        def research_node(state: dict) -> dict:
            """Execute research step"""
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought=f"Executing research step {state['step']}",
                action="research",
                action_input=f"Step {state['step']} of plan"
            ))

            tools_text = (
                "\n".join(f"- {t.name}: {t.description}" for t in state["tools"].values())
                if state["tools"] else "None"
            )
            prompt = f"""You are an AI agent operating in the LangChain framework.

Task:
{state['task']}

Plan:
{state['plan']}

Step:
{state['step']}

Available Tools:
{tools_text}

Instructions:
• Execute this step carefully.
• Provide detailed findings.
• Use available tools where appropriate.
• Do not skip steps.
• Make intermediate decisions explicit.
• If information is missing, state assumptions clearly.
• If the output format is not provided in the task, favor correctness and completeness over brevity.
"""

            try:
                config = ConfigExpert.get_instance()
                findings = self.adapter.generate(prompt, max_tokens=config.get("max_tokens", 1024))
                state["research_results"] = state.get("research_results", []) + [findings]
                state["step"] += 1

                self.reasoning_steps.append(ReasoningStep(
                    step_number=len(self.reasoning_steps) + 1,
                    thought=f"Research step {state['step']-1} completed",
                    observation=f"Findings: {findings[:100]}..."
                ))
            except Exception as e:
                state["research_results"] = state.get("research_results", []) + [f"Research error: {str(e)}"]
                self.reasoning_steps.append(ReasoningStep(
                    step_number=len(self.reasoning_steps) + 1,
                    thought="Error in research phase",
                    observation=str(e)
                ))
            return state

        # Node 3: Synthesis
        def synthesis_node(state: dict) -> dict:
            """Synthesize final report"""
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Synthesizing research findings into final report",
                action="synthesize",
                action_input=f"{len(state['research_results'])} research findings"
            ))

            results_text = "\n\n".join(state["research_results"])
            prompt = f"""
You are an AI agent operating in the LangChain framework.

Task:
{state['task']}

Findings:
{results_text}

Instructions:
• Synthesize these findings into a comprehensive final report.
• Do not skip steps.
• Make intermediate decisions explicit.
• If information is missing, state assumptions clearly.
• If the output format is not provided in the task, favor correctness and completeness over brevity.
"""

            try:
                config = ConfigExpert.get_instance()
                report = self.adapter.generate(prompt, max_tokens=config.get("max_tokens", 1024))
                state["final_report"] = report

                self.reasoning_steps.append(ReasoningStep(
                    step_number=len(self.reasoning_steps) + 1,
                    thought="Final report synthesized successfully",
                    observation=f"Report generated with {len(report)} characters"
                ))
            except Exception as e:
                state["final_report"] = f"Synthesis error: {str(e)}"
                self.reasoning_steps.append(ReasoningStep(
                    step_number=len(self.reasoning_steps) + 1,
                    thought="Error in synthesis phase",
                    observation=str(e)
                ))
            return state

        # Conditional function
        def should_continue(state: dict) -> bool:
            """Decide whether to continue research or synthesize"""
            if state["step"] > 2:  # Limit to 2 research iterations
                self.reasoning_steps.append(ReasoningStep(
                    step_number=len(self.reasoning_steps) + 1,
                    thought="Research iterations complete, moving to synthesis",
                    observation=f"Completed {state['step']-1} research steps"
                ))
                return False

            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought=f"Continuing research (step {state['step']})",
                observation="More research needed"
            ))
            return True

        def workflow(state: dict) -> dict:
            state = planning_node(state)
            state = research_node(state)
            while should_continue(state):
                state = research_node(state)
            state = synthesis_node(state)
            return state

        self.chain = workflow

    def run(self, task: str) -> str:
        """
        Execute the LangChain workflow

        Args:
            task: Research task to execute

        Returns:
            Final research report
        """
        # Wrap task with protocol if provided
        if self.protocol:
            formatted_msg = self.protocol.send_message(task, {})
            task = f"""Protocol: {self.protocol.__class__.__name__}

{json.dumps(formatted_msg, indent=2)}

Execute according to protocol."""

        self.reasoning_steps = []

        try:
            if self.chain is None:
                self.build_workflow()

            self.reasoning_steps.append(ReasoningStep(
                step_number=1,
                thought="Initializing LangChain workflow",
                observation="Chain compiled successfully"
            ))

            initial_state = {
                "task": task,
                "plan": "",
                "research_results": [],
                "final_report": "",
                "step": 0,
                "tools": self.tools
            }

            result = self.chain(initial_state)
            return result.get("final_report", "No report generated")

        except Exception as e:
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Error during workflow execution",
                observation=str(e)
            ))
            return f"LangChain execution error: {str(e)}"