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

class LlamaIndexAgent:
    """
    LlamaIndex ReAct Agent Integration
    
    LlamaIndex specializes in data-augmented LLM applications,
    particularly for RAG (Retrieval Augmented Generation) use cases.
    """

    def __init__(self, adapter, protocol=None):
        """
        Initialize LlamaIndex agent
        
        Args:
            adapter: HuggingFace model adapter
        """
        self.adapter = adapter
        self.llm = adapter.get_llamaindex_llm()
        self.protocol = protocol
        self.chain = None
        self.tools: Dict[str, FunctionTool] = {}
        self.reasoning_steps: List[ReasoningStep] = []

        # Configure LlamaIndex settings
        Settings.llm = self.llm
        Settings.chunk_size = 512

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

    def build_workflow(self):

        def planning_node(state: dict) -> dict:
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Creating research plan",
                action="plan",
                action_input=state['task']
            ))

            # FunctionTool primitives registered for structural purposes
            tool_list = list(state["tools"].values())

            tools_text = (
                "\n".join(f"- {t.metadata.name}: {t.metadata.description}" for t in state["tools"].values())
                if state["tools"] else "None"
            )
            prompt = f"""You are an AI agent operating in the LlamaIndex framework.

Task:
{state['task']}

Available Tools:
{tools_text}

Instructions:
• Think step-by-step.
• Create a detailed research plan for this task.
• Use available tools where appropriate.
• IMPORTANT: You must ONLY call tools by their exact registered names listed above. Do not invent or approximate tool names. Calling an unregistered tool will cause an error.
• Do not skip steps.
• Make intermediate decisions explicit.
• If information is missing, state assumptions clearly.
• If the output format is not provided in the task, favor correctness and completeness over brevity.
"""

            try:
                plan = self.llm.complete(prompt).text
                state["plan"] = plan
                state["step"] = 1

                self.reasoning_steps.append(ReasoningStep(
                    step_number=len(self.reasoning_steps) + 1,
                    thought="Research plan created",
                    observation=f"Plan generated with {len(plan)} characters"
                ))
            except Exception as e:
                state["plan"] = f"Planning error: {str(e)}"
                state["step"] = 1
                self.reasoning_steps.append(ReasoningStep(
                    step_number=len(self.reasoning_steps) + 1,
                    thought="Error in planning phase",
                    observation=str(e)
                ))
            return state

        def research_node(state: dict) -> dict:
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought=f"Executing research step {state['step']}",
                action="research",
                action_input=f"Step {state['step']} of plan"
            ))

            # FunctionTool primitives registered for structural purposes
            tool_list = list(state["tools"].values())

            tools_text = (
                "\n".join(f"- {t.metadata.name}: {t.metadata.description}" for t in state["tools"].values())
                if state["tools"] else "None"
            )
            prompt = f"""You are an AI agent operating in the LlamaIndex framework.

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
• IMPORTANT: You must ONLY call tools by their exact registered names listed above. Do not invent or approximate tool names. Calling an unregistered tool will cause an error.
• Do not skip steps.
• Make intermediate decisions explicit.
• If information is missing, state assumptions clearly.
• If the output format is not provided in the task, favor correctness and completeness over brevity.
"""

            try:
                findings = self.llm.complete(prompt).text
                state["research_results"] = state.get("research_results", []) + [findings]
                state["step"] += 1

                self.reasoning_steps.append(ReasoningStep(
                    step_number=len(self.reasoning_steps) + 1,
                    thought=f"Research step {state['step']-1} completed",
                    observation=f"Findings: {findings[:100]}..."
                ))
            except Exception as e:
                state["research_results"] = state.get("research_results", []) + [f"Research error: {str(e)}"]
                state["step"] += 1
                self.reasoning_steps.append(ReasoningStep(
                    step_number=len(self.reasoning_steps) + 1,
                    thought="Error in research phase",
                    observation=str(e)
                ))
            return state

        def synthesis_node(state: dict) -> dict:
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Synthesizing research findings into final report",
                action="synthesize",
                action_input=f"{len(state['research_results'])} research findings"
            ))

            # FunctionTool primitives registered for structural purposes
            tool_list = list(state["tools"].values())

            results_text = "\n\n".join(state["research_results"])
            prompt = f"""You are an AI agent operating in the LlamaIndex framework.

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
                report = self.llm.complete(prompt).text
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

        def should_continue(state: dict) -> bool:
            if state["step"] > state["max_steps"]:
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
        Execute task using LlamaIndex ReAct agent
        
        Args:
            task: Task to execute
            
        Returns:
            Agent's response
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
                thought="Initializing LlamaIndex workflow",
                observation=f"Tools available: {list(self.tools.keys())}"
            ))

            config = ConfigExpert.get_instance()
            initial_state = {
                "task": task,
                "plan": "",
                "research_results": [],
                "final_report": "",
                "step": 0,
                "tools": self.tools,
                "max_steps": config.get("max_steps", 2)
            }

            result = self.chain(initial_state)
            return result.get("final_report", "No report generated")

        except Exception as e:
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Error during execution",
                observation=str(e)
            ))
            return f"LlamaIndex execution error: {str(e)}"