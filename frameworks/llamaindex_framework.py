"""
LlamaIndex ReAct Agent Integration

LlamaIndex specializes in data-augmented LLM applications,
particularly for RAG use cases.
"""

import json
from typing import List, Dict, Callable, Optional
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
import sys
sys.path.append('..')
from domain_model import ReasoningStep
from tool_pool import ToolPool

# FIXME: LlamaIndex execution error: from_tools

class LlamaIndexAgent:
    """
    LlamaIndex ReAct Agent Integration
    
    LlamaIndex specializes in data-augmented LLM applications,
    particularly for RAG (Retrieval Augmented Generation) use cases.
    """

    # FIXME: There is an ongoing issue with HF API calls in LlamaIndex
    # that causes a 'description' response. There is a need to verify 
    # any conflict with added tools and protocols.
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

    def _get_agent(self, tools: dict) -> ReActAgent:
        """Return a ReActAgent with the given tools, inserting a dummy tool if the list is empty."""
        tool_list = list(tools.values())
        if not tool_list:
            def _noop(query: str) -> str:
                return f"No tools available. Query was: {query}"
            tool_list = [FunctionTool.from_defaults(
                fn=_noop,
                name="no_op",
                description="Placeholder tool used when no tools are configured."
            )]
        return ReActAgent.from_tools(
            tool_list,
            verbose=False,
            max_iterations=5
        )

    def build_research_workflow(self):

        def planning_node(state: dict) -> dict:
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Creating research plan",
                action="plan",
                action_input=state['task']
            ))

            # Create ReActAgent as the LlamaIndex primitive for this node
            planning_agent = self._get_agent(state["tools"])

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
• Do not skip steps.
• Make intermediate decisions explicit.
• If information is missing, state assumptions clearly.
• If the output format is not provided in the task, favor correctness and completeness over brevity.
"""

            try:
                plan = str(planning_agent.chat(prompt))
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

            # Create ReActAgent as the LlamaIndex primitive for this node
            research_agent = self._get_agent(state["tools"])

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
• Do not skip steps.
• Make intermediate decisions explicit.
• If information is missing, state assumptions clearly.
• If the output format is not provided in the task, favor correctness and completeness over brevity.
"""

            try:
                findings = str(research_agent.chat(prompt))
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

            # Create ReActAgent as the LlamaIndex primitive for this node
            synthesis_agent = self._get_agent(state["tools"])

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
                report = str(synthesis_agent.chat(prompt))
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
            if state["step"] > 2:
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
                self.build_research_workflow()

            self.reasoning_steps.append(ReasoningStep(
                step_number=1,
                thought="Initializing LlamaIndex workflow",
                observation=f"Tools available: {list(self.tools.keys())}"
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
                thought="Error during execution",
                observation=str(e)
            ))
            return f"LlamaIndex execution error: {str(e)}"