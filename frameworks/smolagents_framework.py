"""
Smolagents Framework Integration

Smolagents is a lightweight agent framework focusing on simplicity
and efficiency with minimal overhead.
"""

import json
from typing import List, Dict, Callable, Optional
from smolagents import CodeAgent, ToolCallingAgent, Tool
import sys
sys.path.append('..')
from domain_model import ReasoningStep
from tool_pool import ToolPool
from config import ConfigExpert

class SmolAgentWrapper:
    """
    Smolagents Framework Integration
    
    Smolagents is a lightweight agent framework focusing on simplicity
    and efficiency. It emphasizes minimal overhead while maintaining
    tool-use capabilities.
    """

    def __init__(self, adapter, protocol=None):
        """
        Initialize Smolagents wrapper
        
        Args:
            adapter: HuggingFace model adapter
        """
        self.adapter = adapter
        self.protocol = protocol
        self.chain = None
        self.tools: Dict[str, Tool] = {}
        self.reasoning_steps: List[ReasoningStep] = []

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

    def build_workflow(self):

        tool_list = list(self.tools.values())

        planning_agent = ToolCallingAgent(
            model=self.adapter,
            tools=tool_list,
            name="Planner",
            description="Specialized in breaking down complex tasks into actionable research plans."
        )

        research_agent = ToolCallingAgent(
            model=self.adapter,
            tools=tool_list,
            name="Researcher",
            description="Specialized in executing research steps and gathering detailed information."
        )

        synthesis_agent = ToolCallingAgent(
            model=self.adapter,
            tools=tool_list,
            name="Synthesizer",
            description="Specialized in synthesizing findings into coherent and complete reports."
        )

        def planning_node(state: dict) -> dict:
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
            prompt = f"""You are an AI agent operating in the Smolagents framework.
You have access to tools.

If a task requires external information, browsing, interaction, or computation,
you should use the appropriate tool instead of answering directly.

Do not guess when a tool is more appropriate.

Task:
{state['task']}

Available Tools:
{tools_text}

IMPORTANT:
- If the task requires external information, you MUST use a tool.
- Do NOT answer from memory if tools are available.
- Always prefer tool usage over guessing.

Use this format:

Thought: ...
Action: tool_name
Action Input: ...
Observation: ...
... (repeat as needed)
Final Answer: ...

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
                planning_agent.tools = state["tools"]
                plan = self.adapter.generate(prompt, max_tokens=1024, tools=tools_text)
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

        def research_node(state: dict) -> dict:
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
            prompt = f"""You are an AI agent operating in the Smolagents framework.
You have access to tools.

If a task requires external information, browsing, interaction, or computation,
you should use the appropriate tool instead of answering directly.

Do not guess when a tool is more appropriate.

Task:
{state['task']}

Plan:
{state['plan']}

Step:
{state['step']}

Available Tools:
{tools_text}

IMPORTANT:
- If the task requires external information, you MUST use a tool.
- Do NOT answer from memory if tools are available.
- Always prefer tool usage over guessing.

Use this format:

Thought: ...
Action: tool_name
Action Input: ...
Observation: ...
... (repeat as needed)
Final Answer: ...

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
                research_agent.tools = state["tools"]
                findings = self.adapter.generate(prompt, max_tokens=1024, tools=tools_text)
                state["research_results"] = state.get("research_results", []) + [findings]
                state["step"] += 1

                self.reasoning_steps.append(ReasoningStep(
                    step_number=len(self.reasoning_steps) + 1,
                    thought=f"Research step {state['step']-1} completed",
                    observation=f"Findings: {findings}"
                ))
            except Exception as e:
                state["research_results"] = state.get("research_results", []) + [f"Research error: {str(e)}"]
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

            results_text = "\n\n".join(state["research_results"])
            prompt = f"""You are an AI agent operating in the Smolagents framework.

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
                synthesis_agent.tools = state["tools"]
                report = self.adapter.generate(prompt, max_tokens=1024)
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
        Execute task using Smolagents approach
        
        Args:
            task: Task to execute
            
        Returns:
            Execution result
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
                thought="Initializing Smolagents workflow",
                observation="Chain compiled successfully"
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
                thought="Error during workflow execution",
                observation=str(e)
            ))
            return f"Smolagents execution error: {str(e)}"