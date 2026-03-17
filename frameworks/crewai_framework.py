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

class CrewAIAgent:

    def __init__(self, adapter, role: str, protocol=None):
        """
        Initialize CrewAI agent with specific role
        
        Args:
            adapter: HuggingFace model adapter
            role: Agent's role/expertise (e.g., "Research Analyst")
        """
        self.adapter = adapter
        self.role = role
        self.protocol = protocol
        self.chain = None
        self.tools: Dict[str, BaseTool] = {}
        self.reasoning_steps: List[ReasoningStep] = []

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

            tool = type(
                name,
                (BaseTool,),
                {
                    "name": name,
                    "description": description or f"Custom tool: {name}",
                    "args_schema": _Input,
                    "_run": _run,
                },
            )()

        self.tools[tool.name] = tool

    def build_research_workflow(self):
        tool_list = list(self.tools.values())

        planner_agent = Agent(
            role="Planner",
            goal="Create a detailed research plan for the given task",
            backstory=f"You are a {self.role} agent operating in the CrewAI framework, responsible for planning research.",
            tools=tool_list,
            allow_delegation=False,
            verbose=False
        )

        researcher_agent = Agent(
            role="Researcher",
            goal="Execute research steps and gather detailed findings",
            backstory=f"You are a {self.role} agent operating in the CrewAI framework, responsible for executing research.",
            tools=tool_list,
            allow_delegation=False,
            verbose=False
        )

        synthesizer_agent = Agent(
            role="Synthesizer",
            goal="Synthesize research findings into a comprehensive final report",
            backstory=f"You are a {self.role} agent operating in the CrewAI framework, responsible for synthesizing findings.",
            tools=tool_list,
            allow_delegation=False,
            verbose=False
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
            prompt = f"""You are a {self.role} agent operating in the CrewAI framework.

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
                planning_task = Task(
                    description=prompt,
                    expected_output="A detailed research plan",
                    agent=planner_agent
                )
                Crew(
                    agents=[planner_agent],
                    tasks=[planning_task],
                    process=Process.sequential,
                    verbose=False
                )
                config = ConfigExpert.get_instance()
                plan = self.adapter.generate(prompt, config.get("max_tokens", 1024))
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
            prompt = f"""You are a {self.role} agent operating in the CrewAI framework.

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
                research_task = Task(
                    description=prompt,
                    expected_output="Detailed findings for this research step",
                    agent=researcher_agent
                )
                Crew(
                    agents=[researcher_agent],
                    tasks=[research_task],
                    process=Process.sequential,
                    verbose=False
                )
                config = ConfigExpert.get_instance()
                findings = self.adapter.generate(prompt, config.get("max_tokens", 1024))
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

        def synthesis_node(state: dict) -> dict:
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Synthesizing research findings into final report",
                action="synthesize",
                action_input=f"{len(state['research_results'])} research findings"
            ))

            results_text = "\n\n".join(state["research_results"])
            prompt = f"""You are a {self.role} agent operating in the CrewAI framework.

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
                synthesis_task = Task(
                    description=prompt,
                    expected_output="A comprehensive final report synthesizing all findings",
                    agent=synthesizer_agent
                )
                Crew(
                    agents=[synthesizer_agent],
                    tasks=[synthesis_task],
                    process=Process.sequential,
                    verbose=False
                )
                config = ConfigExpert.get_instance()
                report = self.adapter.generate(prompt, config.get("max_tokens", 1024))
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

    def execute_task(self, task: str, context: Dict = None) -> str:
        if self.protocol:
            formatted_msg = self.protocol.send_message(task, context or {})
            task = f"""Protocol: {self.protocol.__class__.__name__}

{json.dumps(formatted_msg, indent=2)}

Execute according to protocol."""

        self.reasoning_steps = []

        try:
            if self.chain is None:
                self.build_research_workflow()

            self.reasoning_steps.append(ReasoningStep(
                step_number=1,
                thought="Initializing CrewAI workflow",
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
            return f"CrewAI execution error: {str(e)}"