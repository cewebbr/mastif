"""
Semantic Kernel Agent Integration

Semantic Kernel (by Microsoft) provides enterprise-grade AI orchestration.
"""

import json
from typing import List, Dict, Callable, Optional
import semantic_kernel as sk
from semantic_kernel.functions import KernelFunction
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from huggingface_hub import InferenceClient
import sys
sys.path.append('..')
from domain_model import ReasoningStep
from tool_pool import ToolPool


class SemanticKernelAgent:
    """
    Semantic Kernel Agent Integration
    
    Semantic Kernel (by Microsoft) provides enterprise-grade AI orchestration
    with a focus on skills/plugins and memory management.
    """

    def __init__(self, adapter, protocol=None):
        """
        Initialize Semantic Kernel agent
        
        Args:
            adapter: HuggingFace model adapter
        """
        self.adapter = adapter
        self.kernel = sk.Kernel()
        self.protocol = protocol
        self.chain = None
        self.tools: Dict[str, KernelFunction] = {}
        self.reasoning_steps: List[ReasoningStep] = []

        # Use InferenceClient for remote Hugging Face inference
        self.inference_client = InferenceClient(
            model=adapter.model_name,
            token=adapter.api_key if hasattr(adapter, "api_key") else None
        )

    def add_tool(self, name: str, func: Optional[Callable] = None, description: Optional[str] = None):
        if name in ToolPool.available_tools:
            # Store pool tool definition; SK wraps it as a KernelFunction via plugin
            tool_def = ToolPool._registry[name]

            # Register as a native kernel function inside an inline plugin
            @sk.kernel_function(name=name, description=tool_def.description)
            def _kernel_func(query: str) -> str:
                return tool_def.func(query)

            plugin = self.kernel.add_plugin(
                plugin=type(name, (), {name: staticmethod(_kernel_func)})(),
                plugin_name=name
            )
            self.tools[name] = plugin[name]
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

            @sk.kernel_function(name=name, description=description or f"Custom tool: {name}")
            def _kernel_func(query: str) -> str:
                return func(query)

            plugin = self.kernel.add_plugin(
                plugin=type(name, (), {name: staticmethod(_kernel_func)})(),
                plugin_name=name
            )
            self.tools[name] = plugin[name]

    def add_semantic_function(self, name: str, prompt_template: str, description: str):
        """
        Add a semantic function (AI-powered) to the kernel
        
        Args:
            name: Function name
            prompt_template: Prompt template with {{$input}} placeholder
            description: Function description
        """
        # Store function info for later use
        if not hasattr(self, "semantic_functions"):
            self.semantic_functions = {}
        self.semantic_functions[name] = {
            "prompt_template": prompt_template,
            "description": description
        }

    def build_research_workflow(self):

        def planning_node(state: dict) -> dict:
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Creating research plan",
                action="plan",
                action_input=state['task']
            ))

            tools_text = (
                "\n".join(f"- {name}: {fn.description}" for name, fn in state["tools"].items())
                if state["tools"] else "None"
            )
            prompt = f"""You are an AI agent operating in the Semantic Kernel framework.

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
                # Register prompt as a semantic function in the kernel
                planning_fn = self.kernel.add_function(
                    plugin_name="research_workflow",
                    function_name="planning",
                    prompt=prompt,
                    description="Creates a detailed research plan for the given task.",
                    prompt_execution_settings=PromptExecutionSettings(max_tokens=1024)
                )

                # Use InferenceClient for remote Hugging Face inference
                plan = self.inference_client.text_generation(prompt, max_new_tokens=1024)
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

            tools_text = (
                "\n".join(f"- {name}: {fn.description}" for name, fn in state["tools"].items())
                if state["tools"] else "None"
            )
            prompt = f"""You are an AI agent operating in the Semantic Kernel framework.

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
                # Register prompt as a semantic function in the kernel
                research_fn = self.kernel.add_function(
                    plugin_name="research_workflow",
                    function_name=f"research_step_{state['step']}",
                    prompt=prompt,
                    description=f"Executes research step {state['step']} of the plan.",
                    prompt_execution_settings=PromptExecutionSettings(max_tokens=1024)
                )

                # Use InferenceClient for remote Hugging Face inference
                findings = self.inference_client.text_generation(prompt, max_new_tokens=1024)
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

            results_text = "\n\n".join(state["research_results"])
            prompt = f"""You are an AI agent operating in the Semantic Kernel framework.

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
                # Register prompt as a semantic function in the kernel
                synthesis_fn = self.kernel.add_function(
                    plugin_name="research_workflow",
                    function_name="synthesis",
                    prompt=prompt,
                    description="Synthesizes all research findings into a comprehensive final report.",
                    prompt_execution_settings=PromptExecutionSettings(max_tokens=1024)
                )

                # Use InferenceClient for remote Hugging Face inference
                report = self.inference_client.text_generation(prompt, max_new_tokens=1024)
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
        Execute task using Hugging Face InferenceClient
        
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
                self.build_research_workflow()

            self.reasoning_steps.append(ReasoningStep(
                step_number=1,
                thought="Initializing Semantic Kernel workflow",
                observation="Kernel configured with HuggingFace InferenceClient"
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
                thought="Error during Semantic Kernel execution",
                observation=str(e)
            ))
            return f"Semantic Kernel execution error: {str(e)}"