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
from config import ConfigExpert
from workflow import WorkflowController


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
        self.tools: Dict[str, KernelFunction] = {}
        self.reasoning_steps: List[ReasoningStep] = []

        # Use InferenceClient for remote Hugging Face inference
        self.inference_client = InferenceClient(
            model=adapter.model_name,
            token=adapter.api_key if hasattr(adapter, "api_key") else None
        )

        # Register workflow nodes as semantic functions in the kernel
        self._register_workflow_functions()

        self._workflow_controller = WorkflowController(
            framework_name="Semantic Kernel",
            generate_fn=self._sk_generate,
            get_tool_payload_fn=self._get_tool_payload,
        )

    def _sk_generate(self, prompt: str, **kwargs) -> str:
        """Generate via InferenceClient chat_completion."""
        return self.inference_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", 1024)
        ).choices[0].message.content

    def _register_workflow_functions(self):
        """Register each configured workflow node as a named SK kernel function."""
        config = ConfigExpert.get_instance()
        workflow_cfg = config.get("workflow", {})
        nodes = workflow_cfg.get("nodes", [])
        for node in nodes:
            self.kernel.add_function(
                plugin_name="workflow",
                function_name=node["name"],
                prompt=f"Execute the {node['name']} step for the given task.",
                description=f"Workflow node: {node['name']}",
                prompt_execution_settings=PromptExecutionSettings(
                    max_tokens=config.get("max_tokens", 1024)
                )
            )

    def _get_tool_payload(self) -> list:
        tool_payload = []
        for name in self.tools:
            try:
                tool_payload.append(ToolPool.get_tool_schema(name))
            except KeyError:
                tool_payload.append({
                    "name": name,
                    "description": "Custom tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string", "description": "Tool input text."}},
                        "required": ["input"],
                    },
                })
        return tool_payload

    def add_tool(self, name: str, func: Optional[Callable] = None, description: Optional[str] = None):
        if name in ToolPool.available_tools:
            kernel_fn = ToolPool.get_tool(name, framework="semantic_kernel")
            self.kernel.add_plugin(
                plugin=type(name, (), {name: staticmethod(kernel_fn)})(),
                plugin_name=name
            )
            self.tools[name] = kernel_fn
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
            self.reasoning_steps.append(ReasoningStep(
                step_number=1,
                thought="Initializing Semantic Kernel workflow",
                observation="Kernel configured with HuggingFace InferenceClient"
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
                thought="Error during Semantic Kernel execution",
                observation=str(e)
            ))
            return f"Semantic Kernel execution error: {str(e)}"
