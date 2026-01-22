"""
Semantic Kernel Agent Integration

Semantic Kernel (by Microsoft) provides enterprise-grade AI orchestration.
"""

import json
from typing import List
import semantic_kernel as sk
from huggingface_hub import InferenceClient
import sys
sys.path.append('..')
from domain_model import ReasoningStep


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
        self.reasoning_steps: List[ReasoningStep] = []
        self.protocol = protocol

        # Use InferenceClient for remote Hugging Face inference
        self.inference_client = InferenceClient(
            model=adapter.model_name,
            token=adapter.api_key if hasattr(adapter, "api_key") else None
        )
    
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
                thought="Initializing Semantic Kernel execution",
                observation="Kernel configured with HuggingFace InferenceClient"
            ))
            
            # Use a reasoning prompt
            reasoning_prompt = """You are an AI assistant operating in the Semantic Kernel framework.

Task:
{input}

Instructions:
• Think through this step-by-step:
  1. Understand the task requirements
  2. Break down the approach
  3. Execute and provide the answer
• Do not skip steps.
• Make intermediate decisions explicit.
• If information is missing, state assumptions clearly.
• If the output format is not provided in the task, favor correctness and completeness over brevity.
  
Response:
Reasoning:
Final Answer:
"""

            
            self.reasoning_steps.append(ReasoningStep(
                step_number=2,
                thought="Preparing prompt for remote inference",
                action="prepare_prompt",
                action_input="reasoning_prompt"
            ))

            # Format the prompt
            prompt = reasoning_prompt.replace("{input}", task)
            
            self.reasoning_steps.append(ReasoningStep(
                step_number=3,
                thought="Sending prompt to HuggingFace InferenceClient",
                action="inference_client.text_generation",
                action_input=prompt
            ))
            
            # Execute the function using InferenceClient
            result = self.inference_client.text_generation(prompt, max_new_tokens=1024, temperature=0.7)
            
            self.reasoning_steps.append(ReasoningStep(
                step_number=4,
                thought="Task execution completed",
                observation=f"Result generated: {str(result)[:100]}..."
            ))
            
            return str(result)
        
        except Exception as e:
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Error during Semantic Kernel execution",
                observation=str(e)
            ))
            return f"Semantic Kernel execution error: {str(e)}"