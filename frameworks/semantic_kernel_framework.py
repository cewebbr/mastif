"""
Semantic Kernel Agent Integration

Semantic Kernel (by Microsoft) provides enterprise-grade AI orchestration.
"""

from typing import List
import semantic_kernel as sk
from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextCompletion
import sys
sys.path.append('..')
from models import ReasoningStep


class SemanticKernelAgent:
    """
    Semantic Kernel Agent Integration
    
    Semantic Kernel (by Microsoft) provides enterprise-grade AI orchestration
    with a focus on skills/plugins and memory management.
    """
    
    def __init__(self, adapter):
        """
        Initialize Semantic Kernel agent
        
        Args:
            adapter: HuggingFace model adapter
        """
        self.adapter = adapter
        self.kernel = sk.Kernel()
        self.reasoning_steps: List[ReasoningStep] = []
        
        # Add HuggingFace service to kernel
        self.kernel.add_text_completion_service(
            "huggingface",
            HuggingFaceTextCompletion(
                model_id=adapter.model_name,
                task="text-generation"
            )
        )
    
    def add_semantic_function(self, name: str, prompt_template: str, description: str):
        """
        Add a semantic function (AI-powered) to the kernel
        
        Args:
            name: Function name
            prompt_template: Prompt template with {{$input}} placeholder
            description: Function description
        """
        self.kernel.create_semantic_function(
            prompt_template=prompt_template,
            function_name=name,
            skill_name="custom_skills",
            description=description,
            max_tokens=300,
            temperature=0.7
        )
    
    def run(self, task: str) -> str:
        """
        Execute task using Semantic Kernel
        
        Args:
            task: Task to execute
            
        Returns:
            Execution result
        """
        self.reasoning_steps = []
        
        try:
            self.reasoning_steps.append(ReasoningStep(
                step_number=1,
                thought="Initializing Semantic Kernel execution",
                observation="Kernel configured with HuggingFace service"
            ))
            
            # Create a reasoning semantic function
            reasoning_prompt = """You are an AI assistant helping to solve a task.

Task: {{$input}}

Think through this step-by-step:
1. Understand the task requirements
2. Break down the approach
3. Execute and provide the answer

Response:"""
            
            self.reasoning_steps.append(ReasoningStep(
                step_number=2,
                thought="Creating semantic function for task execution",
                action="create_semantic_function",
                action_input="reasoning_function"
            ))
            
            reasoning_func = self.kernel.create_semantic_function(
                prompt_template=reasoning_prompt,
                function_name="reasoning",
                skill_name="task_skills",
                max_tokens=500,
                temperature=0.7
            )
            
            self.reasoning_steps.append(ReasoningStep(
                step_number=3,
                thought="Executing semantic function",
                action="invoke_function",
                action_input=task
            ))
            
            # Execute the function
            result = self.kernel.run(reasoning_func, input_str=task)
            
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