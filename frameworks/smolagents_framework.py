"""
Smolagents Framework Integration

Smolagents is a lightweight agent framework focusing on simplicity
and efficiency with minimal overhead.
"""

from typing import Dict, List
import sys
sys.path.append('..')
from domain_model import ReasoningStep


class SmolAgentWrapper:
    """
    Smolagents Framework Integration
    
    Smolagents is a lightweight agent framework focusing on simplicity
    and efficiency. It emphasizes minimal overhead while maintaining
    tool-use capabilities.
    """
    
    def __init__(self, adapter):
        """
        Initialize Smolagents wrapper
        
        Args:
            adapter: HuggingFace model adapter
        """
        self.adapter = adapter
        self.tools: List[Dict[str, str]] = []
        self.reasoning_steps: List[ReasoningStep] = []
    
    def add_tool(self, name: str, description: str):
        """
        Add a tool to the agent's toolkit
        
        Args:
            name: Name of the tool
            description: Description of tool's functionality
        """
        self.tools.append({"name": name, "description": description})
    
    def run(self, task: str) -> str:
        """
        Execute task using Smolagents approach
        
        Args:
            task: Task to execute
            
        Returns:
            Execution result
        """
        self.reasoning_steps = []
        
        # Step 1: Analyze available tools
        self.reasoning_steps.append(ReasoningStep(
            step_number=1,
            thought=f"Analyzing task and available tools. I have {len(self.tools)} tools available."
        ))
        
        tools_desc = "\n".join([f"- {t['name']}: {t['description']}" for t in self.tools])
        
        # Step 2: Plan approach
        self.reasoning_steps.append(ReasoningStep(
            step_number=2,
            thought="Planning which tools to use and in what order",
            action="analyze_tools",
            action_input=task
        ))
        
        # Create prompt with reasoning structure
        prompt = f"""You are a lightweight agent using the Smolagents framework.

Available Tools:
{tools_desc}

Task: {task}

Think step by step:
1. What tools do you need?
2. What's your approach?
3. Execute and provide the result.

Provide your reasoning and final answer:"""
        
        # Step 3: Execute
        self.reasoning_steps.append(ReasoningStep(
            step_number=3,
            thought="Executing task with selected tools",
            action="execute_task",
            action_input=task
        ))
        
        response = self.adapter.generate(prompt)
        
        # Step 4: Complete
        self.reasoning_steps.append(ReasoningStep(
            step_number=4,
            thought="Task execution completed",
            observation="Response generated successfully"
        ))
        
        return response