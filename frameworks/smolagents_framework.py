"""
Smolagents Framework Integration

Smolagents is a lightweight agent framework focusing on simplicity
and efficiency with minimal overhead.
"""

import json
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
    
    def __init__(self, adapter, protocol=None):
        """
        Initialize Smolagents wrapper
        
        Args:
            adapter: HuggingFace model adapter
        """
        self.adapter = adapter
        self.tools: List[Dict[str, str]] = []
        self.reasoning_steps: List[ReasoningStep] = []
        self.protocol = protocol
    
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
        
        # Wrap task with protocol if provided
        if self.protocol:
            formatted_msg = self.protocol.send_message(task, {})
            task = f"""Protocol: {self.protocol.__class__.__name__}

{json.dumps(formatted_msg, indent=2)}

Execute according to protocol."""
            
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
        prompt = f"""You are an AI agent operating in the Smolagents framework.

Task:
{task}

Tools:
{tools_desc}

Instructions:
• Think step-by-step.
• Decide what tools are needed.
• Plan your approach.
• Execute and provide the result.
• Do not skip steps.
• Make intermediate decisions explicit.
• If information is missing, state assumptions clearly.
• If the output format is not provided in the task, favor correctness and completeness over brevity.

Output:
Reasoning:
Final Answer:
"""

        
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