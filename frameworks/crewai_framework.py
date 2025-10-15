"""
CrewAI Framework Integration

CrewAI focuses on role-based agent collaboration where each agent
has a specific role and expertise.
"""

import json
from typing import Dict, List, Optional
import sys
sys.path.append('..')
from models import ReasoningStep


class CrewAIAgent:
    """
    CrewAI Framework Integration
    
    CrewAI focuses on role-based agent collaboration where each agent
    has a specific role and expertise. Agents work together as a crew
    to accomplish complex tasks.
    """
    
    def __init__(self, adapter, role: str):
        """
        Initialize CrewAI agent with specific role
        
        Args:
            adapter: HuggingFace model adapter
            role: Agent's role/expertise (e.g., "Research Analyst")
        """
        self.adapter = adapter
        self.role = role
        self.reasoning_steps: List[ReasoningStep] = []
    
    def execute_task(self, task: str, context: Dict = None) -> str:
        """
        Execute a task using CrewAI-style role-based approach
        
        Args:
            task: Task description
            context: Additional context information
            
        Returns:
            Task execution result
        """
        self.reasoning_steps = []
        
        # Log initial thought
        self.reasoning_steps.append(ReasoningStep(
            step_number=1,
            thought=f"As a {self.role}, I need to analyze this task and determine my approach"
        ))
        
        # Create role-specific prompt
        prompt = f"""You are a {self.role} agent in a CrewAI system.

Your Role: {self.role}
Task: {task}
Context: {json.dumps(context or {}, indent=2)}

Think step-by-step about how to complete this task given your role and expertise.
Provide your reasoning process and final answer."""
        
        # Log the action
        self.reasoning_steps.append(ReasoningStep(
            step_number=2,
            thought="Generating response based on role expertise",
            action="generate_response",
            action_input=task
        ))
        
        # Generate response
        response = self.adapter.generate(prompt)
        
        # Log observation
        self.reasoning_steps.append(ReasoningStep(
            step_number=3,
            thought="Task completed successfully",
            observation=f"Generated response with {len(response)} characters"
        ))
        
        return response