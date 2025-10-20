"""
LlamaIndex ReAct Agent Integration

LlamaIndex specializes in data-augmented LLM applications,
particularly for RAG use cases.
"""

from typing import List, Dict
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
import sys
sys.path.append('..')
from domain_model import ReasoningStep


class LlamaIndexAgent:
    """
    LlamaIndex ReAct Agent Integration
    
    LlamaIndex specializes in data-augmented LLM applications,
    particularly for RAG (Retrieval Augmented Generation) use cases.
    """
    
    def __init__(self, adapter):
        """
        Initialize LlamaIndex agent
        
        Args:
            adapter: HuggingFace model adapter
        """
        self.adapter = adapter
        self.llm = adapter.get_llamaindex_llm()
        self.tools: List[FunctionTool] = []
        self.agent = None
        self.reasoning_steps: List[ReasoningStep] = []
        
        # Configure LlamaIndex settings
        Settings.llm = self.llm
        Settings.chunk_size = 512
    
    def add_tool(self, name: str, description: str, func=None):
        """
        Add a tool to the agent
        
        Args:
            name: Tool name
            description: Tool description
            func: Tool function (uses dummy if not provided)
        """
        if func is None:
            def dummy_func(query: str) -> str:
                result = f"Tool '{name}' processed: {query}"
                self.reasoning_steps.append(ReasoningStep(
                    step_number=len(self.reasoning_steps) + 1,
                    thought=f"Used tool: {name}",
                    action=name,
                    action_input=query,
                    observation=result
                ))
                return result
            func = dummy_func
        
        tool = FunctionTool.from_defaults(
            fn=func,
            name=name,
            description=description
        )
        self.tools.append(tool)
    
    def run(self, task: str) -> str:
        """
        Execute task using LlamaIndex ReAct agent
        
        Args:
            task: Task to execute
            
        Returns:
            Agent's response
        """
        self.reasoning_steps = []
        
        try:
            self.reasoning_steps.append(ReasoningStep(
                step_number=1,
                thought=f"Initializing LlamaIndex ReAct agent with {len(self.tools)} tools",
                observation=f"Tools available: {[t.metadata.name for t in self.tools]}"
            ))
            
            # Create ReAct agent
            if self.agent is None:
                self.agent = ReActAgent.from_tools(
                    self.tools,
                    llm=self.llm,
                    verbose=True,
                    max_iterations=5
                )
            
            self.reasoning_steps.append(ReasoningStep(
                step_number=2,
                thought="Executing task with ReAct reasoning",
                action="query_agent",
                action_input=task
            ))
            
            # Execute query
            response = self.agent.query(task)
            
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Task completed successfully",
                observation=f"Response: {str(response)[:100]}..."
            ))
            
            return str(response)
        
        except Exception as e:
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Error during execution",
                observation=str(e)
            ))
            return f"LlamaIndex execution error: {str(e)}"