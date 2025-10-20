"""
LangChain ReAct Agent Integration

LangChain implements the ReAct (Reasoning + Acting) pattern.
"""

from typing import List, Dict
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
import sys
sys.path.append('..')
from domain_model import ReasoningStep


class LangChainAgent:
    """
    LangChain ReAct Agent Integration
    
    LangChain implements the ReAct (Reasoning + Acting) pattern,
    which interleaves reasoning steps with action execution.
    """
    
    def __init__(self, adapter):
        """
        Initialize LangChain ReAct agent
        
        Args:
            adapter: HuggingFace model adapter
        """
        self.adapter = adapter
        self.tools: List[Tool] = []
        self.llm = adapter.get_langchain_llm()
        self.reasoning_steps: List[ReasoningStep] = []
    
    def add_tool(self, name: str, description: str, func=None):
        """
        Add a tool to the agent
        
        Args:
            name: Tool name
            description: Tool description and usage instructions
            func: Tool function (uses dummy if not provided)
        """
        if func is None:
            # Create a dummy function that logs the call
            def dummy_func(x):
                result = f"Tool '{name}' executed with input: {x}"
                # Log tool execution as reasoning step
                self.reasoning_steps.append(ReasoningStep(
                    step_number=len(self.reasoning_steps) + 1,
                    thought=f"Executed tool: {name}",
                    action=name,
                    action_input=str(x),
                    observation=result
                ))
                return result
            func = dummy_func
        
        tool = Tool(
            name=name,
            func=func,
            description=description
        )
        self.tools.append(tool)
    
    def run(self, task: str) -> str:
        """
        Execute task using LangChain ReAct agent
        
        The ReAct pattern follows: Thought -> Action -> Observation loop
        until the agent determines it has enough information to answer.
        
        Args:
            task: Task to execute
            
        Returns:
            Final answer from the agent
        """
        self.reasoning_steps = []
        
        try:
            # Log initial setup
            self.reasoning_steps.append(ReasoningStep(
                step_number=1,
                thought=f"Initializing ReAct agent with {len(self.tools)} tools",
                observation=f"Tools: {[t.name for t in self.tools]}"
            ))
            
            # Create ReAct prompt template
            template = """Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

            prompt = PromptTemplate.from_template(template)
            
            # Create and execute agent
            agent = create_react_agent(self.llm, self.tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
            
            self.reasoning_steps.append(ReasoningStep(
                step_number=2,
                thought="Starting ReAct reasoning loop",
                action="execute_agent",
                action_input=task
            ))
            
            result = agent_executor.invoke({"input": task})
            
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="ReAct loop completed successfully",
                observation="Final answer generated"
            ))
            
            return result.get("output", str(result))
        
        except Exception as e:
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Error during execution",
                observation=f"Error: {str(e)}"
            ))
            return f"LangChain execution error: {str(e)}"