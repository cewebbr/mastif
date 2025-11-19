"""
LangGraph Stateful Workflow Integration

LangGraph provides graph-based agent orchestration with explicit state management.
"""

from typing import List, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
import sys
sys.path.append('..')
from domain_model import ReasoningStep


class LangGraphAgent:
    """
    LangGraph Stateful Workflow Integration
    
    LangGraph provides graph-based agent orchestration with explicit
    state management. Ideal for complex, multi-step workflows
    with conditional branching.
    """
    
    def __init__(self, adapter):
        """
        Initialize LangGraph agent
        
        Args:
            adapter: HuggingFace model adapter
        """
        self.adapter = adapter
        self.graph = None
        self.reasoning_steps: List[ReasoningStep] = []
    
    def build_research_workflow(self):
        """
        Build a multi-step research workflow using LangGraph
        
        The workflow consists of:
        1. Planning: Create research plan
        2. Research: Execute research steps (iterative)
        3. Synthesis: Compile final report
        """
        
        # Define state structure
        class AgentState(TypedDict):
            task: str
            plan: str
            research_results: Annotated[list, operator.add]
            final_report: str
            step: int
        
        # Node 1: Planning
        def planning_node(state: AgentState) -> AgentState:
            """Create a research plan"""
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Creating research plan",
                action="plan",
                action_input=state['task']
            ))
            
            prompt = f"Create a detailed step-by-step research plan for: {state['task']}"
            try:
                plan = self.adapter.generate(prompt, max_tokens=512)
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
        
        # Node 2: Research
        def research_node(state: AgentState) -> AgentState:
            """Execute research step"""
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought=f"Executing research step {state['step']}",
                action="research",
                action_input=f"Step {state['step']} of plan"
            ))
            
            prompt = f"""Based on this plan: {state['plan']}

Execute research step {state['step']} and provide detailed findings."""
            try:
                findings = self.adapter.generate(prompt, max_tokens=1024)
                state["research_results"] = [findings]
                state["step"] += 1
                
                self.reasoning_steps.append(ReasoningStep(
                    step_number=len(self.reasoning_steps) + 1,
                    thought=f"Research step {state['step']-1} completed",
                    observation=f"Findings: {findings[:100]}..."
                ))
            except Exception as e:
                state["research_results"] = [f"Research error: {str(e)}"]
                self.reasoning_steps.append(ReasoningStep(
                    step_number=len(self.reasoning_steps) + 1,
                    thought="Error in research phase",
                    observation=str(e)
                ))
            return state
        
        # Node 3: Synthesis
        def synthesis_node(state: AgentState) -> AgentState:
            """Synthesize final report"""
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Synthesizing research findings into final report",
                action="synthesize",
                action_input=f"{len(state['research_results'])} research findings"
            ))
            
            results_text = "\n\n".join(state["research_results"])
            prompt = f"""Synthesize these research findings into a comprehensive final report:

{results_text}

Provide a well-structured summary with key insights."""
            try:
                report = self.adapter.generate(prompt, max_tokens=512)
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
        
        # Conditional edge function
        def should_continue(state: AgentState) -> str:
            """Decide whether to continue research or synthesize"""
            if state["step"] > 2:  # Limit to 2 research iterations
                self.reasoning_steps.append(ReasoningStep(
                    step_number=len(self.reasoning_steps) + 1,
                    thought="Research iterations complete, moving to synthesis",
                    observation=f"Completed {state['step']-1} research steps"
                ))
                return "synthesize"
            
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought=f"Continuing research (step {state['step']})",
                observation="More research needed"
            ))
            return "research"
        
        # Build the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan", planning_node)
        workflow.add_node("research", research_node)
        workflow.add_node("synthesize", synthesis_node)
        
        # Add edges
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "research")
        workflow.add_conditional_edges(
            "research",
            should_continue,
            {
                "research": "research",
                "synthesize": "synthesize"
            }
        )
        workflow.add_edge("synthesize", END)
        
        self.graph = workflow.compile()
    
    def run(self, task: str) -> str:
        """
        Execute the LangGraph workflow
        
        Args:
            task: Research task to execute
            
        Returns:
            Final research report
        """
        self.reasoning_steps = []
        
        try:
            if self.graph is None:
                self.build_research_workflow()
            
            self.reasoning_steps.append(ReasoningStep(
                step_number=1,
                thought="Initializing LangGraph workflow",
                observation="Graph compiled successfully"
            ))
            
            initial_state = {
                "task": task,
                "plan": "",
                "research_results": [],
                "final_report": "",
                "step": 0
            }
            
            result = self.graph.invoke(initial_state)
            return result.get("final_report", "No report generated")
        
        except Exception as e:
            self.reasoning_steps.append(ReasoningStep(
                step_number=len(self.reasoning_steps) + 1,
                thought="Error during workflow execution",
                observation=str(e)
            ))
            return f"LangGraph execution error: {str(e)}"