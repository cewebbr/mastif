"""
Mind2Web task evaluator

Handles evaluation of agent performance on Mind2Web tasks.
"""

from typing import Dict, List, Optional
import re
import os
from config import ConfigExpert

class Mind2WebEvaluator:
    """
    Evaluator for Mind2Web benchmark tasks
    
    Evaluates agent responses against with an llm-as-a-judge.
    """
    
    def __init__(self, judge_adapter, judge_model: Optional[str] = None):
        """
        Initialize the evaluator with judge model
        
        Args:
            judge_adapter: Instance for judge model
        """
        self.judge_adapter = judge_adapter
        self.judge_model = judge_model or ConfigExpert.get_instance().get("judge_model", "gpt-4o-mini")
        self.results = []
    
    def evaluate_task(
        self,
        task: Dict,
        agent_response: str,
        reasoning_steps: List
    ) -> Dict:
        """
        Evaluate agent performance using LLM-as-a-judge
        
        Args:
            task: Mind2Web task dictionary with 'confirmed_task'
            agent_response: Agent's final response
            reasoning_steps: Agent's reasoning steps
            
        Returns:
            Evaluation result dictionary
        """
        high_level_task = task.get("confirmed_task", "")
        website = task.get("website", "unknown")
        domain = task.get("domain", "unknown")
        
        # Extract reasoning text from steps
        reasoning_text = self._extract_reasoning_text(reasoning_steps)
        
        # Evaluate three dimensions
        try:
            task_understanding = self._evaluate_task_understanding(
                high_level_task=high_level_task,
                agent_response=agent_response
            )
            
            task_adherence = self._evaluate_task_adherence(
                high_level_task=high_level_task,
                reasoning_steps_text=reasoning_text
            )
            
            task_completion = self._evaluate_task_completion(
                high_level_task=high_level_task,
                final_response=agent_response,
                website=website,
                domain=domain
            )
            
            overall_score = (task_understanding + task_adherence + task_completion) / 3.0
            
            result = {
                "task_id": task["task_id"],
                "website": website,
                "domain": domain,
                "high_level_task": high_level_task,
                "task_understanding": task_understanding,
                "task_adherence": task_adherence,
                "task_completion": task_completion,
                "overall_score": overall_score,
                "reasoning_steps_count": len(reasoning_steps),
            }
            
        except Exception as e:
            print(f"    Warning: Judge evaluation failed: {e}")
            result = {
                "task_id": task["task_id"],
                "website": website,
                "domain": domain,
                "high_level_task": high_level_task,
                "task_understanding": 0.5,
                "task_adherence": 0.5,
                "task_completion": 0.5,
                "overall_score": 0.5,
                "reasoning_steps_count": len(reasoning_steps),
            }
        
        self.results.append(result)
        return result
    
    def _extract_reasoning_text(self, reasoning_steps: List) -> str:
        """Extract text from reasoning steps"""
        reasoning_parts = []
        for step in reasoning_steps:
            if hasattr(step, 'thought') and step.thought:
                reasoning_parts.append(f"[Thought] {step.thought}")
            if hasattr(step, 'observation') and step.observation:
                reasoning_parts.append(f"[Observation] {step.observation}")
        return "\n".join(reasoning_parts)
    
    def _evaluate_task_understanding(
        self,
        high_level_task: str,
        agent_response: str
    ) -> float:
        prompt = f"""Evaluate whether the agent correctly understood the task.

Task:
{high_level_task}

Agent Response:
{agent_response}

Score task_understanding from 0.0 to 1.0 using these anchors:

1.0 = Explicitly captures all goals, constraints, and required outputs
0.7 = Captures main goal but misses minor constraints or details
0.5 = Understands general intent but misses key requirement(s)
0.3 = Misinterprets important parts of the task
0.0 = Completely incorrect or unrelated understanding

Respond with ONLY a single number between 0.0 and 1.0."""

        try:
            response = self.judge_adapter.generate(prompt, max_tokens=50, temperature=0.0)
            return self._extract_score(response)
        except Exception as e:
            print(f"      Warning: Task understanding evaluation failed: {e}")
            return 0.5
    
    def _evaluate_task_adherence(
        self,
        high_level_task: str,
        reasoning_steps_text: str
    ) -> float:
        """
        Evaluate if reasoning steps deviate from the task
        
        Args:
            high_level_task: The task description
            reasoning_steps_text: Combined reasoning steps text
            
        Returns:
            Adherence score (0.0 = completely off-track, 1.0 = fully adherent)
        """
        if not reasoning_steps_text or len(reasoning_steps_text.strip()) < 10:
            return 1.0
        
        prompt = f"""Evaluate how well the agent's reasoning adheres to the task.

Task:
{high_level_task}

Agent Reasoning:
{reasoning_steps_text[:1200]}

Score task_adherence from 0.0 to 1.0 using these anchors:

1.0 = Fully focused on task, all steps directly relevant
0.7 = Mostly on-task, minor irrelevant or redundant steps
0.5 = Mixed relevance, some important off-track reasoning
0.3 = Frequently off-task or distracted reasoning
0.0 = Completely unrelated reasoning

Respond with ONLY a single number between 0.0 and 1.0."""

        try:
            response = self.judge_adapter.generate(prompt, max_tokens=50, temperature=0.0)
            return self._extract_score(response)
        except Exception as e:
            print(f"      Warning: Task adherence evaluation failed: {e}")
            return 0.5
    
    def _evaluate_task_completion(
        self,
        high_level_task: str,
        final_response: str,
        website: str,
        domain: str
    ) -> float:
        prompt = f"""Evaluate whether the agent's response would successfully complete the task.

Task:
{high_level_task}

Website: {website}
Domain: {domain}

Agent Final Response:
{final_response}

Score task_completion from 0.0 to 1.0 using these anchors:

1.0 = Fully completes task with correct and actionable steps/results
0.7 = Likely completes task but with minor gaps or inefficiencies
0.5 = Partially completes task; important steps missing
0.3 = Unlikely to complete task successfully
0.0 = Does not complete task at all or is incorrect

Respond with ONLY a single number between 0.0 and 1.0."""

        try:
            response = self.judge_adapter.generate(prompt, max_tokens=50, temperature=0.0)
            return self._extract_score(response)
        except Exception as e:
            print(f"      Warning: Task completion evaluation failed: {e}")
            return 0.5
    
    def _extract_score(self, response: str) -> float:
        """Extract numerical score from judge response"""
        # Debug log to help refine extraction if needed
        # print(f"      Info: Raw judge response: {repr(response[:200])}")

        # Strip markdown, punctuation and whitespace that may wrap the number
        cleaned = re.sub(r'[`*_~#]', '', response).strip()

        # Match any float or integer in [0, 1] — handles "0.7.", "0.70", "1", "0"
        numbers = re.findall(r'\b(1\.0+|0\.\d+|[01])\b', cleaned)

        if numbers:
            try:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        response_lower = cleaned.lower()
        if any(w in response_lower for w in ["fully", "complete", "perfect"]):
            return 0.9
        elif any(w in response_lower for w in ["mostly", "good"]):
            return 0.7
        elif any(w in response_lower for w in ["partial", "mixed"]):
            return 0.5
        elif any(w in response_lower for w in ["poor", "limited"]):
            return 0.3
        elif any(w in response_lower for w in ["none", "fail", "incorrect"]):
            return 0.1

        # Log unexpected response to aid debugging
        print(f"      Warning: Could not extract score from judge response: {repr(response[:200])}")
        return 0.5
    
    def get_aggregate_metrics(self) -> Dict:
        """Calculate aggregate metrics across all evaluated tasks"""
        if not self.results:
            return {}
        
        total_tasks = len(self.results)
        
        avg_understanding = sum(r["task_understanding"] for r in self.results) / total_tasks
        avg_adherence = sum(r["task_adherence"] for r in self.results) / total_tasks
        avg_completion = sum(r["task_completion"] for r in self.results) / total_tasks
        avg_overall_score = sum(r["overall_score"] for r in self.results) / total_tasks
        avg_reasoning_steps = sum(r["reasoning_steps_count"] for r in self.results) / total_tasks
        
        # Domain-specific metrics
        domain_metrics = {}
        for result in self.results:
            domain = result["domain"]
            if domain not in domain_metrics:
                domain_metrics[domain] = {
                    "count": 0,
                    "total_understanding": 0.0,
                    "total_adherence": 0.0,
                    "total_completion": 0.0
                }
            domain_metrics[domain]["count"] += 1
            domain_metrics[domain]["total_understanding"] += result["task_understanding"]
            domain_metrics[domain]["total_adherence"] += result["task_adherence"]
            domain_metrics[domain]["total_completion"] += result["task_completion"]
        
        # Calculate averages per domain
        for domain in domain_metrics:
            count = domain_metrics[domain]["count"]
            domain_metrics[domain]["avg_understanding"] = domain_metrics[domain]["total_understanding"] / count
            domain_metrics[domain]["avg_adherence"] = domain_metrics[domain]["total_adherence"] / count
            domain_metrics[domain]["avg_completion"] = domain_metrics[domain]["total_completion"] / count
        
        return {
            "total_tasks_evaluated": total_tasks,
            "judge_model": self.judge_model,
            "avg_task_understanding": avg_understanding,
            "avg_task_adherence": avg_adherence,
            "avg_task_completion": avg_completion,
            "avg_overall_score": avg_overall_score,
            "avg_reasoning_steps": avg_reasoning_steps,
            "domain_metrics": domain_metrics
        }