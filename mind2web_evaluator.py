"""
Mind2Web task evaluator

Handles evaluation of agent performance on Mind2Web tasks.
"""

from typing import Dict, List, Optional
import re
import os

class Mind2WebEvaluator:
    """
    Evaluator for Mind2Web benchmark tasks
    
    Evaluates agent responses against with an llm-as-a-judge.
    """
    
    def __init__(self, judge_adapter, judge_model: Optional[str] = None):
        """
        Initialize the evaluator with HuggingFace judge model
        
        Args:
            judge_adapter: HuggingFaceAdapter instance for judge model
        """
        self.judge_adapter = judge_adapter
        self.judge_model = judge_model or os.getenv("JUDGE_MODEL")
        self.results = []
    
    def evaluate_task(
        self,
        task: Dict,
        agent_response: str,
        reasoning_steps: List
    ) -> Dict:
        """
        Evaluate agent performance using HuggingFace LLM-as-a-judge
        
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
            
            task_deviation = self._evaluate_task_deviation(
                high_level_task=high_level_task,
                reasoning_steps_text=reasoning_text
            )
            
            task_completion = self._evaluate_task_completion(
                high_level_task=high_level_task,
                final_response=agent_response,
                website=website,
                domain=domain
            )
            
            # Overall score: average of three metrics
            overall_score = (task_understanding + (1.0 - task_deviation) + task_completion) / 3.0
            
            result = {
                "task_id": task["task_id"],
                "website": website,
                "domain": domain,
                "high_level_task": high_level_task,
                "task_understanding": task_understanding,
                "task_deviation": task_deviation,
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
                "task_deviation": 0.5,
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
                reasoning_parts.append(step.thought)
            if hasattr(step, 'observation') and step.observation:
                reasoning_parts.append(step.observation)
        
        return " ".join(reasoning_parts)
    
    def _evaluate_task_understanding(
        self,
        high_level_task: str,
        agent_response: str
    ) -> float:
        """
        Evaluate if agent understood the task
        
        Args:
            high_level_task: The task description
            agent_response: Agent's response
            
        Returns:
            Understanding score (0.0 to 1.0)
        """
        prompt = f"""Evaluate if the agent understood the task.

Task: {high_level_task}

Agent Response: {agent_response}

Question: Does the agent's response show understanding of what needs to be accomplished?

Scoring:
- 1.0 = Clear understanding of all task requirements
- 0.7 = Good understanding with minor gaps
- 0.5 = Partial understanding
- 0.3 = Poor understanding, misses key points
- 0.0 = No understanding or completely wrong

Respond with ONLY a single number between 0.0 and 1.0, nothing else."""

        try:
            response = self.judge_adapter.generate(prompt, max_tokens=50, temperature=0.0)
            score = self._extract_score(response)
            return score
        except Exception as e:
            print(f"      Warning: Task understanding evaluation failed: {e}")
            return 0.5
    
    def _evaluate_task_deviation(
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
            Deviation score (0.0 = no deviation, 1.0 = completely off-track)
        """
        if not reasoning_steps_text or len(reasoning_steps_text.strip()) < 10:
            return 0.0
        
        prompt = f"""Evaluate if the agent's reasoning stayed focused on the task or went off-track.

Task: {high_level_task}

Agent's Reasoning Steps: {reasoning_steps_text[:1000]}

Question: How much did the reasoning deviate from the task?

Scoring (deviation level):
- 0.0 = No deviation, stayed completely on task
- 0.3 = Minor deviations but mostly relevant
- 0.5 = Moderate deviation, mixed relevance
- 0.7 = Significant deviation, often off-track
- 1.0 = Complete deviation, unrelated to task

Respond with ONLY a single number between 0.0 and 1.0, nothing else."""

        try:
            response = self.judge_adapter.generate(prompt, max_tokens=50, temperature=0.0)
            score = self._extract_score(response)
            return score
        except Exception as e:
            print(f"      Warning: Task deviation evaluation failed: {e}")
            return 0.5
    
    def _evaluate_task_completion(
        self,
        high_level_task: str,
        final_response: str,
        website: str,
        domain: str
    ) -> float:
        """
        Evaluate if final response accomplishes the task
        
        Args:
            high_level_task: The task description
            final_response: Agent's final response
            website: Website name
            domain: Domain category
            
        Returns:
            Completion score (0.0 to 1.0)
        """
        prompt = f"""Evaluate if the agent's response would successfully complete the task.

Task: {high_level_task}
Website: {website}
Domain: {domain}

Agent's Final Response: {final_response}

Question: Would following this response successfully complete the task?

Scoring:
- 1.0 = Would definitely complete the task successfully
- 0.7 = Would likely complete with minor issues
- 0.5 = Might work but has significant gaps
- 0.3 = Unlikely to complete successfully
- 0.0 = Would not complete the task at all

Respond with ONLY a single number between 0.0 and 1.0, nothing else."""

        try:
            response = self.judge_adapter.generate(prompt, max_tokens=50, temperature=0.0)
            score = self._extract_score(response)
            return score
        except Exception as e:
            print(f"      Warning: Task completion evaluation failed: {e}")
            return 0.5
    
    def _extract_score(self, response: str) -> float:
        """Extract numerical score from judge response"""
        # Find number between 0.0 and 1.0
        numbers = re.findall(r'\b[0-1]?\.\d+\b|\b[01]\b', response)
        
        if numbers:
            try:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        # Fallback: keyword matching
        response_lower = response.lower()
        if any(word in response_lower for word in ["excellent", "perfect", "complete"]):
            return 0.9
        elif any(word in response_lower for word in ["good", "adequate"]):
            return 0.7
        elif any(word in response_lower for word in ["partial", "some"]):
            return 0.5
        elif any(word in response_lower for word in ["poor", "weak"]):
            return 0.3
        elif any(word in response_lower for word in ["no", "none", "fail"]):
            return 0.1
        
        return 0.5
    
    def get_aggregate_metrics(self) -> Dict:
        """Calculate aggregate metrics across all evaluated tasks"""
        if not self.results:
            return {}
        
        total_tasks = len(self.results)
        
        avg_understanding = sum(r["task_understanding"] for r in self.results) / total_tasks
        avg_deviation = sum(r["task_deviation"] for r in self.results) / total_tasks
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
                    "total_completion": 0.0
                }
            domain_metrics[domain]["count"] += 1
            domain_metrics[domain]["total_understanding"] += result["task_understanding"]
            domain_metrics[domain]["total_deviation"] += result["task_deviation"]
            domain_metrics[domain]["total_completion"] += result["task_completion"]
        
        # Calculate averages per domain
        for domain in domain_metrics:
            count = domain_metrics[domain]["count"]
            domain_metrics[domain]["avg_understanding"] = domain_metrics[domain]["total_understanding"] / count
            domain_metrics[domain]["avg_deviation"] = domain_metrics[domain]["total_deviation"] / count
            domain_metrics[domain]["avg_completion"] = domain_metrics[domain]["total_completion"] / count
        
        return {
            "total_tasks_evaluated": total_tasks,
            "judge_model": self.judge_model,
            "avg_task_understanding": avg_understanding,
            "avg_task_deviation": avg_deviation,
            "avg_task_completion": avg_completion,
            "avg_overall_score": avg_overall_score,
            "avg_reasoning_steps": avg_reasoning_steps,
            "domain_metrics": domain_metrics
        }