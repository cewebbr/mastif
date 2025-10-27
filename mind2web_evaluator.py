"""
Mind2Web task evaluator

Handles evaluation of agent performance on Mind2Web tasks.
"""

from typing import Dict, List
import os
import re
import openai

class Mind2WebEvaluator:
    """
    Evaluator for Mind2Web benchmark tasks
    
    Evaluates agent responses against ground truth actions and elements.
    """
    
    def __init__(self):
        """Initialize the evaluator"""
        self.results = []
    
    def evaluate_task(
        self,
        task: Dict,
        agent_response: str,
        reasoning_steps: List
    ) -> Dict:
        """
        Evaluate agent performance on a single Mind2Web task
        
        Args:
            task: Mind2Web task dictionary
            agent_response: Agent's response
            reasoning_steps: Agent's reasoning steps
            
        Returns:
            Evaluation result dictionary
        """
        ground_truth_actions = task.get("action_reprs", [])
        
        # Extract actions mentioned in response
        extracted_actions = self._extract_actions_from_response(agent_response)
        
        # Calculate metrics
        action_coverage = self._calculate_action_coverage(
            extracted_actions, 
            ground_truth_actions
        )
        
        action_order_score = self._calculate_action_order_score(
            extracted_actions,
            ground_truth_actions
        )
        
        # Check if task understanding is present
        task_understanding = self._evaluate_task_understanding(
            agent_response,
            task["confirmed_task"]
        )
        
        result = {
            "task_id": task["task_id"],
            "website": task["website"],
            "domain": task["domain"],
            "ground_truth_actions": ground_truth_actions,
            "extracted_actions": extracted_actions,
            "action_coverage": action_coverage,
            "action_order_score": action_order_score,
            "task_understanding_score": task_understanding,
            "reasoning_steps_count": len(reasoning_steps),
            "response_length": len(agent_response),
        }
        
        self.results.append(result)
        return result
    
    def _extract_actions_from_response(self, response: str) -> List[str]:
        """
        Extract action keywords from agent response
        
        Args:
            response: Agent's response text
            
        Returns:
            List of extracted actions
        """
        # Common web actions
        action_keywords = [
            "click", "type", "select", "scroll", "navigate",
            "search", "submit", "fill", "enter", "choose",
            "press", "tap", "input"
        ]
        
        response_lower = response.lower()
        extracted = []
        
        for action in action_keywords:
            if action in response_lower:
                extracted.append(action)
        
        return extracted
    
    def _calculate_action_coverage(
        self,
        extracted: List[str],
        ground_truth: List[str]
    ) -> float:
        """
        Calculate how many ground truth actions are covered
        
        Args:
            extracted: Extracted actions from response
            ground_truth: Ground truth actions
            
        Returns:
            Coverage score (0.0 to 1.0)
        """
        if not ground_truth:
            return 1.0
        
        # Simple keyword matching
        covered = 0
        for gt_action in ground_truth:
            gt_lower = gt_action.lower()
            for extracted_action in extracted:
                if extracted_action in gt_lower:
                    covered += 1
                    break
        
        return covered / len(ground_truth) if ground_truth else 0.0
    
    def _calculate_action_order_score(
        self,
        extracted: List[str],
        ground_truth: List[str]
    ) -> float:
        """
        Calculate if actions are in correct order
        
        Args:
            extracted: Extracted actions
            ground_truth: Ground truth actions
            
        Returns:
            Order score (0.0 to 1.0)
        """
        if not extracted or not ground_truth:
            return 0.0
        
        # Simplified: check if first actions match
        if len(extracted) > 0 and len(ground_truth) > 0:
            first_match = any(
                ext in ground_truth[0].lower() 
                for ext in extracted[:1]
            )
            return 1.0 if first_match else 0.5
        
        return 0.0
    
    # TODO: Metrics should be only task understanding, task deviation (reasoning steps), task completion
    # TODO: Social value consistency should be assessed togheter with task deviation
    def _evaluate_task_understanding(self, response: str, task_description: str) -> float:
        # Set up the OpenAI client with your API key
        openai.api_key = os.getenv("OPEN_AI_KEY")

        # Define the prompt to send to the GPT-5-mini model
        prompt = f"""
        You are a judge assessing the quality of a task response. Your task is to evaluate how well the following response accomplishes the given task.

        Task Description:
        {task_description}

        Response:
        {response}

        Please provide a score between 0 and 1, where 0 means the response completely fails to accomplish the task, and 1 means the response perfectly accomplishes the task.
        """
        # TODO: Test with reasoning explanation and then remove it.
        # Please provide a score between 0 and 1, where 0 means the response completely fails to accomplish the task, and 1 means the response perfectly accomplishes the task. Explain your reasoning.

        try:
            response = openai.ChatCompletion.create(
                model=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}]
            )
            model_response = response.choices[0].message["content"]
        except Exception as e:
            print(f"OpenAI evaluation failed: {e}")
            return 0.0, "Evaluation failed"

        # Extract the score from the model's response
        model_response = response.choices[0].message["content"]
        lines = model_response.strip().split("\n")
        match = re.search(r"([01](?:\.\d+)?)", lines[0]) # Matches: 0, 0.0, 0.25, 0.333, 1, 1.0, 1.00
        score = float(match.group(1)) if match else 0.0

        return score
    
    def get_aggregate_metrics(self) -> Dict:
        """
        Calculate aggregate metrics across all evaluated tasks
        
        Returns:
            Dictionary with aggregate metrics
        """
        if not self.results:
            return {}
        
        total_tasks = len(self.results)
        
        avg_action_coverage = sum(r["action_coverage"] for r in self.results) / total_tasks
        avg_action_order = sum(r["action_order_score"] for r in self.results) / total_tasks
        avg_understanding = sum(r["task_understanding_score"] for r in self.results) / total_tasks
        avg_reasoning_steps = sum(r["reasoning_steps_count"] for r in self.results) / total_tasks
        
        # Domain-specific metrics
        domain_metrics = {}
        for result in self.results:
            domain = result["domain"]
            if domain not in domain_metrics:
                domain_metrics[domain] = {
                    "count": 0,
                    "total_coverage": 0.0,
                    "total_understanding": 0.0
                }
            domain_metrics[domain]["count"] += 1
            domain_metrics[domain]["total_coverage"] += result["action_coverage"]
            domain_metrics[domain]["total_understanding"] += result["task_understanding_score"]
        
        # Calculate averages per domain
        for domain in domain_metrics:
            count = domain_metrics[domain]["count"]
            domain_metrics[domain]["avg_coverage"] = domain_metrics[domain]["total_coverage"] / count
            domain_metrics[domain]["avg_understanding"] = domain_metrics[domain]["total_understanding"] / count
        
        return {
            "total_tasks_evaluated": total_tasks,
            "avg_action_coverage": avg_action_coverage,
            "avg_action_order_score": avg_action_order,
            "avg_task_understanding": avg_understanding,
            "avg_reasoning_steps": avg_reasoning_steps,
            "domain_metrics": domain_metrics
        }