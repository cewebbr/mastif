"""
Mind2Web task evaluator

Handles evaluation of agent performance on Mind2Web tasks.
"""

from typing import Dict, List, Optional
import re
import os
import json
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
            evaluation = self._evaluate_task(
                high_level_task=high_level_task,
                agent_response=agent_response,
                reasoning_steps_text=reasoning_text,
                website=website,
                domain=domain
            )
            
            overall_score = (
                self._get_score(evaluation["task_understanding"])
                + self._get_score(evaluation["task_adherence"])
                + self._get_score(evaluation["task_completion"])
            ) / 3.0
            
            result = {
                "task_id": task["task_id"],
                "website": website,
                "domain": domain,
                "high_level_task": high_level_task,
                "task_understanding": evaluation["task_understanding"],
                "task_adherence": evaluation["task_adherence"],
                "task_completion": evaluation["task_completion"],
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
                "task_understanding": {"score": 0.5, "rationale": "Judge evaluation failed."},
                "task_adherence": {"score": 0.5, "rationale": "Judge evaluation failed."},
                "task_completion": {"score": 0.5, "rationale": "Judge evaluation failed."},
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
    
    def _evaluate_task(
        self,
        high_level_task: str,
        agent_response: str,
        reasoning_steps_text: str,
        website: str,
        domain: str
    ) -> Dict:
        prompt = f"""Evaluate the agent's response and reasoning on the following Mind2Web task.

Task:
{high_level_task}

Website: {website}
Domain: {domain}

Agent Response:
{agent_response}

Agent Reasoning:
{reasoning_steps_text or '(no reasoning provided)'}

For each of the following dimensions, provide a score from 0.0 to 1.0 and a brief rationale:

- task_understanding: how well the agent understood the task goals, constraints, and required output
- task_adherence: how well the agent's reasoning stayed on-task and relevant
- task_completion: whether the final response is sufficient to complete the task successfully

Use these anchors:

task_understanding:
1.0 = Explicitly captures all goals, constraints, and required outputs
0.7 = Captures main goal but misses minor constraints or details
0.5 = Understands general intent but misses key requirement(s)
0.3 = Misinterprets important parts of the task
0.0 = Completely incorrect or unrelated understanding

task_adherence:
1.0 = Fully focused on task, all steps directly relevant
0.7 = Mostly on-task, minor irrelevant or redundant steps
0.5 = Mixed relevance, some important off-track reasoning
0.3 = Frequently off-task or distracted reasoning
0.0 = Completely unrelated reasoning

task_completion:
1.0 = Fully completes task with correct and actionable steps/results
0.7 = Likely completes task but with minor gaps or inefficiencies
0.5 = Partially completes task; important steps missing
0.3 = Unlikely to complete task successfully
0.0 = Does not complete task at all or is incorrect

Respond with ONLY a JSON object with keys `task_understanding`, `task_adherence`, and `task_completion`.
Each value must be an object with keys `score` and `rationale`.
Example:
{
  "task_understanding": {"score": 0.8, "rationale": "..."},
  "task_adherence": {"score": 0.7, "rationale": "..."},
  "task_completion": {"score": 0.9, "rationale": "..."}
}
Do not include any extra text, markdown, or explanation outside the JSON object."""

        try:
            response = self.judge_adapter.generate(prompt, max_tokens=350, temperature=0.0)
            parsed = self._parse_json_object(response)
            return self._normalize_evaluation_results(parsed, response)
        except Exception as e:
            print(f"      Warning: Task evaluation failed: {e}")
            return self._default_evaluation_results(
                "Judge evaluation failed."
            )

    def _normalize_evaluation_results(self, parsed: Optional[dict], raw_response: str) -> Dict:
        if not isinstance(parsed, dict):
            return self._default_evaluation_results(
                "Could not parse judge JSON response."
            )

        evaluation = {}
        for key in ["task_understanding", "task_adherence", "task_completion"]:
            evaluation[key] = self._normalize_evaluation_entry(
                parsed.get(key), key, raw_response
            )

        return evaluation

    def _normalize_evaluation_entry(
        self,
        entry,
        key: str,
        raw_response: str
    ) -> Dict:
        if isinstance(entry, dict):
            score = entry.get("score")
            rationale = entry.get("rationale")

            if isinstance(score, (int, float)):
                score = float(score)
            elif isinstance(score, str):
                try:
                    score = float(score)
                except ValueError:
                    score = self._extract_score(raw_response)
            else:
                score = self._extract_score(raw_response)

            if not isinstance(rationale, str) or not rationale.strip():
                rationale = raw_response.strip() or f"No rationale provided for {key}."

            return {
                "score": max(0.0, min(1.0, score)),
                "rationale": rationale.strip()
            }

        return {
            "score": 0.5,
            "rationale": f"Missing or invalid `{key}` in judge response."
        }

    def _default_evaluation_results(self, message: str) -> Dict:
        return {
            "task_understanding": {"score": 0.5, "rationale": message},
            "task_adherence": {"score": 0.5, "rationale": message},
            "task_completion": {"score": 0.5, "rationale": message}
        }
    
    def _extract_score(self, response: str) -> float:
        """Extract numerical score from judge response"""
        # Debug log to help refine extraction if needed
        # print(f"      Info: Raw judge response: {repr(response)}")

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
        print(f"      Warning: Could not extract score from judge response: {repr(response)}")
        return 0.5

    def _parse_json_object(self, response: str) -> Optional[dict]:
        """Try to extract a JSON object from the judge response."""
        if not response:
            return None

        cleaned = response.strip()
        cleaned = re.sub(r'```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip('`\n ')

        # Try direct parse first
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: find the first JSON object-like substring
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start:end+1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        return None

    def _extract_score_and_rationale(self, response: str) -> dict:
        """Extract score and rationale from judge response."""
        parsed = self._parse_json_object(response)
        if isinstance(parsed, dict):
            score = parsed.get('score')
            rationale = parsed.get('rationale')
            if isinstance(score, (int, float)):
                score = max(0.0, min(1.0, float(score)))
            else:
                score = self._extract_score(response)
            if not isinstance(rationale, str) or not rationale.strip():
                rationale = response.strip()
            return {
                'score': score,
                'rationale': rationale.strip()
            }

        return {
            'score': self._extract_score(response),
            'rationale': response.strip() or 'No rationale provided.'
        }

    def _get_score(self, value):
        if isinstance(value, dict):
            return float(value.get('score', 0.0))
        try:
            return float(value)
        except Exception:
            return 0.0

    def get_aggregate_metrics(self) -> Dict:
        """Calculate aggregate metrics across all evaluated tasks"""
        if not self.results:
            return {}
        
        total_tasks = len(self.results)
        
        avg_understanding = sum(self._get_score(r["task_understanding"]) for r in self.results) / total_tasks
        avg_adherence = sum(self._get_score(r["task_adherence"]) for r in self.results) / total_tasks
        avg_completion = sum(self._get_score(r["task_completion"]) for r in self.results) / total_tasks
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
            domain_metrics[domain]["total_understanding"] += self._get_score(result["task_understanding"])
            domain_metrics[domain]["total_adherence"] += self._get_score(result["task_adherence"])
            domain_metrics[domain]["total_completion"] += self._get_score(result["task_completion"])
        
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