"""
ExperimentLogger

GRASP Information Expert responsible for all experiment logging, reporting,
partial log management, and result export.

Instantiated once per experiment run by Mastif. Handles:
  - Appending and flushing results after each task (partial log)
  - Finding and loading the most recent partial log for a YAML config
  - Prompting the user to resume a partial experiment
  - Exporting final JSON results
  - Printing CLI summaries (overall, by framework, by protocol, by model,
    token usage, tool usage)
  - Deleting the partial log on clean completion
"""

import json
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from domain_model import TestResult, ProtocolType
from transformers import AutoTokenizer
import tiktoken


LOGS_DIR = Path("logs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_to_dict(r: TestResult, task_index: int = -1) -> dict:
    reasoning_dict = [
        {
            "step_number": step.step_number,
            "thought":     step.thought,
            "action":      step.action,
            "action_input": step.action_input,
            "observation": step.observation,
            "timestamp":   step.timestamp,
        }
        for step in r.reasoning_steps
    ]
    return {
        "model_name":            r.model_name,
        "protocol":              r.protocol.value,
        "framework":             r.framework,
        "task_index":            task_index,
        "task":                  r.task,
        "response":              r.response,
        "reasoning_steps":       reasoning_dict,
        "reasoning_steps_count": len(reasoning_dict),
        "latency":               r.latency,
        "success":               r.success,
        "error":                 r.error,
        "metadata":              r.metadata,
        "tool_log":              (r.metadata or {}).get("tool_log", {}),
    }


def _completed_key(model: str, protocol: str, framework: str, task_index: int) -> tuple:
    return (model, protocol, framework, task_index)


# ---------------------------------------------------------------------------
# ExperimentLogger
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """
    GRASP Information Expert for experiment logging.

    Owns all result data and knows how to persist, resume, and report it.
    """

    def __init__(self, yaml_path: str, experiment_metadata: dict):
        """
        Args:
            yaml_path:            Path to the YAML config file driving this experiment.
            experiment_metadata:  Dict with keys: models, protocols, frameworks,
                                  tools, total_tasks, test_mode, experiment_name.
        """
        self._yaml_stem = Path(yaml_path).stem
        self._metadata = experiment_metadata
        self._results: List[TestResult] = []
        self._completed: set = set()   # set of (model, protocol, framework, task_index)
        self._started_at: str = datetime.datetime.now().isoformat()
        self._partial_path: Optional[Path] = None

        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Partial log management
    # ------------------------------------------------------------------

    def find_partial(self) -> Optional[dict]:
        """
        Scan logs/ for the most recent partial log matching this YAML stem.

        Returns the loaded partial log dict, or None if none exists.
        """
        pattern = f"partial-{self._yaml_stem}-*.json"
        candidates = sorted(LOGS_DIR.glob(pattern), reverse=True)
        if not candidates:
            return None
        try:
            with open(candidates[0], encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def load_partial(self, partial: dict):
        """
        Restore state from a partial log dict returned by find_partial().
        Rebuilds the completed set and results list so the test loops can skip
        already-finished combinations.

        Entries whose result carried an error are intentionally excluded from
        _completed so they are retried on the next run.
        """
        self._started_at = partial.get("started_at", self._started_at)
        self._partial_path = LOGS_DIR / f"partial-{self._yaml_stem}-{partial.get('timestamp_key', 'resumed')}.json"

        # Build a quick lookup of results by (model, protocol, framework, task_index)
        # so we can check whether a completed entry produced an error.
        results_by_key = {}
        for r in partial.get("results", []):
            key = _completed_key(
                r.get("model_name", ""), r.get("protocol", ""),
                r.get("framework", ""), r.get("task_index", -1)
            )
            results_by_key[key] = r

        for entry in partial.get("completed", []):
            key = _completed_key(
                entry["model"], entry["protocol"],
                entry["framework"], entry["task_index"]
            )
            # Re-run any entry whose result contained an error
            result = results_by_key.get(key, {})
            if result.get("error") or not result.get("success", True):
                continue
            self._completed.add(key)

        # Restore TestResult objects as plain dicts — sufficient for export/summary
        self._results = partial.get("results", [])

    def is_completed(self, model: str, protocol: str, framework: str, task_index: int) -> bool:
        """Return True if this combination was already completed in a resumed run."""
        return _completed_key(model, protocol, framework, task_index) in self._completed

    def log_result(
        self,
        result: TestResult,
        model: str,
        protocol: str,
        framework: str,
        task_index: int,
    ):
        """
        Append a result, mark the combination as completed, and flush to disk.
        Called after every single task execution.
        """
        self._results.append((result, task_index))
        self._completed.add(_completed_key(model, protocol, framework, task_index))
        self._flush_partial()

    def _flush_partial(self):
        """Write the current partial log to disk immediately."""
        if self._partial_path is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._partial_path = LOGS_DIR / f"partial-{self._yaml_stem}-{ts}.json"

        completed_list = [
            {"model": k[0], "protocol": k[1], "framework": k[2], "task_index": k[3]}
            for k in self._completed
        ]

        results_list = []
        for entry in self._results:
            if isinstance(entry, tuple):
                r, idx = entry
                results_list.append(r if isinstance(r, dict) else _result_to_dict(r, idx))
            else:
                # plain dict restored from a previous partial log
                results_list.append(entry)

        payload = {
            "yaml_file":    f"{self._yaml_stem}.yaml",
            "started_at":   self._started_at,
            "updated_at":   datetime.datetime.now().isoformat(),
            "timestamp_key": self._partial_path.stem.split("-")[-1],
            "metadata":     self._metadata,
            "completed":    completed_list,
            "results":      results_list,
        }

        tmp = self._partial_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp.replace(self._partial_path)   # atomic rename

    def close(self):
        """
        Delete the partial log on clean experiment completion.
        Called by Mastif after all results have been exported.
        """
        if self._partial_path and self._partial_path.exists():
            self._partial_path.unlink()
            print(f"🗑️  Partial log removed: {self._partial_path.name}")

    # ------------------------------------------------------------------
    # Resume prompt (called by Mastif before the test loops)
    # ------------------------------------------------------------------

    @staticmethod
    def prompt_resume(partial: dict, total: int) -> bool:
        """
        Ask the user whether to resume from a partial log.

        Args:
            partial: Partial log dict from find_partial().
            total:   Total expected executions for this run.

        Returns:
            True if the user wants to resume.
        """
        completed_count = len(partial.get("completed", []))
        started_at = partial.get("started_at", "unknown time")
        print(f"\n{'─'*70}")
        print(f"⚠️  Partial log found for this experiment.")
        print(f"   Started at:  {started_at}")
        print(f"   Completed:   {completed_count} / {total} executions")
        print(f"{'─'*70}")
        answer = input("Resume from partial log? (yes/no): ").strip().lower()
        return answer in ("yes", "y")

    def _unwrap_results(self) -> List[TestResult]:
        """
        Return only TestResult objects from _results, unwrapping (result, task_index)
        tuples produced by log_result and skipping plain dicts from resumed partial logs.
        """
        out = []
        for entry in self._results:
            if isinstance(entry, tuple):
                out.append(entry[0])
            elif isinstance(entry, TestResult):
                out.append(entry)
            # plain dicts from resumed partial logs are skipped — they lack
            # reasoning_steps objects and are only needed for JSON export
        return out

    # ------------------------------------------------------------------
    # Final export
    # ------------------------------------------------------------------

    def export_results(self, filename: str = "logs/test_results.json"):
        """Export all results to a final JSON file."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        results_list = []
        for entry in self._results:
            if isinstance(entry, tuple):
                r, idx = entry
                results_list.append(r if isinstance(r, dict) else _result_to_dict(r, idx))
            else:
                results_list.append(entry)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "test_run_timestamp": datetime.datetime.now().isoformat(),
                    "total_tests":        len(results_list),
                    "metadata":           self._metadata,
                    "results":            results_list,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\n✅️ Results exported to {filename}")

    def export_mind2web_results(self, filename: str, aggregate: dict, task_results: list):
        """Export Mind2Web-specific evaluation results."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "benchmark":        "Mind2Web",
                    "timestamp":        datetime.datetime.now().isoformat(),
                    "aggregate_metrics": aggregate,
                    "task_results":     task_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"✅️ Mind2Web results exported to {filename}")

    # ------------------------------------------------------------------
    # CLI summary
    # ------------------------------------------------------------------

    def print_summary(self):
        """Print comprehensive summary of all logged results."""
        results = self._unwrap_results()
        if not results:
            print("No results to summarise.")
            return

        total      = len(results)
        successful = sum(1 for r in results if r.success)
        total_steps = sum(len(r.reasoning_steps) for r in results)

        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"\nOverall Statistics:")
        print(f"  Total Tests:              {total}")
        print(f"  Successful:               {successful} ({successful/total*100:.1f}%)")
        print(f"  Failed:                   {total - successful} ({(total-successful)/total*100:.1f}%)")
        print(f"  Total Reasoning Steps:    {total_steps}")
        print(f"  Avg Reasoning Steps:      {total_steps/total:.1f}")
        print(f"  Tools Used:               {self._format_tool_usage(self._aggregate_tool_usage(results))}")

        self._print_by_framework(results)
        self._print_by_protocol(results)
        self._print_by_model(results)
        self._print_token_usage(results)
        self._print_tool_usage(results)

        print("\n" + "="*70)

    # ------------------------------------------------------------------
    # Internal summary helpers
    # ------------------------------------------------------------------

    def _print_by_framework(self, results: List[TestResult]):
        print("\n" + "-"*70)
        print("Results by Framework:")
        print("-"*70)
        buckets: Dict[str, dict] = {}
        for r in results:
            b = buckets.setdefault(r.framework, {"success": 0, "total": 0, "latency": [], "steps": [], "res": []})
            b["total"] += 1
            b["steps"].append(len(r.reasoning_steps))
            b["res"].append(r)
            if r.success:
                b["success"] += 1
                b["latency"].append(r.latency)
        for fw, b in sorted(buckets.items()):
            sr  = b["success"] / b["total"] * 100
            lat = sum(b["latency"]) / len(b["latency"]) if b["latency"] else 0
            avg = sum(b["steps"]) / len(b["steps"])
            print(f"\n  {fw}:")
            print(f"    Success Rate:       {b['success']}/{b['total']} ({sr:.1f}%)")
            print(f"    Avg Latency:        {lat:.2f}s")
            print(f"    Avg Steps:          {avg:.1f}")
            print(f"    Tools Used:         {self._format_tool_usage(self._aggregate_tool_usage(b['res']))}")

    def _print_by_protocol(self, results: List[TestResult]):
        print("\n" + "-"*70)
        print("Results by Protocol:")
        print("-"*70)
        buckets: Dict[str, dict] = {}
        for r in results:
            p = r.protocol.value
            b = buckets.setdefault(p, {
                "success": 0, "total": 0, "latency": [],
                "total_overhead_ms": [], "send_overhead_ms": [],
                "receive_overhead_ms": [], "message_size_bytes": [], "res": []
            })
            b["total"] += 1
            b["res"].append(r)
            if r.success:
                b["success"] += 1
                b["latency"].append(r.latency)
            for key in ("total_overhead_ms", "send_overhead_ms", "receive_overhead_ms", "message_size_bytes"):
                if r.metadata and key in r.metadata:
                    b[key].append(r.metadata[key])

        def _avg(lst): return sum(lst) / len(lst) if lst else 0

        for proto, b in sorted(buckets.items()):
            sr = b["success"] / b["total"] * 100
            print(f"\n  {proto}:")
            print(f"    Success Rate:          {b['success']}/{b['total']} ({sr:.1f}%)")
            print(f"    Avg Latency:           {_avg(b['latency']):.2f}s")
            print(f"    Avg Total Overhead:    {_avg(b['total_overhead_ms']):.3f}ms")
            print(f"    Avg Send Overhead:     {_avg(b['send_overhead_ms']):.3f}ms")
            print(f"    Avg Receive Overhead:  {_avg(b['receive_overhead_ms']):.3f}ms")
            print(f"    Avg Message Size:      {_avg(b['message_size_bytes']):.0f} bytes")
            print(f"    Tools Used:            {self._format_tool_usage(self._aggregate_tool_usage(b['res']))}")

    def _print_by_model(self, results: List[TestResult]):
        print("\n" + "-"*70)
        print("Results by Model:")
        print("-"*70)
        buckets: Dict[str, dict] = {}
        for r in results:
            b = buckets.setdefault(r.model_name, {"success": 0, "total": 0, "latency": [], "steps": [], "res": []})
            b["total"] += 1
            b["steps"].append(len(r.reasoning_steps))
            b["res"].append(r)
            if r.success:
                b["success"] += 1
                b["latency"].append(r.latency)
        for model, b in sorted(buckets.items()):
            sr  = b["success"] / b["total"] * 100
            lat = sum(b["latency"]) / len(b["latency"]) if b["latency"] else 0
            avg = sum(b["steps"]) / len(b["steps"])
            print(f"\n  {model}:")
            print(f"    Success Rate:       {b['success']}/{b['total']} ({sr:.1f}%)")
            print(f"    Avg Latency:        {lat:.2f}s")
            print(f"    Avg Steps:          {avg:.1f}")
            print(f"    Tools Used:         {self._format_tool_usage(self._aggregate_tool_usage(b['res']))}")

    def _print_token_usage(self, results: List[TestResult]):
        print("\n" + "-"*70)
        print("Token Usage by Model:")
        print("-"*70)
        # TODO: Break down tokens by framework and protocol as well
        for model in sorted(set(r.model_name for r in results)):
            model_results = [r for r in results if r.model_name == model]
            reasoning, output = self._compute_token_metrics(model, model_results)
            print(f"\n  {model}:")
            print(f"    Reasoning tokens: {reasoning}")
            print(f"    Output tokens:    {output}")
            print(f"    Total tokens:     {reasoning + output}")

    def _print_tool_usage(self, results: List[TestResult]):
        print("\n" + "-"*70)
        print("Tool Usage Across All Runs:")
        print("-"*70)
        tool_totals: Dict[str, dict] = {}
        for r in results:
            log = (r.metadata or {}).get("tool_log", {})
            for tool_name, stats in log.get("per_tool", {}).items():
                t = tool_totals.setdefault(tool_name, {"calls": 0, "failures": 0, "duration_ms": []})
                t["calls"]    += stats["calls"]
                t["failures"] += stats["failures"]
                t["duration_ms"].append(stats["avg_duration_ms"])

        if not tool_totals:
            print("  No tool invocations recorded.")
            return

        for name, t in sorted(tool_totals.items(), key=lambda x: -x[1]["calls"]):
            avg_dur   = sum(t["duration_ms"]) / len(t["duration_ms"])
            fail_rate = t["failures"] / t["calls"] * 100 if t["calls"] else 0
            print(f"\n  {name}:")
            print(f"    Total calls:  {t['calls']}")
            print(f"    Failures:     {t['failures']} ({fail_rate:.1f}%)")
            print(f"    Avg duration: {avg_dur:.1f}ms")

        print("\n" + "-"*70)
        print("Tool Usage by Framework:")
        print("-"*70)
        fw_tools: Dict[str, Dict[str, int]] = {}
        for r in results:
            log = (r.metadata or {}).get("tool_log", {})
            for tool_name, stats in log.get("per_tool", {}).items():
                fw_tools.setdefault(r.framework, {})[tool_name] = \
                    fw_tools.get(r.framework, {}).get(tool_name, 0) + stats["calls"]
        for fw, tools in sorted(fw_tools.items()):
            calls_str = ", ".join(
                f"{t} ({c}x)" for t, c in sorted(tools.items(), key=lambda x: -x[1])
            )
            print(f"\n  {fw}: {calls_str}")

    # ------------------------------------------------------------------
    # Tool usage aggregation helpers
    # ------------------------------------------------------------------

    def _aggregate_tool_usage(self, results: List[TestResult]) -> Dict[str, int]:
        usage: Dict[str, int] = {}
        for r in results:
            if not isinstance(r, TestResult):
                continue
            for tool_name, stats in (r.metadata or {}).get("tool_log", {}).get("per_tool", {}).items():
                usage[tool_name] = usage.get(tool_name, 0) + stats.get("calls", 0)
        return usage

    def _format_tool_usage(self, usage: Dict[str, int]) -> str:
        if not usage:
            return "None"
        return ", ".join(f"{n} ({c})" for n, c in sorted(usage.items(), key=lambda x: -x[1]))

    # ------------------------------------------------------------------
    # Token metrics
    # ------------------------------------------------------------------

    def _compute_token_metrics(self, model_name: str, results: List[TestResult]) -> Tuple[int, int]:
        openai_prefixes = ["gpt-", "openai"]
        if any(model_name.startswith(p) for p in openai_prefixes):
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
            encode = enc.encode
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            encode = tokenizer.encode

        reasoning_tokens = output_tokens = 0
        for r in results:
            for step in r.reasoning_steps:
                reasoning_tokens += len(encode(step.thought or ""))
                if step.action:      reasoning_tokens += len(encode(step.action))
                if step.action_input: reasoning_tokens += len(encode(step.action_input))
                if step.observation: reasoning_tokens += len(encode(step.observation))
            output_tokens += len(encode(r.response or ""))
        return reasoning_tokens, output_tokens

    # ------------------------------------------------------------------
    # Accessors used by Mastif
    # ------------------------------------------------------------------

    @property
    def results(self) -> list:
        return self._unwrap_results()

    def append_result(self, result: TestResult):
        """Append without marking completed — used when restoring from partial."""
        self._results.append(result)