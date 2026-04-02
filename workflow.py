"""
WorkflowController

GRASP Controller responsible for reading the workflow configuration from
the YAML file and executing the declared node sequence on behalf of any
agentic framework.

Each framework instantiates one WorkflowController in __init__ and reuses
it across all run() calls, ensuring configuration is read once and the
comparison across frameworks remains fair.
"""

import os
from pathlib import Path
from typing import Dict, List, Callable, Optional

from domain_model import ReasoningStep
from config import ConfigExpert


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prompt(template_path: str) -> str:
    """
    Load a prompt template from a file path.

    Paths are resolved relative to the config file's directory first,
    then relative to the current working directory as a fallback.
    """
    config = ConfigExpert.get_instance()
    config_dir = Path(config.get_config_path()).parent if config.get_config_path() else Path.cwd()

    candidates = [
        config_dir / template_path,
        Path.cwd() / template_path,
        Path(template_path),
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8")

    raise FileNotFoundError(
        f"Prompt template not found: '{template_path}'. "
        f"Searched in: {[str(p) for p in candidates]}"
    )


def _render(template: str, variables: Dict) -> str:
    """
    Render a prompt template by substituting {placeholders} with values.
    Missing keys are left as-is so partial templates still render safely.
    """
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", str(value) if value is not None else "")
    return result


# ---------------------------------------------------------------------------
# NodeConfig — lightweight data class parsed from YAML
# ---------------------------------------------------------------------------

class NodeConfig:
    """Parsed representation of a single workflow node from YAML config."""

    def __init__(self, cfg: dict):
        self.name: str = cfg["name"]
        self.prompt_template: str = cfg["prompt_template"]
        self.output_key: str = cfg["output_key"]
        self.loop: bool = cfg.get("loop", False)
        self._template_text: Optional[str] = None

    @property
    def template_text(self) -> str:
        if self._template_text is None:
            self._template_text = _load_prompt(self.prompt_template)
        return self._template_text


# ---------------------------------------------------------------------------
# WorkflowController
# ---------------------------------------------------------------------------

class WorkflowController:
    """
    GRASP Controller for agentic workflow execution.

    Reads the workflow node sequence once from the YAML configuration,
    compiles it into an executable chain, and exposes a single run()
    method that all framework agents delegate to.

    Instantiate once per agent instance (in __init__) to ensure the
    configuration snapshot is identical across all frameworks throughout
    an experiment run.
    """

    def __init__(self, framework_name: str, generate_fn: Callable, get_tool_payload_fn: Callable):
        """
        Args:
            framework_name:     Human-readable framework label used in prompts
                                (e.g. "LangChain", "CrewAI").
            generate_fn:        Callable(prompt, **kwargs) -> str  — the adapter's
                                generate method bound to the current agent instance.
            get_tool_payload_fn: Callable() -> list  — returns the OpenAI-schema
                                tool list for the current agent's registered tools.
        """
        self.framework_name = framework_name
        self._generate = generate_fn
        self._get_tool_payload = get_tool_payload_fn

        config = ConfigExpert.get_instance()
        workflow_cfg = config.get("workflow", {})

        self.max_steps: int = config.get("max_steps", 2)
        self.max_tokens: int = config.get("max_tokens", 1024)

        raw_nodes: list = workflow_cfg.get("nodes", [])
        if not raw_nodes:
            raise ValueError(
                "No nodes defined under 'workflow.nodes' in the configuration file."
            )

        self._nodes: Dict[str, NodeConfig] = {
            n["name"]: NodeConfig(n) for n in raw_nodes
        }
        self._entry_node: str = workflow_cfg.get("entry_node", raw_nodes[0]["name"])
        self._exit_node: str  = workflow_cfg.get("exit_node",  raw_nodes[-1]["name"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        tools: dict,
        role: str,
        reasoning_steps: List[ReasoningStep],
    ) -> dict:
        """
        Execute the configured workflow and return the final state dict.

        Args:
            task:           The task string passed to the agent.
            tools:          The agent's registered tool objects (keyed by name).
            role:           The agent's role description (used in prompt rendering).
            reasoning_steps: The agent's shared reasoning_steps list — mutated in place.

        Returns:
            Final state dict. The value under the exit node's output_key is
            the agent's final answer.
        """
        state: dict = {
            "task":             task,
            "plan":             "",
            "research_results": [],
            "final_report":     "",
            "step":             0,
            "tools":            tools,
            "max_steps":        self.max_steps,
        }

        # Walk nodes in declaration order
        node_list = list(self._nodes.values())
        for node in node_list:
            if node.loop:
                # First execution always runs; then loop while step <= max_steps
                state = self._execute_node(node, state, role, reasoning_steps)
                while state["step"] <= state["max_steps"]:
                    reasoning_steps.append(ReasoningStep(
                        step_number=len(reasoning_steps) + 1,
                        thought=f"Continuing loop node '{node.name}' (step {state['step']})",
                        observation="More iterations needed"
                    ))
                    state = self._execute_node(node, state, role, reasoning_steps)

                reasoning_steps.append(ReasoningStep(
                    step_number=len(reasoning_steps) + 1,
                    thought=f"Loop node '{node.name}' complete",
                    observation=f"Completed {state['step'] - 1} iteration(s)"
                ))
            else:
                state = self._execute_node(node, state, role, reasoning_steps)

        return state

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    def _execute_node(
        self,
        node: NodeConfig,
        state: dict,
        role: str,
        reasoning_steps: List[ReasoningStep],
    ) -> dict:
        """Render the node's prompt, call generate, and update state."""

        reasoning_steps.append(ReasoningStep(
            step_number=len(reasoning_steps) + 1,
            thought=f"Executing node '{node.name}'",
            action=node.name,
            action_input=state["task"][:200]
        ))

        tools_text = self._build_tools_text(state["tools"])

        # Render research_results as joined text for the report node
        research_results_text = "\n\n".join(
            r for r in state.get("research_results", []) if r
        )

        variables = {
            "framework":       self.framework_name,
            "role":            role,
            "task":            state["task"],
            "plan":            state.get("plan", ""),
            "step":            state.get("step", ""),
            "tools_text":      tools_text,
            "research_results": research_results_text,
        }

        prompt = _render(node.template_text, variables)

        # Nodes that use tools get the tool payload; the report node does not
        use_tools = bool(state["tools"]) and node.name != self._exit_node
        kwargs = {"max_tokens": self.max_tokens}
        if use_tools:
            kwargs["tools"] = self._get_tool_payload()

        try:
            output = self._generate(prompt, **kwargs)
            output = "" if output is None else output

            # Update state based on output_key
            if node.output_key == "research_results":
                state["research_results"] = state.get("research_results", []) + [output]
                state["step"] = state.get("step", 0) + 1
            else:
                state[node.output_key] = output

            reasoning_steps.append(ReasoningStep(
                step_number=len(reasoning_steps) + 1,
                thought=f"Node '{node.name}' completed",
                observation=f"Output: {output[:100]}..."
            ))

        except Exception as e:
            error_msg = f"{node.name} error: {str(e)}"
            if node.output_key == "research_results":
                state["research_results"] = state.get("research_results", []) + [error_msg]
                state["step"] = state.get("step", 0) + 1
            else:
                state[node.output_key] = error_msg

            reasoning_steps.append(ReasoningStep(
                step_number=len(reasoning_steps) + 1,
                thought=f"Error in node '{node.name}'",
                observation=str(e)
            ))

        return state

    def _build_tools_text(self, tools: dict) -> str:
        """Build a human-readable tools listing from the agent's tool dict."""
        if not tools:
            return "None"
        lines = []
        for tool in tools.values():
            name = getattr(tool, "name", None) or getattr(tool, "tool_name", str(tool))
            desc = getattr(tool, "description", "No description available.")
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)
