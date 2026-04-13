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
import re
import json
from pathlib import Path
from typing import Dict, List, Callable, Optional, Tuple

from domain_model import ReasoningStep
from config import ConfigExpert
from tool_pool import ToolPool


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
# ReAct Parser — parse and execute plain-text ReAct format responses
# ---------------------------------------------------------------------------

def _parse_react_actions(response: str) -> List[Tuple[str, str]]:
    """
    Parse plain-text ReAct format responses to extract (action, action_input) pairs.
    
    Looks for patterns like:
        Action: tool_name
        Action Input: {"key": "value"}
    
    Returns list of (action_name, action_input_json_string) tuples.
    """
    actions = []
    # Match Action: tool_name followed by Action Input: {...}
    pattern = r'Action:\s*(\w+)\s*\n\s*Action Input:\s*(.+?)(?=\n(?:Thought:|Observation:|Action:|Final Answer:|$))'
    matches = re.finditer(pattern, response, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        action_name = match.group(1).strip()
        action_input_str = match.group(2).strip()
        actions.append((action_name, action_input_str))
    
    return actions


def _execute_react_actions(response: str, max_iterations: int = 3) -> str:
    """
    Execute tools found in ReAct-format response and append observations.
    
    If the response contains Action/Action Input pairs, execute them,
    collect observations, and append them back to the response.
    
    Args:
        response: The model's text response (may contain ReAct format)
        max_iterations: Max tool execution rounds to prevent infinite loops
    
    Returns:
        Enhanced response with tool execution results appended as Observations
    """
    for iteration in range(max_iterations):
        actions = _parse_react_actions(response)
        if not actions:
            # No more Actions to execute
            break
        
        # Execute each action
        for action_name, action_input_str in actions:
            try:
                # Parse the action input (try JSON, fall back to string)
                try:
                    action_args = json.loads(action_input_str)
                    if isinstance(action_args, dict):
                        result = ToolPool.invoke(action_name, **action_args)
                    else:
                        result = ToolPool.invoke(action_name, action_args)
                except json.JSONDecodeError:
                    # Not JSON, pass as string
                    result = ToolPool.invoke(action_name, action_input_str)
                
                # Append observation to response
                observation_text = f"\nObservation: {result}"
                response += observation_text
                
            except Exception as e:
                error_msg = f"\nObservation: Tool execution error: {str(e)}"
                response += error_msg
    
    return response


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

# TODO: Verify placeholders in prompt templates match the variables we inject in _execute_node
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

        # TODO: Support more complex workflow structures (hierarchies, branching, etc.)
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
        """
        # Initialize state with core variables
        state: dict = {
            "task":       task,
            "step":       0,
            "tools":      tools,
            "max_steps":  self.max_steps,
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
            else:
                state = self._execute_node(node, state, role, reasoning_steps)

        return state

    def _execute_node(
        self,
        node: NodeConfig,
        state: dict,
        role: str,
        reasoning_steps: List[ReasoningStep],
    ) -> dict:
        """Render the node's prompt dynamically, call generate, and update state."""

        reasoning_steps.append(ReasoningStep(
            step_number=len(reasoning_steps) + 1,
            thought=f"Executing node '{node.name}'",
            action=node.name,
            action_input=state["task"][:200]
        ))

        tools_text = self._build_tools_text(state["tools"])

        # --- DYNAMIC VARIABLE INJECTION ---
        # Start with base framework/agent variables
        variables = {
            "framework":  self.framework_name,
            "role":       role,
            "tools_text": tools_text,
        }

        # Inject everything currently in the state dict.
        # This makes any previous output_key (like 'navigation_status') 
        # available to the current node's prompt template.
        for key, value in state.items():
            if key == "research_results" and isinstance(value, list):
                # Special handling for history accumulation: join into a single string
                variables[key] = "\n\n".join(str(r) for r in value if r)
            elif key == "tools":
                continue  # Skip raw tool objects
            else:
                variables[key] = value

        # Render the template using the dynamic variable set
        prompt = _render(node.template_text, variables)

        # Nodes that use tools get the tool payload; the exit node does not
        use_tools = bool(state["tools"]) and node.name != self._exit_node
        kwargs = {"max_tokens": self.max_tokens}
        if use_tools:
            kwargs["tools"] = self._get_tool_payload()

        try:
            output = self._generate(prompt, **kwargs)
            output = "" if output is None else output
            
            # Parse and execute tools if response contains ReAct format
            if use_tools and ("Action:" in output or "Thought:" in output):
                output = _execute_react_actions(output)

            # Update state based on the output_key defined in YAML
            if node.output_key == "research_results":
                # List-based accumulation for history
                state["research_results"] = state.get("research_results", []) + [output]
                state["step"] = state.get("step", 0) + 1
            else:
                # Standard assignment for single-step results
                state[node.output_key] = output

            reasoning_steps.append(ReasoningStep(
                step_number=len(reasoning_steps) + 1,
                thought=f"Node '{node.name}' completed",
                observation=f"Output stored in '{node.output_key}'"
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

    @property
    def final_output_key(self) -> str:
        """Returns the output_key of the configured exit node."""
        return self._nodes[self._exit_node].output_key