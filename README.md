# MASTIF (Multi-Agent System TestIng Framework)

<div align="center">
  <img src="images/mastif-logo.png" width="250px" alt="Black and white logo showing the head of a mastiff breed dog looking forward">
</div>

## Overview

MASTIF is a benchmarking framework for comparing agentic AI stacks across model providers, frameworks, protocols, workflows, and tool ecosystems. It is designed for reproducible evaluations of reasoning, planning, tool use, and web automation on both custom tasks and the [Mind2Web benchmark](https://github.com/OSU-NLP-Group/Mind2Web).

The project supports the same workflow being executed under multiple conditions so you can compare:

- model families and providers
- communication protocols
- framework implementations
- tool availability and prompting strategies
- benchmark vs. custom task performance

## Current key capabilities

MASTIF currently includes the following capabilities, all reflected in the codebase:

- Multi-provider model support for Hugging Face, OpenAI, Anthropic, and local Ollama deployments.
- Multi-framework evaluation across CrewAI, Smolagents, LangChain, LangGraph, LlamaIndex, and Semantic Kernel.
- Protocol comparison for STANDARD, MCP, A2A, and ACP message formats.
- YAML-driven workflow orchestration with node sequencing, looping, output-key propagation, and placeholder validation.
- ReAct-style plain-text tool execution for models that do not expose native structured tool-calling APIs.
- Cross-framework tool pooling where one tool definition is adapted to each framework at runtime.
- Large-scale benchmark support for Mind2Web with judge-based scoring and partial-result resume support.
- Preflight validation for configured models, tokenizers, judge models, and dataset access before a run begins.
- Detailed metrics collection for token usage, latency, reasoning steps, and tool-call statistics.
- JSON result export and CSV conversion utilities for downstream analysis.
- Partial experiment saving so long runs can be resumed without restarting the full benchmark.

## Component and class structure

![Component Diagram](images/component_diagram.png)

![Class Diagram](images/class_diagram.png)

## Installation

### 1. Check Python and create a virtual environment

A Python 3.12 environment is known to work for the current dependency set. From the repository root:

```bash
python3 --version
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Always run MASTIF from the repository root with the virtual environment activated.

For browser-based tooling and web-interaction tasks, install the Chromium runtime:

```bash
python -m playwright install chromium
```

## Environment variables

Set the credentials required by your selected provider:

```bash
export HF_TOKEN='your_huggingface_token'
export OPENAI_API_KEY='your_openai_key'
export ANTHROPIC_API_KEY='your_anthropic_key'
```

For a local Ollama server, the default endpoint is `http://127.0.0.1:11434`. Override it when using a remote or tunneled Ollama service:

```bash
export OLLAMA_URL='http://127.0.0.1:11434'
```

Notes:

- Hugging Face models require `HF_TOKEN`.
- OpenAI models such as `gpt-*` require `OPENAI_API_KEY`.
- Anthropic models such as `claude-*` require `ANTHROPIC_API_KEY`.
- Ollama model IDs contain a colon, such as `llama3.2:latest`, and require a reachable Ollama server; they do not require a cloud API key.
- Mind2Web dataset access requires `HF_TOKEN` and accepted access to `osunlp/Mind2Web`.

## Quick start

Copy the example experiment and customize it:

```bash
cp experiments/example.yaml experiments/my-experiment.yaml
python main.py experiments/my-experiment.yaml
```

You can also run the project with the default example config:

```bash
python main.py
```

The first positional argument is always the YAML experiment file; there is currently no separate `--help` mode or default argument parser. For CSV or prompt-recommendation utilities, use the following commands instead:

### Convert JSON results to CSV

```bash
python main.py csv logs/results-[experiment]-[timestamp].json
```

This writes a sibling CSV next to the original log file.

### Export prompt recommendations to CSV

```bash
python main.py recommend-csv logs/results-[experiment]-[timestamp].json
```

## Supported model providers and example IDs

MASTIF is not limited to one vendor. The runtime adapters support:

- Hugging Face model IDs such as `meta-llama/Llama-3.3-70B-Instruct`
- OpenAI models such as `gpt-4o`, `gpt-4o-mini`, `gpt-oss-*`
- Anthropic models such as `claude-3-5-sonnet-20241022`
- Ollama models using the local format `llama3.2:latest` or similar local names

Model entries in YAML may be plain strings or dictionaries with an optional tokenizer override:

```yaml
models:
  - id: "gpt-4o"
  - id: "meta-llama/Llama-3.3-70B-Instruct"
  - id: "llama3.2:latest"
  - id: "Qwen/Qwen2.5-7B-Instruct"
    tokenizer: "Qwen/Qwen2.5-7B-Instruct"
```

## Supported frameworks

MASTIF evaluates the same workflow across multiple framework implementations:

- Lightweight 
  - Smolagents

- Role-based
  - CrewAI

- Pipeline-based
  - LangChain
  - LlamaIndex

- Graph-based
  - LangGraph
  - SemanticKernel

This is implemented through framework-specific wrappers around a shared tool pool and workflow controller.

## Supported protocols

The framework includes protocol-aware prompt wrapping and overhead measurement for:

- `STANDARD`: direct API call without protocol injection
- `MCP`: Model Context Protocol payloads
- `A2A`: Agent-to-Agent message payloads
- `ACP`: Agent Communication Protocol payloads

These are defined in [protocols.py](protocols.py) and are measured for send/receive overhead and message size.

## Workflow configuration

MASTIF executes workflows from YAML. Each workflow node can reference a prompt template, store output in an output key, and optionally loop until the configured step limit is reached.

Example:

```yaml
workflow:
  nodes:
    - name: "plan"
      prompt_template: "prompts/plan.txt"
      output_key: "plan"
      loop: false

    - name: "research"
      prompt_template: "prompts/research.txt"
      output_key: "research_results"
      loop: true

    - name: "report"
      prompt_template: "prompts/report.txt"
      output_key: "final_report"
      loop: false

  entry_node: "plan"
  exit_node: "report"
```

The workflow engine also validates placeholder continuity between nodes and resolves prompt templates relative to the active experiment config file.

## ReAct and plain-text tool execution

MASTIF includes a built-in ReAct parser that can detect and execute tool calls embedded in plain text, even when a model does not support native tool calling.

Example pattern:

```text
Thought: I need to search for the answer.
Action: web_search
Action Input: {"query": "accessibility guidelines"}
Observation: ...
```

The parser extracts the action and argument payload, invokes the registered tool, and appends the observation back into the model context. This helps broaden compatibility with open-source and fine-tuned models.

## Tool pool and built-in tools

The project uses a shared tool registry and adapts the same tool definition to each framework automatically. The built-in tool set includes:

### Core utility and browsing
- `web_search`
- `web_browser`
- `requests_get`
- `beautifulsoup_scraper`
- `web_interaction`
- `keyboard_interaction`
- `searxng_search`

### Knowledge and information retrieval
- `wikipedia`
- `arxiv`
- `pubmed`
- `youtube_transcript`

### Computation and parsing
- `python_repl`
- `sympy`
- `json_parser`
- `pdf_reader`
- `datetime`

This is implemented in [tool_pool.py](tool_pool.py) and supports per-framework conversions for LangChain, CrewAI, Smolagents, LlamaIndex, and Semantic Kernel.

## Configuration reference

The example config in [experiments/example.yaml](experiments/example.yaml) shows the current supported settings:

- `experiment.name`, `experiment.description`
- `temperature`, `max_tokens`, `max_steps`, `max_tool_rounds`, `timeout`
- `workflow.nodes`, `workflow.entry_node`, `workflow.exit_node`
- `test_mode`: `standard` or `mind2web`
- `requests_soft_limit`
- `models`
- `protocols`
- `frameworks`
- `tools`

The same config file powers the runtime, prompt templates, and logger metadata.

## Benchmark modes

### Standard mode

Runs user-defined tasks across all selected models, protocols, and frameworks.

### Mind2Web mode

Runs the [Mind2Web benchmark](https://github.com/OSU-NLP-Group/Mind2Web) and evaluates task completion with a judge model. In this mode, MASTIF validates the dataset and judge availability before execution.

## Output and logs

Results are written to the `logs/` directory.

### Standard mode

- `logs/results-[experiment]-[timestamp].json`: final results exported on clean completion
- `logs/partial-[experiment]-[timestamp].json`: result-by-result partial progress used for resume prompts

### Mind2Web mode

- `logs/mind2web-results-[timestamp].json`: benchmark-specific aggregate and task-level results
- `logs/results-[experiment]-[timestamp].json`: general result export, also written on clean completion
- `logs/partial-[experiment]-[timestamp].json`: partial benchmark progress used for resume prompts

The logger finds the most recent `logs/partial-[experiment]-*.json` file for the active YAML configuration. Starting the same experiment again offers to resume completed, successful combinations and retries entries that failed or returned retryable errors.

### Metrics collected

- reasoning steps and thought traces
- total, input, and output token counts
- latency and protocol overhead
- task completion and understanding scores
- tool usage summaries and per-tool statistics
- model/framework/protocol breakdowns

## Preflight validation

Before a run starts, [preflight.py](preflight.py) checks:

- configured model availability
- tokenizer availability or fallback behavior
- required API keys for cloud providers
- Mind2Web dataset access when needed
- judge model validity in benchmark mode

This helps catch configuration problems before a long benchmark run begins.

## Current notes and caveats

- Mind2Web requires a valid Hugging Face token and access to the dataset.
- Browser and web-interaction tools require the Playwright browser runtime.
- Long benchmark runs should be allowed to save partial logs or resumed from previous results.
- Tool execution performance depends on the selected model’s native tool-calling behavior; the ReAct fallback expands compatibility for less capable models.

## Example output

- Standard mode example: [out-standard.txt](out-standard.txt)
- Mind2Web example: [out-mind2web.txt](out-mind2web.txt)


## Presentations

### 2026

- Vagner Figueredo de Santana. "[Avaliando Sistemas Multi-Agentes com MASTIF](./presentations/2026-semanadadedados-tutorial.pptx)". Semana de Dados BR 2026. Brasília, Brazil. 

## Publications

### 2026

- Vagner Figueredo de Santana and Diogo Cortiz. "A Baseline Assessment for the Agentic Web Technology Stack." arXiv preprint, 2026. [arXiv:2610.13703](https://arxiv.org/abs/2610.13703)

- Vagner Figueredo de Santana, Diogo Cortiz, Beatriz Rocha, and Henrique S. Xavier. "A Multi-Agent Query Fan-Out Analysis of Popular Google Searches in Brazil." In Proceedings of the 32nd Brazilian Symposium on Multimedia and the Web (WebMedia 2026). DOI: [webmedia.2026.30219](https://sol.sbc.org.br/index.php/webmedia/article/view/46071)

- Vagner Santana, Reinaldo Ferraz, and Jeniffer Vieira de Deus. "On the Limits for Web Accessibility Evaluation Using AI Agents." In Proceedings of the 23rd International Web for All Conference (W4A 2026). DOI: [10.1145/3800424.3800445](https://doi.org/10.1145/3800424.3800445)

## AI attribution

<a href="https://aiattribution.github.io/statements/AIA-HAb-Ce-Hin-R-?model=Copilot,%20ChatGPT%205.2,%20Gemini%203,%20Sonnet%204.5,%20and%20Sonnet%204.6%20v1.0" target="_blank">AIA Human-AI blend, Content edits, Human-initiated, Reviewed, Copilot, ChatGPT 5.2, Gemini 3, Sonnet 4.5, and Sonnet 4.6 v1.0</a>

More info: https://aiattribution.github.io/