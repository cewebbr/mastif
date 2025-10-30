# Agentic Stack Testing Framework

![Class Diagram](class_diagram.png)

## Overview
This framework now supports evaluation on the Mind2Web benchmark, which tests web agent capabilities on real-world website tasks.



## Installation

```bash
# Install additional dependency for Mind2Web
pip install datasets

# Existing dependencies
pip install huggingface_hub transformers torch
pip install openai
pip install crewai smolagents
pip install langchain langchain-community langgraph
pip install llama-index
pip install semantic-kernel
```

## Usage

### Standard Mode (Original Testing)
```bash
export HF_TOKEN='your_token'
export OPEN_AI_KEY='your_key'
export JUDGE_MODEL='gpt-4o-mini'
export TEST_MODE='standard'
python main.py
```

### Mind2Web Evaluation Mode
```bash
export HF_TOKEN='your_token'
export OPEN_AI_KEY='your_key'
export JUDGE_MODEL='gpt-4o-mini'
export TEST_MODE='mind2web'
export MIND2WEB_NUM_TASKS=10  # Use 10 tasks (default)
python main.py
```

### Test All Mind2Web Tasks
```bash
export HF_TOKEN='your_token'
export OPEN_AI_KEY='your_key'
export JUDGE_MODEL='gpt-4o-mini'
export TEST_MODE='mind2web'
export MIND2WEB_NUM_TASKS=0  # 0 means use ALL tasks
python main.py
```

## Configuration

### Environment Variables
- `TEST_MODE`: 'standard' or 'mind2web'
- `MIND2WEB_NUM_TASKS`: Number of tasks (10-2350, or 0 for all)
- `HF_TOKEN`: Your HuggingFace API token
- `OPEN_AI_KEY`: Your OpenAI API key
- `JUDGE_MODEL`: OpenAI llm-as-a-judge model

### Sample Sizes
- **10 tasks**: Quick evaluation (~15 minutes)
- **50 tasks**: Medium evaluation (~1 hour)
- **100 tasks**: Comprehensive sample (~2 hours)
- **All tasks** (2,350): Full benchmark (~24+ hours)

## Output Files

### Mind2Web Mode
- `logs/mind2web-results-TIMESTAMP.json`: Mind2Web specific metrics
- `logs/results-TIMESTAMP.json`: Standard test results

### Metrics Included
- Action Coverage: % of ground truth actions covered
- Action Order Score: Correctness of action sequence
- Task Understanding: Agent's comprehension of the task
- Reasoning Steps: Number of intermediate reasoning steps
- Domain-specific performance breakdowns

## Example Output

```
MIND2WEB BENCHMARK EVALUATION
========================================
Dataset Statistics:
  Total Tasks: 10
  Unique Domains: 5
  Unique Websites: 8
  Avg Actions per Task: 3.2

Testing Model: meta-llama/Llama-3.2-3B-Instruct
Framework: LangChain
  Task 1/10: amazon.com (shopping)
  Goal: Find the cheapest laptop under $800...
    ✓ Completed
      Action Coverage: 75.00%
      Task Understanding: 82.00%
      Reasoning Steps: 5

Overall Performance:
  Tasks Evaluated: 10
  Avg Action Coverage: 68.50%
  Avg Task Understanding: 75.20%
```

## Notes

- Mind2Web requires authentication with HuggingFace
- The test set requires accepting terms on HuggingFace
- Focus is on task understanding and action planning capabilities

AIA Human-AI blend, Content edits, Human-initiated, Reviewed, GPT-4.1 and Sonet 4.5 v1.0
More info: https://aiattribution.github.io/create-attribution