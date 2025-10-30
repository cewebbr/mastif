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
- Task Understanding: Agent's comprehension of the task
- Task Deviation: Agent's adherence to the task in reasoning steps
- Task Completion: Agent's performance on fulfilling the task
- Reasoning Steps: Number of intermediate reasoning steps
- Domain-specific performance breakdowns

## Example Output

```
======================================================================
MIND2WEB BENCHMARK MODE
======================================================================

Configuration:
  Models: 1
  Tasks: 10
  Frameworks: All 6 frameworks
Loading Mind2Web dataset (split: train)...
Attempting to load with streaming mode...
Streaming mode failed: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
Falling back to sample data...

======================================================================
USING SAMPLE MIND2WEB DATA
======================================================================
Note: Full dataset could not be loaded due to formatting issues.
Using representative sample tasks for demonstration.

✓ Loaded 15 sample tasks
These tasks represent typical Mind2Web scenarios.

Sampled 10 tasks from 15 total tasks

======================================================================
MIND2WEB BENCHMARK EVALUATION
======================================================================

Dataset Statistics:
  Total Tasks: 10
  Unique Domains: 5
  Unique Websites: 10
  Avg Actions per Task: 0.0

Top Domains:
  - entertainment: 3 tasks
  - social_media: 3 tasks
  - travel: 2 tasks
  - shopping: 1 tasks
  - general: 1 tasks

======================================================================
Testing Model: mistralai/Mistral-Nemo-Base-2407
======================================================================

----------------------------------------------------------------------
Framework: CrewAI
----------------------------------------------------------------------

  Task 1/10: netflix.com (entertainment)
  Goal: Search for sci-fi movies and add the first result to my list...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 2/10: reddit.com (social_media)
  Goal: Find the top post in r/programming and upvote it...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 3/10: amazon.com (shopping)
  Goal: Find the cheapest laptop under $800 and add it to cart...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 4/10: airbnb.com (travel)
  Goal: Find entire apartments in Tokyo for 2 guests from Jan 10-15...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 5/10: linkedin.com (social_media)
  Goal: Search for Software Engineer jobs in San Francisco and filter by remote...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 6/10: booking.com (travel)
  Goal: Find a hotel in Paris for 2 adults, check-in Dec 15, check-out Dec 20...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 7/10: twitter.com (social_media)
  Goal: Search for tweets about 'AI news' and retweet the most recent one...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 8/10: wikipedia.org (general)
  Goal: Search for 'Artificial Intelligence' and read the introduction...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 9/10: youtube.com (entertainment)
  Goal: Search for 'Python tutorial' videos and play the most viewed one...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 10/10: imdb.com (entertainment)
  Goal: Find the highest rated action movies from 2023...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  CrewAI Mind2Web Summary:
    total_tasks_evaluated: 10
    judge_model: gpt-4o-mini
    avg_task_understanding: 1.0
    avg_task_deviation: 1.0
    avg_task_completion: 1.0
    avg_overall_score: 0.6666666666666666
    avg_reasoning_steps: 3.0

----------------------------------------------------------------------
Framework: Smolagents
----------------------------------------------------------------------

  Task 1/10: netflix.com (entertainment)
  Goal: Search for sci-fi movies and add the first result to my list...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 4

  Task 2/10: reddit.com (social_media)
  Goal: Find the top post in r/programming and upvote it...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 4

  Task 3/10: amazon.com (shopping)
  Goal: Find the cheapest laptop under $800 and add it to cart...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 4

  Task 4/10: airbnb.com (travel)
  Goal: Find entire apartments in Tokyo for 2 guests from Jan 10-15...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 4

  Task 5/10: linkedin.com (social_media)
  Goal: Search for Software Engineer jobs in San Francisco and filter by remote...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 4

  Task 6/10: booking.com (travel)
  Goal: Find a hotel in Paris for 2 adults, check-in Dec 15, check-out Dec 20...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 4

  Task 7/10: twitter.com (social_media)
  Goal: Search for tweets about 'AI news' and retweet the most recent one...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 4

  Task 8/10: wikipedia.org (general)
  Goal: Search for 'Artificial Intelligence' and read the introduction...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 4

  Task 9/10: youtube.com (entertainment)
  Goal: Search for 'Python tutorial' videos and play the most viewed one...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 4

  Task 10/10: imdb.com (entertainment)
  Goal: Find the highest rated action movies from 2023...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 4

  Smolagents Mind2Web Summary:
    total_tasks_evaluated: 20
    judge_model: gpt-4o-mini
    avg_task_understanding: 1.0
    avg_task_deviation: 1.0
    avg_task_completion: 1.0
    avg_overall_score: 0.6666666666666666
    avg_reasoning_steps: 3.5

----------------------------------------------------------------------
Framework: LangChain
----------------------------------------------------------------------

  Task 1/10: netflix.com (entertainment)
  Goal: Search for sci-fi movies and add the first result to my list...
/Users/vsantana/Documents/projects/agentic web/code/agentic-stack-tester/adapters.py:50: LangChainDeprecationWarning: The class `HuggingFaceEndpoint` was deprecated in LangChain 0.0.37 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-huggingface package and should be used instead. To use it run `pip install -U :class:`~langchain-huggingface` and import as `from :class:`~langchain_huggingface import HuggingFaceEndpoint``.
  return HuggingFaceEndpoint(
Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured.


> Entering new AgentExecutor chain...
/Users/vsantana/Documents/projects/agentic web/code/agentic-stack-tester/venv/lib/python3.12/site-packages/huggingface_hub/inference/_client.py:2308: FutureWarning: `stop_sequences` is a deprecated argument for `text_generation` task and will be removed in version '0.28.0'. Use `stop` instead.
  warnings.warn(
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 2/10: reddit.com (social_media)
  Goal: Find the top post in r/programming and upvote it...
Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured.


> Entering new AgentExecutor chain...
/Users/vsantana/Documents/projects/agentic web/code/agentic-stack-tester/venv/lib/python3.12/site-packages/huggingface_hub/inference/_client.py:2308: FutureWarning: `stop_sequences` is a deprecated argument for `text_generation` task and will be removed in version '0.28.0'. Use `stop` instead.
  warnings.warn(
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 3/10: amazon.com (shopping)
  Goal: Find the cheapest laptop under $800 and add it to cart...
Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured.


> Entering new AgentExecutor chain...
/Users/vsantana/Documents/projects/agentic web/code/agentic-stack-tester/venv/lib/python3.12/site-packages/huggingface_hub/inference/_client.py:2308: FutureWarning: `stop_sequences` is a deprecated argument for `text_generation` task and will be removed in version '0.28.0'. Use `stop` instead.
  warnings.warn(
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 4/10: airbnb.com (travel)
  Goal: Find entire apartments in Tokyo for 2 guests from Jan 10-15...
Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured.


> Entering new AgentExecutor chain...
/Users/vsantana/Documents/projects/agentic web/code/agentic-stack-tester/venv/lib/python3.12/site-packages/huggingface_hub/inference/_client.py:2308: FutureWarning: `stop_sequences` is a deprecated argument for `text_generation` task and will be removed in version '0.28.0'. Use `stop` instead.
  warnings.warn(
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 5/10: linkedin.com (social_media)
  Goal: Search for Software Engineer jobs in San Francisco and filter by remote...
Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured.


> Entering new AgentExecutor chain...
/Users/vsantana/Documents/projects/agentic web/code/agentic-stack-tester/venv/lib/python3.12/site-packages/huggingface_hub/inference/_client.py:2308: FutureWarning: `stop_sequences` is a deprecated argument for `text_generation` task and will be removed in version '0.28.0'. Use `stop` instead.
  warnings.warn(
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 6/10: booking.com (travel)
  Goal: Find a hotel in Paris for 2 adults, check-in Dec 15, check-out Dec 20...
Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured.


> Entering new AgentExecutor chain...
/Users/vsantana/Documents/projects/agentic web/code/agentic-stack-tester/venv/lib/python3.12/site-packages/huggingface_hub/inference/_client.py:2308: FutureWarning: `stop_sequences` is a deprecated argument for `text_generation` task and will be removed in version '0.28.0'. Use `stop` instead.
  warnings.warn(
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 7/10: twitter.com (social_media)
  Goal: Search for tweets about 'AI news' and retweet the most recent one...
Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured.


> Entering new AgentExecutor chain...
/Users/vsantana/Documents/projects/agentic web/code/agentic-stack-tester/venv/lib/python3.12/site-packages/huggingface_hub/inference/_client.py:2308: FutureWarning: `stop_sequences` is a deprecated argument for `text_generation` task and will be removed in version '0.28.0'. Use `stop` instead.
  warnings.warn(
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 8/10: wikipedia.org (general)
  Goal: Search for 'Artificial Intelligence' and read the introduction...
Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured.


> Entering new AgentExecutor chain...
/Users/vsantana/Documents/projects/agentic web/code/agentic-stack-tester/venv/lib/python3.12/site-packages/huggingface_hub/inference/_client.py:2308: FutureWarning: `stop_sequences` is a deprecated argument for `text_generation` task and will be removed in version '0.28.0'. Use `stop` instead.
  warnings.warn(
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 9/10: youtube.com (entertainment)
  Goal: Search for 'Python tutorial' videos and play the most viewed one...
Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured.


> Entering new AgentExecutor chain...
/Users/vsantana/Documents/projects/agentic web/code/agentic-stack-tester/venv/lib/python3.12/site-packages/huggingface_hub/inference/_client.py:2308: FutureWarning: `stop_sequences` is a deprecated argument for `text_generation` task and will be removed in version '0.28.0'. Use `stop` instead.
  warnings.warn(
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 10/10: imdb.com (entertainment)
  Goal: Find the highest rated action movies from 2023...
Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured.


> Entering new AgentExecutor chain...
/Users/vsantana/Documents/projects/agentic web/code/agentic-stack-tester/venv/lib/python3.12/site-packages/huggingface_hub/inference/_client.py:2308: FutureWarning: `stop_sequences` is a deprecated argument for `text_generation` task and will be removed in version '0.28.0'. Use `stop` instead.
  warnings.warn(
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  LangChain Mind2Web Summary:
    total_tasks_evaluated: 30
    judge_model: gpt-4o-mini
    avg_task_understanding: 1.0
    avg_task_deviation: 1.0
    avg_task_completion: 1.0
    avg_overall_score: 0.6666666666666666
    avg_reasoning_steps: 3.3333333333333335

----------------------------------------------------------------------
Framework: LangGraph
----------------------------------------------------------------------

  Task 1/10: netflix.com (entertainment)
  Goal: Search for sci-fi movies and add the first result to my list...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 11

  Task 2/10: reddit.com (social_media)
  Goal: Find the top post in r/programming and upvote it...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 11

  Task 3/10: amazon.com (shopping)
  Goal: Find the cheapest laptop under $800 and add it to cart...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 11

  Task 4/10: airbnb.com (travel)
  Goal: Find entire apartments in Tokyo for 2 guests from Jan 10-15...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 11

  Task 5/10: linkedin.com (social_media)
  Goal: Search for Software Engineer jobs in San Francisco and filter by remote...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 11

  Task 6/10: booking.com (travel)
  Goal: Find a hotel in Paris for 2 adults, check-in Dec 15, check-out Dec 20...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 11

  Task 7/10: twitter.com (social_media)
  Goal: Search for tweets about 'AI news' and retweet the most recent one...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 11

  Task 8/10: wikipedia.org (general)
  Goal: Search for 'Artificial Intelligence' and read the introduction...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 11

  Task 9/10: youtube.com (entertainment)
  Goal: Search for 'Python tutorial' videos and play the most viewed one...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 11

  Task 10/10: imdb.com (entertainment)
  Goal: Find the highest rated action movies from 2023...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 11

  LangGraph Mind2Web Summary:
    total_tasks_evaluated: 40
    judge_model: gpt-4o-mini
    avg_task_understanding: 1.0
    avg_task_deviation: 1.0
    avg_task_completion: 1.0
    avg_overall_score: 0.6666666666666666
    avg_reasoning_steps: 5.25

----------------------------------------------------------------------
Framework: LlamaIndex
----------------------------------------------------------------------

  Task 1/10: netflix.com (entertainment)
  Goal: Search for sci-fi movies and add the first result to my list...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 2/10: reddit.com (social_media)
  Goal: Find the top post in r/programming and upvote it...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 3/10: amazon.com (shopping)
  Goal: Find the cheapest laptop under $800 and add it to cart...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 4/10: airbnb.com (travel)
  Goal: Find entire apartments in Tokyo for 2 guests from Jan 10-15...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 5/10: linkedin.com (social_media)
  Goal: Search for Software Engineer jobs in San Francisco and filter by remote...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 6/10: booking.com (travel)
  Goal: Find a hotel in Paris for 2 adults, check-in Dec 15, check-out Dec 20...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 7/10: twitter.com (social_media)
  Goal: Search for tweets about 'AI news' and retweet the most recent one...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 8/10: wikipedia.org (general)
  Goal: Search for 'Artificial Intelligence' and read the introduction...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 9/10: youtube.com (entertainment)
  Goal: Search for 'Python tutorial' videos and play the most viewed one...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 10/10: imdb.com (entertainment)
  Goal: Find the highest rated action movies from 2023...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  LlamaIndex Mind2Web Summary:
    total_tasks_evaluated: 50
    judge_model: gpt-4o-mini
    avg_task_understanding: 1.0
    avg_task_deviation: 0.8
    avg_task_completion: 1.0
    avg_overall_score: 0.7333333333333333
    avg_reasoning_steps: 4.2

----------------------------------------------------------------------
Framework: SemanticKernel
----------------------------------------------------------------------

  Task 1/10: netflix.com (entertainment)
  Goal: Search for sci-fi movies and add the first result to my list...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 2/10: reddit.com (social_media)
  Goal: Find the top post in r/programming and upvote it...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 3/10: amazon.com (shopping)
  Goal: Find the cheapest laptop under $800 and add it to cart...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 4/10: airbnb.com (travel)
  Goal: Find entire apartments in Tokyo for 2 guests from Jan 10-15...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 5/10: linkedin.com (social_media)
  Goal: Search for Software Engineer jobs in San Francisco and filter by remote...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 6/10: booking.com (travel)
  Goal: Find a hotel in Paris for 2 adults, check-in Dec 15, check-out Dec 20...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 7/10: twitter.com (social_media)
  Goal: Search for tweets about 'AI news' and retweet the most recent one...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 8/10: wikipedia.org (general)
  Goal: Search for 'Artificial Intelligence' and read the introduction...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 9/10: youtube.com (entertainment)
  Goal: Search for 'Python tutorial' videos and play the most viewed one...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  Task 10/10: imdb.com (entertainment)
  Goal: Find the highest rated action movies from 2023...
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 0.00%
      Task Completion: 100.00%
      Overall Score: 100.00%
      Reasoning Steps: 0

  SemanticKernel Mind2Web Summary:
    total_tasks_evaluated: 60
    judge_model: gpt-4o-mini
    avg_task_understanding: 1.0
    avg_task_deviation: 0.6666666666666666
    avg_task_completion: 1.0
    avg_overall_score: 0.7777777777777778
    avg_reasoning_steps: 3.5

======================================================================
MIND2WEB EVALUATION SUMMARY
======================================================================

Overall Performance:
  Judge Model: gpt-4o-mini
  Tasks Evaluated: 60
  Avg Task Understanding: 100.00%
  Avg Task Deviation: 66.67% (lower is better)
  Avg Task Completion: 100.00%
  Avg Overall Score: 77.78%
  Avg Reasoning Steps: 3.5

Performance by Domain:
  entertainment:
    Tasks: 18
    Avg Understanding: 100.00%
    Avg Task Deviation: 66.67% (lower is better)
    Avg Completion: 100.00%
  social_media:
    Tasks: 18
    Avg Understanding: 100.00%
    Avg Task Deviation: 66.67% (lower is better)
    Avg Completion: 100.00%
  shopping:
    Tasks: 6
    Avg Understanding: 100.00%
    Avg Task Deviation: 66.67% (lower is better)
    Avg Completion: 100.00%
  travel:
    Tasks: 12
    Avg Understanding: 100.00%
    Avg Task Deviation: 66.67% (lower is better)
    Avg Completion: 100.00%
  general:
    Tasks: 6
    Avg Understanding: 100.00%
    Avg Task Deviation: 66.67% (lower is better)
    Avg Completion: 100.00%

======================================================================
TEST SUMMARY
======================================================================

Overall Statistics:
  Total Tests: 60
  Successful: 40 (66.7%)
  Failed: 20 (33.3%)
  Total Reasoning Steps: 210
  Avg Reasoning Steps per Test: 3.5

----------------------------------------------------------------------
Results by Framework:
----------------------------------------------------------------------

  CrewAI:
    Success Rate: 10/10 (100.0%)
    Avg Latency: 0.15s
    Avg Reasoning Steps: 3.0

  LangChain:
    Success Rate: 10/10 (100.0%)
    Avg Latency: 0.31s
    Avg Reasoning Steps: 3.0

  LangGraph:
    Success Rate: 10/10 (100.0%)
    Avg Latency: 0.66s
    Avg Reasoning Steps: 11.0

  LlamaIndex:
    Success Rate: 0/10 (0.0%)
    Avg Latency: 0.00s
    Avg Reasoning Steps: 0.0

  SemanticKernel:
    Success Rate: 0/10 (0.0%)
    Avg Latency: 0.00s
    Avg Reasoning Steps: 0.0

  Smolagents:
    Success Rate: 10/10 (100.0%)
    Avg Latency: 0.16s
    Avg Reasoning Steps: 4.0

----------------------------------------------------------------------
Results by Protocol:
----------------------------------------------------------------------

  standard:
    Success Rate: 40/60 (66.7%)
    Avg Latency: 0.32s

----------------------------------------------------------------------
Results by Model:
----------------------------------------------------------------------

  mistralai/Mistral-Nemo-Base-2407:
    Success Rate: 40/60 (66.7%)
    Avg Latency: 0.32s
    Avg Reasoning Steps: 3.5

======================================================================
✓ Mind2Web results exported to ./logs/mind2web-results-20251030_091917.json

✓ Results exported to ./logs/results-20251030_091917.json

======================================================================
Mind2Web evaluation complete!
Results: ./logs/mind2web-results-20251030_091917.json
Full logs: ./logs/results-20251030_091917.json
======================================================================
```

## Notes

- Mind2Web requires authentication with HuggingFace
- The test set requires accepting terms on HuggingFace
- Focus is on task understanding and action planning capabilities

## AI Attribution

AIA Human-AI blend, Content edits, Human-initiated, Reviewed, GPT-4.1 and Sonet 4.5 v1.0
More info: https://aiattribution.github.io/create-attribution