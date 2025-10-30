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
PROTOCOL TESTS
----------------------------------------------------------------------

  Testing with MCP on task: Search for sci-fi movies and add the fir...
    ✓ Success (5 reasoning steps)

  Testing with MCP on task: Find the top post in r/programming and u...
    ✓ Success (5 reasoning steps)

  Testing with MCP on task: Find the cheapest laptop under $800 and ...
    ✓ Success (5 reasoning steps)

  Testing with MCP on task: Find entire apartments in Tokyo for 2 gu...
    ✓ Success (5 reasoning steps)

  Testing with MCP on task: Search for Software Engineer jobs in San...
    ✓ Success (5 reasoning steps)

  Testing with MCP on task: Find a hotel in Paris for 2 adults, chec...
    ✓ Success (5 reasoning steps)

  Testing with MCP on task: Search for tweets about 'AI news' and re...
    ✓ Success (5 reasoning steps)

  Testing with MCP on task: Search for 'Artificial Intelligence' and...
    ✓ Success (5 reasoning steps)

  Testing with MCP on task: Search for 'Python tutorial' videos and ...
    ✓ Success (5 reasoning steps)

  Testing with MCP on task: Find the highest rated action movies fro...
    ✓ Success (5 reasoning steps)

  Protocol MCP summary:
    Success rate: 10/10
    Avg latency: 0.16s
    Avg reasoning steps: 5.00
    
-----------------------------------

  Testing with A2A on task: Search for sci-fi movies and add the fir...
    ✓ Success (5 reasoning steps)

  Testing with A2A on task: Find the top post in r/programming and u...
    ✓ Success (5 reasoning steps)

  Testing with A2A on task: Find the cheapest laptop under $800 and ...
    ✓ Success (5 reasoning steps)

  Testing with A2A on task: Find entire apartments in Tokyo for 2 gu...
    ✓ Success (5 reasoning steps)

  Testing with A2A on task: Search for Software Engineer jobs in San...
    ✓ Success (5 reasoning steps)

  Testing with A2A on task: Find a hotel in Paris for 2 adults, chec...
    ✓ Success (5 reasoning steps)

  Testing with A2A on task: Search for tweets about 'AI news' and re...
    ✓ Success (5 reasoning steps)

  Testing with A2A on task: Search for 'Artificial Intelligence' and...
    ✓ Success (5 reasoning steps)

  Testing with A2A on task: Search for 'Python tutorial' videos and ...
    ✓ Success (5 reasoning steps)

  Testing with A2A on task: Find the highest rated action movies fro...
    ✓ Success (5 reasoning steps)

  Protocol A2A summary:
    Success rate: 10/10
    Avg latency: 0.16s
    Avg reasoning steps: 5.00
    
-----------------------------------

  Testing with ACP on task: Search for sci-fi movies and add the fir...
    ✓ Success (5 reasoning steps)

  Testing with ACP on task: Find the top post in r/programming and u...
    ✓ Success (5 reasoning steps)

  Testing with ACP on task: Find the cheapest laptop under $800 and ...
    ✓ Success (5 reasoning steps)

  Testing with ACP on task: Find entire apartments in Tokyo for 2 gu...
    ✓ Success (5 reasoning steps)

  Testing with ACP on task: Search for Software Engineer jobs in San...
    ✓ Success (5 reasoning steps)

  Testing with ACP on task: Find a hotel in Paris for 2 adults, chec...
    ✓ Success (5 reasoning steps)

  Testing with ACP on task: Search for tweets about 'AI news' and re...
    ✓ Success (5 reasoning steps)

  Testing with ACP on task: Search for 'Artificial Intelligence' and...
    ✓ Success (5 reasoning steps)

  Testing with ACP on task: Search for 'Python tutorial' videos and ...
    ✓ Success (5 reasoning steps)

  Testing with ACP on task: Find the highest rated action movies fro...
    ✓ Success (5 reasoning steps)

  Protocol ACP summary:
    Success rate: 10/10
    Avg latency: 0.15s
    Avg reasoning steps: 5.00
    
-----------------------------------

  Testing with standard on task: Search for sci-fi movies and add the fir...
    ✓ Success (3 reasoning steps)

  Testing with standard on task: Find the top post in r/programming and u...
    ✓ Success (3 reasoning steps)

  Testing with standard on task: Find the cheapest laptop under $800 and ...
    ✓ Success (3 reasoning steps)

  Testing with standard on task: Find entire apartments in Tokyo for 2 gu...
    ✓ Success (3 reasoning steps)

  Testing with standard on task: Search for Software Engineer jobs in San...
    ✓ Success (3 reasoning steps)

  Testing with standard on task: Find a hotel in Paris for 2 adults, chec...
    ✓ Success (3 reasoning steps)

  Testing with standard on task: Search for tweets about 'AI news' and re...
    ✓ Success (3 reasoning steps)

  Testing with standard on task: Search for 'Artificial Intelligence' and...
    ✓ Success (3 reasoning steps)

  Testing with standard on task: Search for 'Python tutorial' videos and ...
    ✓ Success (3 reasoning steps)

  Testing with standard on task: Find the highest rated action movies fro...
    ✓ Success (3 reasoning steps)

  Protocol standard summary:
    Success rate: 10/10
    Avg latency: 0.16s
    Avg reasoning steps: 3.00
    
-----------------------------------

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


[1m> Entering new AgentExecutor chain...[0m
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 2/10: reddit.com (social_media)
  Goal: Find the top post in r/programming and upvote it...


[1m> Entering new AgentExecutor chain...[0m
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 3/10: amazon.com (shopping)
  Goal: Find the cheapest laptop under $800 and add it to cart...


[1m> Entering new AgentExecutor chain...[0m
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 4/10: airbnb.com (travel)
  Goal: Find entire apartments in Tokyo for 2 guests from Jan 10-15...


[1m> Entering new AgentExecutor chain...[0m
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 5/10: linkedin.com (social_media)
  Goal: Search for Software Engineer jobs in San Francisco and filter by remote...


[1m> Entering new AgentExecutor chain...[0m
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 6/10: booking.com (travel)
  Goal: Find a hotel in Paris for 2 adults, check-in Dec 15, check-out Dec 20...


[1m> Entering new AgentExecutor chain...[0m
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 7/10: twitter.com (social_media)
  Goal: Search for tweets about 'AI news' and retweet the most recent one...


[1m> Entering new AgentExecutor chain...[0m
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 8/10: wikipedia.org (general)
  Goal: Search for 'Artificial Intelligence' and read the introduction...


[1m> Entering new AgentExecutor chain...[0m
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 9/10: youtube.com (entertainment)
  Goal: Search for 'Python tutorial' videos and play the most viewed one...


[1m> Entering new AgentExecutor chain...[0m
    ✓ Completed
      Task Understanding: 100.00%
      Task Deviation: 100.00%
      Task Completion: 100.00%
      Overall Score: 66.67%
      Reasoning Steps: 3

  Task 10/10: imdb.com (entertainment)
  Goal: Find the highest rated action movies from 2023...


[1m> Entering new AgentExecutor chain...[0m
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
  Total Tests: 100
  Successful: 80 (80.0%)
  Failed: 20 (20.0%)
  Total Reasoning Steps: 390
  Avg Reasoning Steps per Test: 3.9

----------------------------------------------------------------------
Results by Framework:
----------------------------------------------------------------------

  CrewAI:
    Success Rate: 10/10 (100.0%)
    Avg Latency: 0.14s
    Avg Reasoning Steps: 3.0

  LangChain:
    Success Rate: 10/10 (100.0%)
    Avg Latency: 0.31s
    Avg Reasoning Steps: 3.0

  LangGraph:
    Success Rate: 10/10 (100.0%)
    Avg Latency: 0.61s
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
    Avg Latency: 0.15s
    Avg Reasoning Steps: 4.0

  direct:
    Success Rate: 40/40 (100.0%)
    Avg Latency: 0.16s
    Avg Reasoning Steps: 4.5

----------------------------------------------------------------------
Results by Protocol:
----------------------------------------------------------------------

  A2A:
    Success Rate: 10/10 (100.0%)
    Avg Latency: 0.16s

  ACP:
    Success Rate: 10/10 (100.0%)
    Avg Latency: 0.15s

  MCP:
    Success Rate: 10/10 (100.0%)
    Avg Latency: 0.16s

  standard:
    Success Rate: 50/70 (71.4%)
    Avg Latency: 0.27s

----------------------------------------------------------------------
Results by Model:
----------------------------------------------------------------------

  mistralai/Mistral-Nemo-Base-2407:
    Success Rate: 80/100 (80.0%)
    Avg Latency: 0.23s
    Avg Reasoning Steps: 3.9

======================================================================
✓ Mind2Web results exported to ./logs/mind2web-results-20251030_093557.json

✓ Results exported to ./logs/results-20251030_093557.json

======================================================================
Mind2Web evaluation complete!
Results: ./logs/mind2web-results-20251030_093557.json
Full logs: ./logs/results-20251030_093557.json
======================================================================
```

## Notes

- Mind2Web requires authentication with HuggingFace
- The test set requires accepting terms on HuggingFace
- Focus is on task understanding and action planning capabilities

## AI Attribution

AIA Human-AI blend, Content edits, Human-initiated, Reviewed, GPT-4.1 and Sonet 4.5 v1.0

More info: https://aiattribution.github.io/create-attribution