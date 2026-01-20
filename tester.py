"""
Main testing orchestrator for comprehensive agentic technology evaluation

This module orchestrates testing across multiple models, protocols, and frameworks,
collecting detailed metrics including reasoning steps, latency, and success rates.
"""

import json
import time
import datetime
from pathlib import Path
from typing import List, Dict, Optional

from domain_model import TestResult, ProtocolType, ReasoningStep
from adapters import HuggingFaceAdapter, OpenAIAdapter, BaseAdapter
from protocols import MCPProtocol, A2AProtocol, ACPProtocol
from frameworks import (
    CrewAIAgent,
    SmolAgentWrapper,
    LangChainAgent,
    LangGraphAgent,
    LlamaIndexAgent,
    SemanticKernelAgent
)

from mind2web_loader import Mind2WebLoader
from mind2web_evaluator import Mind2WebEvaluator
from transformers import AutoTokenizer
import tiktoken
import yaml

# TODO: Change this to a singleton pattern
class Mastif:
    """
    Main testing framework for comprehensive agentic technology evaluation
    
    This class orchestrates testing across multiple:
    - Models (different HuggingFace and OpenAI models)
    - Protocols (e.g., MCP, A2A, ACP, Standard)
    - Frameworks (CrewAI, Smolagents, LangChain, LangGraph, LlamaIndex, and Semantic Kernel)
    
    Collects detailed metrics including reasoning steps, latency, and success rates.
    """
    
    def __init__(self):
        """Initialize the testing framework"""
        self.results: List[TestResult] = []
        self.protocols = {
            ProtocolType.MCP: MCPProtocol(),
            ProtocolType.A2A: A2AProtocol(),
            ProtocolType.ACP: ACPProtocol()
        }
        self.standard_tools = [
            {"name": "web_browser", "description": "Browse web pages and extract content"},
            {"name": "search", "description": "Search for information on the web"},
            {"name": "calculate", "description": "Perform mathematical calculations"},
            {"name": "analyze", "description": "Analyze data and generate insights"},
            {"name": "wikipedia", "description": "Retrieve information from Wikipedia"},
            {"name": "code_interpreter", "description": "Interpret and execute code snippets"},
            {"name": "click_element", "description": "Click on a web element"},
            {"name": "type_text", "description": "Type text into a field"},
            {"name": "navigate", "description": "Navigate to a URL"},
            {"name": "extract_table", "description": "Extract tables from web pages"},
            {"name": "summarize", "description": "Summarize text or web content"},
            {"name": "translate", "description": "Translate text between languages"},
            {"name": "calendar", "description": "Manage calendar events"},
            {"name": "download", "description": "Download files from the web"}
            # {"name": "file_upload", "description": "Upload a file to a website"}, # Not now
        ]
        # TODO: load config on init and create a get_confg to ease other classes to reuse experiment info

    def load_experiment_config(self, config_path: str) -> Dict:
        """Load experiment configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def test_with_protocol(
        self,
        adapter: BaseAdapter,
        protocol_type: ProtocolType,
        task: str,
        context: Dict = None
    ) -> TestResult:
        """
        Test a model with a specific protocol
        
        Args:
            adapter: Model adapter
            protocol_type: Protocol to use for communication
            task: Task to execute
            context: Additional context information
            
        Returns:
            TestResult with execution details
        """
        start_time = time.time()
        reasoning_steps = []
        
        try:
            reasoning_steps.append(ReasoningStep(
                step_number=1,
                thought=f"Testing with {protocol_type.value} protocol",
                action="initialize_protocol",
                action_input=protocol_type.value
            ))
            
            if protocol_type == ProtocolType.STANDARD:
                # Direct API call without protocol wrapper
                reasoning_steps.append(ReasoningStep(
                    step_number=2,
                    thought="Using standard API call (no protocol overhead)",
                    action="direct_generation",
                    action_input=task
                ))
                response = adapter.generate(task)
            else:
                # Use protocol wrapper
                protocol = self.protocols[protocol_type]
                formatted_msg = protocol.send_message(task, context or {})
                
                reasoning_steps.append(ReasoningStep(
                    step_number=2,
                    thought=f"Formatting message with {protocol_type.value}",
                    action="format_message",
                    action_input=json.dumps(formatted_msg, indent=2)[:200]
                ))
                
                # Simulate protocol-aware communication
                protocol_prompt = f"""Protocol: {protocol_type.value}

Message Structure:
{json.dumps(formatted_msg, indent=2)}

Please respond according to this protocol structure and complete the task."""
                
                reasoning_steps.append(ReasoningStep(
                    step_number=3,
                    thought="Sending formatted message to model",
                    action="generate_with_protocol",
                    action_input=protocol_prompt[:200]
                ))
                
                raw_response = adapter.generate(protocol_prompt)
                response = protocol.receive_message({"content": raw_response})
                
                reasoning_steps.append(ReasoningStep(
                    step_number=4,
                    thought="Received and parsed protocol response",
                    observation=f"Response length: {len(response)} characters"
                ))
            
            latency = time.time() - start_time
            
            reasoning_steps.append(ReasoningStep(
                step_number=len(reasoning_steps) + 1,
                thought="Protocol test completed successfully",
                observation=f"Total latency: {latency:.2f}s"
            ))
            
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol_type,
                framework="direct",
                task=task,
                response=response,
                reasoning_steps=reasoning_steps,
                latency=latency,
                success=True
            )
        
        except Exception as e:
            latency = time.time() - start_time
            reasoning_steps.append(ReasoningStep(
                step_number=len(reasoning_steps) + 1,
                thought="Error during protocol test",
                observation=f"Error: {str(e)}"
            ))
            
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol_type,
                framework="direct",
                task=task,
                response="",
                reasoning_steps=reasoning_steps,
                latency=latency,
                success=False,
                error=str(e)
            )
    
    def test_with_crewai(
        self,
        adapter: BaseAdapter, 
        role: str,
        task: str,
        context: Dict = None,
        protocol: ProtocolType = None
    ) -> TestResult:
        """Test with CrewAI framework"""
        start_time = time.time()
        
        try:
            protocol_instance = self.protocols.get(protocol) if protocol else None
            agent = CrewAIAgent(adapter, role, protocol=protocol_instance)
            response = agent.execute_task(task, context)
            latency = time.time() - start_time
            
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="CrewAI",
                task=task,
                response=response,
                reasoning_steps=agent.reasoning_steps,
                latency=latency,
                success=True,
                metadata={"role": role, "protocol_used": protocol.value if protocol else "none"}
            )
        
        except Exception as e:
            latency = time.time() - start_time
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="CrewAI",
                task=task,
                response="",
                reasoning_steps=[],
                latency=latency,
                success=False,
                error=str(e)
            )
    
    def test_with_smolagents(
        self,
        adapter: BaseAdapter,
        task: str,
        tools: List[Dict[str, str]] = None,
        protocol: ProtocolType = None
    ) -> TestResult:
        """Test with Smolagents framework"""
        start_time = time.time()
        
        try:
            protocol_instance = self.protocols.get(protocol) if protocol else None
            agent = SmolAgentWrapper(adapter, protocol=protocol_instance)

            for tool in (tools or []):
                agent.add_tool(tool["name"], tool["description"])

            response = agent.run(task)
            latency = time.time() - start_time
            
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="Smolagents",
                task=task,
                response=response,
                reasoning_steps=agent.reasoning_steps,
                latency=latency,
                success=True,
                metadata={"tools_count": len(tools or []), "protocol_used": protocol.value if protocol else "none"}
            )
        
        except Exception as e:
            latency = time.time() - start_time
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="Smolagents",
                task=task,
                response="",
                reasoning_steps=[],
                latency=latency,
                success=False,
                error=str(e)
            )
    
    def test_with_langchain(
        self,
        adapter: BaseAdapter,
        task: str,
        tools: List[Dict[str, str]] = None,
        protocol: ProtocolType = None
    ) -> TestResult:
        """Test with LangChain ReAct agent"""
        start_time = time.time()
        
        try:
            protocol_instance = self.protocols.get(protocol) if protocol else None
            agent = LangChainAgent(adapter, protocol=protocol_instance)
            
            # Add tools
            for tool in (tools or []):
                agent.add_tool(tool["name"], tool["description"])
            
            response = agent.run(task)
            latency = time.time() - start_time
            
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="LangChain",
                task=task,
                response=response,
                reasoning_steps=agent.reasoning_steps,
                latency=latency,
                success=True,
                metadata={"tools_count": len(tools or []), "agent_type": "ReAct", "protocol_used": protocol.value if protocol else "none"}
            )
        
        except Exception as e:
            latency = time.time() - start_time
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="LangChain",
                task=task,
                response="",
                reasoning_steps=[],
                latency=latency,
                success=False,
                error=str(e)
            )
    
    def test_with_langgraph(
        self,
        adapter: BaseAdapter,
        task: str,
        protocol: ProtocolType = None
    ) -> TestResult:
        """Test with LangGraph stateful workflow"""
        start_time = time.time()
        
        try:
            protocol_instance = self.protocols.get(protocol) if protocol else None
            agent = LangGraphAgent(adapter, protocol=protocol_instance)
            response = agent.run(task)
            latency = time.time() - start_time
            
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="LangGraph",
                task=task,
                response=response,
                reasoning_steps=agent.reasoning_steps,
                latency=latency,
                success=True,
                metadata={"workflow_type": "research_pipeline", "protocol_used": protocol.value if protocol else "none"  }
            )
        
        except Exception as e:
            latency = time.time() - start_time
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="LangGraph",
                task=task,
                response="",
                reasoning_steps=[],
                latency=latency,
                success=False,
                error=str(e)
            )
    
    def test_with_llamaindex(
        self,
        adapter: BaseAdapter,
        task: str,
        tools: List[Dict[str, str]] = None,
        protocol: ProtocolType = None
    ) -> TestResult:
        """Test with LlamaIndex ReAct agent"""
        start_time = time.time()
        
        try:
            protocol_instance = self.protocols.get(protocol) if protocol else None
            agent = LlamaIndexAgent(adapter, protocol=protocol_instance)
            
            # Add tools
            for tool in (tools or []):
                agent.add_tool(tool["name"], tool["description"])
            
            response = agent.run(task)
            latency = time.time() - start_time
            
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="LlamaIndex",
                task=task,
                response=response,
                reasoning_steps=agent.reasoning_steps,
                latency=latency,
                success=True,
                metadata={"tools_count": len(tools or []), "agent_type": "ReAct", "protocol_used": protocol.value if protocol else "none" }
            )
        
        except Exception as e:
            latency = time.time() - start_time
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="LlamaIndex",
                task=task,
                response="",
                reasoning_steps=[],
                latency=latency,
                success=False,
                error=str(e)
            )
    
    def test_with_semantic_kernel(
        self,
        adapter: BaseAdapter,
        task: str,
        protocol: ProtocolType = None
    ) -> TestResult:
        """Test with Semantic Kernel"""
        start_time = time.time()
        
        try:
            protocol_instance = self.protocols.get(protocol) if protocol else None
            agent = SemanticKernelAgent(adapter, protocol=protocol_instance)
            response = agent.run(task)
            latency = time.time() - start_time
            
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="SemanticKernel",
                task=task,
                response=response,
                reasoning_steps=agent.reasoning_steps,
                latency=latency,
                success=True,
                metadata={"kernel_type": "huggingface", "protocol_used": protocol.value if protocol else "none" }
            )
        
        except Exception as e:
            latency = time.time() - start_time
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="SemanticKernel",
                task=task,
                response="",
                reasoning_steps=[],
                latency=latency,
                success=False,
                error=str(e)
            )
    
    def run_comprehensive_test(self, config_path: str, api_key: Optional[str] = None):
        """
        Run comprehensive tests across all models, protocols, and frameworks.
        Tests all combinations of protocols x frameworks for each model.
        
        Args:
            config_path: Path to the experiment configuration file
            api_key: API token for the model (either HuggingFace or OpenAI)
        """
        
        # Load config
        config = self.load_experiment_config(config_path)
    
        # Extract experiment configuration
        models = config['models']
        protocols = [ProtocolType[p] for p in config['protocols']]
        framework_names = config['frameworks']
        prompt_template = config['prompt_template']
        inner_tasks = config['tasks']
        tools = config.get('tools', self.standard_tools)    
        
        # Format tasks with template
        test_tasks = [prompt_template.format(task=task) for task in inner_tasks]
        
        # Map framework names to functions
        framework_map = {
            "CrewAI": (self.test_with_crewai, {"role": "Web Automation Specialist"}),
            "Smolagents": (self.test_with_smolagents, {"tools": tools}),
            "LangChain": (self.test_with_langchain, {"tools": tools}),
            "LangGraph": (self.test_with_langgraph, {}),
            "LlamaIndex": (self.test_with_llamaindex, {"tools": tools}),
            "SemanticKernel": (self.test_with_semantic_kernel, {})
        }

        frameworks = [(name, *framework_map[name]) for name in framework_names]
        
        # Compute the number of tests and alert the user
        print(f"\n{'-'*70}")
        print("You are about to run a test with the following configuration:")
        print(f"Models: {len(models)}")
        print(f"Protocols: {len(protocols)}")
        print(f"Frameworks: {len(frameworks)}")
        print(f"Tasks: {len(test_tasks)}")
        print(f"Total: {len(models) * len(protocols) * len(frameworks) * len(test_tasks)}")
        print(f"{'-'*70}")
        if( len(models) * len(protocols) * len(frameworks) * len(test_tasks) > 1000 ):
            print("WARNING: This may incur a high number of API calls and associated costs.")
            response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("\n❌ Test execution cancelled by user.")
                return False        
            print("\n✓ Starting test execution...\n")

        for model_name in models:
            print(f"\n{'='*70}")
            print(f"Testing Model: {model_name}")
            print(f"{'='*70}")
            
            if(model_name.startswith("gpt-")):
                adapter = OpenAIAdapter(model_name)
            else:                
                adapter = HuggingFaceAdapter(model_name)
            
            # ===== Test Protocol x Framework Combinations =====
            print(f"\n{'-'*70}")
            print("PROTOCOL x FRAMEWORK COMBINATIONS")
            print(f"{'-'*70}")
            
            for protocol in protocols:
                print(f"\n{'─'*70}")
                print(f"Protocol: {protocol.value}")
                print(f"{'─'*70}")
                
                for framework_name, test_fn, extra_args in frameworks:
                    print(f"\n  {framework_name} with {protocol.value}:")
                    combination_results = []
                    
                    for i, task in enumerate(test_tasks, 1):
                        print(f"    Task {i}/{len(test_tasks)}: {task[:50]}...")
                        
                        # Build arguments for test function
                        args = [adapter, task]
                        kwargs = {"protocol": protocol}  # Add protocol to kwargs
                        
                        # Handle CrewAI's role parameter
                        if framework_name == "CrewAI":
                            args.insert(1, extra_args["role"])
                        
                        # Add tools if framework needs them
                        if "tools" in extra_args:
                            kwargs["tools"] = extra_args["tools"]
                        
                        # Execute test
                        try:
                            result = test_fn(*args, **kwargs)
                            combination_results.append(result)
                            self.results.append(result)
                            
                            status = "✓" if result.success else "❌"
                            print(f"      {status} Latency: {result.latency:.2f}s, Steps: {len(result.reasoning_steps)}")
                        except Exception as e:
                            print(f"      ❌ Error: {str(e)}")
                    
                    # Summary for this protocol-framework combination
                    if combination_results:
                        successes = [r for r in combination_results if r.success]
                        avg_latency = sum(r.latency for r in combination_results) / len(combination_results)
                        avg_steps = sum(len(r.reasoning_steps) for r in combination_results) / len(combination_results)
                        
                        print(f"\n  {framework_name} + {protocol.value} Summary:")
                        print(f"    Success: {len(successes)}/{len(combination_results)} ({len(successes)/len(combination_results)*100:.1f}%)")
                        print(f"    Avg Latency: {avg_latency:.2f}s")
                        print(f"    Avg Steps: {avg_steps:.1f}")
            
            print(f"\n{'='*70}")
            print(f"Model {model_name} Complete")
            print(f"{'='*70}")
    
    def export_results(self, filename: str = "test_results.json"):
        """
        Export test results to JSON file with detailed reasoning steps
        
        Args:
            filename: Output filename (with path)
        """
        # Ensure logs directory exists
        log_dir = Path(filename).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert results to dictionary format
        results_dict = []
        for r in self.results:
            # Convert reasoning steps to dict
            reasoning_dict = [
                {
                    "step_number": step.step_number,
                    "thought": step.thought,
                    "action": step.action,
                    "action_input": step.action_input,
                    "observation": step.observation,
                    "timestamp": step.timestamp
                }
                for step in r.reasoning_steps
            ]
            
            results_dict.append({
                "model_name": r.model_name,
                "protocol": r.protocol.value,
                "framework": r.framework,
                "task": r.task,
                "response": r.response,
                # "response_preview": r.response[:200] + "..." if len(r.response) > 200 else r.response,
                "reasoning_steps": reasoning_dict,
                "reasoning_steps_count": len(reasoning_dict),
                "latency": r.latency,
                "success": r.success,
                "error": r.error,
                "metadata": r.metadata
            })
        
        # Write to file with pretty formatting
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "test_run_timestamp": datetime.datetime.now().isoformat(),
                "total_tests": len(results_dict),
                "results": results_dict
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results exported to {filename}")
    
    def print_summary(self):
        """Print comprehensive summary of test results with detailed statistics"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        total_reasoning_steps = sum(len(r.reasoning_steps) for r in self.results)
        
        print(f"\nOverall Statistics:")
        print(f"  Total Tests: {total}")
        print(f"  Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"  Failed: {total - successful} ({(total-successful)/total*100:.1f}%)")
        print(f"  Total Reasoning Steps: {total_reasoning_steps}")
        print(f"  Avg Reasoning Steps per Test: {total_reasoning_steps/total:.1f}")
        
        # Group by framework
        print("\n" + "-"*70)
        print("Results by Framework:")
        print("-"*70)
        frameworks = {}
        for r in self.results:
            if r.framework not in frameworks:
                frameworks[r.framework] = {
                    "success": 0,
                    "total": 0,
                    "latency": [],
                    "reasoning_steps": []
                }
            frameworks[r.framework]["total"] += 1
            frameworks[r.framework]["reasoning_steps"].append(len(r.reasoning_steps))
            if r.success:
                frameworks[r.framework]["success"] += 1
                frameworks[r.framework]["latency"].append(r.latency)
        
        for fw, stats in sorted(frameworks.items()):
            success_rate = stats['success']/stats['total']*100
            avg_latency = sum(stats['latency'])/len(stats['latency']) if stats['latency'] else 0
            avg_steps = sum(stats['reasoning_steps'])/len(stats['reasoning_steps'])
            
            print(f"\n  {fw}:")
            print(f"    Success Rate: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
            print(f"    Avg Latency: {avg_latency:.2f}s")
            print(f"    Avg Reasoning Steps: {avg_steps:.1f}")
        
        # Group by protocol
        print("\n" + "-"*70)
        print("Results by Protocol:")
        print("-"*70)
        protocols = {}
        for r in self.results:
            if r.protocol.value not in protocols:
                protocols[r.protocol.value] = {
                    "success": 0,
                    "total": 0,
                    "latency": []
                }
            protocols[r.protocol.value]["total"] += 1
            if r.success:
                protocols[r.protocol.value]["success"] += 1
                protocols[r.protocol.value]["latency"].append(r.latency)
        
        for proto, stats in sorted(protocols.items()):
            success_rate = stats['success']/stats['total']*100
            avg_latency = sum(stats['latency'])/len(stats['latency']) if stats['latency'] else 0
            
            print(f"\n  {proto}:")
            print(f"    Success Rate: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
            print(f"    Avg Latency: {avg_latency:.2f}s")
        
        # Results by model
        print("\n" + "-"*70)
        print("Results by Model:")
        print("-"*70)
        models = {}
        for r in self.results:
            if r.model_name not in models:
                models[r.model_name] = {
                    "success": 0,
                    "total": 0,
                    "latency": [],
                    "reasoning_steps": []
                }
            models[r.model_name]["total"] += 1
            models[r.model_name]["reasoning_steps"].append(len(r.reasoning_steps))
            if r.success:
                models[r.model_name]["success"] += 1
                models[r.model_name]["latency"].append(r.latency)
        
        for model, stats in sorted(models.items()):
            success_rate = stats['success']/stats['total']*100
            avg_lat = sum(stats['latency'])/len(stats['latency']) if stats['latency'] else 0
            avg_steps = sum(stats['reasoning_steps'])/len(stats['reasoning_steps'])
            
            print(f"\n  {model}:")
            print(f"    Success Rate: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
            print(f"    Avg Latency: {avg_lat:.2f}s")
            print(f"    Avg Reasoning Steps: {avg_steps:.1f}")
        
        # Print token usage metrics
        print("\n" + "-"*70)
        print("Token Usage by Model:")
        print("-"*70)
        # TODO: Break down tokens by framework and protocol as well
        for model in set(r.model_name for r in self.results):
            model_results = [r for r in self.results if r.model_name == model]
            reasoning, output = self.compute_token_metrics(model, model_results)
            print(f"Model: {model}")
            print(f"  Reasoning tokens: {reasoning}")
            print(f"  Output tokens: {output}")
            print(f"  Total tokens: {reasoning + output}")
        
        print("\n" + "="*70)

    def run_mind2web_evaluation(
        self,
        config_path: str,
        hf_token: Optional[str] = None,
        num_tasks: Optional[int] = 10,
        frameworks: Optional[List[str]] = None
    ):
        """
        Run evaluation on Mind2Web benchmark tasks
        
        Args:
            config_path: Path to the experiment configuration file
            hf_token: HuggingFace API token
            num_tasks: Number of tasks to evaluate (None for all, default 10)
            frameworks: List of frameworks to test (None for all)
        """

        # Initialize Mind2Web loader
        loader = Mind2WebLoader(split="train")
        
        # Load and sample tasks
        tasks = loader.get_task_sample(num_tasks=num_tasks)
        
        # Load config
        config = self.load_experiment_config(config_path)

        # Extract experiment configuration
        models = config['models']
        protocols = [ProtocolType[p] for p in config['protocols']]
        framework_names = config['frameworks']
        tools = config.get('tools', self.standard_tools)

        # Map framework names to functions
        framework_map = {
            "CrewAI": (self.test_with_crewai, {"role": "Web Automation Specialist"}),
            "Smolagents": (self.test_with_smolagents, {"tools": tools}),
            "LangChain": (self.test_with_langchain, {"tools": tools}),
            "LangGraph": (self.test_with_langgraph, {}),
            "LlamaIndex": (self.test_with_llamaindex, {"tools": tools}),
            "SemanticKernel": (self.test_with_semantic_kernel, {})
        }

        frameworks = [(name, *framework_map[name]) for name in framework_names]

        if not tasks:
            print("Failed to load Mind2Web tasks!")
            return
        
        # Print statistics
        stats = loader.get_task_statistics()
        print("\n" + "="*70)
        print("MIND2WEB BENCHMARK EVALUATION")
        print("="*70)
        print(f"\nDataset Statistics:")
        print(f"  Total Tasks: {stats['total_tasks']}")
        print(f"  Unique Domains: {stats['unique_domains']}")
        print(f"  Unique Websites: {stats['unique_websites']}")
        print(f"  Avg Actions per Task: {stats['avg_actions_per_task']:.1f}")
        print(f"\nTop Domains:")
        for domain, count in sorted(stats['domains'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {domain}: {count} tasks")
        
        # Initialize evaluator
        judge_adapter = OpenAIAdapter()
        evaluator = Mind2WebEvaluator(judge_adapter=judge_adapter)

        # Compute the number of tests and alert the user
        print(f"\n{'-'*70}")
        print("You are about to run a Mind2Web test with the following configuration:")
        print(f"Models: {len(models)}")
        print(f"Protocols: {len(protocols)}")
        print(f"Frameworks: {len(frameworks)}")
        print(f"Tasks: {len(tasks)}")
        print(f"Total: {len(models) * len(protocols) * len(frameworks) * len(tasks)}")
        print(f"{'-'*70}")
        if( len(models) * len(protocols) * len(frameworks) * len(tasks) > 1000 ):
            print("WARNING: This may incur a high number of API calls and associated costs.")
            response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("\n❌ Test execution cancelled by user.")
                return False        
            print("\n✓ Starting test execution...\n")

        # Run tests for each model
        for model_name in models:
            print(f"\n{'='*70}")
            print(f"Testing Model: {model_name}")
            print(f"{'='*70}")
            
            adapter = HuggingFaceAdapter(model_name, hf_token)
            
            # ===== Protocol Tests =====
            print(f"\n{'-'*70}")
            print("PROTOCOL TESTS")
            print(f"{'-'*70}")

            for protocol in protocols:
                protocol_results = []
                for task in tasks:
                    print(f"\n  Testing with {protocol.value} on task: {task['confirmed_task'][:40]}...")
                    result = self.test_with_protocol(adapter, protocol, task['confirmed_task'])
                    protocol_results.append(result)
                    self.results.append(result)
                    if result.success:
                        status = "✓ Success"
                    else:
                        status = f"❌ Failed: {result.error}"
                    print(f"    {status} ({len(result.reasoning_steps)} reasoning steps)")

                # Average metrics for this protocol
                successes = [r for r in protocol_results if r.success]
                avg_latency = sum(r.latency for r in protocol_results) / len(protocol_results)
                avg_reasoning_steps = sum(len(r.reasoning_steps) for r in protocol_results) / len(protocol_results)
                print(f"\n  Protocol {protocol.value} summary:")
                print(f"    Success rate: {len(successes)}/{len(protocol_results)}")
                print(f"    Avg latency: {avg_latency:.2f}s")
                print(f"    Avg reasoning steps: {avg_reasoning_steps:.2f}")
                print(f"    \n{'-'*35}")

            # ===== Framework Tests =====
            for framework_name, test_fn, extra_args in frameworks:
                print(f"\n{'-'*70}")
                print(f"Framework: {framework_name}")
                print(f"{'-'*70}")
                framework_results = []
                for i, task in enumerate(tasks, 1):
                    print(f"\n  Task {i}/{len(tasks)}: {task['website']} ({task['domain']})")
                    print(f"  Goal: {task['confirmed_task'][:80]}...")
                    
                    # Format task as prompt
                    task_prompt = f"""You are a web automation agent. Complete this task:

    Website: {task['website']}
    Domain: {task['domain']}
    Task: {task['confirmed_task']}

    If needed, provide a step-by-step plan of actions needed to complete this task.
    You can use external tools and spawn specialized agents as needed.
    The maximum number of agents you can spawn is 3.

    Your response:"""

                    result = None
                    try:
                        args = [adapter, task_prompt]
                        if framework_name == "CrewAI":
                            args.insert(1, extra_args["role"])
                        kwargs = {}
                        if "tools" in extra_args:
                            kwargs["tools"] = extra_args["tools"]
                        result = test_fn(*args, **kwargs)
                        if result:
                            self.results.append(result)
                            
                            # Evaluate result
                            eval_result = evaluator.evaluate_task(
                                task,
                                result.response,
                                result.reasoning_steps
                            )
                            
                            print(f"    ✓ Completed")
                            print(f"      Task Understanding: {eval_result['task_understanding']:.2%}")
                            print(f"      Task Deviation: {eval_result['task_deviation']:.2%}")
                            print(f"      Task Completion: {eval_result['task_completion']:.2%}")
                            print(f"      Overall Score: {eval_result['overall_score']:.2%}")
                            print(f"      Reasoning Steps: {eval_result['reasoning_steps_count']}")
                    except Exception as e:
                        print(f"    ❌ Error: {str(e)}")

                # Summarize metrics for this framework
                metrics = evaluator.get_aggregate_metrics()
                print(f"\n  {framework_name} Mind2Web Summary:")
                for metric, value in metrics.items():
                    if metric == 'domain_metrics':
                        continue
                    print(f"    {metric}: {value}")

        # Print aggregate metrics
        print("\n" + "="*70)
        print("MIND2WEB EVALUATION SUMMARY")
        print("="*70)
        
        aggregate = evaluator.get_aggregate_metrics()
        print(f"\nOverall Performance:")
        print(f"  Judge Model: {aggregate['judge_model']}")
        print(f"  Tasks Evaluated: {aggregate['total_tasks_evaluated']}")
        print(f"  Avg Task Understanding: {aggregate['avg_task_understanding']:.2%}")
        print(f"  Avg Task Deviation: {aggregate['avg_task_deviation']:.2%} (lower is better)")
        print(f"  Avg Task Completion: {aggregate['avg_task_completion']:.2%}")
        print(f"  Avg Overall Score: {aggregate['avg_overall_score']:.2%}")
        print(f"  Avg Reasoning Steps: {aggregate['avg_reasoning_steps']:.1f}")
        
        print(f"\nPerformance by Domain:")
        for domain, metrics in aggregate['domain_metrics'].items():
            print(f"  {domain}:")
            print(f"    Tasks: {metrics['count']}")
            print(f"    Avg Understanding: {metrics['avg_understanding']:.2%}")
            print(f"    Avg Task Deviation: {metrics['avg_deviation']:.2%} (lower is better)")
            print(f"    Avg Completion: {metrics['avg_completion']:.2%}")
        
        # Store Mind2Web specific results
        self.mind2web_results = evaluator.results
        self.mind2web_aggregate = aggregate
        
    def export_mind2web_results(self, filename: str):
        """
        Export Mind2Web evaluation results
        
        Args:
            filename: Output filename
        """
        if not hasattr(self, 'mind2web_results'):
            print("No Mind2Web results to export")
            return
        
        from pathlib import Path
        import json
        import datetime
        
        # Ensure directory exists
        log_dir = Path(filename).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare export data
        export_data = {
            "benchmark": "Mind2Web",
            "timestamp": datetime.datetime.now().isoformat(),
            "aggregate_metrics": self.mind2web_aggregate,
            "task_results": self.mind2web_results
        }
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Mind2Web results exported to {filename}")
    
    def get_supported_protocols(self):
        return [ProtocolType.MCP, ProtocolType.A2A, ProtocolType.ACP, ProtocolType.STANDARD]

    def get_supported_frameworks(self):
        return [
            "CrewAI",
            "Smolagents",
            "LangChain",
            "LangGraph",
            "LlamaIndex",
            "Semantic Kernel"
        ]
    
    def compute_token_metrics(self, model_name: str, results: List[TestResult]):
        # Use tiktoken for OpenAI models
        openai_models = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        if any(model_name.startswith(m) for m in ["gpt-", "openai"]) or model_name in openai_models:
            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")  # fallback

            total_reasoning_tokens = 0
            total_output_tokens = 0

            for result in results:
                for step in result.reasoning_steps:
                    total_reasoning_tokens += len(encoding.encode(step.thought or ""))
                    total_reasoning_tokens += len(encoding.encode(step.action or "")) if step.action else 0
                    total_reasoning_tokens += len(encoding.encode(step.action_input or "")) if step.action_input else 0
                    total_reasoning_tokens += len(encoding.encode(step.observation or "")) if step.observation else 0
                total_output_tokens += len(encoding.encode(result.response or ""))

            return total_reasoning_tokens, total_output_tokens

        # Hugging Face models
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        total_reasoning_tokens = 0
        total_output_tokens = 0

        for result in results:
            for step in result.reasoning_steps:
                total_reasoning_tokens += len(tokenizer.encode(step.thought or ""))
                total_reasoning_tokens += len(tokenizer.encode(step.action or "")) if step.action else 0
                total_reasoning_tokens += len(tokenizer.encode(step.action_input or "")) if step.action_input else 0
                total_reasoning_tokens += len(tokenizer.encode(step.observation or "")) if step.observation else 0
            total_output_tokens += len(tokenizer.encode(result.response or ""))

        return total_reasoning_tokens, total_output_tokens