"""
Main testing orchestrator for comprehensive agentic technology evaluation

This module orchestrates testing across multiple models, protocols, and frameworks,
collecting detailed metrics including reasoning steps, latency, and success rates.
"""

import os
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
from config import ConfigExpert
from tool_pool import ToolPool
from experiment_logger import ExperimentLogger

class Mastif:
    """
    Main testing framework for comprehensive agentic technology evaluation
    
    This class orchestrates testing across multiple:
    - Models (different HuggingFace and OpenAI models)
    - Protocols (e.g., MCP, A2A, ACP, Standard)
    - Frameworks (CrewAI, Smolagents, LangChain, LangGraph, LlamaIndex, and Semantic Kernel)
    
    Collects detailed metrics including reasoning steps, latency, and success rates.
    Delegates all logging, reporting, and partial log management to ExperimentLogger.
    """
    
    def __init__(self, yaml_path: str):
        """
        Initialize the testing framework.

        Args:
            yaml_path: Path to the YAML config file for this experiment.
                       Used to name partial log files.
        """
        self.yaml_path = yaml_path
        self.protocols = {
            ProtocolType.MCP: MCPProtocol(),
            ProtocolType.A2A: A2AProtocol(),
            ProtocolType.ACP: ACPProtocol()
        }
        self.standard_tools = ToolPool.available_tools
        self._logger: Optional[ExperimentLogger] = None
       
    def _init_logger(self, metadata: dict) -> ExperimentLogger:
        """
        Create an ExperimentLogger, check for partial logs, and offer resume.
        Returns the initialised logger.
        """
        logger = ExperimentLogger(self.yaml_path, metadata)
        partial = logger.find_partial()
        if partial:
            total = (
                len(metadata["models"]) *
                len(metadata["protocols"]) *
                len(metadata["frameworks"]) *
                metadata["total_tasks"]
            )
            if ExperimentLogger.prompt_resume(partial, total):
                logger.load_partial(partial)
                print(f"✅️ Resuming from partial log — {len(partial.get('completed', []))} executions already done.\n")
            else:
                print("▶️  Starting fresh (partial log will be overwritten on first result).\n")
        return logger

    def _get_protocol_metrics(self, protocol: ProtocolType, protocol_instance, task: str) -> Dict:
        """
        Measure and return protocol overhead and message size metrics.
        Returns zeroed metrics for STANDARD protocol.
        """
        if protocol_instance is None or protocol == ProtocolType.STANDARD:
            return {
                "send_overhead_ms":    0,
                "receive_overhead_ms": 0,
                "total_overhead_ms":   0,
                "message_size_bytes":  len(task.encode("utf-8")),
            }
        rich_context = protocol_instance.generate_context(
            task, tools=list(ToolPool.available_tools)
        )
        overhead = protocol_instance.measure_overhead(task, rich_context)
        return {
            "send_overhead_ms":    overhead["send_overhead_ms"],
            "receive_overhead_ms": overhead["receive_overhead_ms"],
            "total_overhead_ms":   overhead["total_overhead_ms"],
            "message_size_bytes":  overhead["message_size_bytes"],
        }

    ERROR_PATTERNS = [
        "execution error:",
        "inference error:",
        "planning error:",
        "research error:",
        "synthesis error:",
        "browser error:",
        "api error:",
        "http get error:",
        "tool '",  # tool execution errors
    ]

    # Patterns that indicate an API payment/quota/billing hard stop.
    # Any match aborts the entire experiment immediately.
    PAYMENT_PATTERNS = [
        "payment required",
        "402",
        "insufficient credits",
        "exceeded your current quota",
        "billing",
        "subscription",
        "upgrade your plan",
        "rate limit exceeded",
        "you have run out of",
        "account is not active",
        "payment_required",
    ]

    def _is_payment_error(self, text: str) -> bool:
        """
        Return True if the text contains a payment/quota/billing error
        that makes further API calls futile.
        """
        lower = text.lower()
        return any(p in lower for p in self.PAYMENT_PATTERNS)

    def _check_response_for_errors(self, response: str, framework: str) -> tuple:
        """
        Inspect an agent response for embedded inference or tool errors.

        Returns (success: bool, error: str | None)
        """
        if not response:
            return False, f"{framework} returned an empty response."
        lower = response.lower()
        for pattern in self.ERROR_PATTERNS:
            if pattern in lower:
                # Extract the first line as the error summary
                first_line = response.strip().splitlines()[0][:200]
                return False, first_line
        return True, None

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
                protocol_metrics = {
                    "send_overhead_ms": 0,
                    "receive_overhead_ms": 0,
                    "total_overhead_ms": 0,
                    "message_size_bytes": len(task.encode("utf-8")),
                }
            else:
                # Use protocol wrapper
                protocol = self.protocols[protocol_type]
                rich_context = protocol.generate_context(task, tools=list(ToolPool.available_tools))
                overhead = protocol.measure_overhead(task, rich_context)
                formatted_msg = protocol.send_message(task, rich_context)
                
                protocol_metrics = {
                    "send_overhead_ms":    overhead["send_overhead_ms"],
                    "receive_overhead_ms": overhead["receive_overhead_ms"],
                    "total_overhead_ms":   overhead["total_overhead_ms"],
                    "message_size_bytes":  overhead["message_size_bytes"],
                }

                reasoning_steps.append(ReasoningStep(
                    step_number=2,
                    thought=f"Formatting message with {protocol_type.value}",
                    action="format_message",
                    action_input=json.dumps(formatted_msg, indent=2)
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
                    action_input=protocol_prompt
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
                success=True,
                metadata={
                    "protocol_used": protocol_type.value,
                    **protocol_metrics,
                }
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
    
    def _capture_tool_log(self) -> Dict:
        """Snapshot the ToolPool invocation log and reset it."""
        summary = ToolPool.get_log_summary()
        ToolPool.reset_log()
        return summary

    def test_with_crewai(
        self,
        adapter: BaseAdapter, 
        role: str,
        task: str,
        context: Dict = None,
        protocol: ProtocolType = None,
        tools: List[str] = None
    ) -> TestResult:
        """Test with CrewAI framework"""
        start_time = time.time()
        
        try:
            protocol_instance = self.protocols.get(protocol) if protocol else None
            protocol_metrics = self._get_protocol_metrics(protocol or ProtocolType.STANDARD, protocol_instance, task)
            ToolPool.reset_log()
            agent = CrewAIAgent(adapter, role, protocol=protocol_instance)
            if tools is None:
                tools = list(self.standard_tools)
            for tool_name in (tools or []):
                agent.add_tool(tool_name)
            response = agent.execute_task(task, context)
            latency = time.time() - start_time
            success, error = self._check_response_for_errors(response, "CrewAI")
            if not success:
                print(f"    ⚠️  Inference/tool error detected: {error}")
            
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="CrewAI",
                task=task,
                response=response,
                reasoning_steps=agent.reasoning_steps,
                latency=latency,
                success=success,
                metadata={"role": role, "protocol_used": protocol.value if protocol else "none", **protocol_metrics, "tool_log": self._capture_tool_log()},
                error=error
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
            protocol_metrics = self._get_protocol_metrics(protocol or ProtocolType.STANDARD, protocol_instance, task)
            ToolPool.reset_log()
            agent = SmolAgentWrapper(adapter, protocol=protocol_instance)

            for tool_name in (tools or []):
                agent.add_tool(tool_name)

            response = agent.run(task)
            latency = time.time() - start_time
            success, error = self._check_response_for_errors(response, "Smolagents")
            if not success:
                print(f"    ⚠️  Inference/tool error detected: {error}")
            
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="Smolagents",
                task=task,
                response=response,
                reasoning_steps=agent.reasoning_steps,
                latency=latency,
                success=success,
                metadata={"tools_count": len(tools or []), "protocol_used": protocol.value if protocol else "none", **protocol_metrics, "tool_log": self._capture_tool_log()},
                error=error
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
            protocol_metrics = self._get_protocol_metrics(protocol or ProtocolType.STANDARD, protocol_instance, task)
            ToolPool.reset_log()
            agent = LangChainAgent(adapter, protocol=protocol_instance)

            for tool_name in (tools or []):
                agent.add_tool(tool_name)

            response = agent.run(task)
            latency = time.time() - start_time
            success, error = self._check_response_for_errors(response, "LangChain")
            if not success:
                print(f"    ⚠️  Inference/tool error detected: {error}")
            
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="LangChain",
                task=task,
                response=response,
                reasoning_steps=agent.reasoning_steps,
                latency=latency,
                success=success,
                metadata={"tools_count": len(tools or []), "agent_type": "ReAct", "protocol_used": protocol.value if protocol else "none", **protocol_metrics, "tool_log": self._capture_tool_log()},
                error=error
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
        tools: List[Dict[str, str]] = None,
        protocol: ProtocolType = None
    ) -> TestResult:
        """Test with LangGraph stateful workflow"""
        start_time = time.time()
        
        try:
            protocol_instance = self.protocols.get(protocol) if protocol else None
            protocol_metrics = self._get_protocol_metrics(protocol or ProtocolType.STANDARD, protocol_instance, task)
            ToolPool.reset_log()
            agent = LangGraphAgent(adapter, protocol=protocol_instance)

            for tool_name in (tools or []):
                agent.add_tool(tool_name)

            response = agent.run(task)
            latency = time.time() - start_time
            success, error = self._check_response_for_errors(response, "LangGraph")
            if not success:
                print(f"    ⚠️  Inference/tool error detected: {error}")
            
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="LangGraph",
                task=task,
                response=response,
                reasoning_steps=agent.reasoning_steps,
                latency=latency,
                success=success,
                metadata={"workflow_type": "research_pipeline", "protocol_used": protocol.value if protocol else "none", **protocol_metrics, "tool_log": self._capture_tool_log()},
                error=error
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
            protocol_metrics = self._get_protocol_metrics(protocol or ProtocolType.STANDARD, protocol_instance, task)
            ToolPool.reset_log()
            agent = LlamaIndexAgent(adapter, protocol=protocol_instance)
            
            # Add tools
            for tool_name in (tools or []):
                agent.add_tool(tool_name)
            
            response = agent.run(task)
            latency = time.time() - start_time
            success, error = self._check_response_for_errors(response, "LlamaIndex")
            if not success:
                print(f"    ⚠️  Inference/tool error detected: {error}")
            
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="LlamaIndex",
                task=task,
                response=response,
                reasoning_steps=agent.reasoning_steps,
                latency=latency,
                success=success,
                metadata={"tools_count": len(tools or []), "agent_type": "ReAct", "protocol_used": protocol.value if protocol else "none", **protocol_metrics, "tool_log": self._capture_tool_log()},
                error=error
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
            protocol_metrics = self._get_protocol_metrics(protocol or ProtocolType.STANDARD, protocol_instance, task)
            ToolPool.reset_log()
            agent = SemanticKernelAgent(adapter, protocol=protocol_instance)
            response = agent.run(task)
            latency = time.time() - start_time
            success, error = self._check_response_for_errors(response, "SemanticKernel")
            if not success:
                print(f"    ⚠️  Inference/tool error detected: {error}")
            
            return TestResult(
                model_name=adapter.model_name,
                protocol=protocol or ProtocolType.STANDARD,
                framework="SemanticKernel",
                task=task,
                response=response,
                reasoning_steps=agent.reasoning_steps,
                latency=latency,
                success=success,
                metadata={"kernel_type": "huggingface", "protocol_used": protocol.value if protocol else "none", **protocol_metrics, "tool_log": self._capture_tool_log()},
                error=error
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
    
    def run_comprehensive_test(self):
        """
        Run comprehensive tests across all models, protocols, and frameworks.
        Tests all combinations of protocols x frameworks for each model.
        """
            
        # Extract experiment configuration
        config = ConfigExpert.get_instance()
        models = config.get("models")
        protocols = [ProtocolType[p] for p in config.get("protocols")]
        framework_names = config.get("frameworks")
        prompt_template = config.get("prompt_template")
        inner_tasks = config.get("tasks")
        raw_tools = config.get('tools', self.standard_tools)
        tools = [t["name"] if isinstance(t, dict) else t for t in raw_tools]

        # Format tasks with template
        test_tasks = [prompt_template.format(task=task) for task in inner_tasks]
        
        # Map framework names to functions
        framework_map = {
            "CrewAI": (self.test_with_crewai, {"role": "Web Automation Specialist", "tools": tools}),
            "Smolagents": (self.test_with_smolagents, {"tools": tools}),
            "LangChain": (self.test_with_langchain, {"tools": tools}),
            "LangGraph": (self.test_with_langgraph, {"tools": tools}),
            "LlamaIndex": (self.test_with_llamaindex, {"tools": tools}),
            "SemanticKernel": (self.test_with_semantic_kernel, {})
        }

        frameworks = [(name, *framework_map[name]) for name in framework_names]
        total = len(models) * len(protocols) * len(frameworks) * len(test_tasks)

        # Initialise logger — offers resume if partial log exists
        metadata = {
            "experiment_name": config.get("experiment.name", ""),
            "test_mode":       "standard",
            "models":          models,
            "protocols":       [p.value for p in protocols],
            "frameworks":      framework_names,
            "tools":           tools,
            "total_tasks":     len(test_tasks),
        }
        self._logger = self._init_logger(metadata)
        
        # Compute the number of tests and alert the user
        print(f"\n{'-'*70}")
        print("You are about to run a test with the following configuration:")
        print(f"Models: {len(models)}")
        print(f"Protocols: {len(protocols)}")
        print(f"Frameworks: {len(frameworks)}")
        print(f"Tasks: {len(test_tasks)}")
        print(f"Total executions: {total}")
        print(f"{'-'*70}")
        print(f"Tools ({len(tools)}): {', '.join(tools)}")
        print(f"{'-'*70}")
        if total > config.get("requests_soft_limit", 1000):
            print("WARNING: This may incur a high number of API calls and associated costs.")
            response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("\n❌ Test execution cancelled by user.")
                return False        
            print("\n✅️ Starting test execution...\n")

        for model_name in models:
            print(f"\n{'='*70}")
            print(f"Testing Model: {model_name}")
            print(f"{'='*70}")
            
            if(model_name.startswith("gpt-")):
                adapter = OpenAIAdapter(model_name, api_key=os.getenv("OPENAI_API_KEY"))
            else:                
                adapter = HuggingFaceAdapter(model_name, api_key=os.getenv("HF_TOKEN"))
            
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
                    
                    for i, task in enumerate(test_tasks):
                        print(f"    Task {i+1}/{len(test_tasks)}: {task[:100]}...")

                        # Skip if already completed in a resumed run
                        if self._logger.is_completed(model_name, protocol.value, framework_name, i):
                            print(f"      ⏭️  Skipped (already completed)")
                            continue
                        
                        # Build arguments for test function
                        args = [adapter, task]
                        kwargs = {"protocol": protocol}
                        
                        if framework_name == "CrewAI":
                            args.insert(1, extra_args["role"])
                        
                        if "tools" in extra_args:
                            kwargs["tools"] = extra_args["tools"]
                        
                        # Execute test
                        try:
                            result = test_fn(*args, **kwargs)

                            # Payment/quota hard stop — abort before logging so
                            # this combination is retried on the next resume.
                            if self._is_payment_error(result.response or "") or \
                               self._is_payment_error(result.error or ""):
                                print(f"\n💳 Payment/quota error detected — halting experiment.")
                                print(f"   Partial log preserved. Resume with the same YAML to continue.")
                                raise SystemExit(1)

                            combination_results.append(result)
                            self._logger.log_result(result, model_name, protocol.value, framework_name, i)
                            
                            status = "✅️" if result.success else "❌"
                            error_msg = f" — {result.error}" if not result.success and result.error else ""
                            print(f"      {status} Latency: {result.latency:.2f}s, Steps: {len(result.reasoning_steps)}{error_msg}")
                        except SystemExit:
                            raise  # propagate halt immediately
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
    
    def run_mind2web_evaluation(
        self,
        hf_token: Optional[str] = None
    ):
        """
        Run evaluation on Mind2Web benchmark tasks
        
        Args:
            hf_token: HuggingFace API token
        """

        # Load config
        config = ConfigExpert.get_instance()

        # Initialize Mind2Web loader
        loader = Mind2WebLoader(split="train")
        
        # Load and sample tasks
        # tasks = loader.get_task_sample(num_tasks=config.get("mind2web_num_tasks", 10))
        # Load and sample stratified tasks
        tasks = loader.get_stratified_task_sample(num_tasks=config.get("mind2web_num_tasks", 10))
        
        # Extract experiment configuration
        models = config.get("models")
        protocols = [ProtocolType[p] for p in config.get("protocols")]
        framework_names = config.get("frameworks")
        raw_tools = config.get('tools', self.standard_tools)
        tools = [t["name"] if isinstance(t, dict) else t for t in raw_tools]

        # Map framework names to functions
        framework_map = {
            "CrewAI": (self.test_with_crewai, {"role": "Web Automation Specialist", "tools": tools}),
            "Smolagents": (self.test_with_smolagents, {"tools": tools}),
            "LangChain": (self.test_with_langchain, {"tools": tools}),
            "LangGraph": (self.test_with_langgraph, {"tools": tools}),
            "LlamaIndex": (self.test_with_llamaindex, {"tools": tools}),
            "SemanticKernel": (self.test_with_semantic_kernel, {"tools": tools})
        }

        frameworks = [(name, *framework_map[name]) for name in framework_names]
        total = len(models) * len(protocols) * len(frameworks) * len(tasks)

        if not tasks:
            print("Failed to load Mind2Web tasks!")
            return

        # Initialise logger — offers resume if partial log exists
        metadata = {
            "experiment_name": config.get("experiment.name", ""),
            "test_mode":       "mind2web",
            "models":          models,
            "protocols":       [p.value for p in protocols],
            "frameworks":      framework_names,
            "tools":           tools,
            "total_tasks":     len(tasks),
        }
        self._logger = self._init_logger(metadata)
        
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
        print(f"Total executions: {total}")
        print(f"{'-'*70}")
        print(f"Tools ({len(tools)}): {', '.join(tools)}")
        print(f"{'-'*70}")
        if total > config.get("requests_soft_limit", 1000):
            print("WARNING: This may incur a high number of API calls and associated costs.")
            response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("\n❌ Test execution cancelled by user.")
                return False        
            print("\n✅️ Starting test execution...\n")

        # Run tests for each model
        for model_name in models:
            print(f"\n{'='*70}")
            print(f"Testing Model: {model_name}")
            print(f"{'='*70}")
            
            if model_name.startswith("gpt-"):
                adapter = OpenAIAdapter(model_name, api_key=os.getenv("OPENAI_API_KEY"))
            else:
                adapter = HuggingFaceAdapter(model_name, api_key=os.getenv("HF_TOKEN"))

            # ===== Protocol x Framework Combinations =====
            print(f"\n{'-'*70}")
            print("PROTOCOL x FRAMEWORK COMBINATIONS")
            print(f"{'-'*70}")

            for protocol in protocols:
                print(f"\n{'─'*70}")
                print(f"Protocol: {protocol.value}")
                print(f"{'─'*70}")

                for framework_name, test_fn, extra_args in frameworks:
                    print(f"\n{'-'*70}")
                    print(f"  {framework_name} with {protocol.value}:")
                    print(f"{'-'*70}")
                    combination_results = []

                    for i, task in enumerate(tasks):
                        print(f"\n  Task {i+1}/{len(tasks)}: {task['website']} ({task['domain']})")
                        print(f"  Goal: {task['confirmed_task']}")

                        # Skip if already completed in a resumed run
                        if self._logger.is_completed(model_name, protocol.value, framework_name, i):
                            print(f"    ⏭️  Skipped (already completed)")
                            continue

                        # Construct task context for Workflow (this string maps to {task} in plan-mind2web.txt)
                        task_prompt = (f"Website: {task['website']}\n"
                                    f"Domain: {task['domain']}\n"
                                    f"Goal: {task['confirmed_task']}")

                        try:
                            # Build arguments for test function
                            args = [adapter, task_prompt]
                            if framework_name == "CrewAI":
                                args.insert(1, extra_args["role"])

                            kwargs = {"protocol": protocol}
                            if "tools" in extra_args:
                                kwargs["tools"] = extra_args["tools"]

                            result = test_fn(*args, **kwargs)
                            if result:
                                # Payment/quota hard stop — abort before logging so
                                # this combination is retried on the next resume.
                                if self._is_payment_error(result.response or "") or \
                                   self._is_payment_error(result.error or ""):
                                    print(f"\n💳 Payment/quota error detected — halting experiment.")
                                    print(f"   Partial log preserved. Resume with the same YAML to continue.")
                                    raise SystemExit(1)

                                self._logger.log_result(result, model_name, protocol.value, framework_name, i)
                                combination_results.append(result)

                                eval_result = evaluator.evaluate_task(
                                    task,
                                    result.response,
                                    result.reasoning_steps
                                )

                                status = "✅️ Completed" if result.success else f"⚠️  Completed with errors"
                                print(f"    {status}")
                                if not result.success and result.error:
                                    print(f"      ⚠️  Inference/tool error: {result.error}")

                                understanding = eval_result['task_understanding']
                                adherence = eval_result['task_adherence']
                                completion = eval_result['task_completion']
                                understanding_score = understanding['score'] if isinstance(understanding, dict) else float(understanding)
                                adherence_score = adherence['score'] if isinstance(adherence, dict) else float(adherence)
                                completion_score = completion['score'] if isinstance(completion, dict) else float(completion)
                                understanding_rationale = understanding['rationale'] if isinstance(understanding, dict) else ""
                                adherence_rationale = adherence['rationale'] if isinstance(adherence, dict) else ""
                                completion_rationale = completion['rationale'] if isinstance(completion, dict) else ""
                                print(f"      Task Understanding: {understanding_score:.2%}")
                                print(f"       ↳ Rationale: {understanding_rationale}")
                                print(f"      Task Adherence: {adherence_score:.2%}")
                                print(f"       ↳ Rationale: {adherence_rationale}")
                                print(f"      Task Completion: {completion_score:.2%}")
                                print(f"       ↳ Rationale: {completion_rationale}")
                                print(f"      Overall Score: {eval_result['overall_score']:.2%}")
                                print(f"      Reasoning Steps: {eval_result['reasoning_steps_count']}")
                        except SystemExit:
                            raise  # propagate halt immediately
                        except Exception as e:
                            print(f"    ❌ Error: {str(e)}")

                    if combination_results:
                        successes = [r for r in combination_results if r.success]
                        avg_latency = sum(r.latency for r in combination_results) / len(combination_results)
                        avg_reasoning_steps = sum(len(r.reasoning_steps) for r in combination_results) / len(combination_results)

                        print(f"\n  {framework_name} + {protocol.value} Summary:")
                        print(f"    Success: {len(successes)}/{len(combination_results)} ({len(successes)/len(combination_results)*100:.1f}%)")
                        print(f"    Avg Latency: {avg_latency:.2f}s")
                        print(f"    Avg Reasoning Steps: {avg_reasoning_steps:.1f}")

        # Print aggregate metrics
        print("\n" + "="*70)
        print("MIND2WEB EVALUATION SUMMARY")
        print("="*70)
        
        aggregate = evaluator.get_aggregate_metrics()
        print(f"\nOverall Performance:")
        print(f"  Judge Model: {aggregate['judge_model']}")
        print(f"  Tasks Evaluated: {aggregate['total_tasks_evaluated']}")
        print(f"  Avg Task Understanding: {aggregate['avg_task_understanding']:.2%}")
        print(f"  Avg Task Adherence: {aggregate['avg_task_adherence']:.2%}")
        print(f"  Avg Task Completion: {aggregate['avg_task_completion']:.2%}")
        print(f"  Avg Overall Score: {aggregate['avg_overall_score']:.2%}")
        print(f"  Avg Reasoning Steps: {aggregate['avg_reasoning_steps']:.1f}")
        
        print(f"\nPerformance by Domain:")
        for domain, metrics in aggregate['domain_metrics'].items():
            print(f"  {domain}:")
            print(f"    Tasks: {metrics['count']}")
            print(f"    Avg Understanding: {metrics['avg_understanding']:.2%}")
            print(f"    Avg Task Adherence: {metrics['avg_adherence']:.2%}")
            print(f"    Avg Completion: {metrics['avg_completion']:.2%}")
        
        # Store Mind2Web specific results
        self.mind2web_results = evaluator.results
        self.mind2web_aggregate = aggregate
        
    def export_mind2web_results(self, filename: str):
        """Export Mind2Web evaluation results via ExperimentLogger."""
        if not hasattr(self, 'mind2web_results'):
            print("No Mind2Web results to export")
            return
        if self._logger:
            self._logger.export_mind2web_results(filename, self.mind2web_aggregate, self.mind2web_results)

    def export_results(self, filename: str = "logs/test_results.json"):
        """Export all results to JSON via ExperimentLogger."""
        if self._logger:
            self._logger.export_results(filename)

    def print_summary(self):
        """Print comprehensive summary via ExperimentLogger."""
        if self._logger:
            self._logger.print_summary()

    def close(self):
        """
        Finalise the experiment: export results, print summary, and remove
        the partial log on clean completion.
        """
        if self._logger:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            yaml_stem = Path(self.yaml_path).stem
            self._logger.export_results(f"logs/results-{yaml_stem}-{ts}.json")
            self._logger.print_summary()
            self._logger.close()

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