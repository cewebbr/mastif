"""
HuggingFace and OpenAI model adapters for unified model access
"""

import os
from typing import Any, Dict, List, Optional
from huggingface_hub import InferenceClient
import openai
from langchain_community.llms import HuggingFaceEndpoint
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from abc import ABC, abstractmethod
from config import ConfigExpert
from tool_pool import ToolPool
import anthropic

class BaseAdapter(ABC):
    """
    Abstract base class for all model adapters.
    Ensures a consistent interface for all adapters.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the model given a prompt.
        """
        pass

class HuggingFaceAdapter(BaseAdapter):
    """
    Unified adapter for HuggingFace models
    
    Provides a consistent interface for interacting with HuggingFace models
    across different frameworks. Handles API authentication, request formatting,
    and response parsing.
    """

    @staticmethod
    def _normalize_tool_choice(tool_choice):
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            normalized = tool_choice.strip().lower()
            if normalized in ("auto", "none"):
                return normalized
            return tool_choice
        if isinstance(tool_choice, dict):
            t_type = tool_choice.get("type")
            if t_type in ("auto", "none"):
                return t_type
            if t_type == "function" and isinstance(tool_choice.get("function"), dict):
                return tool_choice
        return None
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize the HuggingFace adapter
        
        Args:
            model_name: HuggingFace model identifier
            api_key: HuggingFace API token
        """
        self._model_name = model_name
        self.api_key = api_key or os.getenv("HF_TOKEN")
        self.client = InferenceClient(model=model_name, token=self.api_key)
    
    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate conversational response from the model, executing tool calls if requested."""
        try:
            config = ConfigExpert.get_instance()
            tools = kwargs.get("tools")
            if not isinstance(tools, list) or len(tools) == 0:
                tools = None
            else:
                normalised = []
                for t in tools:
                    if isinstance(t, dict) and t.get("type") == "function":
                        normalised.append(t)
                    elif isinstance(t, str):
                        schemas = ToolPool.get_openai_schemas([t])
                        normalised.extend(schemas)
                    else:
                        name = getattr(t, "name", None) or getattr(t, "tool_name", None)
                        if name:
                            schemas = ToolPool.get_openai_schemas([name])
                            normalised.extend(schemas)
                tools = normalised if normalised else None

            messages = [{"role": "user", "content": prompt}]
            max_tokens = kwargs.get("max_tokens", config.get("max_tokens", 1024))
            temperature = kwargs.get("temperature", config.get("temperature", 0.7))
            max_tool_rounds = config.get("max_tool_rounds", 5)

            for round_num in range(max_tool_rounds):
                request_kwargs = {
                    "messages": messages,
                    "model": self.model_name,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "trust_remote_code": True
                }
                if tools is not None:
                    request_kwargs["tools"] = tools
                    normalized_choice = None
                    if "tool_choice" in kwargs:
                        normalized_choice = self._normalize_tool_choice(kwargs["tool_choice"])
                        if normalized_choice is None and os.getenv("DEBUG_TOOL_CALLS", "false").lower() in ("1", "true", "yes", "on"):
                            print(f"⚠️ HuggingFace adapter dropped invalid tool_choice: {kwargs['tool_choice']}")
                    request_kwargs["tool_choice"] = normalized_choice if normalized_choice is not None else "auto"
                    if os.getenv("DEBUG_TOOL_CALLS", "false").lower() in ("1", "true", "yes", "on"):
                        print(f"🔧 HuggingFace tool request round={round_num} | model={self.model_name} | tool_choice={request_kwargs.get('tool_choice')}")

                response = self.client.chat_completion(**request_kwargs)

                if isinstance(response, dict) and "error" in response:
                    return f"API Error: {response.get('error')} - {response.get('description', 'No description provided')}"

                if not (hasattr(response, 'choices') and len(response.choices) > 0):
                    return "Error: Unexpected response format."

                message = response.choices[0].message

                # Model produced a text response — we're done
                if message.content is not None:
                    return message.content

                # Model requested tool calls — execute them and continue the loop
                tool_calls = getattr(message, "tool_calls", None)
                if not tool_calls:
                    return ""

                # Append assistant message with tool_calls to conversation
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in tool_calls
                    ]
                })

                # Execute each tool call and append results
                for tc in tool_calls:
                    tool_name = tc.function.name
                    try:
                        import json as _json
                        args = _json.loads(tc.function.arguments)
                        query = args.get("query") or args.get("input") or str(args)
                    except Exception:
                        query = tc.function.arguments

                    if os.getenv("DEBUG_TOOL_CALLS", "false").lower() in ("1", "true", "yes", "on"):
                        print(f"🛠️  Executing tool: {tool_name}({query!r})")

                    try:
                        tool_def = ToolPool._registry.get(tool_name)
                        if tool_def:
                            result = tool_def.func(query)
                        else:
                            result = f"Tool '{tool_name}' not found in pool."
                    except Exception as e:
                        result = f"Tool execution error: {str(e)}"

                    if os.getenv("DEBUG_TOOL_CALLS", "false").lower() in ("1", "true", "yes", "on"):
                        print(f"📋 Tool result ({tool_name}): {str(result)[:200]}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    })

            # Exhausted max_tool_rounds without a text response
            return "Error: Maximum tool execution rounds reached without a final text response."

        except Exception as e:
            return f"Inference Error: {str(e)}"
    
    def get_langchain_llm(self):
        """Create LangChain-compatible LLM instance"""
        config = ConfigExpert.get_instance()
        return HuggingFaceEndpoint(
            repo_id=self.model_name,
            huggingfacehub_api_token=self.api_key,
            temperature=config.get("temperature", 0.7),
            max_new_tokens=config.get("max_tokens", 1024)
        )
    
    def get_llamaindex_llm(self):
        """Create LlamaIndex-compatible LLM instance using CustomLLM"""
        
        class HuggingFaceLLM(CustomLLM):
            """Custom LlamaIndex LLM wrapper for HuggingFace models"""
            
            model_name: str
            adapter: 'HuggingFaceAdapter'
            
            def __init__(self, adapter: 'HuggingFaceAdapter'):
                super().__init__(
                    model_name=adapter.model_name,
                    adapter=adapter
                )
            
            @property
            def metadata(self) -> LLMMetadata:
                """Get LLM metadata"""
                return LLMMetadata(
                    context_window=2048,
                    num_output=512,
                    model_name=self.model_name,
                )
            
            @llm_completion_callback()
            def complete(self, prompt: str, **kwargs) -> CompletionResponse:
                """Generate completion"""
                response = self.adapter.generate(prompt, **kwargs)
                return CompletionResponse(text=response)
            
            @llm_completion_callback()
            def stream_complete(self, prompt: str, **kwargs):
                """Stream completion"""
                response = self.complete(prompt, **kwargs)
                yield response
        
        return HuggingFaceLLM(adapter=self)
    
class OpenAIAdapter(BaseAdapter):
    """
    Unified adapter for OpenAI models
    
    Provides a consistent interface for interacting with OpenAI models
    across different frameworks. Handles API authentication, request formatting,
    and response parsing.
    """

    @staticmethod
    def _normalize_tool_choice(tool_choice):
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            normalized = tool_choice.strip().lower()
            if normalized in ("auto", "none"):
                return normalized
            return tool_choice
        if isinstance(tool_choice, dict):
            t_type = tool_choice.get("type")
            if t_type in ("auto", "none"):
                return t_type
            if t_type == "function" and isinstance(tool_choice.get("function"), dict):
                return tool_choice
        return None
    
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the OpenAI adapter
        
        Args:
            model_name: OpenAI model identifier
            api_key: OpenAI API key
        """
        self._model_name = model_name or ConfigExpert.get_instance().get("judge_model", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion from the model, executing tool calls if requested."""
        try:
            client = openai.OpenAI(api_key=self.api_key)
            config = ConfigExpert.get_instance()
            tools = kwargs.get("tools")
            if not isinstance(tools, list) or len(tools) == 0:
                tools = None
            else:
                normalised = []
                for t in tools:
                    if isinstance(t, dict) and t.get("type") == "function":
                        normalised.append(t)
                    elif isinstance(t, str):
                        schemas = ToolPool.get_openai_schemas([t])
                        normalised.extend(schemas)
                    else:
                        name = getattr(t, "name", None) or getattr(t, "tool_name", None)
                        if name:
                            schemas = ToolPool.get_openai_schemas([name])
                            normalised.extend(schemas)
                tools = normalised if normalised else None

            messages = [{"role": "user", "content": prompt}]
            max_tokens = kwargs.get("max_tokens", config.get("max_tokens", 1024))
            temperature = kwargs.get("temperature", config.get("temperature", 0.7))
            max_tool_rounds = config.get("max_tool_rounds", 5)

            for round_num in range(max_tool_rounds):
                request_kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if tools is not None:
                    request_kwargs["tools"] = tools
                    normalized_choice = None
                    if "tool_choice" in kwargs:
                        normalized_choice = self._normalize_tool_choice(kwargs["tool_choice"])
                        if normalized_choice is None and os.getenv("DEBUG_TOOL_CALLS", "false").lower() in ("1", "true", "yes", "on"):
                            print(f"⚠️ OpenAI adapter dropped invalid tool_choice: {kwargs['tool_choice']}")
                    request_kwargs["tool_choice"] = normalized_choice if normalized_choice is not None else "auto"
                    if os.getenv("DEBUG_TOOL_CALLS", "false").lower() in ("1", "true", "yes", "on"):
                        print(f"🔧 OpenAI tool request round={round_num} | model={self.model_name} | tool_choice={request_kwargs.get('tool_choice')}")

                response = client.chat.completions.create(**request_kwargs)
                message = response.choices[0].message

                # Model produced a text response — we're done
                if message.content is not None:
                    return message.content.strip()

                # Model requested tool calls — execute them and continue the loop
                tool_calls = getattr(message, "tool_calls", None)
                if not tool_calls:
                    return ""

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in tool_calls
                    ]
                })

                for tc in tool_calls:
                    tool_name = tc.function.name
                    try:
                        import json as _json
                        args = _json.loads(tc.function.arguments)
                        query = args.get("query") or args.get("input") or str(args)
                    except Exception:
                        query = tc.function.arguments

                    if os.getenv("DEBUG_TOOL_CALLS", "false").lower() in ("1", "true", "yes", "on"):
                        print(f"🛠️  Executing tool: {tool_name}({query!r})")

                    try:
                        tool_def = ToolPool._registry.get(tool_name)
                        if tool_def:
                            result = tool_def.func(query)
                        else:
                            result = f"Tool '{tool_name}' not found in pool."
                    except Exception as e:
                        result = f"Tool execution error: {str(e)}"

                    if os.getenv("DEBUG_TOOL_CALLS", "false").lower() in ("1", "true", "yes", "on"):
                        print(f"📋 Tool result ({tool_name}): {str(result)[:200]}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    })

            return "Error: Maximum tool execution rounds reached without a final text response."

        except Exception as e:
            return f"Error: {str(e)}"

class AnthropicAdapter(BaseAdapter):
    """
    Unified adapter for Anthropic Claude models.
    Handles API communication, tool calling, and response parsing for the Claude 3 family.
    """

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the Anthropic adapter.
        
        Args:
            model_name: Anthropic model identifier (e.g., 'claude-3-5-sonnet-20240620')
            api_key: Anthropic API key
        """
        self._model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _map_tools_to_anthropic(self, tools: List) -> List[Dict[str, Any]]:
        """Maps generic tool definitions or OpenAI schemas to Anthropic tool format."""
        anthropic_tools = []
        for t in tools:
            # If it's already an OpenAI-style schema, convert it
            if isinstance(t, dict) and "function" in t:
                func = t["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                })
            # If it's a tool name string, fetch the schema from ToolPool
            elif isinstance(t, str):
                schemas = ToolPool.get_openai_schemas([t])
                for s in schemas:
                    func = s["function"]
                    anthropic_tools.append({
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                    })
        return anthropic_tools

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic's Messages API with support for tool loops."""
        try:
            config = ConfigExpert.get_instance()
            raw_tools = kwargs.get("tools")
            anthropic_tools = self._map_tools_to_anthropic(raw_tools) if raw_tools else []
            
            messages = [{"role": "user", "content": prompt}]
            max_tokens = kwargs.get("max_tokens", config.get("max_tokens", 1024))
            temperature = kwargs.get("temperature", config.get("temperature", 0.7))
            max_tool_rounds = config.get("max_tool_rounds", 5)

            for round_num in range(max_tool_rounds):
                request_kwargs = {
                    "model": self.model_name,
                    "max_tokens": max_tokens,
                    "messages": messages,
                    "temperature": temperature,
                }
                if anthropic_tools:
                    request_kwargs["tools"] = anthropic_tools

                response = self.client.messages.create(**request_kwargs)
                
                # Check for tool use in the content blocks
                tool_requests = [b for b in response.content if b.type == "tool_use"]
                text_outputs = [b for b in response.content if b.type == "text"]

                # If no tool use, return the text content
                if not tool_requests:
                    return text_outputs[0].text if text_outputs else ""

                # Append assistant's tool-use blocks to the message history
                messages.append({"role": "assistant", "content": response.content})

                # Process tool outputs
                for tool_call in tool_requests:
                    tool_name = tool_call.name
                    tool_input = tool_call.input
                    
                    # Handle varying input formats (query, input, or raw dict)
                    query = tool_input.get("query") or tool_input.get("input") or str(tool_input)

                    if os.getenv("DEBUG_TOOL_CALLS", "false").lower() in ("1", "true"):
                        print(f"🛠️ Anthropic executing: {tool_name}({query!r})")

                    try:
                        tool_def = ToolPool._registry.get(tool_name)
                        result = tool_def.func(query) if tool_def else f"Tool '{tool_name}' not found."
                    except Exception as e:
                        result = f"Error: {str(e)}"

                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": str(result),
                            }
                        ],
                    })

            return "Error: Maximum tool rounds reached."

        except Exception as e:
            return f"Anthropic Error: {str(e)}"