"""
HuggingFace and OpenAI model adapters for unified model access
"""

import os
from typing import Optional
from huggingface_hub import InferenceClient
import openai
from langchain_community.llms import HuggingFaceEndpoint
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from abc import ABC, abstractmethod
from config import ConfigExpert

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
        """Generate conversational response from the model"""
        try:
            # Conversational models expect a list of message objects
            messages = [{"role": "user", "content": prompt}]
            config = ConfigExpert.get_instance()
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", config.get("max_tokens", 1024)),
                temperature=kwargs.get("temperature", config.get("temperature", 0.7))
            )
            
            # Check if response is the expected object or an error dict
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            
            # If Hugging Face returns an error dictionary
            if isinstance(response, dict) and "error" in response:
                return f"API Error: {response.get('error')} - {response.get('description', 'No description provided')}"
            
            return "Error: Unexpected response format."
        
        except Exception as e:
            return f"Inference Error: {str(e)}"
    
    def get_langchain_llm(self):
        """Create LangChain-compatible LLM instance"""
        return HuggingFaceEndpoint(
            repo_id=self.model_name,
            huggingfacehub_api_token=self.api_key,
            temperature=0.7,
            max_new_tokens=1024
        )
    
    def get_llamaindex_llm(self):
        """Create LlamaIndex-compatible LLM instance using CustomLLM"""
        
        class HuggingFaceLLM(CustomLLM):
            """Custom LlamaIndex LLM wrapper for HuggingFace models"""
            
            model_name: str
            adapter: 'HuggingFaceAdapter'
            
            def __init__(self, adapter: 'HuggingFaceAdapter'):
                super().__init__()
                self.adapter = adapter
                self.model_name = adapter.model_name
            
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
        
        return HuggingFaceLLM(adapter=self) #FIXME It seems that model name not propagating correctly
    
class OpenAIAdapter(BaseAdapter):
    """
    Unified adapter for OpenAI models
    
    Provides a consistent interface for interacting with OpenAI models
    across different frameworks. Handles API authentication, request formatting,
    and response parsing.
    """
    
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the OpenAI adapter
        
        Args:
            model_name: OpenAI model identifier
            api_key: OpenAI API key
        """
        self._model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion from the model"""
        try:
            client = openai.OpenAI(api_key=self.api_key)
            config = ConfigExpert.get_instance()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", config.get("max_tokens", 1024)),
                temperature=kwargs.get("temperature", config.get("temperature", 0.7)),
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"