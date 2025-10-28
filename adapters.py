"""
HuggingFace model adapter for unified model access
"""

import os
from typing import Optional
from huggingface_hub import InferenceClient
import openai
from langchain_community.llms import HuggingFaceEndpoint
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback

class HuggingFaceAdapter:
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
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HF_TOKEN")
        self.client = InferenceClient(token=self.api_key)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion from the model"""
        try:
            response = self.client.text_generation(
                prompt,
                model=self.model_name,
                max_new_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                return_full_text=False
            )
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_langchain_llm(self):
        """Create LangChain-compatible LLM instance"""
        return HuggingFaceEndpoint(
            repo_id=self.model_name,
            huggingfacehub_api_token=self.api_key,
            temperature=0.7,
            max_new_tokens=512
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
        
        return HuggingFaceLLM(adapter=self)
    
class OpenAIAdapter:
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
        self.model_name = model_name or os.getenv("JUDGE_MODEL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion from the model"""
        try:
            openai.api_key = self.api_key
            response = openai.Completion.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
            )
            return response.choices[0].text.strip()
        except Exception as e:
            return f"Error: {str(e)}"