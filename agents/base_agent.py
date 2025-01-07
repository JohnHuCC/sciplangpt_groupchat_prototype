# agents/base_agent.py
from typing import Dict, Any
import os
from openai import OpenAI

class BaseAgent:
    """Base class for all agents in the system"""
    def __init__(self, llm_config: Dict[str, Any]):
        self._validate_config(llm_config)
        self.llm_config = llm_config
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _validate_config(self, llm_config: Dict[str, Any]):
        """Validate agent configuration"""
        if not llm_config:
            raise ValueError("LLM configuration is required")
        if not isinstance(llm_config, dict):
            raise TypeError("LLM configuration must be a dictionary")
        required_keys = ['model']
        missing_keys = [key for key in required_keys if key not in llm_config]
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    def _create_chat_completion(self, messages: list, temperature: float = 0.7) -> str:
        """Centralized method for creating chat completions"""
        try:
            response = self.client.chat.completions.create(
                model=self.llm_config['model'],
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Failed to create chat completion: {str(e)}")