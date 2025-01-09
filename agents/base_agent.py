from openai import AsyncOpenAI
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import os

class BaseAgent:
    """Base class for all agents in the system"""
    def __init__(self, name: str, base_prompt: str, docs_dir: str, 
                 llm_config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.base_prompt = base_prompt
        self.docs_dir = docs_dir
        self.llm_config = llm_config or {
            "model": "gpt-4",
            "temperature": 0.7
        }
        self._executor = ThreadPoolExecutor()
        self._validate_config(self.llm_config)
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize AsyncOpenAI client"""
        try:
            self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
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
            
    async def _create_chat_completion_async(self, messages: list, temperature: float = 0.7) -> str:
        """非同步方式創建聊天完成"""
        try:
            response = await self.client.chat.completions.create(
                model=self.llm_config['model'],
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Failed to create chat completion: {str(e)}")
    
    def to_dict(self) -> Dict:
        """Convert agent instance to dictionary representation"""
        return {
            "name": self.name,
            "base_prompt": self.base_prompt,
            "docs_dir": self.docs_dir,
            "llm_config": self.llm_config
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'BaseAgent':
        """Create agent instance from configuration dictionary"""
        return cls(
            name=config["name"],
            base_prompt=config["base_prompt"],
            docs_dir=config["docs_dir"],
            llm_config=config.get("llm_config")
        )
        
    def __del__(self):
        """確保程序結束時關閉執行器"""
        self._executor.shutdown(wait=False)