import logging
from openai import AsyncOpenAI
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import os

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all agents in the system"""
    def __init__(self, name: str, base_prompt: str, docs_dir: str, 
                 parameters: Dict[str, Any] = None,
                 llm_config: Optional[Dict[str, Any]] = None):
        # 參數驗證
        if not name or not isinstance(name, str):
            raise ValueError("Name must be a non-empty string")
        if not base_prompt or not isinstance(base_prompt, str):
            raise ValueError("Base prompt must be a non-empty string")
        if not docs_dir or not isinstance(docs_dir, str):
            raise ValueError("Docs directory must be a non-empty string")
        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary")
            
        self.name = name
        self.base_prompt = base_prompt
        self.docs_dir = docs_dir
        self.parameters = parameters or {}
        self.llm_config = llm_config or {
            "model": "gpt-4",
            "temperature": self.parameters.get("temperature", 0.7)
        }
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._validate_config(self.llm_config)
        self._initialize_client()
        logger.info(f"Successfully initialized agent {self.name}")
    
    def _initialize_client(self):
        """Initialize AsyncOpenAI client"""
        try:
            self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            logger.info(f"Successfully initialized OpenAI client for agent {self.name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
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
            
    async def _create_chat_completion_async(self, messages: list, 
                                          temperature: Optional[float] = None) -> str:
        """非同步方式創建聊天完成"""
        try:
            temp = temperature or self.parameters.get("temperature", 0.7)
            
            response = await self.client.chat.completions.create(
                model=self.llm_config['model'],
                messages=messages,
                temperature=temp
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to create chat completion: {str(e)}")
            raise RuntimeError(f"Failed to create chat completion: {str(e)}")
    
    def _format_prompt(self, base_prompt: str) -> str:
        """根據參數格式化提示文本"""
        style = self.parameters.get("prompt_style", "default")
        output_format = self.parameters.get("output_format")
        
        formatted_prompt = base_prompt
        
        if style == "formal":
            formatted_prompt = "Please provide a formal and professional response.\n" + formatted_prompt
        elif style == "creative":
            formatted_prompt = "Please provide a creative and innovative response.\n" + formatted_prompt
            
        if output_format:
            if output_format == "json":
                formatted_prompt += "\nPlease provide the response in JSON format."
            elif output_format == "markdown":
                formatted_prompt += "\nPlease provide the response in Markdown format."
            elif output_format == "bullet_points":
                formatted_prompt += "\nPlease provide the response as bullet points."
                
        return formatted_prompt
    
    def _apply_parameters(self, content: str) -> str:
        """根據參數處理內容"""
        max_length = self.parameters.get("max_length")
        language = self.parameters.get("language", "default")
        
        processed_content = content
        
        if language != "default":
            # 這裡可以添加語言處理邏輯
            pass
            
        if max_length and len(processed_content) > max_length:
            processed_content = processed_content[:max_length] + "..."
            
        return processed_content
    
    def to_dict(self) -> Dict:
        """Convert agent instance to dictionary representation"""
        # 只返回可序列化的資料
        return {
            "name": self.name,
            "base_prompt": self.base_prompt,
            "docs_dir": self.docs_dir,
            "parameters": self.parameters.copy(),  # 使用 copy 避免修改原始數據
            "type": self.__class__.__name__,  # 添加類型資訊
            "created_at": datetime.now().isoformat()  # 添加創建時間
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'BaseAgent':
        """Create agent instance from configuration dictionary"""
        return cls(
            name=config["name"],
            base_prompt=config["base_prompt"],
            docs_dir=config["docs_dir"],
            parameters=config.get("parameters", {}),
            llm_config=config.get("llm_config")
        )
    
    async def cleanup(self):
        """非同步清理資源"""
        try:
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)
                logger.info(f"Successfully cleaned up resources for agent {self.name}")
        except Exception as e:
            logger.error(f"Error during cleanup for agent {self.name}: {str(e)}")
        
    def __del__(self):
        """確保程序結束時關閉執行器"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)