import logging
from openai import AsyncOpenAI
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all agents in the system"""
    def __init__(self, name: str, base_prompt: str, docs_dir: str, 
                 description: str,
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
        self.description = description
        self.base_prompt = base_prompt
        self.docs_dir = docs_dir
        self.parameters = parameters or {}
        self.llm_config = llm_config or {
            "model": "gpt-4",
            "temperature": self.parameters.get("temperature", 0.7)
        }
        
        # 新增的屬性
        self.available_agents = {}  # 可用的其他 agents
        self.knowledge_embedding = None
        self.embedding_model = parameters.get("embedding_model", "text-embedding-ada-002")
        
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._validate_config(self.llm_config)
        self._initialize_client()
        logger.info(f"Successfully initialized agent {self.name}")
        
    def register_available_agents(self, agents: Dict[str, 'BaseAgent']):
        """註冊可用的其他 agents"""
        self.available_agents = {name: agent for name, agent in agents.items() 
                               if name != self.name}
        
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
            
    async def decide_next_agent(self, message: str, current_result: str) -> Optional[str]:
        """決定下一個處理的 agent"""
        try:
            if not self.available_agents:
                return None

            prompt = f"""基於當前的處理結果，決定下一步應該由哪個 agent 處理。請考慮每個 agent 的專長來做決定。

    當前消息: {message}
    當前處理結果: {current_result}

    可用的 agents:
    {chr(10).join([f"- {name}: {agent.description}" for name, agent in self.available_agents.items()])}

    重要規則：
    1. 除非真的完全不需要其他 agent 的專業知識，否則應該選擇一個最適合的 agent 繼續處理
    2. 要充分利用每個 agent 的專長
    3. 確保答案的完整性和全面性

    請從上述 agents 中選擇一個最適合處理下一步的 agent。
    只需要回答 agent 的名稱，如果確實不需要進一步處理才回答 "NONE"。"""

            messages = [
                {"role": "system", "content": "你是一個專業的任務分配器，需要確保工作流程的完整性和效率。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._create_chat_completion_async(messages, temperature=0.3)
            next_agent = response.strip()
            
            logger.info(f"Agent {self.name} decided next agent: {next_agent}")
            
            if next_agent == "NONE":
                logger.warning(f"Agent {self.name} decided to end the chain. Available agents were: {list(self.available_agents.keys())}")
                return None
            elif next_agent in self.available_agents:
                return next_agent
            else:
                logger.warning(f"Invalid next agent decision: {next_agent}")
                # 如果決策無效，選擇第一個可用的 agent
                return next(iter(self.available_agents))
                
        except Exception as e:
            logger.error(f"Error deciding next agent: {str(e)}")
            return None

    async def create_embedding(self, text: str) -> Optional[np.ndarray]:
        """創建文本的 embedding"""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            return None

    async def initialize_knowledge_embedding(self, knowledge_text: str):
        """初始化 agent 的知識 embedding"""
        try:
            self.knowledge_embedding = await self.create_embedding(knowledge_text)
            logger.info(f"Successfully initialized knowledge embedding for agent {self.name}")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge embedding: {str(e)}")
            
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
            
    async def process_query(self, message: str, context: Optional[Dict] = None) -> Dict:
        """處理查詢"""
        try:
            message_trail = context.get('message_trail', []) if context else []
            
            # 處理當前消息
            current_result = await self._process_message(message, context)
            
            # 記錄當前處理結果
            current_step = {
                "agent_name": self.name,
                "input": message,
                "output": current_result,
                "timestamp": datetime.now().isoformat()
            }
            message_trail.append(current_step)
            
            # 決定下一個處理者
            next_agent_name = await self.decide_next_agent(message, current_result)
            current_step["next_agent"] = next_agent_name
            
            response = {
                "agent": self.name,
                "content": current_result,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "message_trail": message_trail
            }

            # 如果有下一個處理者，繼續處理
            if next_agent_name and next_agent_name in self.available_agents:
                try:
                    next_agent = self.available_agents[next_agent_name]
                    context = context or {}
                    context['message_trail'] = message_trail
                    
                    next_result = await next_agent.process_query(current_result, context)
                    response["next_agent_result"] = next_result
                    if "message_trail" in next_result:
                        response["message_trail"] = next_result["message_trail"]
                except Exception as e:
                    logger.error(f"Error in next agent {next_agent_name}: {str(e)}")
                    response["next_agent_error"] = str(e)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query in agent {self.name}: {str(e)}")
            return {
                "agent": self.name,
                "error": str(e),
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "message_trail": message_trail
            }
    
    async def _process_message(self, message: str, context: Optional[Dict] = None) -> str:
        """實際處理消息的方法，子類需要實現此方法"""
        raise NotImplementedError("Subclasses must implement _process_message method")
    
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
        return {
            "name": self.name,
            "base_prompt": self.base_prompt,
            "docs_dir": self.docs_dir,
            "description": self.description,
            "parameters": self.parameters.copy(),
            "type": self.__class__.__name__,
            "created_at": datetime.now().isoformat()
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'BaseAgent':
        """Create agent instance from configuration dictionary"""
        return cls(
            name=config["name"],
            base_prompt=config["base_prompt"],
            docs_dir=config["docs_dir"],
            description=config.get("description", ""),
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