import logging
from openai import AsyncOpenAI
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
from datetime import datetime
import aiohttp
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import asyncio
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all agents in the system"""
    def __init__(self, name: str, base_prompt: str, docs_dir: str, 
                 description: str,
                 parameters: Dict[str, Any] = None,
                 llm_config: Optional[Dict[str, Any]] = None):
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
        
        self.available_agents = {}
        self.knowledge_embedding = None
        self.embedding_model = parameters.get("embedding_model", "text-embedding-ada-002")
        
        # 爬蟲相關屬性
        self._driver = None
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._validate_config(self.llm_config)
        self._initialize_client()
        logger.info(f"Successfully initialized agent {self.name}")

    async def evaluate_capability(self, prompt: str) -> Dict:
        """評估 agent 處理當前訊息的能力"""
        try:
            messages = [
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._create_chat_completion_async(messages, temperature=0.3)
            
            try:
                score = float(response.split('\n')[0])
                reason = '\n'.join(response.split('\n')[1:])
            except:
                score = 0.0
                reason = response
                
            return {
                "score": min(max(score, 0.0), 1.0),
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"Error evaluating capability: {str(e)}")
            return {
                "score": 0.0,
                "reason": f"評估時發生錯誤: {str(e)}"
            }

    async def make_decision(self, prompt: str) -> str:
        """做出是/否決定"""
        try:
            messages = [
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._create_chat_completion_async(
                messages,
                temperature=0.1
            )
            
            return response.strip().lower()
            
        except Exception as e:
            logger.error(f"Error making decision: {str(e)}")
            return "no"
            
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

    async def process_query(self, message: str, context: Optional[Dict] = None) -> Dict:
        """處理查詢"""
        try:
            self.context = context or {}
            message_trail = self.context.get('message_trail', [])
            processed_agents = self.context.get('processed_agents', [])
            
            current_result = await self._process_message(message, self.context)
            
            current_step = {
                "agent_name": self.name,
                "input": message,
                "output": current_result,
                "timestamp": datetime.now().isoformat()
            }
            message_trail.append(current_step)
            processed_agents.append(self.name)
            
            self.context.update({
                'message_trail': message_trail,
                'processed_agents': processed_agents
            })
            
            # 評估其他 agents 的能力
            if self.available_agents:
                evaluations = []
                for agent_name, agent in self.available_agents.items():
                    if agent_name not in processed_agents:
                        eval_result = await agent.evaluate_capability(current_result)
                        evaluations.append((agent_name, eval_result))
                
                # 選擇評分最高的 agent
                if evaluations:
                    next_agent_name, best_eval = max(evaluations, key=lambda x: x[1]["score"])
                    current_step["next_agent"] = next_agent_name
                    current_step["next_agent_reason"] = best_eval["reason"]
            
            response = {
                "agent": self.name,
                "content": current_result,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "message_trail": message_trail,
                "processed_agents": processed_agents
            }

            # 如果有下一個處理者，繼續處理
            if current_step.get("next_agent") and current_step["next_agent"] in self.available_agents:
                try:
                    next_agent = self.available_agents[current_step["next_agent"]]
                    should_continue = await next_agent.make_decision(
                        f"Do you need to process this message further?\nMessage: {current_result}"
                    )
                    
                    if should_continue == "yes":
                        next_result = await next_agent.process_query(current_result, self.context)
                        response.update({
                            "next_agent_result": next_result,
                            "message_trail": next_result.get("message_trail", message_trail),
                            "processed_agents": next_result.get("processed_agents", processed_agents)
                        })
                except Exception as e:
                    logger.error(f"Error in next agent {current_step['next_agent']}: {str(e)}")
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

    async def _create_chat_completion_async(self, messages: List[Dict[str, str]], 
                                          temperature: Optional[float] = None) -> str:
        """建立非同步 chat completion"""
        try:
            temp = temperature if temperature is not None else self.llm_config.get("temperature", 0.7)
            response = await self.client.chat.completions.create(
                model=self.llm_config["model"],
                messages=messages,
                temperature=temp
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error creating chat completion: {str(e)}")
            raise

    async def create_embedding(self, text: str) -> Optional[np.ndarray]:
        """建立文本嵌入"""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            return None
    
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
            pass
            
        if max_length and len(processed_content) > max_length:
            processed_content = processed_content[:max_length] + "..."
            
        return processed_content

    @property
    def driver(self):
        """懶加載 selenium driver"""
        if self._driver is None:
            from seleniumbase import Driver
            self._driver = Driver(uc=True, headless=True)
        return self._driver

    def cleanup_driver(self):
        """清理 selenium driver"""
        if self._driver:
            self._driver.quit()
            self._driver = None

    def find_element_safe(self, by: By, selector: str) -> Optional[Any]:
        """安全地查找元素"""
        try:
            return self.driver.find_element(by, selector)
        except Exception as e:
            logger.error(f"Error finding element {selector}: {e}")
            return None

    async def fetch_page(self, url: str) -> Optional[str]:
        """抓取網頁內容"""
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    async def download_file(self, url: str, filename: str, folder: str) -> Optional[str]:
        """下載文件"""
        try:
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, filename)
            
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        with open(filepath, 'wb') as f:
                            while True:
                                chunk = await response.content.read(8192)
                                if not chunk:
                                    break
                                f.write(chunk)
                        return filepath
                    return None
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None

    def selenium_get(self, url: str, wait_time: float = 2) -> bool:
        """使用 Selenium 訪問頁面"""
        try:
            self.driver.get(url)
            time.sleep(wait_time)
            return True
        except Exception as e:
            logger.error(f"Error navigating to {url}: {e}")
            return False
    
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
            self.cleanup_driver()
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)
                logger.info(f"Successfully cleaned up resources for agent {self.name}")
        except Exception as e:
            logger.error(f"Error during cleanup for agent {self.name}: {str(e)}")
        
    def __del__(self):
        """確保程序結束時關閉執行器"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
            self.cleanup_driver()