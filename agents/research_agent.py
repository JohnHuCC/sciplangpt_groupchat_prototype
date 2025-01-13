from typing import Any, Dict, List, Optional
from .base_agent import BaseAgent
import logging
import json
import os

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """
    研究助手 Agent 實現類
    繼承自 BaseAgent，專門處理研究相關的查詢和任務
    """
    
    def __init__(self, 
                 name: str, 
                 base_prompt: str, 
                 docs_dir: str,
                 parameters: Dict[str, Any] = None,
                 llm_config: Optional[Dict[str, Any]] = None,
                 knowledge_base: Optional[List[str]] = None):
        """
        初始化研究助手 Agent
        
        Args:
            name (str): Agent 的名稱
            base_prompt (str): 基礎提示文本
            docs_dir (str): 文檔目錄
            parameters (Dict[str, Any], optional): 額外參數
            llm_config (Dict[str, Any], optional): LLM 配置
            knowledge_base (List[str], optional): 知識庫文件列表
        """
        super().__init__(
            name=name,
            base_prompt=base_prompt,
            docs_dir=docs_dir,
            parameters=parameters,
            llm_config=llm_config
        )
        self.knowledge_base = knowledge_base or []
        self.context = []
        self.research_history = []
        
    async def initialize(self):
        """
        初始化 Agent，加載知識庫和其他必要資源
        """
        try:
            # 加載知識庫文件
            if self.knowledge_base:
                for file_path in self.knowledge_base:
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            self.context.append({
                                'source': file_path,
                                'content': content
                            })
            logger.info(f"Research Agent {self.name} initialized with {len(self.context)} knowledge base files")
        except Exception as e:
            logger.error(f"Error initializing Research Agent: {str(e)}")
            raise

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        處理接收到的消息
        
        Args:
            message (Dict[str, Any]): 輸入消息
            
        Returns:
            Dict[str, Any]: 處理後的回應
        """
        try:
            # 記錄接收到的消息
            self.research_history.append({
                'role': 'user',
                'content': message.get('content', '')
            })
            
            # 準備完整的提示文本
            prompt = self._format_prompt(self.base_prompt)
            
            # 添加知識庫內容到提示中
            if self.context:
                prompt += "\n\n相關參考資料：\n"
                for doc in self.context:
                    prompt += f"\n---\n來源：{doc['source']}\n內容：{doc['content']}\n---"
            
            # 構建消息列表
            messages = [
                {"role": "system", "content": prompt},
                *self.research_history[-5:]  # 只保留最近的5條對話歷史
            ]
            
            # 使用 OpenAI API 生成回應
            response_content = await self._create_chat_completion_async(messages)
            
            # 應用參數處理
            processed_response = self._apply_parameters(response_content)
            
            # 記錄回應
            self.research_history.append({
                'role': 'assistant',
                'content': processed_response
            })
            
            return {
                'content': processed_response,
                'type': 'text',
                'metadata': {
                    'sources': [doc.get('source') for doc in self.context]
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing message in Research Agent: {str(e)}")
            return {
                'content': f"抱歉，處理您的請求時發生錯誤: {str(e)}",
                'error': True
            }

    async def _analyze_context(self, query: str) -> List[Dict[str, Any]]:
        """
        分析查詢相關的上下文
        
        Args:
            query (str): 用戶查詢
            
        Returns:
            List[Dict[str, Any]]: 相關的上下文信息列表
        """
        relevant_docs = []
        
        try:
            # 簡單的關鍵詞匹配，可以替換為更複雜的相關性分析
            for doc in self.context:
                content = doc.get('content', '').lower()
                if query.lower() in content:
                    relevant_docs.append(doc)
            
            return relevant_docs
        except Exception as e:
            logger.error(f"Error analyzing context: {str(e)}")
            return []

    async def reset(self):
        """
        重置 Agent 狀態
        """
        self.research_history = []
        logger.info(f"Research Agent {self.name} has been reset")

    async def save_state(self) -> Dict[str, Any]:
        """
        保存 Agent 的當前狀態
        
        Returns:
            Dict[str, Any]: Agent 的狀態數據
        """
        state = {
            'name': self.name,
            'base_prompt': self.base_prompt,
            'docs_dir': self.docs_dir,
            'parameters': self.parameters,
            'llm_config': self.llm_config,
            'knowledge_base': self.knowledge_base,
            'context': self.context,
            'research_history': self.research_history
        }
        return state
        
    async def load_state(self, state: Dict[str, Any]):
        """
        加載 Agent 的狀態
        
        Args:
            state (Dict[str, Any]): 要加載的狀態數據
        """
        self.name = state.get('name', self.name)
        self.base_prompt = state.get('base_prompt', self.base_prompt)
        self.docs_dir = state.get('docs_dir', self.docs_dir)
        self.parameters = state.get('parameters', self.parameters)
        self.llm_config = state.get('llm_config', self.llm_config)
        self.knowledge_base = state.get('knowledge_base', self.knowledge_base)
        self.context = state.get('context', self.context)
        self.research_history = state.get('research_history', self.research_history)

    def to_dict(self) -> Dict:
        """Convert agent instance to dictionary representation"""
        base_dict = super().to_dict()
        base_dict.update({
            "knowledge_base": self.knowledge_base,
            "context": self.context,
            "research_history": self.research_history
        })
        return base_dict

    @classmethod
    def from_config(cls, config: Dict) -> 'ResearchAgent':
        """Create agent instance from configuration dictionary"""
        return cls(
            name=config["name"],
            base_prompt=config["base_prompt"],
            docs_dir=config["docs_dir"],
            parameters=config.get("parameters", {}),
            llm_config=config.get("llm_config"),
            knowledge_base=config.get("knowledge_base", [])
        )