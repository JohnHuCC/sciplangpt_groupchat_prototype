from typing import Any, Dict, List, Optional
from .base_agent import BaseAgent
import logging
import json
import os

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """研究助手 Agent 實現類"""
    
    def __init__(self, 
                 name: str, 
                 base_prompt: str, 
                 docs_dir: str,
                 parameters: Dict[str, Any] = None,
                 llm_config: Optional[Dict[str, Any]] = None,
                 knowledge_base: Optional[List[str]] = None):
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
        """初始化 Agent"""
        try:
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
            
            # 初始化 embedding
            if self.context:
                combined_text = "\n\n".join(doc['content'] for doc in self.context)
                await self.initialize_knowledge_embedding(combined_text)
                
        except Exception as e:
            logger.error(f"Error initializing Research Agent: {str(e)}")
            raise

    async def _process_message(self, message: str, context: Optional[Dict] = None) -> str:
        """實現父類的抽象方法"""
        try:
            # 記錄消息
            self.research_history.append({
                'role': 'user',
                'content': message
            })
            
            # 分析相關上下文
            relevant_docs = await self._analyze_context(message)
            
            # 準備提示
            prompt = self._format_prompt(self.base_prompt)
            
            # 添加相關文檔
            if relevant_docs:
                prompt += "\n\n相關參考資料：\n"
                for doc in relevant_docs:
                    prompt += f"\n---\n來源：{doc['source']}\n內容：{doc['content']}\n---"
            
            # 構建消息列表
            messages = [
                {"role": "system", "content": prompt},
                *self.research_history[-5:]  # 只保留最近的5條對話
            ]
            
            # 生成回應
            response = await self._create_chat_completion_async(messages)
            
            # 記錄回應
            self.research_history.append({
                'role': 'assistant',
                'content': response
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message in Research Agent: {str(e)}")
            raise

    async def _analyze_context(self, query: str) -> List[Dict[str, Any]]:
        """分析查詢相關的上下文"""
        relevant_docs = []
        
        try:
            # 創建查詢的 embedding
            query_embedding = await self.create_embedding(query)
            if query_embedding is None or self.knowledge_embedding is None:
                return []
            
            # 計算相似度
            similarity = np.dot(query_embedding, self.knowledge_embedding)
            
            # 如果相似度超過閾值，添加到相關文檔
            threshold = self.parameters.get("similarity_threshold", 0.5)
            if similarity > threshold:
                relevant_docs.extend(self.context)
            
            return relevant_docs
        except Exception as e:
            logger.error(f"Error analyzing context: {str(e)}")
            return []

    async def reset(self):
        """重置 Agent 狀態"""
        self.research_history = []
        logger.info(f"Research Agent {self.name} has been reset")

    def to_dict(self) -> Dict:
        """轉換為字典表示"""
        base_dict = super().to_dict()
        base_dict.update({
            "knowledge_base": self.knowledge_base,
            "context": self.context,
            "research_history": self.research_history
        })
        return base_dict

    @classmethod
    def from_config(cls, config: Dict) -> 'ResearchAgent':
        """從配置創建實例"""
        return cls(
            name=config["name"],
            base_prompt=config["base_prompt"],
            docs_dir=config["docs_dir"],
            parameters=config.get("parameters", {}),
            llm_config=config.get("llm_config"),
            knowledge_base=config.get("knowledge_base", [])
        )