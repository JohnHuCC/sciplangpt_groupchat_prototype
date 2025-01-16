from typing import Dict, Any, List, Optional, Tuple
import os
import logging
from fastapi import HTTPException
from .base_agent import BaseAgent
from utils.query_knowledge import query_knowledge_async
import create_embedding

logger = logging.getLogger(__name__)

class DynamicAgent(BaseAgent):
    def __init__(self, 
                 name: str,
                 base_prompt: str,
                 docs_dir: str,
                 description: str = "Dynamic processing agent for general tasks",
                 parameters: Dict[str, Any] = None,
                 llm_config: Optional[Dict[str, Any]] = None,
                 **kwargs):  # 添加 **kwargs 來接收額外的參數
        super().__init__(
            name=name,
            base_prompt=base_prompt,
            docs_dir=docs_dir,
            description=description,
            parameters=parameters,
            llm_config=llm_config
        )
        
        self.similarity_threshold = parameters.get("similarity_threshold", 0.0)
        self.max_knowledge_items = parameters.get("max_knowledge_items", 3)

    async def _process_message(self, message: str, context: Optional[Dict] = None) -> str:
        """實現父類的抽象方法，處理具體的消息邏輯"""
        try:
            # 查詢知識庫
            texts, sources, scores = await self._query_knowledge_base(message)
            
            # 格式化結果
            knowledge_results = self._format_knowledge_results(texts, sources, scores)
            
            # 構建提示
            prompt = self._build_prompt(message, knowledge_results)
            
            # 生成回應
            messages = [
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": prompt}
            ]

            # 使用可能在參數中指定的溫度值
            temperature = context.get("temperature") if context else None
            response = await self._create_chat_completion_async(messages, temperature)
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    async def _query_knowledge_base(self, query: str, k: Optional[int] = None) -> Tuple[List[str], List[str], List[float]]:
        """包裝知識庫查詢的輔助函數"""
        try:
            k = k or self.max_knowledge_items
            
            texts, sources, scores = await create_embedding.query_index_async(
                query=query,
                k=k,
                docs_dir=self.docs_dir
            )
            
            # 根據相似度閾值過濾結果
            if self.similarity_threshold > 0:
                filtered_results = [
                    (text, source, score)
                    for text, source, score in zip(texts, sources, scores)
                    if score >= self.similarity_threshold
                ]
                if filtered_results:
                    texts, sources, scores = zip(*filtered_results)
                    return list(texts), list(sources), list(scores)
                return [], [], []
                
            return texts, sources, scores
            
        except Exception as e:
            logger.error(f"Knowledge base query error: {str(e)}")
            return [], [], []

    def _format_knowledge_results(self, texts: List[str], sources: List[str], 
                                scores: List[float]) -> List[Dict[str, Any]]:
        """格式化知識查詢結果"""
        include_scores = self.parameters.get("include_scores", True)
        include_sources = self.parameters.get("include_sources", True)
        
        results = []
        for text, source, score in zip(texts, sources, scores):
            result = {"text": text}
            if include_sources:
                result["source"] = os.path.basename(source)
            if include_scores:
                result["similarity"] = float(score)
            results.append(result)
            
        return results

    def _build_prompt(self, query: str, knowledge_results: List[Dict[str, Any]]) -> str:
        """構建提示文本"""
        formatted_base_prompt = self._format_prompt(self.base_prompt)
        knowledge_format = self.parameters.get("knowledge_format", "detailed")
        
        prompt = f"{formatted_base_prompt}\n\nQuery: {query}\n"
        
        if knowledge_results:
            prompt += "\nRelevant Knowledge:\n"
            
            if knowledge_format == "simple":
                for result in knowledge_results:
                    prompt += f"- {result['text']}\n"
            else:
                for result in knowledge_results:
                    prompt += f"- {result['text']}\n"
                    if "source" in result:
                        prompt += f"  Source: {result['source']}"
                    if "similarity" in result:
                        prompt += f" (Similarity: {result['similarity']:.4f})"
                    prompt += "\n"
        
        return prompt

    @classmethod
    def from_config(cls, config: Dict) -> 'DynamicAgent':
        """從配置創建實例"""
        return cls(
            name=config["name"],
            base_prompt=config["base_prompt"],
            docs_dir=config["docs_dir"],
            description=config.get("description", "Dynamic processing agent for general tasks"),
            parameters=config.get("parameters", {}),
            llm_config=config.get("llm_config")
        )