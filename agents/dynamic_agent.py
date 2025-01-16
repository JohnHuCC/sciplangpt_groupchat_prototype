from typing import Dict, Any, List, Optional, Tuple
import os
from fastapi import HTTPException
from .base_agent import BaseAgent
from utils.query_knowledge import query_knowledge_async
import create_embedding

class DynamicAgent(BaseAgent):
    def __init__(self, 
                 name: str,
                 base_prompt: str,
                 docs_dir: str,
                 parameters: Dict[str, Any] = None,
                 llm_config: Optional[Dict[str, Any]] = None,
                 **kwargs):  # 添加 **kwargs 來接收額外的參數
        super().__init__(name, base_prompt, docs_dir, parameters, llm_config)

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
            raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

    async def _query_knowledge_base(self, query: str, k: Optional[int] = None) -> Tuple[List[str], List[str], List[float]]:
        """包裝知識庫查詢的輔助函數"""
        try:
            k = k or self.parameters.get("max_knowledge_items", 3)
            similarity_threshold = self.parameters.get("similarity_threshold", 0.0)
            
            texts, sources, scores = await create_embedding.query_index_async(
                query=query,
                k=k,
                docs_dir=self.docs_dir
            )
            
            if similarity_threshold > 0:
                filtered_results = [
                    (text, source, score)
                    for text, source, score in zip(texts, sources, scores)
                    if score >= similarity_threshold
                ]
                if filtered_results:
                    texts, sources, scores = zip(*filtered_results)
                    return list(texts), list(sources), list(scores)
                return [], [], []
                
            return texts, sources, scores
            
        except Exception as e:
            print(f"Knowledge base query error: {str(e)}")
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
            parameters=config.get("parameters", {}),
            llm_config=config.get("llm_config")
        )