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
                 llm_config: Optional[Dict[str, Any]] = None):
        super().__init__(name, base_prompt, docs_dir, parameters, llm_config)

    async def _query_knowledge_base(self, query: str, k: Optional[int] = None) -> Tuple[List[str], List[str], List[float]]:
        """包裝知識庫查詢的輔助函數"""
        try:
            # 從參數中獲取 k 值，如果沒有設置則使用默認值
            k = k or self.parameters.get("max_knowledge_items", 3)
            
            # 從參數中獲取相似度閾值
            similarity_threshold = self.parameters.get("similarity_threshold", 0.0)
            
            texts, sources, scores = await create_embedding.query_index_async(
                query=query,
                k=k,
                docs_dir=self.docs_dir
            )
            
            # 根據相似度閾值過濾結果
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
        # 從參數中獲取格式化選項
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
        # 處理基礎提示
        formatted_base_prompt = self._format_prompt(self.base_prompt)
        
        # 獲取知識展示格式
        knowledge_format = self.parameters.get("knowledge_format", "detailed")
        
        prompt = f"{formatted_base_prompt}\n\nQuery: {query}\n"
        
        if knowledge_results:
            prompt += "\nRelevant Knowledge:\n"
            
            if knowledge_format == "simple":
                # 簡單格式只展示文本
                for result in knowledge_results:
                    prompt += f"- {result['text']}\n"
            else:
                # 詳細格式包含來源和分數
                for result in knowledge_results:
                    prompt += f"- {result['text']}\n"
                    if "source" in result:
                        prompt += f"  Source: {result['source']}"
                    if "similarity" in result:
                        prompt += f" (Similarity: {result['similarity']:.4f})"
                    prompt += "\n"
        
        return prompt

    async def process_query(self, query: str,
                          temperature: Optional[float] = None) -> Dict[str, Any]:
        """非同步處理查詢"""
        try:
            # 查詢知識庫
            texts, sources, scores = await self._query_knowledge_base(query)
            
            # 格式化結果
            knowledge_results = self._format_knowledge_results(texts, sources, scores)
            
            # 構建提示
            prompt = self._build_prompt(query, knowledge_results)
            
            # 生成回應
            messages = [
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": prompt}
            ]

            # 使用可能在參數中指定的溫度值
            response = await self._create_chat_completion_async(messages, temperature)
            
            # 處理回應
            processed_response = self._apply_parameters(response)

            return {
                "content": processed_response,
                "metadata": {
                    "sources": [
                        {
                            "text": k["text"],
                            "source": k.get("source", "unknown"),
                            "similarity": k.get("similarity", 0.0)
                        }
                        for k in knowledge_results
                    ] if self.parameters.get("include_knowledge_details", True) else []
                }
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
            
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