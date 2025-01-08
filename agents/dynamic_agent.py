from typing import Dict, Any, List, Optional, Tuple
import os
import asyncio
from .base_agent import BaseAgent
from utils.query_knowledge import query_knowledge
import create_embedding

class DynamicAgent(BaseAgent):
    def __init__(self, 
                 name: str,
                 base_prompt: str,
                 docs_dir: str,
                 query_templates: List[str],
                 llm_config: Optional[Dict[str, Any]] = None):
        super().__init__(name, base_prompt, docs_dir, llm_config)
        self.query_templates = query_templates

    async def _query_knowledge_base(self, query: str, k: int = 3) -> Tuple[List[str], List[str], List[float]]:
        """包裝知識庫查詢的輔助函數"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: create_embedding.query_index(
                    query=query,
                    k=k,
                    docs_dir=self.docs_dir
                )
            )
        except Exception as e:
            print(f"Knowledge base query error: {str(e)}")
            # 返回空結果而不是拋出異常，讓主流程可以繼續
            return [], [], []

    def _format_knowledge_results(self, texts: List[str], sources: List[str], 
                                scores: List[float]) -> List[Dict[str, Any]]:
        """格式化知識查詢結果"""
        return [
            {
                'text': text,
                'source': source,
                'similarity': float(score)
            }
            for text, source, score in zip(texts, sources, scores)
        ]

    def _build_prompt(self, query: str, knowledge_results: List[Dict[str, Any]]) -> str:
        """構建提示文本"""
        prompt = f"""{self.base_prompt}

Query: {query}

Relevant Knowledge:
"""
        for result in knowledge_results:
            prompt += f"- {result['text']}\n"
            prompt += f"  (Source: {os.path.basename(result['source'])}, Score: {result['similarity']:.4f})\n"
        
        return prompt

    async def process_query_async(self, query: str) -> Dict[str, Any]:
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

            response = await self._create_chat_completion_async(messages)

            return {
                "response": response,
                "used_knowledge": [
                    {
                        "text": k["text"],
                        "source": os.path.basename(k["source"]),
                        "similarity": k["similarity"]
                    }
                    for k in knowledge_results
                ]
            }

        except Exception as e:
            raise Exception(f"Error processing query: {str(e)}")
            
    def process_query(self, query: str) -> Dict[str, Any]:
        """同步處理查詢（保留向後相容性）"""
        try:
            # 直接使用 create_embedding 的同步方法
            texts, sources, scores = create_embedding.query_index(
                query=query,
                k=3,
                docs_dir=self.docs_dir
            )

            # 複用格式化和提示構建邏輯
            knowledge_results = self._format_knowledge_results(texts, sources, scores)
            prompt = self._build_prompt(query, knowledge_results)

            messages = [
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": prompt}
            ]

            response = self._create_chat_completion(messages)

            return {
                "response": response,
                "used_knowledge": [
                    {
                        "text": k["text"],
                        "source": os.path.basename(k["source"]),
                        "similarity": k["similarity"]
                    }
                    for k in knowledge_results
                ]
            }

        except Exception as e:
            raise Exception(f"Error processing query: {str(e)}")