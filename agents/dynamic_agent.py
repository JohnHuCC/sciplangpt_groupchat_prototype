# dynamic_agent.py

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
                 **kwargs):
        # 先調用父類的初始化方法
        super().__init__(
            name=name,
            base_prompt=base_prompt,
            docs_dir=docs_dir,
            description=description,
            parameters=parameters or {},
            llm_config=llm_config
        )
        
        # 設定本類別特有的參數
        self.similarity_threshold = self.parameters.get("similarity_threshold", 0.0)
        self.max_knowledge_items = self.parameters.get("max_knowledge_items", 3)

    async def _process_message(self, message: str, context: Optional[Dict] = None) -> str:
        """處理具體的消息邏輯"""
        try:
            # 查詢知識庫
            texts, sources, scores = await self._query_knowledge_base(message)
            knowledge_results = self._format_knowledge_results(texts, sources, scores)
            prompt = self._build_prompt(message, knowledge_results, context)
            
            # 生成回應
            messages = [
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": prompt}
            ]

            temperature = context.get("temperature") if context else None
            
            try:
                response = await self.client.chat.completions.create(
                    model=self.llm_config['model'],
                    messages=messages,
                    temperature=temperature or self.parameters.get("temperature", 0.7)
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI API error: {str(e)}")
                raise
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    async def evaluate_capability(self, message: str) -> Dict:
        """評估處理能力"""
        try:
            texts, sources, scores = await self._query_knowledge_base(message, k=1)
            knowledge_score = max(scores) if scores else 0.0
            
            eval_prompt = f"""請評估你處理以下訊息的能力:

訊息: {message}

知識庫相關度: {knowledge_score:.2f}

請考慮:
1. 訊息的類型和複雜度
2. 你的專長和經驗
3. 知識庫中的相關資訊
4. 處理這類訊息的成功經驗

請提供:
1. 能力評分 (0-1的數字，越高表示越適合)
2. 評估理由

只需要回答一個數字評分，然後換行給出理由。"""

            messages = [
                {"role": "system", "content": "你是一個專業的動態處理代理，需要評估自己處理訊息的能力。"},
                {"role": "user", "content": eval_prompt}
            ]
            
            try:
                response = await self.client.chat.completions.create(
                    model=self.llm_config['model'],
                    messages=messages,
                    temperature=0.3
                )
                result = response.choices[0].message.content
                
                try:
                    score = float(result.split('\n')[0])
                    reason = '\n'.join(result.split('\n')[1:])
                except:
                    score = 0.5
                    reason = result
                    
                return {
                    "score": min(max(score, 0.0), 1.0),
                    "reason": reason
                }
            except Exception as e:
                logger.error(f"OpenAI API error: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error evaluating capability: {str(e)}")
            return {
                "score": 0.0,
                "reason": f"評估時發生錯誤: {str(e)}"
            }
            
    async def make_decision(self, message: str) -> str:
        """決定是否需要繼續處理"""
        try:
            texts, _, scores = await self._query_knowledge_base(message, k=2)
            has_more_knowledge = len(scores) > 1 and scores[1] >= self.similarity_threshold

            decision_prompt = f"""請決定是否需要處理以下訊息:

訊息: {message}

知識庫狀態: {"還有其他相關資訊" if has_more_knowledge else "無更多相關資訊"}

請考慮:
1. 訊息是否已經得到完整處理
2. 是否還需要補充或改進
3. 繼續處理是否能帶來更多價值
4. 知識庫中是否還有相關資訊需要使用

請直接回答 'yes' 或 'no'。"""

            messages = [
                {"role": "system", "content": "你是一個專業的動態處理代理，需要決定是否需要繼續處理訊息。"},
                {"role": "user", "content": decision_prompt}
            ]
            
            try:
                response = await self.client.chat.completions.create(
                    model=self.llm_config['model'],
                    messages=messages,
                    temperature=0.1
                )
                return response.choices[0].message.content.strip().lower()
            except Exception as e:
                logger.error(f"OpenAI API error: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error making decision: {str(e)}")
            return "no"

    async def _query_knowledge_base(self, query: str, k: Optional[int] = None) -> Tuple[List[str], List[str], List[float]]:
        """包裝知識庫查詢的輔助函數"""
        try:
            k = k or self.max_knowledge_items
            
            texts, sources, scores = await create_embedding.query_index_async(
                query=query,
                k=k,
                docs_dir=self.docs_dir
            )
            
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

    def _build_prompt(self, query: str, knowledge_results: List[Dict[str, Any]], 
                     context: Optional[Dict] = None) -> str:
        """構建提示文本"""
        formatted_base_prompt = self._format_prompt(self.base_prompt)
        knowledge_format = self.parameters.get("knowledge_format", "detailed")
        
        prompt = f"{formatted_base_prompt}\n\nQuery: {query}\n"
        
        if context and "history" in context:
            history = context["history"][-5:]
            prompt += "\n相關歷史:\n" + "\n".join([
                f"- {msg.get('sender', 'unknown')}: {msg.get('content', '')}"
                for msg in history
            ])
        
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