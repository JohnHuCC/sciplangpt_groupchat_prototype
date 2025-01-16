from typing import Dict, Any, List, Optional
import os
import re
from fastapi import HTTPException
from pydantic import BaseModel
from agents.base_agent import BaseAgent
from utils.query_knowledge import query_knowledge_async
import logging

logger = logging.getLogger(__name__)

class ResearchQuestion(BaseModel):
    question: str
    knowledge: List[Dict[str, Any]]

class QuestionBasedResearchAgent(BaseAgent):
    def __init__(self, name: str, base_prompt: str, docs_dir: str,
                 query_templates: List[str], 
                 parameters: Dict[str, Any] = None,
                 llm_config: Optional[Dict[str, Any]] = None):
        super().__init__(name, base_prompt, docs_dir, parameters, llm_config)
        self.query_templates = query_templates

    async def _process_message(self, message: str, context: Optional[Dict] = None) -> Dict:
        """實現具體的消息處理邏輯"""
        try:
            # 1. 使用 embedding 進行相關性查詢
            knowledge_results = await query_knowledge_async(message, 3, self.docs_dir)
            logger.info(f"Found {len(knowledge_results)} relevant knowledge pieces")
            
            # 2. 構建提示
            prompt = self._build_prompt(message, knowledge_results)
            logger.info("Built prompt with knowledge context")
            
            # 3. 生成回應
            messages = [
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._create_chat_completion_async(messages)
            logger.info("Generated response from LLM")
            
            # 4. 處理回應並加入相關知識來源
            processed_response = self._apply_parameters(response)
            
            result = {
                "content": processed_response,
                "knowledge_used": knowledge_results,
                "metadata": {
                    "sources": [
                        {
                            "file": os.path.basename(result["source"]),
                            "similarity": result["similarity"],
                            "content": result["text"][:200] + "..."  # 只包含前200字符
                        } for result in knowledge_results
                    ],
                    "prompt_tokens": len(prompt.split()),
                    "response_tokens": len(processed_response.split())
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in _process_message: {str(e)}")
            raise

    def _build_prompt(self, message: str, knowledge_results: List[Dict]) -> str:
        """構建包含知識上下文的提示"""
        try:
            # 使用模板（如果有）或使用默認模板
            template = self.query_templates[0] if self.query_templates else """
請基於以下相關知識來回答問題。回答時請：
1. 確保回答準確且與問題相關
2. 適當引用相關知識來源
3. 如果知識庫中沒有足夠信息，請明確指出

問題：
{question}

相關知識：
{knowledge}

請提供完整且準確的回答。
"""
            # 格式化知識內容
            knowledge_text = ""
            for i, result in enumerate(knowledge_results, 1):
                knowledge_text += f"\n來源 {i} ({os.path.basename(result['source'])})：\n{result['text']}\n"
                knowledge_text += f"相關度：{result['similarity']:.4f}\n"
            
            # 構建最終提示
            prompt = template.format(
                question=message,
                knowledge=knowledge_text
            )
            
            return self._format_prompt(prompt)
            
        except Exception as e:
            logger.error(f"Error building prompt: {str(e)}")
            raise

    async def generate_research_questions(self, research_area: str) -> str:
        """生成研究問題"""
        template = self.query_templates[0] if self.query_templates else """
請針對研究領域「{research_area}」生成具體的研究問題。

請按照以下格式生成，確保每個問題都以問號結尾：

問題1：[第一個研究問題]？
問題2：[第二個研究問題]？
問題3：[第三個研究問題]？
問題4：[第四個研究問題]？

注意事項：
1. 每個問題都必須以「問題X：」開頭
2. 每個問題都必須以問號「？」結尾
3. 問題需要涵蓋：
   - 研究背景和現況
   - 技術方法和創新點
   - 實驗設計和驗證方法
   - 應用價值和影響
"""
        try:
            prompt = template.format(research_area=research_area)
            
            messages = [
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": prompt}
            ]
            
            questions = await self._create_chat_completion_async(messages)
            logger.info(f"Generated research questions for {research_area}")
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate questions: {str(e)}"
            )

    def parse_questions(self, text: str) -> List[str]:
        """從文本中提取問題"""
        questions = []
        pattern = r'問題\d+：(.*?)[？\?]'
        
        try:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                question = match.group(1).strip() + "？"
                questions.append(question)
                logger.info(f"Parsed question: {question}")
            
            logger.info(f"Total {len(questions)} questions parsed")
            return questions
            
        except Exception as e:
            logger.error(f"Error parsing questions: {str(e)}")
            return []

    async def query_knowledge_for_questions(self, questions: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """為每個問題查詢相關知識"""
        results = {}
        
        try:
            # 並行處理所有問題的查詢
            for question in questions:
                try:
                    query_results = await query_knowledge_async(question, 3, self.docs_dir)
                    if query_results:
                        results[question] = query_results
                        logger.info(f"Found {len(query_results)} results for question: {question}")
                    else:
                        logger.warning(f"No results found for question: {question}")
                except Exception as e:
                    logger.error(f"Error querying knowledge for question '{question}': {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in query_knowledge_for_questions: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error querying knowledge: {str(e)}"
            )

    async def synthesize_research_plan(self, questions: str, knowledge_results: Dict[str, List[Dict[str, str]]]) -> str:
        """合成研究計畫"""
        try:
            template = self.query_templates[-1] if len(self.query_templates) > 1 else """
請根據以下研究問題和相關知識，生成一個完整的研究計畫：

===== 研究問題 =====
{questions}

===== 相關知識 =====
{knowledge}

請根據以上內容，生成一個結構完整的研究計畫，包含：
1. 研究背景與動機
2. 研究目標
3. 研究方法與步驟
4. 預期成果與應用價值
5. 創新點與貢獻

要求：
- 整合所有問題和相關知識
- 在適當位置引用知識來源
- 注意研究計畫的可行性和完整性
"""
            # 格式化知識內容
            knowledge_content = ""
            for question, results in knowledge_results.items():
                knowledge_content += f"\n針對問題：{question}\n相關知識：\n"
                for result in results:
                    knowledge_content += f"- {result['text']}\n"
                    knowledge_content += f"  (來源: {os.path.basename(result['source'])}, 相關度: {result['similarity']:.4f})\n"

            prompt = template.format(
                questions=questions,
                knowledge=knowledge_content
            )

            messages = [
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": prompt}
            ]
            
            return await self._create_chat_completion_async(messages)
            
        except Exception as e:
            logger.error(f"Error synthesizing research plan: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to synthesize research plan: {str(e)}"
            )