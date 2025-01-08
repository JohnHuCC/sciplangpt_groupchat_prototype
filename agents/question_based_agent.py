from typing import Dict, Any, List
import os
import re
import asyncio
from openai import OpenAI
from agents.base_agent import BaseAgent
from utils.query_knowledge import query_knowledge
from typing import Optional

class QuestionBasedResearchAgent(BaseAgent):
    def __init__(self, name: str, base_prompt: str, docs_dir: str,
                 query_templates: List[str], llm_config: Optional[Dict[str, Any]] = None):
        super().__init__(name, base_prompt, docs_dir, llm_config)
        self.query_templates = query_templates
    
    def to_dict(self) -> Dict:
        """Convert agent instance to dictionary representation"""
        base_dict = super().to_dict()
        base_dict.update({
            "type": "question_based_research",
            "query_templates": self.query_templates
        })
        return base_dict
    
    @classmethod
    def from_config(cls, config: Dict) -> 'QuestionBasedResearchAgent':
        """Create agent instance from configuration dictionary"""
        return cls(
            name=config["name"],
            base_prompt=config["base_prompt"],
            docs_dir=config["docs_dir"],
            query_templates=config["query_templates"],
            llm_config=config.get("llm_config")
        )
        
    async def generate_research_questions_async(self, research_area: str) -> str:
        """非同步生成研究問題"""
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
        prompt = template.format(research_area=research_area)
        
        try:
            messages = [
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": prompt}
            ]
            
            questions = await self._create_chat_completion_async(messages)
            print(f"\n=== 使用 {self.name} 生成的研究問題 ===")
            print(questions)
            print("======================")
            return questions
            
        except Exception as e:
            print(f"生成問題時發生錯誤: {str(e)}")
            raise

    def parse_questions(self, text: str) -> List[str]:
        """從文本中提取問題"""
        questions = []
        pattern = r'問題\d+：(.*?)[？\?]'
        
        print("\n=== 開始解析問題 ===")
        matches = re.finditer(pattern, text, re.MULTILINE)
        for match in matches:
            question = match.group(1).strip() + "？"
            questions.append(question)
            print(f"找到問題: {question}")
        
        print(f"總共解析出 {len(questions)} 個問題")
        return questions

    async def query_knowledge_for_questions_async(self, questions: str) -> Dict[str, List[Dict[str, str]]]:
        """非同步查詢知識庫"""
        results = {}
        
        question_list = self.parse_questions(questions)
        if not question_list:
            raise ValueError(f"未能從文本中解析出有效問題:\n{questions}")
            
        print(f"\n開始查詢 {self.name} 的知識庫...")
        
        # 創建所有問題的查詢任務
        tasks = []
        for question in question_list:
            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(
                self._executor,
                query_knowledge,
                question,
                3,
                self.docs_dir
            )
            tasks.append((question, task))
            
        # 等待所有查詢完成
        for question, task in tasks:
            try:
                query_results = await task
                if query_results:
                    results[question] = query_results
                    print(f"✓ 問題「{question}」找到 {len(query_results)} 條相關知識")
                    for idx, result in enumerate(query_results, 1):
                        print(f"  知識 {idx} 相似度: {result['similarity']:.4f}")
                else:
                    print(f"! 問題「{question}」未找到相關知識")
            except Exception as e:
                print(f"查詢問題「{question}」時發生錯誤: {str(e)}")
                
        if not results:
            raise ValueError("沒有找到任何問題的相關知識")
            
        return results

    async def synthesize_plan_async(self, questions: str, knowledge_results: Dict[str, List[Dict[str, str]]]) -> str:
        """非同步合成最終的研究計畫"""
        try:
            # 使用最後一個模板作為計畫生成模板，如果沒有則使用默認模板
            template = self.query_templates[-1] if len(self.query_templates) > 1 else """\
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
- 標明引用的知識來源
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
            raise RuntimeError(f"Failed to synthesize research plan: {str(e)}")
        
    async def generate_research_plan_async(self, research_area: str) -> Dict[str, Any]:
        """非同步生成研究計畫的主要流程"""
        try:
            print(f"\n==== 開始使用 {self.name} 為 '{research_area}' 生成研究計畫 ====")
            
            # 1. 生成問題
            print("\n1. 生成研究問題...")
            questions = await self.generate_research_questions_async(research_area)
            
            # 2. 查詢知識庫
            print("\n2. 查詢知識庫...")
            knowledge_results = await self.query_knowledge_for_questions_async(questions)
            
            # 3. 生成計畫
            print("\n3. 生成研究計畫...")
            research_plan = await self.synthesize_plan_async(questions, knowledge_results)
            
            return {
                "questions": questions,
                "knowledge_results": knowledge_results,
                "research_plan": research_plan
            }
            
        except Exception as e:
            print(f"\n❌ 生成研究計畫時發生錯誤: {str(e)}")
            raise RuntimeError(f"Failed to generate research plan: {str(e)}")
            
    # 保留同步方法以保持向後相容性
    def generate_research_questions(self, research_area: str) -> str:
        """同步方式生成研究問題"""
        return asyncio.run(self.generate_research_questions_async(research_area))
        
    def query_knowledge_for_questions(self, questions: str) -> Dict[str, List[Dict[str, str]]]:
        """同步方式查詢知識庫"""
        return asyncio.run(self.query_knowledge_for_questions_async(questions))
        
    def generate_research_plan(self, research_area: str) -> Dict[str, Any]:
        """同步方式生成研究計畫"""
        return asyncio.run(self.generate_research_plan_async(research_area))