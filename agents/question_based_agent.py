from typing import Dict, Any, List
import os
import re
from openai import OpenAI
from .base_agent import BaseAgent
from utils.query_knowledge import query_knowledge

class QuestionBasedResearchAgent(BaseAgent):
    def __init__(self, llm_config: Dict[str, Any]):
        super().__init__(llm_config)
        
    def generate_research_questions(self, research_area: str) -> str:
        """生成研究問題"""
        prompt = f"""
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
            messages = [
                {"role": "system", "content": "你是一個研究問題生成專家。"},
                {"role": "user", "content": prompt}
            ]
            
            questions = self._create_chat_completion(messages)
            print("\n=== 生成的研究問題 ===")
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

    def query_knowledge_for_questions(self, questions: str) -> Dict[str, List[Dict[str, str]]]:
        """查詢知識庫"""
        results = {}
        
        # 解析問題
        question_list = self.parse_questions(questions)
        if not question_list:
            raise ValueError(f"未能從文本中解析出有效問題:\n{questions}")
            
        print(f"\n開始查詢知識庫...")
        for i, question in enumerate(question_list, 1):
            print(f"\n處理問題 {i}: {question}")
            try:
                # 使用新的 query_knowledge 函數格式
                query_results = query_knowledge(question, k=3)
                if query_results:
                    results[question] = query_results
                    print(f"✓ 找到 {len(query_results)} 條相關知識")
                    # 印出找到的知識的相似度分數
                    for idx, result in enumerate(query_results, 1):
                        print(f"  知識 {idx} 相似度: {result['similarity']:.4f}")
                else:
                    print(f"! 該問題未找到相關知識")
            except Exception as e:
                print(f"查詢問題 {i} 時發生錯誤: {str(e)}")
                
        if not results:
            raise ValueError("沒有找到任何問題的相關知識")
            
        return results
        
    def generate_research_plan(self, research_area: str) -> Dict[str, Any]:
        """生成研究計畫的主要流程"""
        try:
            print(f"\n==== 開始為 '{research_area}' 生成研究計畫 ====")
            
            # 1. 生成問題
            print("\n1. 生成研究問題...")
            questions = self.generate_research_questions(research_area)
            
            # 2. 查詢知識庫
            print("\n2. 查詢知識庫...")
            knowledge_results = self.query_knowledge_for_questions(questions)
            
            # 3. 生成計畫
            print("\n3. 生成研究計畫...")
            research_plan = self.synthesize_plan(questions, knowledge_results)
            
            return {
                "questions": questions,
                "knowledge_results": knowledge_results,
                "research_plan": research_plan
            }
            
        except Exception as e:
            print(f"\n❌ 生成研究計畫時發生錯誤: {str(e)}")
            raise RuntimeError(f"Failed to generate research plan: {str(e)}")
            
    def synthesize_plan(self, questions: str, knowledge_results: Dict[str, List[Dict[str, str]]]) -> str:
        """合成最終的研究計畫"""
        try:
            prompt = f"""請根據以下研究問題和相關知識，生成一個完整的研究計畫：

===== 研究問題 =====
{questions}

===== 相關知識 =====
"""
            # 添加每個問題的相關知識
            for question, results in knowledge_results.items():
                prompt += f"\n針對問題：{question}\n相關知識：\n"
                for result in results:
                    prompt += f"- {result['text']}\n"
                    prompt += f"  (來源: {os.path.basename(result['source'])}, 相關度: {result['similarity']:.4f})\n"

            prompt += """\n請根據以上內容，生成一個結構完整的研究計畫，包含：
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

            messages = [
                {"role": "system", "content": "你是一位專業的研究計畫撰寫專家。"},
                {"role": "user", "content": prompt}
            ]
            
            return self._create_chat_completion(messages)
            
        except Exception as e:
            raise RuntimeError(f"Failed to synthesize research plan: {str(e)}")