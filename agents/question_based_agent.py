from typing import Dict, Any, List
import os
import re
from .base_agent import BaseAgent
import create_embedding
import numpy as np
import faiss

class QuestionBasedResearchAgent(BaseAgent):
    def __init__(self, llm_config: Dict[str, Any]):
        super().__init__(llm_config)
        self.embedding_processor = None
        self.texts = []
        self.sources = []
        self.metadata = []
        
    def set_knowledge_base(self, agent_dir: str):
        """設置知識庫"""
        self.embedding_processor = create_embedding.EmbeddingProcessor(agent_dir)
        self.texts, self.sources, self.metadata = self.embedding_processor.load_index()
        print(f"Loaded {len(self.texts)} texts from knowledge base")

    def query_knowledge_for_questions(self, questions: str) -> Dict[str, List[Dict[str, Any]]]:
        """針對每個問題查詢知識庫"""
        results = {}
        
        # 確認知識庫已經初始化
        if not self.embedding_processor:
            raise ValueError("Knowledge base not initialized")
        
        # 解析問題
        question_list = self.parse_questions(questions)
        if not question_list:
            print("Warning: No valid questions found")
            return results
            
        print(f"\n開始查詢知識庫...")
        found_any = False
        
        for i, question in enumerate(question_list, 1):
            print(f"\n處理問題 {i}: {question}")
            try:
                # 生成問題的 embedding
                query_embedding = self.embedding_processor.embed_text(question)
                query_embedding = query_embedding.reshape(1, -1)
                
                # 使用 FAISS 搜索最相關的文本
                D, I = self.embedding_processor.index.search(query_embedding, 3)  # top_k=3
                
                # 收集結果
                query_results = []
                for score, idx in zip(D[0], I[0]):
                    if idx < len(self.texts):
                        query_results.append({
                            'text': self.texts[idx],
                            'source': self.sources[idx],
                            'similarity': float(score)
                        })
                
                if query_results:
                    results[question] = query_results
                    found_any = True
                    print(f"✓ 找到 {len(query_results)} 條相關知識")
                    # 印出相似度分數
                    for idx, result in enumerate(query_results, 1):
                        print(f"  知識 {idx} 相似度: {result['similarity']:.4f}")
                else:
                    print(f"! 該問題未找到相關知識")
                    
            except Exception as e:
                print(f"查詢問題 {i} 時發生錯誤: {str(e)}")
        
        if not found_any:
            print("\n注意：沒有找到任何相關知識，將生成基於問題的研究計畫")
        
        return results

    def generate_research_plan(self, research_area: str) -> Dict[str, Any]:
        """生成研究計畫的主要流程"""
        try:
            print(f"\n==== 開始為 '{research_area}' 生成研究計畫 ====")
            
            # 1. 生成問題
            print("\n1. 生成研究問題...")
            questions = self.generate_research_questions(research_area)
            print("生成的問題：")
            print(questions)
            
            # 2. 查詢知識庫
            print("\n2. 查詢知識庫...")
            knowledge_results = self.query_knowledge_for_questions(questions)
            
            # 3. 生成計畫 - 即使沒有找到知識也繼續執行
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
    
    def generate_research_questions(self, research_area: str) -> str:
        """生成研究問題"""
        prompt = f"""
請針對研究領域「{research_area}」生成具體的研究問題。

請按照以下格式生成，確保每個問題都以問號結尾：

問題1：[在此填寫背景或現況相關問題]？
問題2：[在此填寫技術或方法相關問題]？
問題3：[在此填寫實驗設計相關問題]？
問題4：[在此填寫應用價值相關問題]？

注意事項：
1. 每個問題都必須以「問題X：」開頭
2. 每個問題都必須以問號「？」結尾
3. 確保問題具體且有深度
"""
        try:
            messages = [
                {"role": "system", "content": "你是一個研究問題生成專家。"},
                {"role": "user", "content": prompt}
            ]
            
            questions = self._create_chat_completion(messages)
            return questions
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate research questions: {str(e)}")

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

    def synthesize_plan(self, questions: str, knowledge_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """合成研究計畫"""
        try:
            base_prompt = f"""請根據以下研究問題{
            "和相關知識" if knowledge_results else ""}生成一個完整的研究計畫：

研究問題：
{questions}
"""
            # 如果有找到相關知識，加入知識部分
            if knowledge_results:
                base_prompt += "\n相關知識：\n"
                for question, results in knowledge_results.items():
                    base_prompt += f"\n關於「{question}」：\n"
                    for result in results:
                        text = result['text'][:500] + "..." if len(result['text']) > 500 else result['text']
                        base_prompt += f"- {text}\n"
                        base_prompt += f"  (來源: {os.path.basename(result['source'])}, 相關度: {result['similarity']:.4f})\n"

            base_prompt += """\n請根據以上內容，生成一個結構完整的研究計畫，包含：
1. 研究背景與動機
2. 研究目標
3. 研究方法與步驟
4. 預期成果與應用價值
5. 創新點與貢獻

要求：
- 請生成具體且可行的研究計畫
- 確保計畫的邏輯性和完整性
- 如有相關知識可引用，請在適當位置引用"""

            messages = [
                {"role": "system", "content": "你是一位專業的研究計畫撰寫專家。"},
                {"role": "user", "content": base_prompt}
            ]

            try:
                return self._create_chat_completion(messages)
            except Exception as e:
                if "context_length_exceeded" in str(e):
                    # 如果超出長度限制，使用簡化版本重試
                    simplified_prompt = f"""請根據以下研究問題生成一個精簡的研究計畫：

{questions}

請包含：研究背景、目標、方法、預期成果與創新點。"""
                    
                    messages = [
                        {"role": "system", "content": "你是一位專業的研究計畫撰寫專家。請生成簡潔的研究計畫。"},
                        {"role": "user", "content": simplified_prompt}
                    ]
                    return self._create_chat_completion(messages)
                else:
                    raise
            
        except Exception as e:
            raise RuntimeError(f"Failed to synthesize research plan: {str(e)}")