from typing import Any, Dict, List, Optional
from .base_agent import BaseAgent
import logging
import json
import os
import numpy as np
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from bs4 import BeautifulSoup
import PyPDF2
import asyncio

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
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
        self.confidence_threshold = parameters.get("confidence_threshold", 0.7)
        self.research_storage = os.path.join(docs_dir, "research_papers")
        os.makedirs(self.research_storage, exist_ok=True)
        
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
            
            if self.context:
                combined_text = "\n\n".join(doc['content'] for doc in self.context)
                await self.initialize_knowledge_embedding(combined_text)
                
        except Exception as e:
            logger.error(f"Error initializing Research Agent: {str(e)}")
            raise

    async def _process_message(self, message: str, context: Optional[Dict] = None) -> str:
        try:
            # 1. 先用現有知識生成初始回答和評估
            initial_result = await self.evaluate_capability(message)
            confidence = initial_result.get("score", 0.0)
            initial_response = initial_result.get("reason", "")

            # 2. 如果信心度低於閾值，啟動爬蟲研究
            if confidence < self.parameters.get("confidence_threshold", 0.7):
                logger.info(f"Confidence {confidence} below threshold, initiating research")
                try:
                    # 開始研究過程
                    research_materials = await self._conduct_research(message)
                    if research_materials:
                        # 使用研究結果生成新的回答
                        enhanced_response = await self._generate_enhanced_response(
                            message, initial_response, research_materials
                        )
                        response = enhanced_response
                    else:
                        response = initial_response
                except Exception as e:
                    logger.error(f"Error during research: {e}")
                    response = initial_response
            else:
                response = initial_response

            # 3. 記錄到研究歷史
            self.research_history.append({
                'role': 'assistant',
                'content': response,
                'confidence': confidence,
                'used_research': confidence < self.parameters.get("confidence_threshold", 0.7)
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error in research agent: {e}")
            raise

    async def _generate_initial_response(self, message: str) -> str:
        """生成初始回答"""
        relevant_docs = await self._analyze_context(message)
        prompt = self._format_prompt(self.base_prompt)
        
        if relevant_docs:
            prompt += "\n\n相關參考資料：\n"
            for doc in relevant_docs:
                prompt += f"\n---\n來源：{doc['source']}\n內容：{doc['content']}\n---"
        
        messages = [
            {"role": "system", "content": prompt},
            *self.research_history[-5:],
            {"role": "user", "content": message}
        ]
        
        return await self._create_chat_completion_async(messages)

    async def _evaluate_response_quality(self, query: str, response: str) -> float:
        """評估回答品質"""
        evaluation_prompt = f"""
        請評估以下回答的品質和完整性。評分標準：
        1. 專業性 (是否包含專業知識和術語)
        2. 完整性 (是否完整回答問題)
        3. 準確性 (資訊是否正確且有根據)
        4. 細節程度 (是否提供足夠細節)

        問題：{query}
        回答：{response}

        請給出0到1之間的分數，僅回覆分數，不需要解釋。
        """
        
        messages = [
            {"role": "system", "content": "You are a critical evaluator."},
            {"role": "user", "content": evaluation_prompt}
        ]
        
        try:
            evaluation = await self._create_chat_completion_async(messages)
            return float(evaluation.strip())
        except:
            logger.error("Error in response evaluation")
            return 0.0

    async def _conduct_research(self, query: str) -> List[Dict[str, Any]]:
        """進行研究並收集資料"""
        try:
            papers = []
            logger.info("Starting Google Scholar research")
            
            # 初始化搜索
            if not self.selenium_get("https://scholar.google.com"):
                return papers
                
            # 搜索查詢
            search_box = self.find_element_safe(
                By.XPATH, '//*[@class="gs_in_txt gs_in_ac"]'
            )
            if not search_box:
                return papers
                
            # 修改搜索詞以更好地匹配學術文獻
            search_query = f"{query} research methodology experimental"
            search_box.send_keys(search_query)
            search_box.send_keys(Keys.RETURN)
            await asyncio.sleep(2)
            
            # 收集論文
            paper_elements = self.driver.find_elements(
                By.XPATH, '//*[@class="gs_r gs_or gs_scl"]'
            )[:5]  # 限制為前5篇
            
            for paper in paper_elements:
                try:
                    paper_data = await self._extract_paper_info(paper)
                    if paper_data:
                        papers.append(paper_data)
                        # 將新資料加入上下文
                        self.context.append({
                            'source': paper_data['link'],
                            'content': paper_data['content'][:1000]  # 限制內容長度
                        })
                except Exception as e:
                    logger.error(f"Error extracting paper info: {e}")
                    continue
            
            logger.info(f"Collected {len(papers)} research papers")
            return papers
            
        except Exception as e:
            logger.error(f"Research error: {e}")
            return []
        finally:
            self.cleanup_driver()

    async def _extract_paper_info(self, paper_element) -> Optional[Dict]:
        """提取論文資訊"""
        try:
            title = paper_element.find_element(By.XPATH, './/*[@class="gs_rt"]').text
            link = paper_element.find_element(
                By.XPATH, './/*[@class="gs_rt"]/a'
            ).get_attribute('href')
            
            content = await self._download_and_extract_content(link)
            
            return {
                "title": title,
                "link": link,
                "content": content
            }
        except Exception as e:
            logger.error(f"Error extracting paper info: {e}")
            return None

    async def _download_and_extract_content(self, url: str) -> str:
        """下載並提取內容"""
        try:
            if url.endswith('.pdf'):
                pdf_path = await self.download_file(
                    url,
                    f"{hash(url)}.pdf",
                    self.research_storage
                )
                if pdf_path:
                    return self._extract_pdf_content(pdf_path)
            else:
                content = await self.fetch_page(url)
                if content:
                    soup = BeautifulSoup(content, 'html.parser')
                    return soup.get_text()
            return ""
        except Exception as e:
            logger.error(f"Content extraction error: {e}")
            return ""

    async def _generate_enhanced_response(
        self,
        query: str,
        initial_response: str,
        research_materials: List[Dict[str, Any]]
    ) -> str:
        """使用研究資料生成增強的回答"""
        try:
            research_context = "\n\n".join([
                f"Title: {paper['title']}\nContent: {paper['content']}\n"
                for paper in research_materials
            ])
            
            prompt = f"""
            基於原始問題、初始評估和新的研究資料，生成更完整和準確的回答。

            原始問題：{query}
            初始評估：{initial_response}
            
            研究資料：
            {research_context}

            請提供一個專業且全面的回答，包含：
            1. 研究發現和關鍵概念
            2. 可能的應用場景
            3. 技術細節和方法
            4. 潛在的限制和注意事項
            """
            
            messages = [
                {"role": "system", "content": "You are a research expert."},
                {"role": "user", "content": prompt}
            ]
            
            enhanced_response = await self._create_chat_completion_async(messages)
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error generating enhanced response: {e}")
            return initial_response

    async def reset(self):
        """重置 Agent 狀態"""
        self.research_history = []
        self.cleanup_driver()
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