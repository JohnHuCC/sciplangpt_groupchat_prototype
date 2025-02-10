from typing import Dict, Any, List, Optional, Tuple
import os
import logging
import asyncio
import re 
from fastapi import HTTPException
from .base_agent import BaseAgent
from utils.query_knowledge import query_knowledge_async
import create_embedding
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import aiohttp
from bs4 import BeautifulSoup
import io
import PyPDF2
import requests

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
        parameters = parameters or {}
        # 添加研究相關的預設參數
        default_research_params = {
            "similarity_threshold": 0.0,
            "max_knowledge_items": 3,
            "confidence_threshold": 0.7,
            "max_research_papers": 5,
            "research_wait_time": 2.0,
            "enable_research": True  # 控制是否啟用研究功能
        }
        # 合併用戶參數和預設參數
        parameters = {**default_research_params, **parameters}
        
        super().__init__(
            name=name,
            base_prompt=base_prompt,
            docs_dir=docs_dir,
            description=description,
            parameters=parameters,
            llm_config=llm_config
        )
        
        self.similarity_threshold = self.parameters.get("similarity_threshold", 0.0)
        self.max_knowledge_items = self.parameters.get("max_knowledge_items", 3)
        self.research_history = []

    async def _process_message(self, message: str, context: Optional[Dict] = None) -> str:
        """處理具體的消息邏輯"""
        try:
            logger.info(f"Processing message: {message}")
            logger.info(f"Research enabled: {self.parameters.get('enable_research', True)}")
            logger.info(f"Current parameters: {self.parameters}")

            # 1. 先用現有知識庫查詢
            texts, sources, scores = await self._query_knowledge_base(message)
            knowledge_results = self._format_knowledge_results(texts, sources, scores)
            initial_prompt = self._build_prompt(message, knowledge_results, context)
            
            # 2. 生成初始回應
            messages = [
                {"role": "system", "content": self.base_prompt},
                {"role": "user", "content": initial_prompt}
            ]
            
            temperature = context.get("temperature") if context else None
            initial_response = await self._create_chat_completion_async(messages, temperature)
            logger.info("Generated initial response")

            # 3. 強制進行研究 (用於測試)
            force_research = True  # 添加這行來強制執行研究
            if self.parameters.get("enable_research", True) and force_research:
                logger.info("Starting response quality evaluation")
                confidence = await self._evaluate_response_quality(message, initial_response)
                logger.info(f"Response quality score: {confidence}")
                
                # 4. 降低閾值確保研究觸發
                threshold = 0.9  # 設置較高的閾值以確保觸發研究
                if confidence < threshold:
                    logger.info(f"Confidence {confidence} below threshold {threshold}, conducting research")
                    try:
                        research_materials = await self._conduct_research(message)
                        logger.info(f"Research completed, found {len(research_materials)} materials")
                        
                        if research_materials:
                            logger.info("Generating enhanced response with research materials")
                            enhanced_response = await self._generate_enhanced_response(
                                message, initial_response, research_materials
                            )
                            return enhanced_response
                        else:
                            logger.info("No research materials found")
                    except Exception as e:
                        logger.error(f"Error during research: {e}")
                        # 即使研究失敗也返回初始回應
                        return initial_response

            logger.info("Returning initial response")
            return initial_response
            
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
                response = await self._create_chat_completion_async(messages, 0.3)
                result = response
                
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

        請給出0到1之間的分數，僅回覆分數。
        """
        
        try:
            messages = [
                {"role": "system", "content": "You are a critical evaluator."},
                {"role": "user", "content": evaluation_prompt}
            ]
            
            eval_response = await self._create_chat_completion_async(messages, temperature=0.1)
            # return float(eval_response.strip())
            return 0.5
        except:
            return 0.0
            
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
                response = await self._create_chat_completion_async(messages, 0.1)
                return response.strip().lower()
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

    async def _conduct_research(self, query: str) -> List[Dict[str, Any]]:
        """進行研究並收集資料"""
        max_retries = 3
        retry_count = 0
        research_storage = os.path.join(self.docs_dir, "research_papers")
        os.makedirs(research_storage, exist_ok=True)
        
        while retry_count < max_retries:
            try:
                papers = []
                url_list = []
                
                # 1. 搜索 Google Scholar
                if not self.selenium_get("https://scholar.google.com"):
                    logger.error("Failed to access Google Scholar")
                    retry_count += 1
                    await asyncio.sleep(1)
                    continue
                    
                search_box = self.find_element_safe(By.XPATH, '//*[@class="gs_in_txt gs_in_ac"]')
                if not search_box:
                    logger.error("Search box not found")
                    retry_count += 1
                    await asyncio.sleep(1)
                    continue
                
                search_query = f"{query} research methodology"
                logger.info(f"Searching for: {search_query}")
                search_box.send_keys(search_query)
                search_box.send_keys(Keys.RETURN)
                await asyncio.sleep(2)
                
                while len(papers) < 5:  # 持續搜索直到找到5篇論文
                    # 2. 收集論文資訊
                    paper_elements = self.driver.find_elements(
                        By.XPATH, '//*[@class="gs_r gs_or gs_scl"]'
                    )
                    
                    if not paper_elements:
                        logger.error("No papers found on current page")
                        break

                    logger.info(f"Found {len(paper_elements)} papers on current page")
                    
                    for paper in paper_elements:
                        try:
                            paper_info = {}
                            # 獲取標題和連結
                            title_elem = paper.find_element(By.XPATH, './/*[@class="gs_rt"]')
                            paper_info['title'] = title_elem.text
                            
                            # 獲取連結
                            link_elem = title_elem.find_element(By.XPATH, './/a')
                            paper_info['link'] = link_elem.get_attribute('href')
                            
                            # 獲取摘要
                            try:
                                abstract_elem = paper.find_element(By.CLASS_NAME, 'gs_rs')
                                paper_info['abstract'] = abstract_elem.text
                            except:
                                paper_info['abstract'] = ""
                            
                            # 查找PDF連結
                            try:
                                pdf_links = paper.find_elements(By.XPATH, './/a[contains(@href, ".pdf")]')
                                if pdf_links:
                                    paper_info['pdf_link'] = pdf_links[0].get_attribute('href')
                            except:
                                paper_info['pdf_link'] = None

                            url_list.append(paper_info)
                        except Exception as e:
                            logger.error(f"Error extracting paper info: {e}")
                            continue

                    # 3. 嘗試下載文件
                    for paper_info in url_list:
                        if len(papers) >= 5:  # 檢查是否已達到目標
                            break
                            
                        success = False
                        
                        # 首先嘗試直接 PDF 連結
                        if paper_info.get('pdf_link'):
                            file_path = await self._download_pdf(paper_info['pdf_link'], research_storage, f"paper_{len(papers)}.pdf")
                            if file_path:
                                success = True
                        
                        # 如果直接下載失敗，嘗試 DOI 方法
                        if not success:
                            try:
                                # 嘗試從頁面中提取 DOI
                                response = requests.get(
                                    paper_info['link'], 
                                    headers={'User-Agent': 'Mozilla/5.0'},
                                    timeout=10
                                )
                                if response.status_code == 200:
                                    doi_pattern = r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+'
                                    soup = BeautifulSoup(response.text, 'html.parser')
                                    
                                    # 方法1：從 meta 標籤找 DOI
                                    doi_meta = soup.find("meta", {"name": "citation_doi"})
                                    if doi_meta and doi_meta.get("content"):
                                        doi = doi_meta["content"]
                                    else:
                                        # 方法2：從頁面內容找 DOI
                                        doi_match = re.search(doi_pattern, response.text)
                                        doi = doi_match.group(0) if doi_match else None
                                    
                                    if doi:
                                        pdf_url = await self._get_unpaywall_pdf(doi)
                                        if pdf_url:
                                            file_path = await self._download_pdf(
                                                pdf_url, 
                                                research_storage, 
                                                f"paper_{len(papers)}.pdf"
                                            )
                                            if file_path:
                                                success = True
                            except Exception as e:
                                logger.error(f"Error processing paper link: {e}")
                        
                        # 如果上述方法都失敗，保存網頁內容
                        if not success and paper_info.get('abstract'):
                            try:
                                file_path = os.path.join(research_storage, f"paper_{len(papers)}.txt")
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write(f"Title: {paper_info['title']}\n\n")
                                    f.write(f"URL: {paper_info['link']}\n\n")
                                    f.write(f"Abstract:\n{paper_info['abstract']}\n")
                                success = True
                            except Exception as e:
                                logger.error(f"Error saving text content: {e}")
                        
                        if success:
                            papers.append({
                                "title": paper_info['title'],
                                "link": paper_info['link'],
                                "file_path": file_path,
                                "content": paper_info.get('abstract', '')
                            })
                            logger.info(f"Successfully saved paper {len(papers)} / 5")

                    # 如果還沒收集到5篇論文，嘗試下一頁
                    if len(papers) < 5:
                        try:
                            next_buttons = self.driver.find_elements(By.CSS_SELECTOR, "#gs_n a")
                            next_page_found = False
                            for button in next_buttons:
                                if button.text == "Next" or button.text == "»":
                                    button.click()
                                    next_page_found = True
                                    await asyncio.sleep(2)
                                    break
                            
                            if not next_page_found:
                                logger.info("No more pages available")
                                break
                        except Exception as e:
                            logger.error(f"Error navigating to next page: {e}")
                            break

                if papers:  # 如果有找到任何論文就返回
                    logger.info(f"Returning with {len(papers)} papers")
                    return papers
                
                retry_count += 1
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Research error: {e}")
                retry_count += 1
                await asyncio.sleep(1)
            finally:
                self.cleanup_driver()
        
        return []
    
    async def _get_unpaywall_pdf(self, doi: str, email: str = "example@email.com") -> Optional[str]:
        """使用 Unpaywall API 獲取 PDF URL"""
        try:
            api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("best_oa_location"):
                            return data["best_oa_location"]["url_for_pdf"]
            return None
        except Exception as e:
            logger.error(f"Error getting PDF URL from Unpaywall: {e}")
            return None
        

    async def _download_pdf(self, url: str, folder: str, filename: str) -> Optional[str]:
        """下載 PDF 文件"""
        try:
            file_path = os.path.join(folder, filename)
            
            # 使用 requests 進行下載
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/pdf,*/*'
            }
            
            response = requests.get(url, headers=headers, stream=True, timeout=15, allow_redirects=True)
            if response.status_code == 200:
                # 檢查內容類型
                if "application/pdf" in response.headers.get("Content-Type", ""):
                    # 以二進制方式保存文件
                    with open(file_path, "wb") as pdf_file:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                pdf_file.write(chunk)
                    logger.info(f"PDF downloaded successfully: {file_path}")
                    return file_path
                else:
                    logger.error(f"URL does not point to a PDF: {response.headers.get('Content-Type')}")
                    return None
            else:
                logger.error(f"Failed to download PDF, status code: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            return None

        
    async def _download_and_extract_content(self, url: str) -> Optional[str]:
        """下載並提取內容"""
        try:
            logger.info(f"Attempting to download content from: {url}")
            
            # ResearchGate 特殊處理
            if 'researchgate.net' in url:
                try:
                    logger.info("Using Selenium for ResearchGate")
                    self.selenium_get(url)
                    await asyncio.sleep(3)  # 等待頁面載入
                    
                    # 嘗試多個可能的按鈕選擇器
                    button_selectors = [
                        "//button[contains(@class, 'download-button')]",
                        "//button[contains(@class, 'pdf-download')]",
                        "//a[contains(@class, 'download-fulltext')]",
                        "//div[contains(@class, 'download-link')]",
                        # ResearchGate 的替代內容定位
                        "//div[contains(@class, 'research-detail-header')]",
                        "//div[contains(@class, 'publication-abstract')]",
                        "//div[contains(@class, 'publication-content')]"
                    ]
                    
                    content_found = False
                    text_content = []
                    
                    # 收集頁面上的相關內容
                    for selector in button_selectors:
                        try:
                            elements = self.driver.find_elements(By.XPATH, selector)
                            for element in elements:
                                if element.is_displayed():
                                    # 如果是按鈕，嘗試點擊
                                    if element.tag_name in ['button', 'a']:
                                        try:
                                            element.click()
                                            await asyncio.sleep(2)
                                        except:
                                            pass
                                    # 如果是內容元素，提取文本
                                    else:
                                        text = element.text
                                        if text and len(text.strip()) > 50:  # 確保內容有意義
                                            text_content.append(text.strip())
                                            content_found = True
                        except:
                            continue
                    
                    # 如果找到了內容
                    if content_found:
                        final_content = "\n\n".join(text_content)
                        logger.info(f"Successfully extracted content from ResearchGate, length: {len(final_content)}")
                        return final_content
                    
                    # 如果上述方法都失敗，提取整個頁面文本
                    page_source = self.driver.page_source
                    soup = BeautifulSoup(page_source, 'html.parser')
                    
                    # 移除不需要的元素
                    for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                        element.decompose()
                    
                    text = soup.get_text(separator=' ', strip=True)
                    if len(text) > 100:  # 確保提取的內容有意義
                        return text
                        
                except Exception as e:
                    logger.error(f"Selenium extraction failed for ResearchGate: {e}")
                finally:
                    self.cleanup_driver()
            
            # 非 ResearchGate 網站的處理（保持原有的處理邏輯）
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/pdf,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch URL: {url}")
                        return None
                    
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'pdf' in content_type:
                        try:
                            raw_data = await response.read()
                            pdf_file = io.BytesIO(raw_data)
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            text = ""
                            for page in pdf_reader.pages:
                                text += page.extract_text() or ""
                            return text
                        except Exception as e:
                            logger.error(f"Error processing PDF: {e}")
                    else:
                        try:
                            text = await response.text(errors='ignore')
                            soup = BeautifulSoup(text, 'html.parser')
                            
                            # 移除不需要的元素
                            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                                element.decompose()
                            
                            content = soup.get_text(separator=' ', strip=True)
                            if len(content) > 100:
                                return content
                                
                        except Exception as e:
                            logger.error(f"Error processing HTML: {e}")
                    
            return None
                    
        except Exception as e:
            logger.error(f"Error in content extraction: {e}")
            return None
    
    async def _extract_paper_info(self, paper_element, storage_path: str, count: int) -> Optional[Dict]:
        """提取論文資訊"""
        try:
            logger.info("Attempting to extract paper info")
            
            # 提取標題和連結
            title_element = paper_element.find_element(By.XPATH, './/*[@class="gs_rt"]')
            title = title_element.text.strip()
            logger.info(f"Found title: {title}")
            
            # 先嘗試找 PDF 連結
            pdf_link = None
            try:
                pdf_links = paper_element.find_elements(By.XPATH, './/a[contains(@href, ".pdf")]')
                if pdf_links:
                    pdf_link = pdf_links[0].get_attribute('href')
                    logger.info(f"Found PDF link: {pdf_link}")
            except Exception as e:
                logger.error(f"Error finding PDF link: {e}")

            # 獲取主要連結
            try:
                link_element = title_element.find_element(By.XPATH, './/a')
                link = link_element.get_attribute('href')
                logger.info(f"Found main link: {link}")
            except Exception as e:
                logger.error(f"Error getting main link: {e}")
                if not pdf_link:
                    return None

            # 使用 PDF 連結或主要連結
            content_link = pdf_link or link

            # 提取摘要
            try:
                abstract_element = paper_element.find_element(By.CLASS_NAME, 'gs_rs')
                abstract = abstract_element.text.strip()
                logger.info(f"Found abstract: {abstract[:100]}...")
            except Exception as e:
                logger.error(f"Error getting abstract: {e}")
                abstract = ""

            # 存儲文件
            file_path = os.path.join(storage_path, f"paper_{count}.txt")
            try:
                content = await self._download_and_extract_content(content_link)
                if content:
                    with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
                        f.write(f"Title: {title}\n\n")
                        f.write(f"URL: {content_link}\n\n")
                        f.write(f"Abstract: {abstract}\n\n")
                        f.write("Content:\n")
                        f.write(content)
                    logger.info(f"Successfully saved content to {file_path}")
                else:
                    # 如果無法獲取內容，至少保存標題和摘要
                    with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
                        f.write(f"Title: {title}\n\n")
                        f.write(f"URL: {content_link}\n\n")
                        f.write(f"Abstract: {abstract}\n\n")
            except Exception as e:
                logger.error(f"Error saving content: {e}")
                
            return {
                "title": title,
                "link": content_link,
                "file_path": file_path,
                "abstract": abstract
            }

        except Exception as e:
            logger.error(f"Error in paper extraction: {e}")
            return None

    async def _generate_enhanced_response(
        self,
        query: str,
        initial_response: str,
        research_materials: List[Dict[str, Any]]
    ) -> str:
        """生成增強的回答"""
        try:
            # 讀取保存的文件內容
            research_context = []
            for paper in research_materials:
                try:
                    if paper.get('file_path') and os.path.exists(paper['file_path']):
                        with open(paper['file_path'], 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            research_context.append(f"Title: {paper['title']}\n\n{content[:1000]}...")
                    else:
                        research_context.append(f"Title: {paper['title']}\nAbstract: {paper.get('abstract', '')}")
                except Exception as e:
                    logger.error(f"Error reading research material: {e}")
                    continue
            
            if not research_context:
                return initial_response

            research_text = "\n\n---\n\n".join(research_context)
            
            enhancement_prompt = f"""
            基於原始問題、初始回答和新的研究資料，生成更完整和準確的回答。

            原始問題：{query}
            初始回答：{initial_response}
            
            研究資料：
            {research_text}

            請生成一個更全面的回答，包含：
            1. 更多專業細節和術語
            2. 具體的研究支持
            3. 可能的應用場景
            4. 潛在的限制或注意事項
            """
            
            messages = [
                {"role": "system", "content": "You are a research expert."},
                {"role": "user", "content": enhancement_prompt}
            ]
            
            return await self._create_chat_completion_async(messages)
        except Exception as e:
            logger.error(f"Error generating enhanced response: {e}")
            return initial_response

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