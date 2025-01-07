import os
from typing import List, Optional, Tuple, Dict
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
import fitz  # PyMuPDF
import openai
import pickle
import dotenv
from openai import OpenAI
import re

dotenv.load_dotenv()

class EmbeddingProcessor:
    EMBEDDING_MODEL = "text-embedding-3-large"
    EMBEDDING_SIZE = 3072
    
    def __init__(self, base_dir: str):
        """
        初始化 EmbeddingProcessor
        
        Args:
            base_dir (str): agent 的基礎目錄
        """
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.base_dir = base_dir
        self.embedding_dir = os.path.join(base_dir, "embedding")
        self.index_file = os.path.join(self.embedding_dir, "faiss_index.index")
        self.metadata_file = os.path.join(self.embedding_dir, "metadata.pkl")
        
        # 創建必要的目錄
        os.makedirs(self.embedding_dir, exist_ok=True)
        
        # 初始化 FAISS 索引
        self.index = faiss.IndexFlatIP(self.EMBEDDING_SIZE)
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def preprocess_text(self, text: str) -> str:
        """預處理文本，移除參考文獻和引用"""
        sections_to_remove = [
            'References', 'Bibliography', 'Citations', 
            'Works Cited', 'Acknowledgments', 'Appendix', 'Appendices'
        ]
        
        pattern = r'\n\s*(' + '|'.join(sections_to_remove) + r')\s*\n'
        text = re.split(pattern, text, flags=re.IGNORECASE)[0]
        
        # 移除引用
        text = re.sub(r'\([^)]*\b(?:et al\.,?\s*\d{4}|\d{4})[^)]*\)', '', text)
        text = re.sub(r'\[\d+(?:-\d+)?\]', '', text)
        text = re.sub(r'\^\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def embed_text(self, text: str) -> np.ndarray:
        """生成文本的 embedding"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.EMBEDDING_MODEL
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return np.zeros(self.EMBEDDING_SIZE, dtype=np.float32)

    def read_file(self, filepath: str) -> Optional[str]:
        """根據文件類型讀取內容"""
        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext == '.pdf':
                return self._read_pdf(filepath)
            elif ext == '.txt':
                return self._read_txt(filepath)
            elif ext == '.docx':
                return self._read_docx(filepath)
            else:
                print(f"Unsupported file format: {ext}")
                return None
        except Exception as e:
            print(f"Error reading file {filepath}: {str(e)}")
            return None

    def _read_pdf(self, filepath: str) -> str:
        doc = fitz.open(filepath)
        text = [page.get_text() for page in doc]
        return "\n".join(text)

    def _read_txt(self, filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()

    def _read_docx(self, filepath: str) -> str:
        doc = Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])

    def process_files(self) -> bool:
        """處理目錄中的所有文件並創建索引"""
        texts = []
        sources = []
        metadata = []
        all_embeddings = []

        # 獲取目錄中的所有文件
        files = [f for f in os.listdir(self.base_dir) 
                if os.path.isfile(os.path.join(self.base_dir, f)) and 
                f.lower().endswith(('.pdf', '.txt', '.docx'))]
                
        print(f"Found {len(files)} files in {self.base_dir}")

        for filename in files:
            filepath = os.path.join(self.base_dir, filename)
            print(f"Processing file: {filepath}")
            
            content = self.read_file(filepath)
            if content:
                cleaned_content = self.preprocess_text(content)
                print(f"Original content length: {len(content)} characters")
                print(f"Cleaned content length: {len(cleaned_content)} characters")

                chunks = self.text_splitter.split_text(cleaned_content)
                print(f"Split file into {len(chunks)} chunks")

                # 創建 embeddings
                embeddings = [self.embed_text(chunk) for chunk in chunks]
                valid_embeddings = [emb for emb in embeddings if np.linalg.norm(emb) > 0]
                
                if valid_embeddings:
                    all_embeddings.extend(valid_embeddings)
                    texts.extend(chunks[:len(valid_embeddings)])
                    sources.extend([filepath] * len(valid_embeddings))
                    metadata.extend([{"source": filepath}] * len(valid_embeddings))
                else:
                    print(f"No valid embeddings for file {filepath}")

        # 將 embeddings 添加到索引
        if all_embeddings:
            all_embeddings = np.array(all_embeddings, dtype=np.float32)
            self.index.add(all_embeddings)
            print(f"Added {len(all_embeddings)} embeddings to index")
            
            # 保存索引和元數據
            self.save_index(texts, sources, metadata)
            return True
        
        print("No embeddings were generated")
        return False

    def save_index(self, texts: List[str], sources: List[str], metadata: List[dict]):
        """保存索引和元數據"""
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump({
                    'texts': texts,
                    'sources': sources,
                    'metadata': metadata
                }, f)
            print(f"Index and metadata saved successfully")
        except Exception as e:
            print(f"Error saving index: {str(e)}")

    def load_index(self) -> Tuple[List[str], List[str], List[Dict]]:
        """載入索引和元數據"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                self.index = faiss.read_index(self.index_file)
                with open(self.metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    return data['texts'], data['sources'], data['metadata']
            return [], [], []
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return [], [], []

def process_files_for_agent(agent_dir: str) -> bool:
    """處理特定 agent 目錄中的文件"""
    processor = EmbeddingProcessor(agent_dir)
    return processor.process_files()

def load_agent_index(agent_dir: str) -> Tuple[List[str], List[str], List[Dict]]:
    """載入特定 agent 的索引"""
    processor = EmbeddingProcessor(agent_dir)
    return processor.load_index()




