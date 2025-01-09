import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
from openai import AsyncOpenAI
import numpy as np
import pickle
import faiss
from typing import List, Dict, Tuple
import create_embedding
from fastapi import HTTPException
import dotenv

dotenv.load_dotenv()

# Set OpenAI API Key
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Constants
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_SIZE = 3072
CONFIDENCE_THRESHOLD = 0.4
EMBEDDING_FOLDER = "embedding"

async def embed_text(text: str) -> np.ndarray:
    """Generate an embedding for a given text asynchronously."""
    try:
        response = await client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return np.zeros(EMBEDDING_SIZE, dtype=np.float32)

async def load_index_and_metadata(embedding_folder: str):
    """Load the FAISS index and metadata from the specified folder asynchronously."""
    index_path = os.path.join(embedding_folder, "faiss_index.index")
    metadata_path = os.path.join(embedding_folder, "metadata.pkl")

    # Load FAISS index
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail=f"FAISS index not found at {index_path}.")
    faiss_index = faiss.read_index(index_path)

    # Load metadata
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail=f"Metadata file not found at {metadata_path}.")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # Extract texts and sources from metadata
    texts = metadata.get("texts", [])
    sources = metadata.get("sources", [])

    print(f"Loaded FAISS index and metadata from {embedding_folder}.")
    return faiss_index, texts, sources

async def ensure_embeddings_exist(docs_dir: str) -> bool:
    """確保 embedding 文件存在，如果不存在則生成"""
    try:
        embedding_dir = os.path.join(docs_dir, "embedding")
        index_path = os.path.join(embedding_dir, "faiss_index.index")
        metadata_path = os.path.join(embedding_dir, "metadata.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            print(f"\nEmbedding files not found in {docs_dir}, generating now...")
            await create_embedding.process_files_async(docs_dir)
            return True
        return True
    except Exception as e:
        print(f"Error ensuring embeddings: {str(e)}")
        return False

async def query_knowledge_async(query: str, k: int = 3, docs_dir: str = None) -> List[Dict]:
    """
    查詢知識庫，返回最相關的內容（非同步版本）
    
    Args:
        query (str): 查詢文本
        k (int): 返回結果數量
        docs_dir (str): 文檔目錄路徑
    """
    try:
        if not docs_dir:
            raise HTTPException(status_code=400, detail="docs_dir is required")
            
        # 檢查並確保有 embedding 文件
        if not await ensure_embeddings_exist(docs_dir):
            raise HTTPException(status_code=500, detail=f"Failed to ensure embeddings in {docs_dir}")

        # 使用 create_embedding 查詢
        texts, sources, scores = await create_embedding.query_index_async(
            query=query,
            k=k,
            docs_dir=docs_dir
        )
        
        # 組織返回結果
        results = []
        for text, source, score in zip(texts, sources, scores):
            results.append({
                'text': text,
                'source': source,
                'similarity': float(score)
            })
            
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge base query error: {str(e)}")