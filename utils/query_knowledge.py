# Prompt formation function
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
import openai
import dotenv
from openai import OpenAI
import numpy as np
import pickle
import faiss
from typing import List, Dict, Tuple
import create_embedding

dotenv.load_dotenv()

# Set OpenAI API Key

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Constants
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_SIZE = 3072 #default for text-embedding-3-large
CONFIDENCE_THRESHOLD = 0.4
EMBEDDING_FOLDER = "embedding"

def embed_text(text: str) -> np.ndarray:
    """Generate an embedding for a given text."""
    try:
        response = client.embeddings.create(
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

def load_index_and_metadata(embedding_folder: str):
    """Load the FAISS index and metadata from the specified folder."""
    index_path = os.path.join(embedding_folder, "faiss_index.index")
    metadata_path = os.path.join(embedding_folder, "metadata.pkl")

    # Load FAISS index
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}.")
    faiss_index = faiss.read_index(index_path)

    # Load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}.")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    # Extract texts and sources from metadata
    texts = metadata.get("texts", [])
    sources = metadata.get("sources", [])

    print(f"Loaded FAISS index and metadata from {embedding_folder}.")
    return faiss_index, texts, sources


# def query_knowledge(question: str, k: int = 5) -> list[dict[str, str]]:
#     """
#     Embed the question and query the FAISS index to retrieve relevant documents.

#     :param question: The query or research question.
#     :param k: The number of top results to retrieve.
#     :return: A list of dictionaries with retrieved text, source, and similarity score.
#     """
#     # Load FAISS index and metadata
#     faiss_index, texts, sources = load_index_and_metadata(EMBEDDING_FOLDER)

#     # Generate embedding for the question
#     print("Generating embedding for the query...")
#     query_embedding = embed_text(question).reshape(1, -1).astype('float32')

#     # Ensure index is trained
#     # if not faiss_index.is_trained:
#     #     print("FAISS index is not trained. Cannot perform queries.")
#     #     return []

#     # Perform the search
#     print("Performing FAISS search...")
#     D, I = faiss_index.search(query_embedding, k)

#     results = []
#     for distance, idx in zip(D[0], I[0]):
#         if idx < len(texts):
#             result = {
#                 'text': texts[idx],
#                 'source': sources[idx],
#                 'similarity': distance
#             }
#             results.append(result)
#     # print(result)

#     # Optionally, filter results based on similarity threshold
#     filtered_results = [res for res in results if res['similarity'] >= CONFIDENCE_THRESHOLD]

#     if filtered_results:
#         print(f"Query successful. Retrieved {len(filtered_results)} relevant documents.")
#     else:
#         print("Query successful, but no relevant documents exceeded the similarity threshold.")

#     return filtered_results[:k]
def ensure_embeddings_exist(docs_dir: str) -> bool:
    """確保 embedding 文件存在，如果不存在則生成"""
    try:
        embedding_dir = os.path.join(docs_dir, "embedding")
        index_path = os.path.join(embedding_dir, "faiss_index.index")
        metadata_path = os.path.join(embedding_dir, "metadata.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            print(f"\nEmbedding files not found in {docs_dir}, generating now...")
            create_embedding.process_files(docs_dir)
            return True
        return True
    except Exception as e:
        print(f"Error ensuring embeddings: {str(e)}")
        return False

def query_knowledge(query: str, k: int = 3, docs_dir: str = None) -> List[Dict]:
    """
    查詢知識庫，返回最相關的內容
    
    Args:
        query (str): 查詢文本
        k (int): 返回結果數量
        docs_dir (str): 文檔目錄路徑
    """
    try:
        if not docs_dir:
            raise ValueError("docs_dir is required")
            
        # 檢查並確保有 embedding 文件
        if not ensure_embeddings_exist(docs_dir):
            print(f"Failed to ensure embeddings in {docs_dir}")
            return []

        # 使用 create_embedding 查詢
        texts, sources, scores = create_embedding.query_index(
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
        print(f"Knowledge base query error: {str(e)}")
        return []