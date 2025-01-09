import os
from typing import List, Optional, Tuple, Dict
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
import fitz  # PyMuPDF
import pickle
import dotenv
from openai import AsyncOpenAI
import re
from fastapi import HTTPException

dotenv.load_dotenv()

# Set OpenAI API Key
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Constants
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_SIZE = 3072

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def preprocess_text(text: str) -> str:
    """Preprocess the text by removing references and citations."""
    sections_to_remove = ['References', 'Bibliography', 'Citations', 'Works Cited', 
                         'Acknowledgments', 'Appendix', 'Appendices']
    
    pattern = r'\n\s*(' + '|'.join(sections_to_remove) + r')\s*\n'
    text = re.split(pattern, text, flags=re.IGNORECASE)[0]
    text = re.sub(r'\([^)]*\b(?:et al\.,?\s*\d{4}|\d{4})[^)]*\)', '', text)
    text = re.sub(r'\[\d+(?:-\d+)?\]', '', text)
    text = re.sub(r'\^\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

async def embed_text(text: str) -> Optional[np.ndarray]:
    """Generate an embedding for a given text asynchronously."""
    if not text.strip():
        print("Warning: Empty text provided for embedding")
        return None
        
    try:
        response = await client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else None
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

def get_embedding_paths(docs_dir: str) -> Tuple[str, str, str]:
    """Get paths for embedding related files."""
    embedding_dir = os.path.join(docs_dir, "embedding")
    index_path = os.path.join(embedding_dir, "faiss_index.index")
    metadata_path = os.path.join(embedding_dir, "metadata.pkl")
    return embedding_dir, index_path, metadata_path

def ensure_embedding_dir(docs_dir: str) -> str:
    """Ensure embedding directory exists and return its path."""
    embedding_dir = os.path.join(docs_dir, "embedding")
    os.makedirs(embedding_dir, exist_ok=True)
    return embedding_dir

async def read_file(filepath: str) -> Optional[str]:
    """Read file content based on its extension asynchronously."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
        
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.pdf':
            doc = fitz.open(filepath)
            text = [page.get_text() for page in doc]
            return "\n".join(text)
        elif ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.docx':
            doc = Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            print(f"Unsupported file format: {ext}")
            return None
    except Exception as e:
        print(f"Error reading file {filepath}: {str(e)}")
        return None

async def process_files_async(docs_dir: str):
    """Process files and create embeddings asynchronously."""
    if not os.path.exists(docs_dir):
        raise HTTPException(status_code=404, detail=f"Directory not found: {docs_dir}")

    files = [f for f in os.listdir(docs_dir) 
             if os.path.isfile(os.path.join(docs_dir, f))]
             
    if not files:
        print(f"No files found in {docs_dir}")
        return

    print(f"Processing {len(files)} files in {docs_dir}")
    
    # Initialize storage
    index = faiss.IndexFlatIP(EMBEDDING_SIZE)
    texts = []
    sources = []
    metadata = []
    embeddings_buffer = []

    for filename in files:
        filepath = os.path.join(docs_dir, filename)
        print(f"Processing: {filename}")
        
        content = await read_file(filepath)
        if not content:
            continue
            
        # Process content
        cleaned_content = preprocess_text(content)
        chunks = text_splitter.split_text(cleaned_content)
        print(f"Generated {len(chunks)} chunks from {filename}")

        # Generate embeddings
        chunk_embeddings = []
        for chunk in chunks:
            embedding = await embed_text(chunk)
            if embedding is not None:
                chunk_embeddings.append(embedding)
                texts.append(chunk)
                sources.append(filepath)
                metadata.append({"source": filepath})

        if chunk_embeddings:
            embeddings_buffer.extend(chunk_embeddings)
            print(f"Added {len(chunk_embeddings)} embeddings from {filename}")
        else:
            print(f"No valid embeddings generated for {filename}")

    # Add all embeddings to index
    if embeddings_buffer:
        embeddings_array = np.array(embeddings_buffer, dtype=np.float32)
        index.add(embeddings_array)
        print(f"Total embeddings added to index: {len(embeddings_buffer)}")
        
        # Save files
        embedding_dir = ensure_embedding_dir(docs_dir)
        index_path = os.path.join(embedding_dir, "faiss_index.index")
        metadata_path = os.path.join(embedding_dir, "metadata.pkl")
        
        try:
            faiss.write_index(index, index_path)
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'texts': texts,
                    'sources': sources,
                    'metadata': metadata
                }, f)
            print(f"Saved index and metadata to {embedding_dir}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving index files: {str(e)}")
    else:
        print("No valid embeddings generated from any files")

async def load_index(docs_dir: str) -> Tuple[faiss.Index, List[str], List[str]]:
    """Load index and metadata from the specified directory asynchronously."""
    embedding_dir, index_path, metadata_path = get_embedding_paths(docs_dir)
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail=f"Index files not found in {embedding_dir}")

    try:
        # Load metadata
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            texts = data['texts']
            sources = data['sources']
        
        # Load index
        index = faiss.read_index(index_path)
        return index, texts, sources
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading index: {str(e)}")

async def query_index_async(query: str, k: int = 3, docs_dir: str = None) -> Tuple[List[str], List[str], List[float]]:
    """Query the index for relevant documents asynchronously."""
    if not docs_dir:
        raise HTTPException(status_code=400, detail="docs_dir parameter is required")

    try:
        # Load index
        index, texts, sources = await load_index(docs_dir)
        
        # Generate query embedding
        query_embedding = await embed_text(query)
        if query_embedding is None:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        # Search
        D, I = index.search(query_embedding.reshape(1, -1), k)
        
        # Prepare results
        results = [(texts[i], sources[i], float(d)) 
                  for d, i in zip(D[0], I[0]) 
                  if i < len(texts)]
        
        if not results:
            return [], [], []
            
        return zip(*results)  # Unzip into separate lists
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during query: {str(e)}")