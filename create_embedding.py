import os
from typing import List, Optional
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

# Set OpenAI API Key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Constants
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_SIZE = 3072

# Directory paths
DOCS_DIR = "docs"
EMBEDDING_DIR = "embedding"
INDEX_FILE = os.path.join(EMBEDDING_DIR, "faiss_index.index")
METADATA_FILE = os.path.join(EMBEDDING_DIR, "metadata.pkl")

# Initialize FAISS index (no training required)
index = faiss.IndexFlatIP(EMBEDDING_SIZE)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def preprocess_text(text: str) -> str:
    """
    Preprocess the text by removing the References section and in-text citations.
    """
    # Define sections to remove
    sections_to_remove = ['References', 'Bibliography', 'Citations', 'Works Cited', 'Acknowledgments', 'Appendix', 'Appendices']
    
    # Create a regex pattern to split at these sections
    pattern = r'\n\s*(' + '|'.join(sections_to_remove) + r')\s*\n'
    text = re.split(pattern, text, flags=re.IGNORECASE)[0]

    # Remove various in-text citations
    # Parenthetical citations (e.g., (Smith et al., 2020))
    text = re.sub(r'\([^)]*\b(?:et al\.,?\s*\d{4}|\d{4})[^)]*\)', '', text)
    
    # Bracketed citations (e.g., [1], [2-4])
    text = re.sub(r'\[\d+(?:-\d+)?\]', '', text)
    
    # Superscript citations (e.g., ^1)
    text = re.sub(r'\^\d+', '', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


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


def read_file(filepath: str) -> Optional[str]:
    """Read the content of a file based on its extension."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.pdf':
            return read_pdf(filepath)
        elif ext == '.txt':
            return read_txt(filepath)
        elif ext == '.docx':
            return read_docx(filepath)
        else:
            print(f"Unsupported file format: {ext}")
            return None
    except Exception as e:
        print(f"Error reading file {filepath}: {str(e)}")
        return None


def read_pdf(filepath: str) -> str:
    """Read content from a PDF file."""
    doc = fitz.open(filepath)
    text = [page.get_text() for page in doc]
    full_text = "\n".join(text)
    return full_text


def read_txt(filepath: str) -> str:
    """Read content from a TXT file."""
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()


def read_docx(filepath: str) -> str:
    """Read content from a DOCX file."""
    doc = Document(filepath)
    return "\n".join([para.text for para in doc.paragraphs])


def process_files(directory: str):
    """Process all files in a directory and add them to the FAISS index."""
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    texts = []
    sources = []
    metadata = []
    all_embeddings = []

    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    print(f"Found {len(files)} files in {directory}.")

    for filepath in files:
        print(f"Processing file: {filepath}")
        content = read_file(filepath)
        if content:
            # Preprocess the text to remove references and citations
            cleaned_content = preprocess_text(content)
            print(f"Original content length: {len(content)} characters.")
            print(f"Cleaned content length: {len(cleaned_content)} characters.")

            chunks = text_splitter.split_text(cleaned_content)
            print(f"Split file into {len(chunks)} chunks.")

            # Embed text chunks
            embeddings = [embed_text(chunk) for chunk in chunks]

            # Optional: Filter out zero embeddings if any
            valid_embeddings = [emb for emb in embeddings if np.linalg.norm(emb) > 0]
            if not valid_embeddings:
                print(f"No valid embeddings for file {filepath}. Skipping.")
                continue

            all_embeddings.extend(valid_embeddings)
            texts.extend(chunks[:len(valid_embeddings)])
            sources.extend([filepath] * len(valid_embeddings))
            metadata.extend([{"source": filepath}] * len(valid_embeddings))
        else:
            print(f"Failed to read content from {filepath}.")

    # Convert all embeddings to a single NumPy array
    if all_embeddings:
        all_embeddings = np.array(all_embeddings, dtype=np.float32)

        # Check embedding dimensions
        if all_embeddings.shape[1] != EMBEDDING_SIZE:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {EMBEDDING_SIZE}, got {all_embeddings.shape[1]}."
            )

        # Add embeddings to the index
        index.add(all_embeddings)
        print(f"Added {all_embeddings.shape[0]} embeddings to the FAISS index.")
    else:
        print("No embeddings were generated. The index will remain empty.")

    save_index(texts, sources, metadata)


def save_index(texts: List[str], sources: List[str], metadata: List[dict]):
    """Save the FAISS index and metadata to the embedding directory."""
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    try:
        faiss.write_index(index, INDEX_FILE)
        with open(METADATA_FILE, 'wb') as f:
            pickle.dump({'texts': texts, 'sources': sources, 'metadata': metadata}, f)
        print(f"Index and metadata saved to {INDEX_FILE} and {METADATA_FILE}.")
    except Exception as e:
        print(f"Error saving index: {str(e)}")


def load_index():
    """Load the FAISS index and metadata from the embedding directory."""
    try:
        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            global index
            index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, 'rb') as f:
                data = pickle.load(f)
                texts = data['texts']
                sources = data['sources']
                metadata = data['metadata']
            print(f"Loaded index and metadata from {INDEX_FILE} and {METADATA_FILE}.")
            return texts, sources, metadata
        else:
            print("No existing index or metadata found.")
            return [], [], []
    except Exception as e:
        print(f"Error loading index: {str(e)}")
        return [], [], []


if __name__ == "__main__":
    process_files(DOCS_DIR)



