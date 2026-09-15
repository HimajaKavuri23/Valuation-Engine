# RAG Retriever Module

import numpy as np
import re
import requests
from sentence_transformers import SentenceTransformer
from sec_retriever import get_company_cik, get_latest_10k_accession

# Load model once at module level
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_full_10k_text(ticker: str) -> tuple[str, str]:
    """Fetch full 10-K text and return (text, source_url)."""
    try:
        cik = get_company_cik(ticker)
        if not cik:
            return None, None

        accession, primary_doc = get_latest_10k_accession(cik)
        if not accession or not primary_doc:
            return None, None

        cik_int = str(int(cik))
        accession_clean = accession.replace("-", "")
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/{primary_doc}"

        headers = {"User-Agent": "Himaja Kavuri hkavuri@usc.edu"}
        response = requests.get(doc_url, headers=headers)
        text = response.text

        # Clean HTML
        clean = re.sub(r'<[^>]+>', ' ', text)
        clean = re.sub(r'&#\d+;', ' ', clean)
        clean = re.sub(r'&[a-z]+;', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()

        return clean, doc_url

    except Exception as e:
        return None, None


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Split text into overlapping chunks with position tracking."""
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        chunks.append({
            'text': chunk_text,
            'start_word': i,
            'end_word': min(i + chunk_size, len(words))
        })
        i += chunk_size - overlap

    return chunks


def find_item1a_chunks(chunks: list[dict]) -> list[dict]:
    """Filter chunks to only those in the Item 1A Risk Factors section."""
    in_section = False
    relevant = []

    for chunk in chunks:
        text_lower = chunk['text'].lower()

        # Start of Item 1A
        if not in_section and ('item 1a' in text_lower and 'risk' in text_lower):
            in_section = True

        # End of section
        if in_section and ('item 1b' in text_lower or
                           ('item 2' in text_lower and 'item 1' not in text_lower)):
            break

        if in_section:
            relevant.append(chunk)

    return relevant if relevant else chunks[:50]  # Fallback to first 50 chunks


def retrieve_relevant_chunks(query: str, chunks: list[dict],
                              embeddings: np.ndarray, top_k: int = 5) -> list[dict]:
    """Find most relevant chunks for a query using cosine similarity."""
    query_embedding = model.encode([query])[0]

    # Cosine similarity
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
    )

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'text': chunks[idx]['text'],
            'similarity': float(similarities[idx]),
            'chunk_index': int(idx)
        })

    return results


def get_rag_context(ticker: str, query: str = None, top_k: int = 5) -> dict:
    """
    Main RAG function. Returns relevant chunks with metadata.
    
    Returns:
        {
            'context': str,  # Combined relevant text
            'chunks': list,  # Individual chunks with scores
            'source_url': str,  # SEC filing URL
            'success': bool
        }
    """
    try:
        # Fetch and chunk document
        full_text, source_url = get_full_10k_text(ticker)
        if not full_text:
            return {'context': '', 'chunks': [], 'source_url': '', 'success': False}

        # Chunk the text
        all_chunks = chunk_text(full_text)

        # Filter to Item 1A section
        relevant_chunks = find_item1a_chunks(all_chunks)

        if not relevant_chunks:
            return {'context': '', 'chunks': [], 'source_url': source_url, 'success': False}

        # Embed all chunks
        chunk_texts = [c['text'] for c in relevant_chunks]
        embeddings = model.encode(chunk_texts, show_progress_bar=False)

        # Use query or default to risk factors query
        if not query:
            query = "key risk factors material risks business operations financial condition"

        # Retrieve top chunks
        top_chunks = retrieve_relevant_chunks(query, relevant_chunks, embeddings, top_k)

        # Build context with chunk numbers for citation
        context_parts = []
        for i, chunk in enumerate(top_chunks):
            context_parts.append(
                f"[Passage {i+1} | Relevance: {chunk['similarity']:.2f}]\n{chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        return {
            'context': context,
            'chunks': top_chunks,
            'source_url': source_url,
            'success': True
        }

    except Exception as e:
        return {
            'context': f'RAG retrieval failed: {str(e)}',
            'chunks': [],
            'source_url': '',
            'success': False
        }