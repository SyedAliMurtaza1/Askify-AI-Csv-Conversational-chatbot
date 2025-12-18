import faiss
import numpy as np
import pickle
from typing import List, Dict, Any
import logging
import re

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.chunks = []
        self.metadata = []
    
    def add_chunks(self, chunks: List[Dict[str, Any]], embedder):
        """Add chunks to vector store"""
        contents = [chunk["content"] for chunk in chunks]
        embeddings = embedder.embed_texts(contents)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.chunks.extend(contents)
        self.metadata.extend([chunk["metadata"] for chunk in chunks])
        
        logger.info(f"Added {len(chunks)} chunks to vector store")
    
    def search(self, query: str, embedder, top_k: int = 3, similarity_threshold: float = 0.0):
        """Search for similar chunks - NO HARDCODING"""
        
        query_clean = query.strip().lower()
        
        # 1. Try direct text matching first
        text_matches = []
        for idx, chunk in enumerate(self.chunks):
            chunk_lower = chunk.lower()
            
            # Check if query words appear in chunk
            query_words = set(re.findall(r'\w+', query_clean))
            chunk_words = set(re.findall(r'\w+', chunk_lower))
            
            if query_words & chunk_words:  # Any overlap
                overlap = len(query_words & chunk_words) / len(query_words) if query_words else 0
                text_matches.append({
                    "content": self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "similarity": min(1.0, overlap + 0.3),  # Boost text matches
                    "rank": len(text_matches) + 1
                })
        
        if text_matches:
            text_matches.sort(key=lambda x: x["similarity"], reverse=True)
            return text_matches[:top_k]
        
        # 2. Fall back to embedding search
        try:
            query_embedding = embedder.embed_text(query)
            query_embedding = np.array([query_embedding])
            faiss.normalize_L2(query_embedding)
            
            similarities, indices = self.index.search(query_embedding, top_k * 2)
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.chunks):
                    results.append({
                        "content": self.chunks[idx],
                        "metadata": self.metadata[idx],
                        "similarity": float(similarity),
                        "rank": i + 1
                    })
            
            return results[:top_k]
        except:
            return []
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        data = {
            "chunks": self.chunks,
            "metadata": self.metadata,
            "index": faiss.serialize_index(self.index)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.chunks = data["chunks"]
        self.metadata = data["metadata"]
        self.index = faiss.deserialize_index(data["index"])