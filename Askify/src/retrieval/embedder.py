from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Embedding model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed list of texts"""
        return self.model.encode(texts, convert_to_numpy=True)
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed single text"""
        return self.model.encode([text], convert_to_numpy=True)[0]