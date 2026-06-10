"""Pluggable embedding providers for the Resume RAG system."""
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 32):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.batch_size = batch_size
        logger.info(f"OpenAI embedding provider initialized: {model}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            for attempt in range(3):
                try:
                    response = self.client.embeddings.create(input=batch, model=self.model)
                    embeddings.extend(item.embedding for item in response.data)
                    time.sleep(0.05)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"Embedding attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)
        return embeddings


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        self.batch_size = batch_size
        logger.info(f"HuggingFace embedding provider initialized: {model}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.extend(batch_embeddings.tolist())
        return embeddings


class CohereEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "embed-english-v3.0", batch_size: int = 32):
        import cohere
        self.client = cohere.Client()
        self.model = model
        self.batch_size = batch_size
        logger.info(f"Cohere embedding provider initialized: {model}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            for attempt in range(3):
                try:
                    response = self.client.embed(
                        texts=batch, model=self.model, input_type="search_document"
                    )
                    embeddings.extend(response.embeddings)
                    time.sleep(0.05)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"Embedding attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)
        return embeddings


def create_embedding_provider(config: Dict[str, Any]) -> EmbeddingProvider:
    provider = config.get("provider", "huggingface")
    model_config = config.get("model", {})
    batch_size = config.get("batch_size", 32)

    if provider == "openai":
        return OpenAIEmbeddingProvider(
            model=model_config.get("openai", "text-embedding-3-small"),
            batch_size=batch_size,
        )
    elif provider == "cohere":
        return CohereEmbeddingProvider(
            model=model_config.get("cohere", "embed-english-v3.0"),
            batch_size=batch_size,
        )
    else:
        return HuggingFaceEmbeddingProvider(
            model=model_config.get("huggingface", "sentence-transformers/all-MiniLM-L6-v2"),
            batch_size=batch_size,
        )
