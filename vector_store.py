"""ChromaDB-backed vector store for resume embeddings."""
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        collection_name: str = "resumes",
        persist_directory: str = "./chroma_db",
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"VectorStore ready — collection '{collection_name}' "
            f"at '{persist_directory}' ({self.collection.count()} vectors)"
        )

    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> int:
        if not documents:
            return 0

        if ids is None:
            ids = [
                self._generate_id(doc, meta)
                for doc, meta in zip(documents, metadatas)
            ]

        existing = set(self.collection.get(ids=ids)["ids"])

        new_docs, new_embs, new_metas, new_ids = [], [], [], []
        for doc, emb, meta, id_ in zip(documents, embeddings, metadatas, ids):
            if id_ not in existing:
                new_docs.append(doc)
                new_embs.append(emb)
                new_metas.append(self._serialize_meta(meta))
                new_ids.append(id_)

        if new_docs:
            self.collection.add(
                documents=new_docs,
                embeddings=new_embs,
                metadatas=new_metas,
                ids=new_ids,
            )
            skipped = len(documents) - len(new_docs)
            logger.info(
                f"Stored {len(new_docs)} vectors "
                f"({'skipped ' + str(skipped) + ' duplicates' if skipped else 'no duplicates'})"
            )

        return len(new_docs)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        count = self.collection.count()
        if count == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        n_results = min(top_k, count)
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        return self.collection.query(**kwargs)

    def delete_by_source(self, source_file: str) -> None:
        results = self.collection.get(where={"source_file": source_file})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} vectors for '{source_file}'")

    def get_all_candidates(self) -> List[str]:
        results = self.collection.get(include=["metadatas"])
        return sorted(
            {m["candidate_name"] for m in results["metadatas"] if "candidate_name" in m}
        )

    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def _generate_id(document: str, metadata: Dict[str, Any]) -> str:
        key = (
            f"{metadata.get('source_file', '')}"
            f"|{metadata.get('section', '')}"
            f"|{metadata.get('chunk_index', 0)}"
            f"|{document[:80]}"
        )
        return hashlib.md5(key.encode()).hexdigest()

    @staticmethod
    def _serialize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: json.dumps(v) if isinstance(v, (list, dict)) else v
            for k, v in meta.items()
        }
