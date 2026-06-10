#!/usr/bin/env python3
"""Resume RAG System — load, chunk, embed, and index resumes into ChromaDB."""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml
from tqdm import tqdm

from chunker import ResumeChunker
from embeddings import create_embedding_provider
from metadata_extractor import MetadataExtractor
from vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".pdf", ".docx", ".txt"}


class ResumeLoader:
    def load(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        loaders = {
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".txt": self._load_txt,
        }
        if ext not in loaders:
            raise ValueError(f"Unsupported format: {ext}")
        return loaders[ext](file_path)

    def _load_pdf(self, file_path: str) -> str:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _load_docx(self, file_path: str) -> str:
        from docx import Document
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)

    def _load_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def load_directory(
        self,
        directory: str,
        supported_formats: List[str] = None,
    ) -> List[Dict[str, str]]:
        formats = set(supported_formats or SUPPORTED_FORMATS)
        resumes = []
        for path in Path(directory).iterdir():
            if path.suffix.lower() not in formats:
                continue
            try:
                text = self.load(str(path))
                resumes.append({"file_path": str(path), "text": text})
                logger.info(f"Loaded: {path.name}")
            except Exception as e:
                logger.error(f"Failed to load {path.name}: {e}")
        return resumes


class ResumeProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loader = ResumeLoader()
        self.chunker = ResumeChunker(
            chunk_size=config.get("indexing", {}).get("chunk_size", 512),
            chunk_overlap=config.get("indexing", {}).get("chunk_overlap", 50),
        )
        llm_cfg = config.get("llm") if config.get("llm", {}).get("provider") else None
        self.extractor = MetadataExtractor(llm_config=llm_cfg)
        self.embedding_provider = create_embedding_provider(config.get("embedding", {}))
        self.vector_store = VectorStore(
            collection_name=config.get("vector_store", {}).get("collection_name", "resumes"),
            persist_directory=config.get("vector_store", {}).get("persist_directory", "./chroma_db"),
        )

    def index_directory(self, directory: str) -> Dict[str, int]:
        supported_formats = self.config.get("indexing", {}).get(
            "supported_formats", list(SUPPORTED_FORMATS)
        )
        resumes = self.loader.load_directory(directory, supported_formats)

        if not resumes:
            logger.warning(f"No resumes found in '{directory}'")
            return {"resumes": 0, "chunks": 0, "stored": 0}

        logger.info(f"Loaded {len(resumes)} resumes")

        total_chunks = 0
        total_stored = 0
        total_skipped = 0

        for resume in tqdm(resumes, desc="Indexing resumes"):
            result = self._index_resume(resume["file_path"], resume["text"])
            total_chunks += result["chunks"]
            total_stored += result["stored"]
            total_skipped += result["skipped"]

        logger.info(f"INFO: Loaded {len(resumes)} resumes")
        logger.info(f"INFO: Generated {total_chunks} chunks")
        logger.info(f"INFO: Stored {total_stored} new vectors ({total_skipped} already indexed)")

        return {
            "resumes": len(resumes),
            "chunks": total_chunks,
            "stored": total_stored,
            "skipped": total_skipped,
        }

    def index_file(self, file_path: str) -> Dict[str, int]:
        try:
            text = self.loader.load(file_path)
            return self._index_resume(file_path, text)
        except Exception as e:
            logger.error(f"Failed to index '{file_path}': {e}")
            return {"chunks": 0, "stored": 0, "skipped": 0}

    def _index_resume(self, file_path: str, text: str) -> Dict[str, int]:
        try:
            metadata = self.extractor.extract(text, file_path)
            candidate_name = metadata.get("name") or Path(file_path).stem

            chunks = self.chunker.chunk(text, candidate_name, file_path)
            if not chunks:
                logger.warning(f"No chunks generated for {file_path}")
                return {"chunks": 0, "stored": 0, "skipped": 0}

            documents = [c["content"] for c in chunks]
            embeddings = self.embedding_provider.embed(documents)

            metadatas = []
            for chunk in chunks:
                meta = {**metadata, **chunk}
                meta.pop("content", None)
                metadatas.append(meta)

            counts = self.vector_store.add(documents, embeddings, metadatas)
            logger.debug(
                f"{candidate_name}: {len(chunks)} chunks, "
                f"{counts['stored']} stored, {counts['skipped']} already indexed"
            )
            return {"chunks": len(chunks), **counts}

        except Exception as e:
            logger.error(f"Failed to index '{file_path}': {e}")
            return {"chunks": 0, "stored": 0, "skipped": 0}


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    logger.warning(f"Config file not found: {config_path}. Using defaults.")
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume RAG System — index resumes into a vector database"
    )
    parser.add_argument(
        "--index",
        type=str,
        help="Directory or single file to index",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="List all indexed candidates",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    processor = ResumeProcessor(config)

    if args.list_candidates:
        candidates = processor.vector_store.get_all_candidates()
        if candidates:
            print(f"Indexed candidates ({len(candidates)}):")
            for name in candidates:
                print(f"  • {name}")
        else:
            print("No candidates indexed yet.")
        return

    if args.index:
        target = Path(args.index)
        if target.is_dir():
            results = processor.index_directory(str(target))
        elif target.is_file():
            results = processor.index_file(str(target))
            results["resumes"] = 1
        else:
            logger.error(f"Path not found: {args.index}")
            sys.exit(1)

        skipped = results.get("skipped", 0)
        skip_note = f", {skipped} already indexed" if skipped else ""
        print(
            f"\nIndexing complete — "
            f"{results.get('resumes', 1)} resume(s), "
            f"{results['chunks']} chunks, "
            f"{results['stored']} new vectors stored{skip_note}."
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
