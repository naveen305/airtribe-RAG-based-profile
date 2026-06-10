"""Section-aware resume chunker."""
import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

SECTION_HEADERS = {
    "summary", "professional summary", "career summary", "objective",
    "career objective", "professional objective", "profile",
    "experience", "work experience", "professional experience",
    "employment history", "employment", "work history",
    "education", "academic background", "academic qualifications",
    "qualifications",
    "skills", "technical skills", "core competencies", "competencies",
    "expertise", "key skills", "technologies",
    "certifications", "certificates", "licenses", "accreditations",
    "projects", "personal projects", "key projects", "notable projects",
    "achievements", "accomplishments", "awards", "honors", "recognition",
    "publications", "research", "papers",
    "languages", "interests", "hobbies", "activities",
    "references", "volunteer", "volunteering",
}


class ResumeChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, candidate_name: str, source_file: str) -> List[Dict[str, Any]]:
        sections = self._split_into_sections(text)
        chunks: List[Dict[str, Any]] = []

        for section_name, section_content in sections:
            if not section_content.strip():
                continue
            chunks.extend(
                self._chunk_section(section_content, section_name, candidate_name, source_file)
            )

        if not chunks:
            chunks = self._chunk_section(text, "General", candidate_name, source_file)

        return chunks

    def _split_into_sections(self, text: str) -> List[Tuple[str, str]]:
        sections: List[Tuple[str, str]] = []
        current_section = "Header"
        current_lines: List[str] = []

        for line in text.split("\n"):
            stripped = line.strip()
            if self._is_section_header(stripped):
                if current_lines:
                    sections.append((current_section, "\n".join(current_lines)))
                current_section = stripped.rstrip(":").title()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_section, "\n".join(current_lines)))

        return sections

    def _is_section_header(self, line: str) -> bool:
        if not line or len(line) > 60:
            return False
        cleaned = line.rstrip(":").lower().strip()
        return cleaned in SECTION_HEADERS

    def _chunk_section(
        self,
        content: str,
        section_name: str,
        candidate_name: str,
        source_file: str,
    ) -> List[Dict[str, Any]]:
        words = content.split()

        if not words:
            return []

        if len(words) <= self.chunk_size:
            return [
                {
                    "section": section_name,
                    "content": content.strip(),
                    "candidate_name": candidate_name,
                    "source_file": source_file,
                    "chunk_index": 0,
                }
            ]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                {
                    "section": section_name,
                    "content": chunk_text,
                    "candidate_name": candidate_name,
                    "source_file": source_file,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1
            if end == len(words):
                break
            start = end - self.chunk_overlap

        return chunks
