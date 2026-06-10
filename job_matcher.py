#!/usr/bin/env python3
"""Job Matching Engine — semantic + hybrid search with explainable scoring."""
import argparse
import ast
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from embeddings import create_embedding_provider
from metadata_extractor import SKILL_KEYWORDS, SKILL_SYNONYMS
from vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class JobMatcher:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_provider = create_embedding_provider(config.get("embedding", {}))
        self.vector_store = VectorStore(
            collection_name=config.get("vector_store", {}).get("collection_name", "resumes"),
            persist_directory=config.get("vector_store", {}).get("persist_directory", "./chroma_db"),
        )

        search_cfg = config.get("search", {})
        self.top_k: int = search_cfg.get("top_k", 10)
        self.semantic_weight: float = search_cfg.get("semantic_weight", 0.7)
        self.keyword_weight: float = search_cfg.get("keyword_weight", 0.3)

        score_cfg = config.get("scoring", {})
        self.w_semantic: float = score_cfg.get("semantic_weight", 0.7)
        self.w_skill: float = score_cfg.get("skill_weight", 0.2)
        self.w_experience: float = score_cfg.get("experience_weight", 0.1)

    def match(
        self,
        job_description: str,
        min_experience: int = 0,
        required_skills: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self.vector_store.count() == 0:
            logger.error("Vector store is empty. Run resume_rag.py --index first.")
            return {
                "job_description": job_description,
                "top_matches": [],
                "error": "No resumes indexed.",
            }

        jd_embedding = self.embedding_provider.embed_single(job_description)
        jd_skills = self._extract_skills(job_description)
        jd_experience = self._extract_experience_requirement(job_description)

        search_results = self.vector_store.search(
            jd_embedding, top_k=self.top_k * 5
        )

        candidate_chunks: Dict[str, List[Dict]] = defaultdict(list)
        for doc_id, doc, meta, dist in zip(
            search_results["ids"][0],
            search_results["documents"][0],
            search_results["metadatas"][0],
            search_results["distances"][0],
        ):
            name = meta.get("candidate_name", "Unknown")
            semantic_sim = max(0.0, 1.0 - float(dist))
            keyword_sim = self._keyword_overlap(job_description, doc)
            hybrid_score = (
                self.semantic_weight * semantic_sim
                + self.keyword_weight * keyword_sim
            )
            candidate_chunks[name].append(
                {
                    "id": doc_id,
                    "content": doc,
                    "metadata": meta,
                    "semantic_score": semantic_sim,
                    "hybrid_score": hybrid_score,
                }
            )

        candidates = []
        for name, chunks in candidate_chunks.items():
            candidate = self._build_candidate(name, chunks, jd_skills, jd_experience)

            # Must-have filters
            if min_experience > 0 and candidate["experience_years"] < min_experience:
                logger.debug(f"Filtered {name}: {candidate['experience_years']} < {min_experience} yrs")
                continue
            if required_skills:
                cand_lower = {s.lower() for s in candidate["all_skills"]}
                missing = [s for s in required_skills if s.lower() not in cand_lower]
                if missing:
                    logger.debug(f"Filtered {name}: missing required skills {missing}")
                    continue

            candidates.append(candidate)

        candidates.sort(key=lambda x: x["match_score"], reverse=True)
        top = candidates[: self.top_k]

        for match in top:
            match["reasoning"] = self._generate_reasoning(match, jd_skills, jd_experience)

        return {
            "job_description": job_description,
            "required_skills": jd_skills,
            "min_experience_required": jd_experience,
            "total_candidates_evaluated": len(candidate_chunks),
            "top_matches": [self._format_match(m) for m in top],
        }

    # ------------------------------------------------------------------ #
    #  Candidate aggregation                                               #
    # ------------------------------------------------------------------ #

    def _build_candidate(
        self,
        name: str,
        chunks: List[Dict],
        jd_skills: List[str],
        jd_experience: int,
    ) -> Dict[str, Any]:
        hybrid_scores = [c["hybrid_score"] for c in chunks]
        sem_scores = [c["semantic_score"] for c in chunks]

        # Weighted mix of best and average scores
        max_hybrid = max(hybrid_scores)
        avg_hybrid = sum(hybrid_scores) / len(hybrid_scores)
        agg_semantic = 0.6 * max_hybrid + 0.4 * avg_hybrid

        first_meta = chunks[0]["metadata"]

        candidate_skills = self._parse_skills_field(first_meta.get("skills", "[]"))
        matched_skills = self._match_skills(jd_skills, candidate_skills)
        skill_ratio = len(matched_skills) / max(len(jd_skills), 1)

        experience_years = int(first_meta.get("experience_years", 0))
        exp_score = (
            min(experience_years / jd_experience, 1.0) if jd_experience > 0 else 0.5
        )

        match_score = (
            self.w_semantic * agg_semantic
            + self.w_skill * skill_ratio
            + self.w_experience * exp_score
        ) * 100

        best_chunks = sorted(chunks, key=lambda x: x["hybrid_score"], reverse=True)[:3]
        excerpts = [
            c["content"][:250].rstrip() + ("…" if len(c["content"]) > 250 else "")
            for c in best_chunks
            if len(c["content"].strip()) > 30
        ]

        return {
            "candidate_name": name,
            "resume_path": first_meta.get("source_file", ""),
            "match_score": round(match_score, 1),
            "semantic_score": round(agg_semantic * 100, 1),
            "skill_score": round(skill_ratio * 100, 1),
            "experience_score": round(exp_score * 100, 1),
            "matched_skills": matched_skills,
            "all_skills": candidate_skills,
            "experience_years": experience_years,
            "education": first_meta.get("education", "Not specified"),
            "email": first_meta.get("email"),
            "relevant_excerpts": excerpts,
            "reasoning": "",
        }

    # ------------------------------------------------------------------ #
    #  Scoring helpers                                                     #
    # ------------------------------------------------------------------ #

    def _extract_skills(self, text: str) -> List[str]:
        found: set = set()
        for skill in SKILL_KEYWORDS:
            if re.search(r"\b" + re.escape(skill) + r"\b", text, re.IGNORECASE):
                found.add(skill)
        for abbr, full in SKILL_SYNONYMS.items():
            if re.search(r"\b" + re.escape(abbr) + r"\b", text):
                found.add(full)
        return sorted(found)

    def _extract_experience_requirement(self, text: str) -> int:
        matches = re.findall(
            r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)", text, re.I
        )
        return max((int(m) for m in matches), default=0)

    def _keyword_overlap(self, jd: str, chunk: str) -> float:
        jd_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", jd.lower()))
        chunk_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", chunk.lower()))
        if not jd_tokens:
            return 0.0
        return len(jd_tokens & chunk_tokens) / len(jd_tokens)

    def _match_skills(self, jd_skills: List[str], cand_skills: List[str]) -> List[str]:
        cand_lower = {s.lower() for s in cand_skills}
        return [s for s in jd_skills if s.lower() in cand_lower]

    @staticmethod
    def _parse_skills_field(raw: Any) -> List[str]:
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return []

    def _generate_reasoning(
        self, candidate: Dict, jd_skills: List[str], jd_experience: int
    ) -> str:
        parts = []

        if candidate["matched_skills"]:
            listed = ", ".join(candidate["matched_skills"][:4])
            parts.append(f"Strong skill alignment with {listed}")

        yrs = candidate["experience_years"]
        if yrs > 0:
            suffix = "meeting" if jd_experience == 0 or yrs >= jd_experience else "below"
            parts.append(f"{yrs} years of experience ({suffix} requirement)")

        edu = candidate["education"]
        if edu and edu != "Not specified":
            parts.append(f"{edu} degree")

        missing = [s for s in jd_skills if s not in candidate["matched_skills"]]
        if missing:
            parts.append(f"Skill gaps: {', '.join(missing[:3])}")

        if not parts:
            return (
                f"Candidate matches the job with a {candidate['match_score']}% overall score "
                "based on semantic similarity."
            )

        return ". ".join(parts) + "."

    @staticmethod
    def _format_match(m: Dict) -> Dict[str, Any]:
        return {
            "candidate_name": m["candidate_name"],
            "resume_path": m["resume_path"],
            "match_score": m["match_score"],
            "matched_skills": m["matched_skills"],
            "relevant_excerpts": m["relevant_excerpts"],
            "reasoning": m["reasoning"],
            "details": {
                "semantic_score": m["semantic_score"],
                "skill_score": m["skill_score"],
                "experience_score": m["experience_score"],
                "experience_years": m["experience_years"],
                "education": m["education"],
                "email": m["email"],
            },
        }


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    logger.warning(f"Config '{config_path}' not found. Using defaults.")
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Job Matching Engine — rank resumes against a job description"
    )
    parser.add_argument("jd_file", nargs="?", help="Path to job description .txt file")
    parser.add_argument("--jd", type=str, help="Job description as a string (alternative to file)")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--min-experience", type=int, default=0, metavar="YEARS")
    parser.add_argument("--required-skills", type=str, nargs="+", metavar="SKILL")
    parser.add_argument("--output", type=str, help="Write JSON results to this file")
    args = parser.parse_args()

    config = load_config(args.config)

    job_description: Optional[str] = None
    if args.jd_file:
        with open(args.jd_file) as f:
            job_description = f.read()
    elif args.jd:
        job_description = args.jd
    else:
        parser.print_help()
        sys.exit(1)

    matcher = JobMatcher(config)
    results = matcher.match(
        job_description,
        min_experience=args.min_experience,
        required_skills=args.required_skills,
    )

    output = json.dumps(results, indent=2, ensure_ascii=False)
    print(output)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        logger.info(f"Results saved to '{args.output}'")


if __name__ == "__main__":
    main()
