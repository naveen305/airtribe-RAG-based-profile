"""Metadata extraction from resume text using regex, NLP, and optional LLM."""
import datetime
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SKILL_KEYWORDS: List[str] = [
    "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "C++", "C#", "Ruby", "PHP",
    "React", "Vue", "Angular", "Node.js", "Django", "Flask", "FastAPI", "Spring", "Rails",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "CI/CD", "Jenkins",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "TensorFlow", "PyTorch",
    "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Kafka",
    "Git", "Linux", "REST", "GraphQL", "Microservices", "Agile", "Scrum",
    "Data Science", "Data Analysis", "Statistics", "Pandas", "NumPy", "Scikit-learn",
    "Spark", "Hadoop", "Airflow", "dbt", "Tableau", "Power BI",
]

SKILL_SYNONYMS: Dict[str, str] = {
    "ML": "Machine Learning",
    "AI": "Artificial Intelligence",
    "JS": "JavaScript",
    "TS": "TypeScript",
    "Py": "Python",
    "K8s": "Kubernetes",
    "DL": "Deep Learning",
    "GCP": "Google Cloud Platform",
}

EDUCATION_LEVELS = [
    "PhD", "Ph.D", "Doctorate",
    "Master", "M.S.", "M.Sc", "MBA", "M.Tech",
    "Bachelor", "B.S.", "B.Sc", "B.Tech", "B.E.",
    "Associate",
]


class MetadataExtractor:
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        self.llm_config = llm_config
        self._email_re = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
        self._phone_re = re.compile(
            r"(\+?\d{1,3}[\s.-]?)?(\(?\d{3}\)?[\s.-]?)(\d{3}[\s.-]?\d{4})"
        )
        self._exp_re = [
            re.compile(r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)", re.I),
            re.compile(r"(\d+)\+?\s*(?:years?|yrs?)\s+(?:in|with)\s+", re.I),
        ]

    def extract(self, text: str, source_file: str) -> Dict[str, Any]:
        metadata = {
            "name": self._extract_name(text),
            "email": self._extract_email(text),
            "phone": self._extract_phone(text),
            "skills": self._extract_skills(text),
            "experience_years": self._extract_experience_years(text),
            "education": self._extract_education(text),
            "source_file": source_file,
        }

        if self.llm_config:
            try:
                llm_meta = self._llm_extract(text)
                for key, value in llm_meta.items():
                    if value and not metadata.get(key):
                        metadata[key] = value
            except Exception as e:
                logger.warning(f"LLM extraction failed, using regex results: {e}")

        return metadata

    def _extract_name(self, text: str) -> str:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        skip = {"resume", "cv", "curriculum vitae", "email", "phone", "address", "linkedin"}
        for line in lines[:6]:
            words = line.split()
            if (
                2 <= len(words) <= 4
                and not any(c.isdigit() for c in line)
                and line.lower() not in skip
                and not any(kw in line.lower() for kw in skip)
                and all(w[0].isupper() for w in words if w and w[0].isalpha())
            ):
                return line
        return "Unknown"

    def _extract_email(self, text: str) -> Optional[str]:
        matches = self._email_re.findall(text)
        return matches[0] if matches else None

    def _extract_phone(self, text: str) -> Optional[str]:
        match = self._phone_re.search(text)
        if match:
            return "".join(g for g in match.groups() if g)
        return None

    def _extract_skills(self, text: str) -> List[str]:
        found: set = set()
        text_lower = text.lower()

        for skill in SKILL_KEYWORDS:
            if skill.lower() in text_lower:
                found.add(skill)

        for abbr, full in SKILL_SYNONYMS.items():
            if re.search(r"\b" + re.escape(abbr) + r"\b", text):
                found.add(full)

        return sorted(found)

    def _extract_experience_years(self, text: str) -> int:
        max_years = 0

        for pattern in self._exp_re:
            for match in pattern.findall(text):
                try:
                    max_years = max(max_years, int(match))
                except ValueError:
                    pass

        date_re = re.compile(
            r"(\d{4})\s*[-–]\s*(?:(\d{4})|Present|Current|Now)", re.I
        )
        current_year = datetime.datetime.now().year
        for start_str, end_str in date_re.findall(text):
            start = int(start_str)
            end = int(end_str) if end_str else current_year
            years = end - start
            if 0 < years < 50:
                max_years = max(max_years, years)

        return max_years

    def _extract_education(self, text: str) -> str:
        for level in EDUCATION_LEVELS:
            if re.search(r"\b" + re.escape(level) + r"\b", text, re.I):
                return level
        return "Not specified"

    def _llm_extract(self, text: str) -> Dict[str, Any]:
        import anthropic

        client = anthropic.Anthropic()
        prompt = f"""Extract structured metadata from this resume. Return ONLY valid JSON.

Resume:
{text[:3000]}

Return JSON with these exact keys:
{{
  "name": "full name or null",
  "email": "email or null",
  "phone": "phone or null",
  "skills": ["skill1", "skill2"],
  "experience_years": <integer>,
  "education": "highest degree or 'Not specified'"
}}"""

        response = client.messages.create(
            model=self.llm_config.get("model", "claude-opus-4-8"),
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```")
        return json.loads(raw)

    @staticmethod
    def normalize_skill(skill: str) -> str:
        return SKILL_SYNONYMS.get(skill, skill)
