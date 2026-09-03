"""Application service that composes safety, retrieval, and generation."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .generation import GroundedTemplateGenerator, TransformersGenerator
from .retrieval import KnowledgeBase
from .safety import assess_safety, safety_prefix


@dataclass(frozen=True)
class ChatResult:
    answer: str
    risk_level: str
    sources: list[dict[str, str]]
    model_backend: str

    def to_dict(self) -> dict:
        return asdict(self)


class ChatService:
    def __init__(self, knowledge_path: Path | None = None):
        root = Path(__file__).resolve().parents[2]
        self.knowledge = KnowledgeBase.from_json(knowledge_path or root / "knowledge" / "guidance.json")
        model_path = os.getenv("FIT_MODEL_PATH", "").strip()
        if model_path:
            self.generator = TransformersGenerator(model_path)
            self.backend = "transformers"
        else:
            self.generator = GroundedTemplateGenerator()
            self.backend = "grounded-template"

    def chat(self, message: str, profile: dict[str, str] | None = None) -> ChatResult:
        question = message.strip()
        if not question:
            raise ValueError("message must not be empty")
        decision = assess_safety(question)
        if not decision.should_generate:
            return ChatResult(decision.response or "", decision.level, [], "safety-policy")

        context = self.knowledge.search(question)
        answer = safety_prefix(decision.level) + self.generator.generate(question, context, profile or {})
        sources = [{"title": item.document.title, "url": item.document.source_url} for item in context]
        return ChatResult(answer, decision.level, sources, self.backend)
