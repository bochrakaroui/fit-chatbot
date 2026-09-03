"""Grounded response generators with an optional local Transformers backend."""

from __future__ import annotations

from typing import Protocol

from .retrieval import RetrievedDocument


class Generator(Protocol):
    def generate(self, question: str, context: list[RetrievedDocument], profile: dict[str, str]) -> str: ...


class GroundedTemplateGenerator:
    """Safe zero-setup fallback that makes the repository runnable without weights."""

    def generate(self, question: str, context: list[RetrievedDocument], profile: dict[str, str]) -> str:
        if not context:
            return (
                "I need a little more context to make that useful. Share your goal, experience "
                "level, available equipment, schedule, and any limitations you want considered."
            )
        lead = context[0].document.text
        personal = []
        if profile.get("goal"):
            personal.append(f"goal: {profile['goal']}")
        if profile.get("experience"):
            personal.append(f"experience: {profile['experience']}")
        if profile.get("equipment"):
            personal.append(f"equipment: {profile['equipment']}")
        suffix = f" I accounted for your {', '.join(personal)}." if personal else ""
        return f"{lead}{suffix} Start conservatively, track how it feels, and adjust one variable at a time."


class TransformersGenerator:
    def __init__(self, model_path: str, *, max_new_tokens: int = 320):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("Install the 'ml' project extra to use local model inference.") from exc
        self.max_new_tokens = max_new_tokens
        self.pipeline = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path,
            device_map="auto",
        )

    def generate(self, question: str, context: list[RetrievedDocument], profile: dict[str, str]) -> str:
        sources = "\n".join(f"- {item.document.text}" for item in context)
        profile_text = (
            ", ".join(f"{key}: {value}" for key, value in profile.items() if value) or "not provided"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Form, a cautious fitness education assistant. Use only the supplied "
                    "guidance for factual claims. Ask for missing context. Never diagnose or prescribe."
                ),
            },
            {
                "role": "user",
                "content": f"Profile: {profile_text}\nReviewed guidance:\n{sources}\n\nQuestion: {question}",
            },
        ]
        result = self.pipeline(messages, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = result[0]["generated_text"]
        if isinstance(generated, list):
            return generated[-1]["content"].strip()
        return str(generated).strip()
