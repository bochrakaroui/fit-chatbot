"""Small dependency-free BM25-style retriever for reviewed guidance."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    title: str
    text: str
    source_url: str
    topics: tuple[str, ...]


@dataclass(frozen=True)
class RetrievedDocument:
    document: KnowledgeDocument
    score: float


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


class KnowledgeBase:
    def __init__(self, documents: list[KnowledgeDocument]):
        self.documents = documents
        self._tokens = [tokenize(f"{doc.title} {doc.text} {' '.join(doc.topics)}") for doc in documents]
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            self._document_frequency.update(set(tokens))
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens) / len(self._tokens) if self._tokens else 1.0
        )

    @classmethod
    def from_json(cls, path: Path) -> KnowledgeBase:
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents = [
            KnowledgeDocument(
                id=item["id"],
                title=item["title"],
                text=item["text"],
                source_url=item["source_url"],
                topics=tuple(item.get("topics", [])),
            )
            for item in payload
        ]
        return cls(documents)

    def search(self, query: str, limit: int = 3) -> list[RetrievedDocument]:
        query_tokens = set(tokenize(query))
        if not query_tokens or not self.documents:
            return []
        scores: list[RetrievedDocument] = []
        total = len(self.documents)
        for document, tokens in zip(self.documents, self._tokens, strict=False):
            counts = Counter(tokens)
            length_adjustment = 0.25 + 0.75 * len(tokens) / self._average_length
            score = 0.0
            for token in query_tokens:
                frequency = counts[token]
                if not frequency:
                    continue
                inverse_frequency = math.log(
                    1
                    + (total - self._document_frequency[token] + 0.5)
                    / (self._document_frequency[token] + 0.5)
                )
                score += inverse_frequency * frequency * 2.2 / (frequency + 1.2 * length_adjustment)
            if score > 0:
                scores.append(RetrievedDocument(document, score))
        return sorted(scores, key=lambda result: result.score, reverse=True)[:limit]
