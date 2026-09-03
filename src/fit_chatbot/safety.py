"""Deterministic safety triage applied before any generative model call."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyDecision:
    level: str
    should_generate: bool
    response: str | None = None


EMERGENCY_PATTERNS = (
    re.compile(r"\b(chest pain|pressure in (?:my|the) chest)\b", re.I),
    re.compile(r"\b(fainted|fainting|passed out|cannot breathe|can't breathe)\b", re.I),
    re.compile(r"\b(sudden numbness|sudden weakness|severe bleeding)\b", re.I),
)

CLINICAL_PATTERNS = (
    re.compile(r"\b(pregnan|postpartum|medication|diabetes|heart condition)\w*\b", re.I),
    re.compile(r"\b(sharp pain|severe pain|swelling|numbness|injur(?:y|ed))\b", re.I),
    re.compile(r"\b(eating disorder|purging|starv(?:e|ing)|laxatives)\b", re.I),
)


def assess_safety(message: str) -> SafetyDecision:
    if any(pattern.search(message) for pattern in EMERGENCY_PATTERNS):
        return SafetyDecision(
            level="urgent",
            should_generate=False,
            response=(
                "Stop exercising now. These symptoms can require urgent medical attention. "
                "Contact your local emergency service or seek urgent in-person care; do not "
                "rely on this chat to assess the cause."
            ),
        )
    if any(pattern.search(message) for pattern in CLINICAL_PATTERNS):
        return SafetyDecision(level="high", should_generate=True)
    return SafetyDecision(level="standard", should_generate=True)


def safety_prefix(level: str) -> str:
    if level == "high":
        return (
            "I can offer general fitness education, but this needs advice tailored by a "
            "qualified clinician who knows your history. "
        )
    return ""
