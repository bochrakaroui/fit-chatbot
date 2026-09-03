from pathlib import Path

from fit_chatbot.retrieval import KnowledgeBase

ROOT = Path(__file__).resolve().parents[1]


def test_squat_query_returns_squat_guidance_first() -> None:
    knowledge = KnowledgeBase.from_json(ROOT / "knowledge" / "guidance.json")
    result = knowledge.search("How should my knees move during a squat?")
    assert result
    assert result[0].document.id == "form-squat"


def test_empty_query_has_no_results() -> None:
    knowledge = KnowledgeBase.from_json(ROOT / "knowledge" / "guidance.json")
    assert knowledge.search("---") == []
