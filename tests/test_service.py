from fit_chatbot.service import ChatService


def test_service_returns_grounded_source() -> None:
    result = ChatService().chat("How can I improve my squat form?")
    assert result.risk_level == "standard"
    assert result.sources
    assert result.model_backend == "grounded-template"


def test_service_uses_profile_without_inference() -> None:
    result = ChatService().chat(
        "How should I begin strength training?",
        {"goal": "build strength", "experience": "beginner", "equipment": "dumbbells"},
    )
    assert "build strength" in result.answer


def test_service_blocks_emergency_generation() -> None:
    result = ChatService().chat("I have chest pain and feel faint")
    assert result.model_backend == "safety-policy"
    assert result.sources == []
