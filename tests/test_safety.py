from fit_chatbot.safety import assess_safety


def test_emergency_language_bypasses_generation() -> None:
    decision = assess_safety("I have chest pain and feel faint during this workout")
    assert decision.level == "urgent"
    assert decision.should_generate is False
    assert "emergency" in (decision.response or "").lower()


def test_clinical_context_is_high_risk() -> None:
    decision = assess_safety("How should I exercise while pregnant?")
    assert decision.level == "high"
    assert decision.should_generate is True


def test_normal_training_question_is_standard() -> None:
    decision = assess_safety("How many sets should I do?")
    assert decision.level == "standard"
