from fastapi.testclient import TestClient

from fit_chatbot.api import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_endpoint() -> None:
    response = client.post(
        "/api/chat",
        json={"message": "How should I start strength training?", "profile": {"experience": "beginner"}},
    )
    assert response.status_code == 200
    assert response.json()["answer"]


def test_chat_rejects_empty_message() -> None:
    response = client.post("/api/chat", json={"message": "", "profile": {}})
    assert response.status_code == 422
