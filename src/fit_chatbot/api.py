"""FastAPI application for the Fit Coach service."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .service import ChatService


class UserProfile(BaseModel):
    goal: str = Field(default="", max_length=200)
    experience: str = Field(default="", max_length=100)
    equipment: str = Field(default="", max_length=300)
    schedule: str = Field(default="", max_length=200)
    limitations: str = Field(default="", max_length=500)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    profile: UserProfile = Field(default_factory=UserProfile)


@lru_cache(maxsize=1)
def get_service() -> ChatService:
    return ChatService()


app = FastAPI(title="Form Fitness Coach API", version="0.1.0")
origins = [
    item.strip() for item in os.getenv("FIT_CORS_ORIGINS", "http://localhost:5173").split(",") if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "form-fitness-coach"}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    try:
        return get_service().chat(request.message, request.profile.model_dump()).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
