"""Typed memory schemas for structured long-term storage."""

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Persistent user profile stored in long-term memory."""

    user_id: str
    name: str | None = None
    preferences: list[str] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)


class SessionSummary(BaseModel):
    """Summary of a completed session for episodic memory."""

    thread_id: str
    user_id: str
    topic: str
    summary: str
    actions_taken: list[str] = Field(default_factory=list)
    timestamp: str
