"""HTTP client for the KBTU user API at https://api.kbtucare.site/api/v1/users"""

import logging
from typing import Optional

import httpx

BASE_URL = "https://api.kbtucare.site"

logger = logging.getLogger(__name__)

MOOD_VALUES = ["Amazing", "Nice", "Not bad", "Sad", "Anxiously", "Stressed"]

# Mapping from natural speech to API value
MOOD_MAP = {
    # Russian
    "отлично": "Amazing", "замечательно": "Amazing", "прекрасно": "Amazing",
    "хорошо": "Nice", "неплохо": "Not bad", "нормально": "Not bad",
    "грустно": "Sad", "плохо": "Sad", "тревожно": "Anxiously",
    "беспокойно": "Anxiously", "стресс": "Stressed", "стрессово": "Stressed",
    # Kazakh
    "керемет": "Amazing", "жақсы": "Nice", "жаман емес": "Not bad",
    "қайғылы": "Sad", "мазасыз": "Anxiously", "стресс": "Stressed",
    # English
    "amazing": "Amazing", "great": "Amazing", "nice": "Nice", "good": "Nice",
    "not bad": "Not bad", "okay": "Not bad", "ok": "Not bad",
    "sad": "Sad", "anxious": "Anxiously", "stressed": "Stressed",
}


def _client(token: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=10,
        headers={"Authorization": f"Bearer {token}"},
    )


async def get_profile(token: str) -> dict:
    """GET /api/v1/users/me"""
    async with _client(token) as c:
        r = await c.get("/api/v1/users/me")
        r.raise_for_status()
        return r.json()


async def update_profile(token: str, full_name: Optional[str] = None,
                          phone: Optional[str] = None, bio: Optional[str] = None,
                          gender: Optional[str] = None) -> dict:
    """PUT /api/v1/users/me"""
    body = {k: v for k, v in {
        "full_name": full_name, "phone": phone, "bio": bio, "gender": gender,
    }.items() if v is not None}
    async with _client(token) as c:
        r = await c.put("/api/v1/users/me", json=body)
        r.raise_for_status()
        return r.json()


async def log_mood(token: str, mood: str) -> dict:
    """POST /api/v1/users/me/mood  mood must be one of MOOD_VALUES."""
    # Normalize: allow natural-language values
    normalized = MOOD_MAP.get(mood.lower().strip(), mood)
    if normalized not in MOOD_VALUES:
        # Best-effort fuzzy match
        for val in MOOD_VALUES:
            if val.lower() in mood.lower():
                normalized = val
                break
    async with _client(token) as c:
        r = await c.post("/api/v1/users/me/mood", json={"mood": normalized})
        r.raise_for_status()
        return r.json() if r.content else {"mood": normalized, "status": "logged"}


async def get_mood_history(token: str, filter: str = "last_week") -> dict:
    """GET /api/v1/users/me/mood/graphic"""
    async with _client(token) as c:
        r = await c.get("/api/v1/users/me/mood/graphic", params={"filter": filter})
        r.raise_for_status()
        return r.json()


async def list_psychologists(token: str) -> list[dict]:
    """GET /api/v1/users/psychologists"""
    async with _client(token) as c:
        r = await c.get("/api/v1/users/psychologists")
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else data.get("psychologists", [])
