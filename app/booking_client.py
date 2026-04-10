"""HTTP client for the KBTU booking API at https://api.kbtucare.site/api/v1"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx

from app.config import settings

BASE_URL = "https://api.kbtucare.site"
ALMATY_TZ = timezone(timedelta(hours=5))

logger = logging.getLogger(__name__)

_DAYS = {
    "ru": ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"],
    "kk": ["Дүйсенбі", "Сейсенбі", "Сәрсенбі", "Бейсенбі", "Жұма", "Сенбі", "Жексенбі"],
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
}
_MONTHS = {
    "ru": ["", "января", "февраля", "марта", "апреля", "мая", "июня",
           "июля", "августа", "сентября", "октября", "ноября", "декабря"],
    "kk": ["", "қаңтар", "ақпан", "наурыз", "сәуір", "мамыр", "маусым",
           "шілде", "тамыз", "қыркүйек", "қазан", "қараша", "желтоқсан"],
    "en": ["", "January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"],
}

_STATUS_LABELS = {
    "ru": {"available": "свободно", "booked": "забронировано", "cancelled": "отменено",
           "completed": "завершено", "confirmed": "подтверждено", "pending": "ожидает"},
    "kk": {"available": "бос", "booked": "брондалған", "cancelled": "бекітілмеген",
           "completed": "аяқталған", "confirmed": "расталған", "pending": "күтілуде"},
    "en": {"available": "available", "booked": "booked", "cancelled": "cancelled",
           "completed": "completed", "confirmed": "confirmed", "pending": "pending"},
}


def _to_almaty(dt_str: str) -> datetime:
    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    return dt.astimezone(ALMATY_TZ)


def _fmt_dt(dt_str: str, lang: str = "ru") -> str:
    """Format ISO datetime as natural speech: 'Пятница, 11 апреля, 10:30'."""
    dt = _to_almaty(dt_str)
    days = _DAYS.get(lang, _DAYS["ru"])
    months = _MONTHS.get(lang, _MONTHS["ru"])
    if lang == "en":
        return f"{days[dt.weekday()]}, {months[dt.month]} {dt.day}, {dt.strftime('%H:%M')}"
    return f"{days[dt.weekday()]}, {dt.day} {months[dt.month]}, {dt.strftime('%H:%M')}"


def _status(status: str, lang: str = "ru") -> str:
    return _STATUS_LABELS.get(lang, _STATUS_LABELS["ru"]).get(status, status)


def _client(token: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=10,
        headers={"Authorization": f"Bearer {token}"},
    )


# ── Psychologists ──────────────────────────────────────────────

async def list_psychologists(token: str) -> list[dict]:
    """GET /api/v1/users/psychologists — list all psychologists."""
    async with _client(token) as c:
        r = await c.get("/api/v1/users/psychologists")
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else data.get("psychologists", [])


async def get_psychologist_id(token: str) -> str:
    """Return configured PSYCHOLOGIST_ID or auto-discover from API."""
    if settings.psychologist_id:
        return settings.psychologist_id
    psychologists = await list_psychologists(token)
    if not psychologists:
        raise RuntimeError("No psychologists found in API")
    pid = psychologists[0].get("id") or psychologists[0].get("user_id", "")
    logger.info("Auto-discovered psychologist_id: %s", pid)
    return pid


# ── Slots ──────────────────────────────────────────────────────

async def get_available_slots(token: str, days_ahead: int = 14) -> list[dict]:
    """Fetch free slots for the next `days_ahead` days, sorted by start_time."""
    psychologist_id = await get_psychologist_id(token)
    now = datetime.now(ALMATY_TZ)
    free_slots: list[dict] = []

    async with _client(token) as c:
        months_to_query: set[tuple[int, int]] = set()
        for offset in range(days_ahead + 1):
            d = now + timedelta(days=offset)
            months_to_query.add((d.year, d.month))

        available_dates: set[str] = set()
        for year, month in sorted(months_to_query):
            try:
                r = await c.get("/api/v1/slots/calendar", params={
                    "psychologist_id": psychologist_id, "year": str(year), "month": str(month),
                })
                r.raise_for_status()
                for d in (r.json().get("available_dates") or []):
                    available_dates.add(d)
            except Exception as exc:
                logger.error("calendar error %s-%s: %s", year, month, exc)

        cutoff = (now + timedelta(days=days_ahead)).date()
        target_dates = sorted(
            d for d in available_dates
            if datetime.fromisoformat(d).date() <= cutoff
        )

        for date_str in target_dates:
            try:
                r = await c.get("/api/v1/slots", params={
                    "psychologist_id": psychologist_id, "date": date_str,
                })
                r.raise_for_status()
                slots = r.json()
                if isinstance(slots, list):
                    for s in slots:
                        if s.get("status") == "available":
                            free_slots.append(s)
            except Exception as exc:
                logger.error("slots error for %s: %s", date_str, exc)

    free_slots.sort(key=lambda s: s["start_time"])
    return free_slots


def format_slots_for_llm(slots: list[dict], lang: str = "ru") -> str:
    """Format slot list as numbered text for LLM context injection."""
    if not slots:
        return {"ru": "Свободных слотов нет.", "kk": "Бос уақыт табылмады.", "en": "No available slots."}.get(lang, "No slots.")
    lines = [f"{i}. {_fmt_dt(s['start_time'], lang)}  [slot_id: {s['id']}]"
             for i, s in enumerate(slots, 1)]
    return "\n".join(lines)


async def get_formatted_slots(token: str, lang: str = "ru", max_slots: int = 6) -> str:
    try:
        slots = await get_available_slots(token)
        return format_slots_for_llm(slots[:max_slots], lang)
    except Exception as exc:
        logger.error("get_formatted_slots: %s", exc)
        return ""


async def get_formatted_psychologists(token: str) -> str:
    """Format psychologist list for LLM context."""
    try:
        psychologists = await list_psychologists(token)
        if not psychologists:
            return ""
        lines = []
        for p in psychologists:
            name = p.get("full_name") or p.get("name", "")
            spec = p.get("specialization") or p.get("specialty") or p.get("position", "")
            pid = p.get("id") or p.get("user_id", "")
            exp = p.get("experience_years")
            line = f"- {name}"
            if spec:
                line += f", {spec}"
            if exp:
                line += f", опыт {exp} лет"
            line += f"  [psychologist_id: {pid}]"
            lines.append(line)
        return "\n".join(lines)
    except Exception as exc:
        logger.error("get_formatted_psychologists: %s", exc)
        return ""


# ── Student appointments ───────────────────────────────────────

async def get_appointments(token: str) -> list[dict]:
    """GET /api/v1/student/appointments"""
    async with _client(token) as c:
        r = await c.get("/api/v1/student/appointments")
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []


def format_appointments_for_llm(appointments: list[dict], lang: str = "ru") -> str:
    """Format student's appointments for LLM context."""
    if not appointments:
        return {"ru": "Нет записей.", "kk": "Жазбалар жоқ.", "en": "No appointments."}.get(lang, "No appointments.")
    lines = []
    for i, a in enumerate(appointments, 1):
        status = _status(a.get("status") or "booked", lang)
        dt = _fmt_dt(a["start_time"], lang) if a.get("start_time") else "?"
        psych = a.get("psychologist_name", "")
        line = f"{i}. {dt} — {status}"
        if psych:
            line += f" ({psych})"
        line += f"  [slot_id: {a['id']}]"
        lines.append(line)
    return "\n".join(lines)


async def get_formatted_appointments(token: str, lang: str = "ru") -> str:
    try:
        return format_appointments_for_llm(await get_appointments(token), lang)
    except Exception as exc:
        logger.error("get_formatted_appointments: %s", exc)
        return ""


# ── Booking ────────────────────────────────────────────────────

class BookingError(Exception):
    """Raised when booking API returns a known error."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


def _handle_booking_error(r, action: str = "booking"):
    """Raise BookingError with human-readable message for known HTTP errors."""
    if r.status_code == 401:
        raise BookingError(401, "token_expired")
    if r.status_code == 409:
        raise BookingError(409, "slot_taken")
    if r.status_code == 404:
        raise BookingError(404, "slot_not_found")
    r.raise_for_status()


async def book_appointment(
    token: str,
    slot_id: str,
    booking_type: str = "online",
    phone_number: str = "",
    questionnaire: Optional[dict] = None,
) -> dict:
    """Full booking flow: POST /reserve → POST /confirm."""
    async with _client(token) as c:
        # Step 1: reserve
        r = await c.post(f"/api/v1/student/slots/{slot_id}/reserve",
                         json={"booking_type": booking_type})
        _handle_booking_error(r, "reserve")
        reserve_result = r.json()
        logger.info("Slot reserved: %s", reserve_result.get("message", ""))

        # Step 2: confirm
        if not phone_number:
            phone_number = "+70000000000"

        answers: dict = {
            "main_topic": (questionnaire or {}).get("main_topic", ""),
            "avoid_topics": (questionnaire or {}).get("avoid_topics", ""),
            "sleep": (questionnaire or {}).get("sleep", "Okay"),
            "appetite": (questionnaire or {}).get("appetite", "Okay"),
            "mood": (questionnaire or {}).get("mood", ""),
        }

        body = {
            "phone_number": phone_number,
            "answers": json.dumps(answers, ensure_ascii=False),
        }
        r = await c.post(f"/api/v1/student/slots/{slot_id}/confirm", json=body)
        _handle_booking_error(r, "confirm")
        confirm_result = r.json()
        logger.info("Slot confirmed: %s", confirm_result.get("message", ""))

    return {
        "slot_id": slot_id,
        "reserve": reserve_result,
        "confirm": confirm_result,
    }


# ── Cancel ─────────────────────────────────────────────────────

async def cancel_appointment(slot_id: str, token: str,
                              reason_topic: str, reason_message: str = "") -> dict:
    """POST /api/v1/student/slots/{id}/cancel"""
    body = {"reason_topic": reason_topic}
    if reason_message:
        body["reason_message"] = reason_message
    async with _client(token) as c:
        r = await c.post(f"/api/v1/student/slots/{slot_id}/cancel", json=body)
        r.raise_for_status()
        return r.json()


# ── Confirm ────────────────────────────────────────────────────

async def confirm_appointment(slot_id: str, token: str,
                               phone_number: str, answers: Optional[dict] = None) -> dict:
    """POST /api/v1/student/slots/{id}/confirm"""
    body: dict = {"phone_number": phone_number}
    if answers:
        body["answers"] = json.dumps(answers, ensure_ascii=False)
    async with _client(token) as c:
        r = await c.post(f"/api/v1/student/slots/{slot_id}/confirm", json=body)
        r.raise_for_status()
        return r.json()


# ── Rate ───────────────────────────────────────────────────────

async def rate_session(slot_id: str, token: str, rating: int, review: str = "") -> dict:
    """POST /api/v1/student/slots/{id}/rate"""
    body: dict = {"rating": max(1, min(5, rating))}
    if review:
        body["review"] = review
    async with _client(token) as c:
        r = await c.post(f"/api/v1/student/slots/{slot_id}/rate", json=body)
        r.raise_for_status()
        return r.json()


# ── Reschedule ─────────────────────────────────────────────────

async def reschedule_appointment(slot_id: str, new_slot_id: str, token: str) -> dict:
    """POST /api/v1/student/slots/{id}/reschedule"""
    async with _client(token) as c:
        r = await c.post(f"/api/v1/student/slots/{slot_id}/reschedule",
                         json={"new_slot_id": new_slot_id})
        r.raise_for_status()
        return r.json()


# ── Waitlist ───────────────────────────────────────────────────

async def get_waitlist(token: str) -> list[dict]:
    """GET /api/v1/student/waitlist"""
    async with _client(token) as c:
        r = await c.get("/api/v1/student/waitlist")
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []


async def join_waitlist(token: str, date: str, psychologist_id: Optional[str] = None) -> dict:
    """POST /api/v1/student/waitlist"""
    pid = psychologist_id or await get_psychologist_id(token)
    async with _client(token) as c:
        r = await c.post("/api/v1/student/waitlist",
                         json={"psychologist_id": pid, "date": date})
        r.raise_for_status()
        return r.json()


async def leave_waitlist(waitlist_id: str, token: str) -> dict:
    """DELETE /api/v1/student/waitlist/{id}"""
    async with _client(token) as c:
        r = await c.delete(f"/api/v1/student/waitlist/{waitlist_id}")
        r.raise_for_status()
        return r.json() if r.content else {"status": "removed"}
