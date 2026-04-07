"""Appointment storage — saves bookings locally and/or via teammate's API."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
APPOINTMENTS_FILE = DATA_DIR / "appointments.json"


def _load_appointments() -> list[dict]:
    if not APPOINTMENTS_FILE.exists():
        return []
    return json.loads(APPOINTMENTS_FILE.read_text(encoding="utf-8"))


def _save_appointments(appointments: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    APPOINTMENTS_FILE.write_text(
        json.dumps(appointments, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _save_locally(student_data: dict, language: str, api_result: Optional[dict] = None) -> dict:
    """Save appointment record to local JSON file."""
    appointments = _load_appointments()

    appointment = {
        "id": str(uuid.uuid4())[:8],
        "first_name": student_data.get("first_name", student_data.get("name", "")),
        "last_name": student_data.get("last_name", ""),
        "specialty": student_data.get("specialty", student_data.get("faculty", "")),
        "problem_summary": student_data.get("problem_summary", student_data.get("concern", "")),
        "appointment_date": student_data.get("appointment_date", student_data.get("preferred_time", "")),
        "language": language,
        "status": "booked" if api_result else "pending",
        "created_at": datetime.now().isoformat(),
    }

    if api_result:
        appointment["slot_id"] = api_result.get("slot_id")
        appointment["slot_info"] = api_result.get("slot_info")

    appointments.append(appointment)
    _save_appointments(appointments)
    logger.info("Appointment saved locally: id=%s", appointment["id"])
    return appointment


async def create_appointment(
    student_data: dict,
    language: str = "ru",
    token: Optional[str] = None,
) -> dict:
    """Create appointment: calls teammate's API if token provided, saves locally either way.

    Args:
        student_data: Dict with first_name, last_name, specialty,
                      problem_summary, appointment_date (preferred).
        language: Conversation language.
        token: JWT token from the student's session (from teammate's platform).

    Returns:
        The created appointment record.
    """
    api_result = None

    if token:
        try:
            from app.booking_client import book_appointment
            slot_id = student_data.get("slot_id")
            if not slot_id:
                raise ValueError("slot_id missing in student_data — cannot book")
            api_result = await book_appointment(
                token=token,
                slot_id=slot_id,
                booking_type=student_data.get("booking_type", "online"),
                phone_number=student_data.get("phone_number", ""),
                questionnaire={
                    "main_topic": student_data.get("problem_summary", ""),
                    "avoid_topics": student_data.get("avoid_topics", ""),
                    "sleep": student_data.get("sleep", "Okay"),
                    "appetite": student_data.get("appetite", "Okay"),
                    "mood": student_data.get("mood_note", ""),
                },
            )
            logger.info("Appointment booked via API: slot_id=%s", api_result.get("slot_id"))
        except Exception as exc:
            logger.error("API booking failed, saving locally only: %s", exc)

    return _save_locally(student_data, language, api_result)


def get_appointments() -> list[dict]:
    """Get all locally saved appointments."""
    return _load_appointments()
