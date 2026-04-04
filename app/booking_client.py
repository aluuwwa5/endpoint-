"""HTTP client for teammate's booking API at http://167.71.75.254"""

import logging
from typing import Optional

import httpx

BASE_URL = "http://167.71.75.254"
logger = logging.getLogger(__name__)


async def get_available_slots(token: str, date: Optional[str] = None) -> list[dict]:
    """Get available slots.

    Calls GET /slots/calendar to find first available date (if date not given),
    then GET /slots?date=YYYY-MM-DD to get slots for that date.
    """
    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10) as client:

            # If no date given — get first available date from calendar
            if not date:
                r = await client.get("/slots/calendar", headers=headers)
                r.raise_for_status()
                data = r.json()
                # Handle both list and dict response shapes
                if isinstance(data, list) and data:
                    first = data[0]
                    date = first if isinstance(first, str) else first.get("date")
                elif isinstance(data, dict):
                    dates = data.get("dates") or data.get("available_dates") or []
                    if dates:
                        first = dates[0]
                        date = first if isinstance(first, str) else first.get("date")

            if not date:
                logger.warning("booking_client: no available dates in calendar")
                return []

            r = await client.get("/slots", params={"date": date}, headers=headers)
            r.raise_for_status()
            slots = r.json()

            # Normalise to list
            if isinstance(slots, dict):
                slots = slots.get("slots") or slots.get("data") or []

            return slots if isinstance(slots, list) else []

    except Exception as exc:
        logger.error("booking_client.get_available_slots error: %s", exc)
        return []


async def book_slot(slot_id: int | str, token: str) -> dict:
    """POST /student/slots/{id}/book"""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10) as client:
            r = await client.post(f"/student/slots/{slot_id}/book", headers=headers)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "booking_client.book_slot HTTP %s: %s",
            exc.response.status_code,
            exc.response.text,
        )
        raise
    except Exception as exc:
        logger.error("booking_client.book_slot error: %s", exc)
        raise


async def book_appointment(token: str, preferred_date: Optional[str] = None) -> dict:
    """Find first available slot and book it.

    Returns dict with slot_id, slot_info, booking keys.
    Raises RuntimeError if no slots available or booking fails.
    """
    slots = await get_available_slots(token, date=preferred_date)
    if not slots:
        raise RuntimeError("No available slots found")

    slot = slots[0]
    slot_id = slot.get("id") or slot.get("slot_id")
    if not slot_id:
        raise RuntimeError("Cannot determine slot ID from API response")

    booking = await book_slot(slot_id, token)

    return {
        "slot_id": slot_id,
        "slot_info": slot,
        "booking": booking,
    }
