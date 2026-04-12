"""Voice Pipeline: orchestrates STT → RAG+context → LLM → TTS → actions."""

import asyncio
import logging
import time

from app.stt.whisper_stt import whisper_stt
from app.llm.gemini_llm import gemini_llm
from app.tts.azure_tts import azure_tts
from app.rag.knowledge_base import knowledge_base
from app.appointments import create_appointment

logger = logging.getLogger(__name__)


class VoicePipeline:
    """Full voice conversation pipeline: audio in → audio out."""

    def __init__(self):
        knowledge_base.initialize()

    # ── Context fetching ──────────────────────────────────────

    async def _fetch_context(self, text: str, language: str, token: str) -> tuple[str, str, str, str]:
        """Fetch RAG context, available slots, student appointments, and psychologists in parallel.

        Returns (rag_context, slots_context, appointments_context, psychologists_context).
        """
        from app.booking_client import get_formatted_slots, get_formatted_appointments, get_formatted_psychologists

        async def _no_context() -> str:
            return ""

        rag_task = asyncio.to_thread(knowledge_base.retrieve, text)
        slots_task = get_formatted_slots(token, lang=language) if token else _no_context()
        appointments_task = get_formatted_appointments(token, lang=language) if token else _no_context()
        psychologists_task = get_formatted_psychologists(token) if token else _no_context()

        rag_ctx, slots_ctx, appts_ctx, psychologists_ctx = await asyncio.gather(
            rag_task, slots_task, appointments_task, psychologists_task
        )

        if slots_ctx:
            logger.info("Slots context: %d chars", len(slots_ctx))
        if appts_ctx:
            logger.info("Appointments context: %d chars", len(appts_ctx))
        if psychologists_ctx:
            logger.info("Psychologists context: %d chars", len(psychologists_ctx))

        return rag_ctx, slots_ctx, appts_ctx, psychologists_ctx

    # ── Action handlers ───────────────────────────────────────

    async def _handle_action(
        self, action: str, student_data: dict | None, language: str, token: str
    ) -> dict | None:
        """Execute the action returned by the LLM. Returns action result or None."""
        if not student_data:
            return None

        try:
            if action == "book":
                from datetime import datetime, timezone, timedelta
                # Validate appointment date is not in the past
                appt_date_str = student_data.get("appointment_date", "")
                slot_start = student_data.get("slot_start_time", "")
                check_str = slot_start or appt_date_str
                almaty_now = datetime.now(timezone(timedelta(hours=5)))
                if check_str:
                    try:
                        # Try parsing ISO format from slot_start_time
                        dt = datetime.fromisoformat(check_str.replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone(timedelta(hours=5)))
                        if dt < almaty_now:
                            return {
                                "ru": "Нельзя записаться на прошедшую дату. Давай выберем ближайшее свободное время?",
                                "kk": "Өткен күнге жазылу мүмкін емес. Жақын бос уақытты таңдайық?",
                                "en": "Cannot book a past date. Shall we pick an upcoming slot?",
                            }.get(language, "Cannot book a past date.")
                    except (ValueError, TypeError):
                        pass  # Can't parse — let the API decide

                return await create_appointment(
                    student_data=student_data,
                    language=language,
                    token=token or None,
                )

            if action == "cancel" and token:
                from app.booking_client import cancel_appointment
                VALID_REASON_TOPICS = {
                    "Schedule Conflict", "Personal Circumstances",
                    "Found Another Specialist", "Health Issues", "Other"
                }
                reason_topic = student_data.get("reason_topic", "Other")
                if reason_topic not in VALID_REASON_TOPICS:
                    reason_topic = "Other"
                return await cancel_appointment(
                    slot_id=student_data["slot_id"],
                    token=token,
                    reason_topic=reason_topic,
                    reason_message=student_data.get("reason_message", ""),
                )

            if action == "reschedule" and token:
                from app.booking_client import reschedule_appointment
                return await reschedule_appointment(
                    slot_id=student_data["old_slot_id"],
                    new_slot_id=student_data["new_slot_id"],
                    token=token,
                )

            if action == "confirm_appointment" and token:
                from app.booking_client import confirm_appointment
                answers = {"reason": student_data.get("reason", "")} if student_data.get("reason") else None
                return await confirm_appointment(
                    slot_id=student_data["slot_id"],
                    token=token,
                    phone_number=student_data["phone_number"],
                    answers=answers,
                )

            if action == "rate" and token:
                from app.booking_client import rate_session
                return await rate_session(
                    slot_id=student_data["slot_id"],
                    token=token,
                    rating=int(student_data.get("rating", 5)),
                    review=student_data.get("review", ""),
                )

            if action == "log_mood" and token:
                from app.user_client import log_mood
                return await log_mood(token=token, mood=student_data["mood"])

            if action == "join_waitlist" and token:
                from app.booking_client import join_waitlist
                return await join_waitlist(
                    token=token,
                    date=student_data["date"],
                )

        except KeyError as exc:
            logger.error("Action '%s' missing field: %s", action, exc)
        except Exception as exc:
            from app.booking_client import BookingError
            if isinstance(exc, BookingError):
                if exc.status_code == 409:
                    return {
                        "ru": "К сожалению, этот слот уже занят. Давай выберем другое время?",
                        "kk": "Өкінішке орай, бұл уақыт бос емес. Басқа уақыт таңдайық?",
                        "en": "Sorry, that slot is already taken. Shall we pick another time?",
                    }.get(language, "Slot already taken.")
                if exc.status_code == 401:
                    return {
                        "ru": "Твоя сессия истекла. Пожалуйста, войди в систему заново.",
                        "kk": "Сессияңыздың мерзімі аяқталды. Қайта кіріңіз.",
                        "en": "Your session has expired. Please log in again.",
                    }.get(language, "Session expired.")
            logger.error("Action '%s' failed: %s", action, exc)

        return None

    # ── Core pipeline ─────────────────────────────────────────

    async def _run(
        self, text: str, language: str, session_id: str,
        male: bool = False, token: str = ""
    ) -> dict:
        """RAG + context + LLM + action execution."""
        timings: dict[str, int] = {}

        t0 = time.time()
        rag_ctx, slots_ctx, appts_ctx, psychologists_ctx = await self._fetch_context(text, language, token)
        timings["rag_ms"] = int((time.time() - t0) * 1000)

        t0 = time.time()
        llm_result = await gemini_llm.generate_response(
            text=text,
            language=language,
            session_id=session_id,
            rag_context=rag_ctx,
            male=male,
            slots_context=slots_ctx,
            appointments_context=appts_ctx,
            psychologists_context=psychologists_ctx,
        )
        timings["llm_ms"] = int((time.time() - t0) * 1000)

        action = llm_result["action"]
        action_result = await self._handle_action(action, llm_result.get("student_data"), language, token)

        if action_result:
            logger.info("Action '%s' completed: %s", action, str(action_result)[:120])

        return {
            "text_out": llm_result["reply"],
            "action": action,
            "action_result": action_result,
            "timings": timings,
        }

    # ── Public API ────────────────────────────────────────────

    async def process_audio(
        self,
        audio_data: bytes,
        session_id: str = "default",
        sample_rate: int = 16000,
        male: bool = False,
        token: str = "",
    ) -> dict:
        """Process raw audio through the full pipeline."""
        timings: dict[str, int] = {}

        t0 = time.time()
        stt_result = await whisper_stt.transcribe(audio_data, sample_rate)
        timings["stt_ms"] = int((time.time() - t0) * 1000)

        text_in = stt_result["text"]
        language = stt_result["language"]

        if not text_in.strip():
            return {
                "text_in": "", "language": language, "text_out": "",
                "audio_out": b"", "action": "none", "action_result": None, "timings": timings,
            }

        result = await self._run(text_in, language, session_id, male=male, token=token)
        timings.update(result["timings"])

        t0 = time.time()
        audio_out = await azure_tts.synthesize(result["text_out"], language=language, male=male)
        timings["tts_ms"] = int((time.time() - t0) * 1000)
        timings["total_ms"] = sum(timings.values())

        logger.info("Pipeline done: lang=%s action=%s timings=%s", language, result["action"], timings)

        return {
            "text_in": text_in,
            "language": language,
            "text_out": result["text_out"],
            "audio_out": audio_out,
            "action": result["action"],
            "action_result": result["action_result"],
            "timings": timings,
        }

    def prime_session(self, session_id: str, greeting: str) -> None:
        """Inject greeting into LLM session so it won't repeat it."""
        gemini_llm.prime_session(session_id, greeting)

    async def process_text(
        self,
        text: str,
        language: str = "ru",
        session_id: str = "default",
        token: str = "",
    ) -> dict:
        """Process text input (for testing without audio)."""
        timings: dict[str, int] = {}

        result = await self._run(text, language, session_id, token=token)
        timings.update(result["timings"])

        t0 = time.time()
        audio_out = await azure_tts.synthesize(result["text_out"], language=language)
        timings["tts_ms"] = int((time.time() - t0) * 1000)
        timings["total_ms"] = sum(timings.values())

        return {
            "text_in": text,
            "language": language,
            "text_out": result["text_out"],
            "audio_out": audio_out,
            "action": result["action"],
            "action_result": result["action_result"],
            "timings": timings,
        }
