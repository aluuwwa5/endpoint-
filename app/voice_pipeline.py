"""Voice Pipeline: orchestrates STT → RAG → LLM → TTS flow with action handling."""

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

    async def _run_llm_and_actions(
        self, text: str, language: str, session_id: str, male: bool = False, token: str = ""
    ) -> dict:
        """Run RAG + LLM + handle actions (booking, crisis). Returns timings + parsed result."""
        timings = {}

        # RAG retrieval
        t0 = time.time()
        rag_context = knowledge_base.retrieve(text)
        timings["rag_ms"] = int((time.time() - t0) * 1000)

        # LLM — now returns {reply, action, student_data}
        t0 = time.time()
        llm_result = await gemini_llm.generate_response(
            text=text,
            language=language,
            session_id=session_id,
            rag_context=rag_context,
            male=male,
        )
        timings["llm_ms"] = int((time.time() - t0) * 1000)

        text_out = llm_result["reply"]
        action = llm_result["action"]
        appointment = None

        # Handle actions
        if action == "book" and llm_result.get("student_data"):
            appointment = await create_appointment(
                student_data=llm_result["student_data"],
                language=language,
                token=token or None,
            )
            logger.info("Appointment booked: %s", appointment["id"])

        return {
            "text_out": text_out,
            "action": action,
            "appointment": appointment,
            "timings": timings,
        }

    async def process_audio(
        self,
        audio_data: bytes,
        session_id: str = "default",
        sample_rate: int = 16000,
        male: bool = False,
        token: str = "",
    ) -> dict:
        """Process raw audio through the full pipeline.

        Returns:
            dict with: text_in, language, text_out, audio_out, action, appointment, timings
        """
        timings = {}

        # Step 1: STT (auto-detect language from speech)
        t0 = time.time()
        stt_result = await whisper_stt.transcribe(audio_data, sample_rate)
        timings["stt_ms"] = int((time.time() - t0) * 1000)

        text_in = stt_result["text"]
        language = stt_result["language"]

        if not text_in.strip():
            return {
                "text_in": "",
                "language": language,
                "text_out": "",
                "audio_out": b"",
                "action": "none",
                "appointment": None,
                "timings": timings,
            }

        # Steps 2-3: RAG + LLM + actions
        llm_result = await self._run_llm_and_actions(text_in, language, session_id, male=male, token=token)
        timings.update(llm_result["timings"])

        # Step 4: TTS
        t0 = time.time()
        audio_out = await azure_tts.synthesize(llm_result["text_out"], language=language, male=male)
        timings["tts_ms"] = int((time.time() - t0) * 1000)
        timings["total_ms"] = sum(timings.values())

        logger.info(
            "Pipeline: lang=%s, action=%s, timings=%s",
            language, llm_result["action"], timings,
        )

        return {
            "text_in": text_in,
            "language": language,
            "text_out": llm_result["text_out"],
            "audio_out": audio_out,
            "action": llm_result["action"],
            "appointment": llm_result["appointment"],
            "timings": timings,
        }

    async def process_text(
        self,
        text: str,
        language: str = "ru",
        session_id: str = "default",
        token: str = "",
    ) -> dict:
        """Process text input (for testing without audio)."""
        timings = {}

        # RAG + LLM + actions
        llm_result = await self._run_llm_and_actions(text, language, session_id, token=token)
        timings.update(llm_result["timings"])

        # TTS
        t0 = time.time()
        audio_out = await azure_tts.synthesize(llm_result["text_out"], language=language)
        timings["tts_ms"] = int((time.time() - t0) * 1000)
        timings["total_ms"] = sum(timings.values())

        return {
            "text_in": text,
            "language": language,
            "text_out": llm_result["text_out"],
            "audio_out": audio_out,
            "action": llm_result["action"],
            "appointment": llm_result["appointment"],
            "timings": timings,
        }
