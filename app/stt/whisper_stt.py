"""Speech-to-Text: Groq Whisper API (primary) with local faster-whisper fallback."""

import io
import logging
import wave

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {"kk", "ru", "en"}

# Kazakh-specific characters not found in Russian or English
_KK_CHARS = set("әғқңөұүһі")


def _detect_language_from_text(text: str) -> str | None:
    """Detect language from transcribed text using character analysis.

    Returns 'kk', 'ru', 'en', or None if unsure.
    """
    lower = text.lower()

    # Kazakh has unique characters that Russian doesn't have
    if any(ch in _KK_CHARS for ch in lower):
        return "kk"

    # If text is purely ASCII — likely English
    try:
        lower.encode("ascii")
        if len(lower.split()) >= 2:
            return "en"
    except UnicodeEncodeError:
        pass

    # Cyrillic text without Kazakh chars — Russian
    cyrillic_count = sum(1 for ch in lower if "\u0400" <= ch <= "\u04ff")
    if cyrillic_count > len(lower) * 0.3:
        return "ru"

    return None


HALLUCINATION_PHRASES = {
    "thanks for watching", "thank you for watching", "subscribe",
    "so so so", "the end", "bye bye", "you", "thank you",
    "подписывайтесь", "спасибо за просмотр",
}


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    """Convert raw PCM 16-bit mono to WAV format for Groq API."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


class GroqWhisperSTT:
    """Groq cloud Whisper API — fast, accurate, supports Kazakh well."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=settings.groq_api_key)
        return self._client

    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> dict:
        # Check audio energy
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_array ** 2))
        if rms < 0.005:
            lang = language or "ru"
            logger.info("STT: audio too quiet (rms=%.4f), skipping", rms)
            return {"text": "", "language": lang, "confidence": 0.0}

        wav_data = _pcm_to_wav(audio_data, sample_rate)
        client = self._get_client()

        lang_param = language if language in SUPPORTED_LANGUAGES else None

        try:
            result = client.audio.transcriptions.create(
                file=("audio.wav", wav_data),
                model="whisper-large-v3-turbo",
                language=lang_param,
                response_format="verbose_json",
            )

            text = result.text.strip() if result.text else ""
            whisper_lang = getattr(result, "language", None) or ""

            # Map full language names to codes
            lang_map = {"kazakh": "kk", "russian": "ru", "english": "en"}
            if whisper_lang in lang_map:
                whisper_lang = lang_map[whisper_lang]

            # Text-based detection overrides Whisper when it's more reliable
            text_lang = _detect_language_from_text(text)
            detected_lang = text_lang or whisper_lang or "ru"

            if detected_lang not in SUPPORTED_LANGUAGES:
                detected_lang = "ru"

            if text_lang and text_lang != whisper_lang:
                logger.info("STT: text-based lang=%s overrides Whisper lang=%s", text_lang, whisper_lang)

            # Filter hallucinations
            if text.lower() in HALLUCINATION_PHRASES:
                logger.info("STT: filtered hallucination: '%s'", text)
                return {"text": "", "language": detected_lang, "confidence": 0.0}

            logger.info("STT (Groq): lang=%s, text='%s'", detected_lang, text[:100])
            return {"text": text, "language": detected_lang, "confidence": 0.95}

        except Exception as e:
            logger.error("Groq Whisper API error: %s", e)
            raise


class LocalWhisperSTT:
    """Local faster-whisper fallback."""

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            logger.info("Loading local Whisper model: %s", settings.whisper_model_size)
            self._model = WhisperModel(
                settings.whisper_model_size,
                device="auto",
                compute_type="auto",
            )
            logger.info("Local Whisper model loaded")
        return self._model

    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> dict:
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        rms = np.sqrt(np.mean(audio_array ** 2))
        if rms < 0.005:
            lang = language or "ru"
            return {"text": "", "language": lang, "confidence": 0.0}

        model = self._load_model()
        lang_param = language if language in SUPPORTED_LANGUAGES else None

        segments, info = model.transcribe(
            audio_array,
            beam_size=5,
            language=lang_param,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=300),
        )

        full_text = " ".join(seg.text.strip() for seg in segments)

        if full_text.strip().lower() in HALLUCINATION_PHRASES:
            detected_lang = language or "ru"
            return {"text": "", "language": detected_lang, "confidence": 0.0}

        if lang_param:
            detected_lang = lang_param
        else:
            detected_lang = info.language if info.language in SUPPORTED_LANGUAGES else "ru"

        logger.info("STT (local): lang=%s (prob=%.2f), text='%s'", detected_lang, info.language_probability, full_text[:100])
        return {"text": full_text, "language": detected_lang, "confidence": info.language_probability}


class WhisperSTT:
    """STT with Groq API primary, local fallback."""

    def __init__(self):
        self._groq = GroqWhisperSTT() if settings.groq_api_key else None
        self._local = LocalWhisperSTT()

    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> dict:
        if self._groq:
            try:
                return await self._groq.transcribe(audio_data, sample_rate, language)
            except Exception:
                logger.warning("Groq STT failed, falling back to local Whisper")

        return await self._local.transcribe(audio_data, sample_rate, language)


# Singleton
whisper_stt = WhisperSTT()
