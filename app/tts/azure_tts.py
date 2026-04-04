"""Text-to-Speech module using Azure Cognitive Services Speech SDK."""

import logging
import azure.cognitiveservices.speech as speechsdk

from app.config import settings

logger = logging.getLogger(__name__)

# Voice mapping by language
VOICE_MAP = {
    "kk": "kk-KZ-AigulNeural",       # Kazakh female
    "ru": "ru-RU-DariyaNeural",       # Russian female
    "en": "en-US-JennyNeural",        # English female
}

# Alternate voices (male)
VOICE_MAP_MALE = {
    "kk": "kk-KZ-DauletNeural",
    "ru": "ru-RU-DmitryNeural",
    "en": "en-US-GuyNeural",
}


class AzureTTS:
    """Azure Speech TTS with multilingual support."""

    def __init__(self):
        self._speech_config: speechsdk.SpeechConfig | None = None

    def _get_config(self) -> speechsdk.SpeechConfig:
        if self._speech_config is None:
            self._speech_config = speechsdk.SpeechConfig(
                subscription=settings.azure_speech_key,
                region=settings.azure_speech_region,
            )
            self._speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
            )
        return self._speech_config

    async def synthesize(
        self,
        text: str,
        language: str = "ru",
        male: bool = False,
    ) -> bytes:
        """Synthesize text to raw PCM audio bytes.

        Args:
            text: Text to speak.
            language: Language code (kk, ru, en).
            male: Use male voice if True.

        Returns:
            Raw 16kHz 16-bit mono PCM audio bytes.
        """
        config = self._get_config()
        voice_map = VOICE_MAP_MALE if male else VOICE_MAP
        voice_name = voice_map.get(language, VOICE_MAP["ru"])
        config.speech_synthesis_voice_name = voice_name

        # Use in-memory stream (no speaker output on server)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=config,
            audio_config=None,  # No audio output — we capture the bytes
        )

        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info(
                "TTS done: voice=%s, text='%s', audio_bytes=%d",
                voice_name,
                text[:60],
                len(result.audio_data),
            )
            return result.audio_data

        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            logger.error(
                "TTS canceled: reason=%s, error=%s",
                cancellation.reason,
                cancellation.error_details,
            )
            raise RuntimeError(f"TTS failed: {cancellation.error_details}")

        raise RuntimeError(f"TTS unexpected result: {result.reason}")

    async def synthesize_ssml(
        self,
        text: str,
        language: str = "ru",
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> bytes:
        """Synthesize with SSML for finer control over prosody.

        Args:
            text: Text to speak.
            language: Language code.
            rate: Speech rate adjustment (e.g., "+10%", "-5%").
            pitch: Pitch adjustment.

        Returns:
            Raw PCM audio bytes.
        """
        voice_name = VOICE_MAP.get(language, VOICE_MAP["ru"])
        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{language}">
            <voice name="{voice_name}">
                <prosody rate="{rate}" pitch="{pitch}">
                    {text}
                </prosody>
            </voice>
        </speak>
        """

        config = self._get_config()
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=config, audio_config=None
        )

        result = synthesizer.speak_ssml_async(ssml).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data

        raise RuntimeError(f"SSML TTS failed: {result.reason}")


# Singleton
azure_tts = AzureTTS()
