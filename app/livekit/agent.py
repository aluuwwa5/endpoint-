"""LiveKit Voice Agent — real-time WebRTC voice bot using LiveKit Agents framework."""

import logging

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli

from app.voice_pipeline import VoicePipeline
from app.rag.knowledge_base import knowledge_base
from app.config import settings

logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
SAMPLES_PER_CHANNEL = 480  # 30ms frames at 16kHz


class VoiceBotAgent:
    """LiveKit agent that handles real-time voice conversations."""

    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.pipeline = VoicePipeline()
        self._audio_buffer = bytearray()
        self._is_processing = False
        # Accumulate ~2 seconds of audio before processing
        self._buffer_threshold = SAMPLE_RATE * 2 * 2  # 2 sec * 2 bytes/sample

    async def handle_track(self, track: rtc.Track, participant: rtc.RemoteParticipant):
        """Handle incoming audio track from a participant."""
        audio_stream = rtc.AudioStream(track)
        session_id = participant.identity or "anonymous"

        logger.info("Audio track received from participant: %s", session_id)

        async for event in audio_stream:
            if isinstance(event, rtc.AudioFrameEvent):
                # Accumulate audio frames
                frame = event.frame
                self._audio_buffer.extend(frame.data)

                # Process when we have enough audio (silence detection handled by Whisper VAD)
                if len(self._audio_buffer) >= self._buffer_threshold and not self._is_processing:
                    await self._process_buffer(session_id)

    async def _process_buffer(self, session_id: str):
        """Process accumulated audio buffer through the pipeline."""
        self._is_processing = True

        try:
            audio_data = bytes(self._audio_buffer)
            self._audio_buffer.clear()

            result = await self.pipeline.process_audio(
                audio_data=audio_data,
                session_id=session_id,
                sample_rate=SAMPLE_RATE,
            )

            if result["audio_out"]:
                await self._send_audio(result["audio_out"])
                logger.info(
                    "Sent response: '%s' (timings: %s)",
                    result["text_out"][:60],
                    result["timings"],
                )

        except Exception as e:
            logger.error("Pipeline error: %s", e)

        finally:
            self._is_processing = False

    async def _send_audio(self, audio_data: bytes):
        """Send audio response back through LiveKit."""
        source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
        track = rtc.LocalAudioTrack.create_audio_track("voice-bot-response", source)

        options = rtc.TrackPublishOptions()
        publication = await self.ctx.room.local_participant.publish_track(track, options)

        # Send audio in chunks
        chunk_size = SAMPLES_PER_CHANNEL * NUM_CHANNELS * 2  # 2 bytes per sample
        offset = 0

        while offset < len(audio_data):
            chunk = audio_data[offset : offset + chunk_size]
            if len(chunk) < chunk_size:
                chunk = chunk + b"\x00" * (chunk_size - len(chunk))

            frame = rtc.AudioFrame(
                data=chunk,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=SAMPLES_PER_CHANNEL,
            )
            await source.capture_frame(frame)
            offset += chunk_size

        await self.ctx.room.local_participant.unpublish_track(publication.sid)


async def entrypoint(ctx: JobContext):
    """LiveKit agent entrypoint."""
    logger.info("Voice bot agent starting...")

    # Initialize knowledge base on startup
    knowledge_base.initialize()

    bot = VoiceBotAgent(ctx)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            ctx.create_task(bot.handle_track(track, participant))

    await ctx.connect()
    logger.info("Voice bot agent connected to room: %s", ctx.room.name)


def run_agent():
    """Run the LiveKit agent worker."""
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_url,
        )
    )


if __name__ == "__main__":
    run_agent()
