"""FastAPI application — HTTP + WebSocket endpoints for the voice bot."""

import base64
import json
import logging
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

from fastapi import FastAPI, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
from app.appointments import get_appointments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KBTU Voice Bot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Lazy-init pipeline to avoid slow startup
_pipeline = None

# JWT tokens per session (set by client via WS config message)
_session_tokens: dict[str, str] = {}


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from app.voice_pipeline import VoicePipeline
        _pipeline = VoicePipeline()
    return _pipeline


def _build_ws_response(result: dict) -> dict:
    """Build a JSON response dict for WebSocket from pipeline result."""
    resp = {
        "type": "transcription",
        "text_in": result["text_in"],
        "text_out": result["text_out"],
        "language": result["language"],
        "action": result.get("action", "none"),
        "timings": result["timings"],
    }
    if result.get("appointment"):
        resp["appointment"] = result["appointment"]
    return resp


# ── Root ──────────────────────────────────────────────────────


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/system-design")
async def system_design():
    return FileResponse(BASE_DIR / "static" / "system_design.html")


# ── Health check ──────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "services": {
            "groq": bool(settings.groq_api_key),
            "gemini": bool(settings.gemini_api_key),
            "azure_speech": bool(settings.azure_speech_key),
        },
    }


# ── Text endpoint (for testing without audio) ────────────────


class TextRequest(BaseModel):
    text: str
    language: str = "ru"
    session_id: str = ""


@app.post("/api/chat")
async def chat_text(req: TextRequest, authorization: str = Header(default="")):
    """Text-only chat endpoint (no audio). Useful for testing the LLM + RAG."""
    pipeline = get_pipeline()
    session_id = req.session_id or str(uuid.uuid4())
    token = authorization.removeprefix("Bearer ").strip()

    result = await pipeline.process_text(
        text=req.text,
        language=req.language,
        session_id=session_id,
        token=token,
    )

    return {
        "text_in": result["text_in"],
        "text_out": result["text_out"],
        "language": result["language"],
        "action": result.get("action", "none"),
        "appointment": result.get("appointment"),
        "audio_out_base64": base64.b64encode(result["audio_out"]).decode()
        if result["audio_out"]
        else None,
        "timings": result["timings"],
        "session_id": session_id,
    }


# ── Appointments API (for teammate's platform) ───────────────


@app.get("/api/appointments")
async def list_appointments():
    """Get all booked appointments. For teammate's platform integration."""
    return {"appointments": get_appointments()}


# ── WebSocket endpoint (real-time audio streaming) ───────────


@app.websocket("/ws/voice")
async def websocket_voice(ws: WebSocket):
    """WebSocket endpoint for push-to-talk voice conversation.

    Protocol:
    1. Client sends JSON {"type": "config", "gender": "female"|"male"}
    2. Client sends binary audio frames (all at once or streamed)
    3. Client sends JSON {"type": "end"} when done speaking
    4. Server processes entire audio, responds with JSON + binary TTS audio
    """
    await ws.accept()
    session_id = None  # will be set from client config or generated
    pipeline = get_pipeline()
    audio_buffer = bytearray()
    gender = "female"
    token = ""  # JWT token passed by frontend from teammate's platform

    logger.info("WebSocket connected")

    try:
        while True:
            data = await ws.receive()

            # Handle disconnect
            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data:
                audio_buffer.extend(data["bytes"])

            elif "text" in data:
                msg = json.loads(data["text"])

                if msg.get("type") == "config":
                    gender = msg.get("gender", "female")
                    if not session_id:
                        session_id = msg.get("session_id") or str(uuid.uuid4())
                    # Store JWT token for this session
                    if msg.get("token"):
                        token = msg["token"]
                        _session_tokens[session_id] = token
                    logger.info("Session %s: gender=%s, has_token=%s", session_id, gender, bool(token))

                elif msg.get("type") == "end":
                    if not session_id:
                        session_id = str(uuid.uuid4())
                    # Retrieve token if stored for this session
                    if not token and session_id in _session_tokens:
                        token = _session_tokens[session_id]
                    # Process ALL accumulated audio at once
                    if len(audio_buffer) > 1600:
                        audio_data = bytes(audio_buffer)
                        audio_buffer.clear()

                        logger.info(
                            "Processing %d bytes of audio (%.1fs), session=%s",
                            len(audio_data),
                            len(audio_data) / (16000 * 2),
                            session_id,
                        )

                        result = await pipeline.process_audio(
                            audio_data=audio_data,
                            session_id=session_id,
                            male=(gender == "male"),
                            token=token,
                        )

                        if result["text_in"]:
                            await ws.send_json(_build_ws_response(result))
                            if result["audio_out"]:
                                await ws.send_bytes(result["audio_out"])
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session=%s", session_id)
    except Exception as e:
        logger.error("WebSocket error: session=%s, %s", session_id, e)
    finally:
        try:
            await ws.close()
        except Exception:
            pass
