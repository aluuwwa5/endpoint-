import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE anything else so HF_HOME/TORCH_HOME are set early
load_dotenv(Path(__file__).parent.parent / ".env")

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    groq_api_key: str = ""
    gemini_api_key: str = ""

    # Azure Speech
    azure_speech_key: str = ""
    azure_speech_region: str = "westeurope"

    # LiveKit
    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = ""
    livekit_api_secret: str = ""

    # Whisper
    whisper_model_size: str = "medium"

    # Booking API
    booking_api_url: str = "https://api.kbtucare.site"
    psychologist_id: str = ""

    # Database (empty = use SQLite locally; set to postgresql://... in Docker)
    database_url: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
